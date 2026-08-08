from __future__ import annotations

import math
from collections import Counter

from .ingest import load_source_snapshots
from .model import (
    ChoiceResult,
    ContestResult,
    DecisionAnalysis,
    ElectionAnalysis,
    ElectionConfig,
    ElectionReportError,
    RaceAnalysis,
    RiskAssessment,
    Snapshot,
    SourceConfig,
)


RISK_ORDER = {
    "not_applicable": -1,
    "unknown": 0,
    "very_low": 1,
    "low": 2,
    "meaningful": 3,
    "high": 4,
    "toss_up": 5,
}
FLIP_BANDS = {
    "not_applicable": "not applicable",
    "unknown": "not estimable",
    "very_low": "under 2% modeled probability",
    "low": "2–10% modeled probability",
    "meaningful": "10–25% modeled probability",
    "high": "25–40% modeled probability",
    "toss_up": "40% or higher modeled probability",
}
PROBABILITY_MODEL = "tempered_beta_binomial_v1"
# Mail-ballot batches are compositionally different. These caps preserve the observed
# vote share while preventing raw ballot volume from creating false precision.
CUMULATIVE_EFFECTIVE_SAMPLE_CAP = 300
LATEST_BATCH_EFFECTIVE_SAMPLE_CAP = 200
EXACT_TAIL_MAX_REMAINING = 5_000
MEASURE_CHOICES = {
    "yes",
    "no",
    "approved",
    "rejected",
    "accept",
    "reject",
    "maintained",
    "repealed",
}
JUDICIAL_MARKERS = (
    "supreme court",
    "justice position",
    "court of appeals",
    "superior court",
    "district court",
    "electoral district judge",
    "superintendent of public instruction",
)
BROADER_THAN_COUNTY_MARKERS = (
    "u.s. representative",
    "united states representative",
    "legislative district",
    "state senator",
    "supreme court",
    "justice position",
    "court of appeals",
    "superintendent of public instruction",
)


def _is_write_in(choice: ChoiceResult, config: ElectionConfig) -> bool:
    return choice.name.casefold() in config.write_in_labels


def _ranked_choices(
    contest: ContestResult, config: ElectionConfig
) -> tuple[ChoiceResult, ...]:
    return tuple(
        sorted(
            (choice for choice in contest.choices if not _is_write_in(choice, config)),
            key=lambda item: (-item.votes, item.order, item.name.casefold()),
        )
    )


def _infer_rule(
    source: SourceConfig, contest: ContestResult, config: ElectionConfig
) -> str:
    override = config.rule_overrides.get(
        f"{source.source_id}:{contest.contest_id}",
        config.rule_overrides.get(f"*:{contest.contest_id}"),
    )
    if override is not None:
        return override
    name = contest.name.casefold()
    choice_names = {
        choice.name.casefold()
        for choice in contest.choices
        if not _is_write_in(choice, config)
    }
    if "precinct committee officer" in name:
        return "winner"
    if choice_names and choice_names <= MEASURE_CHOICES:
        return "winner"
    if any(marker in name for marker in JUDICIAL_MARKERS):
        return "wa_judicial"
    return "top_two"


def _contest_in_snapshot(
    snapshot: Snapshot, contest_key: str
) -> ContestResult | None:
    return snapshot.contests.get(contest_key)


def _choice_votes(contest: ContestResult, name: str) -> int | None:
    for choice in contest.choices:
        if choice.name == name:
            return choice.votes
    return None


def _source_reported_ballots(snapshot: Snapshot) -> int:
    return max(contest.ballots for contest in snapshot.contests.values())


def _remaining_contest_ballots(
    source: SourceConfig, snapshot: Snapshot, contest: ContestResult
) -> int | None:
    if source.expected_final_ballots is None:
        return None
    reported = _source_reported_ballots(snapshot)
    if reported <= 0 or source.expected_final_ballots <= reported:
        return 0
    return round(
        contest.ballots * (source.expected_final_ballots / reported - 1.0)
    )


def _probability_level(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    if probability >= 0.40:
        return "toss_up"
    if probability >= 0.25:
        return "high"
    if probability >= 0.10:
        return "meaningful"
    if probability >= 0.02:
        return "low"
    return "very_low"


def _tempered_beta_binomial_probability(
    change_votes: int,
    current_votes: int,
    remaining_votes: int,
    margin: int,
    latest_batch_share: float | None,
    latest_batch_votes: int | None,
    tie_changes: bool,
) -> tuple[float, float]:
    """Return the predictive tail probability and exact required future share."""
    if remaining_votes <= 0:
        return 0.0, 1.0
    if tie_changes:
        threshold = math.ceil((remaining_votes + margin) / 2)
    else:
        threshold = math.floor((remaining_votes + margin) / 2) + 1
    required_share = threshold / remaining_votes
    if threshold <= 0:
        return 1.0, required_share
    if threshold > remaining_votes:
        return 0.0, required_share

    observed_total = change_votes + current_votes
    observed_share = change_votes / observed_total if observed_total else 0.5
    cumulative_weight = min(observed_total, CUMULATIVE_EFFECTIVE_SAMPLE_CAP)
    alpha = 1.0 + observed_share * cumulative_weight
    beta = 1.0 + (1.0 - observed_share) * cumulative_weight
    if (
        latest_batch_share is not None
        and latest_batch_votes is not None
        and latest_batch_votes > 0
    ):
        batch_weight = min(latest_batch_votes, LATEST_BATCH_EFFECTIVE_SAMPLE_CAP)
        alpha += latest_batch_share * batch_weight
        beta += (1.0 - latest_batch_share) * batch_weight

    if remaining_votes <= EXACT_TAIL_MAX_REMAINING:
        log_probabilities = []
        for future_change_votes in range(threshold, remaining_votes + 1):
            future_current_votes = remaining_votes - future_change_votes
            log_probability = (
                math.lgamma(remaining_votes + 1)
                - math.lgamma(future_change_votes + 1)
                - math.lgamma(future_current_votes + 1)
                + math.lgamma(future_change_votes + alpha)
                + math.lgamma(future_current_votes + beta)
                - math.lgamma(remaining_votes + alpha + beta)
                + math.lgamma(alpha + beta)
                - math.lgamma(alpha)
                - math.lgamma(beta)
            )
            log_probabilities.append(log_probability)
        maximum = max(log_probabilities)
        probability = math.exp(maximum) * sum(
            math.exp(value - maximum) for value in log_probabilities
        )
        return max(0.0, min(1.0, probability)), required_share

    shape_total = alpha + beta
    mean = remaining_votes * alpha / shape_total
    variance = (
        remaining_votes
        * alpha
        * beta
        * (shape_total + remaining_votes)
        / (shape_total * shape_total * (shape_total + 1.0))
    )
    if variance <= 0:
        return (1.0 if mean >= threshold else 0.0), required_share
    z_score = (threshold - 0.5 - mean) / math.sqrt(variance)
    probability = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return max(0.0, min(1.0, probability)), required_share


def _reliability(
    source: SourceConfig,
    probability: float | None,
    history_points: int,
    observed_votes: int,
    denominator_reconciled: bool,
    controlling: bool,
) -> tuple[int, str, str]:
    """Score input evidence separately from the modeled change probability."""
    if probability is None:
        return (
            0,
            "Insufficient",
            "No remaining-ballot forecast is available, so the probability is not "
            "estimable.",
        )
    history_score = 25 if history_points <= 1 else 65 if history_points == 2 else 85
    sample_score = min(100.0, 20.0 * math.log10(observed_votes + 1.0))
    raw_score = (
        0.55 * source.forecast_reliability
        + 0.25 * history_score
        + 0.20 * sample_score
    )
    adjustments: list[str] = []
    if observed_votes < 500:
        raw_score -= 10.0
        adjustments.append("small decision-vote pool")
    elif observed_votes < 2_000:
        raw_score -= 5.0
        adjustments.append("limited decision-vote pool")
    if denominator_reconciled:
        raw_score -= 5.0
        adjustments.append("reconciled ballot denominator")
    if not controlling:
        raw_score = min(raw_score, 25.0)
        adjustments.append("noncontrolling county slice")
    score = max(0, min(85, round(raw_score)))
    grade = (
        "High"
        if score >= 75
        else "Moderate"
        if score >= 50
        else "Low"
        if score >= 25
        else "Insufficient"
    )
    adjustment_text = (
        f" Adjustments: {', '.join(adjustments)}." if adjustments else ""
    )
    rationale = (
        f"Forecast evidence {source.forecast_reliability}/100; "
        f"{history_points} compatible snapshot(s); {observed_votes:,} observed "
        f"decision votes. Basis: {source.forecast_basis}{adjustment_text}"
    )
    return score, grade, rationale


def _risk_from_requirement(
    margin: int,
    remaining_votes: int | None,
    latest_batch_share: float | None,
    latest_batch_votes: int | None,
    lead_changed: bool,
    change_votes: int,
    current_votes: int,
    source: SourceConfig,
    history_points: int,
    denominator_reconciled: bool,
    controlling: bool,
    tie_changes: bool = False,
) -> RiskAssessment:
    if remaining_votes is None:
        probability = None
        required_share = None
        rationale = "No expected-final-ballot forecast is configured for this source."
    else:
        probability, required_share = _tempered_beta_binomial_probability(
            change_votes,
            current_votes,
            remaining_votes,
            margin,
            latest_batch_share,
            latest_batch_votes,
            tie_changes,
        )
        if remaining_votes <= 0:
            rationale = (
                "The configured final-ballot forecast leaves no estimated decision "
                "votes."
            )
        else:
            rationale = (
                f"The tempered beta-binomial model estimates a {probability:.1%} "
                f"chance of change across {remaining_votes:,} estimated remaining "
                f"decision votes; the change side needs {required_share:.1%}."
            )
        if lead_changed:
            rationale += " The boundary changed in an earlier supplied snapshot."

    level = _probability_level(probability)
    reliability_score, reliability_grade, reliability_rationale = _reliability(
        source,
        probability,
        history_points,
        change_votes + current_votes,
        denominator_reconciled,
        controlling,
    )

    return RiskAssessment(
        level=level,
        flip_band=FLIP_BANDS[level],
        change_probability=probability,
        probability_model=(
            PROBABILITY_MODEL if probability is not None else "not_available"
        ),
        reliability_score=reliability_score,
        reliability_grade=reliability_grade,
        reliability_rationale=reliability_rationale,
        required_share=required_share,
        estimated_remaining_votes=remaining_votes,
        latest_batch_share=latest_batch_share,
        latest_batch_votes=latest_batch_votes,
        lead_changed=lead_changed,
        rationale=rationale,
    )


def _pair_decision(
    kind: str,
    current: ChoiceResult,
    challenger: ChoiceResult,
    contest: ContestResult,
    source: SourceConfig,
    latest: Snapshot,
    history: tuple[Snapshot, ...],
    controlling: bool,
) -> DecisionAnalysis:
    margin = current.votes - challenger.votes
    remaining_ballots = _remaining_contest_ballots(source, latest, contest)
    if remaining_ballots is None:
        remaining_pair = None
    elif contest.ballots <= 0:
        remaining_pair = 0
    else:
        pair_rate = (current.votes + challenger.votes) / contest.ballots
        remaining_pair = round(remaining_ballots * pair_rate)

    margin_history: list[tuple[str, int]] = []
    for snapshot in history:
        historical = _contest_in_snapshot(snapshot, contest.key)
        if historical is None:
            continue
        current_votes = _choice_votes(historical, current.name)
        challenger_votes = _choice_votes(historical, challenger.name)
        if current_votes is None or challenger_votes is None:
            continue
        margin_history.append(
            (snapshot.timestamp.isoformat(timespec="minutes"), current_votes - challenger_votes)
        )
    currently_ahead = margin > 0
    lead_changed = any(
        (value > 0) != currently_ahead for _stamp, value in margin_history[:-1]
    )

    latest_batch_share = None
    latest_batch_votes = None
    if len(history) >= 2:
        previous = _contest_in_snapshot(history[-2], contest.key)
        if previous is not None:
            previous_current = _choice_votes(previous, current.name)
            previous_challenger = _choice_votes(previous, challenger.name)
            if previous_current is not None and previous_challenger is not None:
                current_delta = current.votes - previous_current
                challenger_delta = challenger.votes - previous_challenger
                pair_delta = current_delta + challenger_delta
                if current_delta >= 0 and challenger_delta >= 0 and pair_delta > 0:
                    latest_batch_votes = pair_delta
                    latest_batch_share = challenger_delta / pair_delta

    risk = _risk_from_requirement(
        margin,
        remaining_pair,
        latest_batch_share,
        latest_batch_votes,
        lead_changed,
        challenger.votes,
        current.votes,
        source,
        len(margin_history),
        contest.ballots != contest.reported_ballots,
        controlling,
    )
    if margin == 0:
        current_state = f"{current.name} and {challenger.name} are tied."
    else:
        vote_word = "vote" if margin == 1 else "votes"
        current_state = (
            f"{current.name} leads {challenger.name} by {margin:,} {vote_word}."
        )
    return DecisionAnalysis(
        kind=kind,
        current_state=current_state,
        current_side=current.name,
        change_side=challenger.name,
        margin=margin,
        risk=risk,
        margin_history=tuple(margin_history),
    )


def _majority_decision(
    leader: ChoiceResult,
    contest: ContestResult,
    source: SourceConfig,
    latest: Snapshot,
    history: tuple[Snapshot, ...],
    controlling: bool,
) -> DecisionAnalysis:
    total_votes = sum(choice.votes for choice in contest.choices)
    signed_margin = 2 * leader.votes - total_votes
    has_majority = signed_margin > 0
    margin = abs(signed_margin)
    remaining_ballots = _remaining_contest_ballots(source, latest, contest)
    if remaining_ballots is None:
        remaining_valid = None
    elif contest.ballots <= 0:
        remaining_valid = 0
    else:
        valid_rate = total_votes / contest.ballots
        remaining_valid = round(remaining_ballots * valid_rate)

    history_values: list[tuple[str, int]] = []
    for snapshot in history:
        historical = _contest_in_snapshot(snapshot, contest.key)
        if historical is None:
            continue
        historical_leader = _choice_votes(historical, leader.name)
        if historical_leader is None:
            continue
        historical_total = sum(choice.votes for choice in historical.choices)
        value = 2 * historical_leader - historical_total
        history_values.append((snapshot.timestamp.isoformat(timespec="minutes"), value))
    lead_changed = any(
        (value > 0) != has_majority for _stamp, value in history_values[:-1]
    )

    latest_batch_share = None
    latest_batch_votes = None
    if len(history) >= 2:
        previous = _contest_in_snapshot(history[-2], contest.key)
        if previous is not None:
            previous_leader = _choice_votes(previous, leader.name)
            if previous_leader is not None:
                previous_total = sum(choice.votes for choice in previous.choices)
                leader_delta = leader.votes - previous_leader
                total_delta = total_votes - previous_total
                if leader_delta >= 0 and total_delta > 0 and leader_delta <= total_delta:
                    latest_batch_votes = total_delta
                    latest_batch_share = (
                        (total_delta - leader_delta) / total_delta
                        if has_majority
                        else leader_delta / total_delta
                    )

    change_side = "all other choices" if has_majority else leader.name
    current_side = leader.name if has_majority else "no majority"
    state = (
        f"{leader.name} has {leader.votes / total_votes:.2%} and is above 50%."
        if has_majority and total_votes
        else f"{leader.name} has {leader.votes / total_votes:.2%}; no candidate is above 50%."
        if total_votes
        else "No valid votes are reported."
    )
    risk = _risk_from_requirement(
        margin,
        remaining_valid,
        latest_batch_share,
        latest_batch_votes,
        lead_changed,
        total_votes - leader.votes if has_majority else leader.votes,
        leader.votes if has_majority else total_votes - leader.votes,
        source,
        len(history_values),
        contest.ballots != contest.reported_ballots,
        controlling,
        tie_changes=has_majority,
    )
    return DecisionAnalysis(
        kind="majority_status",
        current_state=state,
        current_side=current_side,
        change_side=change_side,
        margin=margin,
        risk=risk,
        margin_history=tuple(history_values),
    )


def _scope_status(source: SourceConfig, contest: ContestResult) -> tuple[bool, str]:
    name = contest.name.casefold()
    if source.scope == "county" and any(
        marker in name for marker in BROADER_THAN_COUNTY_MARKERS
    ):
        return (
            False,
            "This is a county-only slice of a broader contest. Add the controlling "
            "district or statewide export before treating the risk as an outcome forecast.",
        )
    return True, "This source is treated as controlling for the reported contest."


def _overall_risk(decisions: tuple[DecisionAnalysis, ...]) -> str:
    if not decisions:
        return "not_applicable"
    return max((decision.risk.level for decision in decisions), key=RISK_ORDER.get)


def analyze_election(config: ElectionConfig) -> ElectionAnalysis:
    source_map = {source.source_id: source for source in config.sources}
    history_map: dict[str, tuple[Snapshot, ...]] = {}
    all_snapshots: list[Snapshot] = []
    for source in config.sources:
        snapshots = load_source_snapshots(source)
        history_map[source.source_id] = snapshots
        all_snapshots.extend(snapshots)
    if not all_snapshots:
        locations = ", ".join(str(source.path) for source in config.sources)
        raise ElectionReportError(f"no dated CSV snapshots found in: {locations}")

    selected: dict[str, tuple[SourceConfig, Snapshot, ContestResult]] = {}
    for source in config.sources:
        snapshots = history_map[source.source_id]
        if not snapshots:
            continue
        latest = snapshots[-1]
        for key, contest in latest.contests.items():
            candidate = (source, latest, contest)
            existing = selected.get(key)
            if existing is None or (source.priority, contest.ballots) > (
                existing[0].priority,
                existing[2].ballots,
            ):
                selected[key] = candidate

    races: list[RaceAnalysis] = []
    for source, latest, contest in selected.values():
        ranked = _ranked_choices(contest, config)
        rule = _infer_rule(source, contest, config)
        history = history_map[source.source_id]
        controlling, scope_note = _scope_status(source, contest)
        decisions: list[DecisionAnalysis] = []
        if rule == "winner" and len(ranked) >= 2:
            decisions.append(
                _pair_decision(
                    "winner",
                    ranked[0],
                    ranked[1],
                    contest,
                    source,
                    latest,
                    history,
                    controlling,
                )
            )
        elif rule in {"top_two", "wa_judicial"} and len(ranked) >= 3:
            decisions.append(
                _pair_decision(
                    "top_two_cutoff",
                    ranked[1],
                    ranked[2],
                    contest,
                    source,
                    latest,
                    history,
                    controlling,
                )
            )
        if rule == "wa_judicial" and ranked:
            decisions.append(
                _majority_decision(
                    ranked[0], contest, source, latest, history, controlling
                )
            )

        races.append(
            RaceAnalysis(
                contest_id=contest.contest_id,
                contest=contest.name,
                contest_order=contest.order,
                source_id=source.source_id,
                jurisdiction=source.jurisdiction,
                snapshot=latest.timestamp.isoformat(timespec="minutes"),
                ballots=contest.ballots,
                reported_ballots=contest.reported_ballots,
                overvotes=contest.overvotes,
                undervotes=contest.undervotes,
                valid_votes=sum(choice.votes for choice in contest.choices),
                rule=rule,
                controlling=controlling,
                scope_note=scope_note,
                choices=tuple(sorted(contest.choices, key=lambda item: (-item.votes, item.order))),
                decisions=tuple(decisions),
                overall_risk=_overall_risk(tuple(decisions)),
            )
        )

    races.sort(key=lambda race: (race.contest_order, race.contest.casefold()))
    as_of = max(snapshot.timestamp for snapshot in all_snapshots).isoformat(
        timespec="minutes"
    )
    return ElectionAnalysis(
        title=config.title,
        as_of=as_of,
        races=tuple(races),
        snapshots=tuple(
            sorted(all_snapshots, key=lambda item: (item.timestamp, item.source_id))
        ),
    )


def risk_counts(analysis: ElectionAnalysis) -> Counter[str]:
    return Counter(
        race.overall_risk for race in analysis.races if race.controlling
    )
