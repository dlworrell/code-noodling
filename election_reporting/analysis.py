from __future__ import annotations

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
    "very_low": "under 5%",
    "low": "5–15%",
    "meaningful": "15–35%",
    "high": "25–45%",
    "toss_up": "35–65%",
}
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


def _risk_from_requirement(
    margin: int,
    remaining_votes: int | None,
    latest_batch_share: float | None,
    latest_batch_votes: int | None,
    lead_changed: bool,
) -> RiskAssessment:
    required_share: float | None = None
    if remaining_votes is None:
        level = "unknown"
        rationale = "No expected-final-ballot forecast is configured for this source."
    elif remaining_votes <= 0:
        level = "very_low"
        rationale = "The configured final-ballot forecast leaves no estimated votes."
    elif margin <= 0:
        required_share = 0.5
        level = "toss_up"
        rationale = "The current decision boundary is tied."
    else:
        required_share = 0.5 + margin / (2.0 * remaining_votes)
        if required_share <= 0.505:
            level = "toss_up"
        elif required_share <= 0.52:
            level = "high"
        elif required_share <= 0.55:
            level = "meaningful"
        elif required_share <= 0.60:
            level = "low"
        else:
            level = "very_low"
        rationale = (
            f"The change side needs about {required_share:.1%} of the estimated "
            "remaining decision votes."
        )
        if margin <= 3 and remaining_votes >= margin:
            level = "toss_up"
            rationale += " The absolute margin is three votes or fewer."

    if lead_changed and level not in {"unknown", "not_applicable"}:
        level = "toss_up"
        rationale += " The lead changed in an earlier supplied snapshot."
    elif (
        required_share is not None
        and latest_batch_share is not None
        and latest_batch_votes is not None
        and latest_batch_votes > 0
    ):
        if latest_batch_share >= required_share + 0.005:
            promotion = {
                "very_low": "low",
                "low": "meaningful",
                "meaningful": "high",
                "high": "toss_up",
                "toss_up": "toss_up",
            }
            level = promotion.get(level, level)
            rationale += " The latest batch exceeded that required share."
        elif latest_batch_share <= 0.47:
            demotion = {
                "toss_up": "high",
                "high": "meaningful",
                "meaningful": "low",
                "low": "very_low",
                "very_low": "very_low",
            }
            level = demotion.get(level, level)
            rationale += " The latest batch moved materially toward the current side."

    return RiskAssessment(
        level=level,
        flip_band=FLIP_BANDS[level],
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
    lead_changed = any(value <= 0 for _stamp, value in margin_history[:-1])

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
    )
    return DecisionAnalysis(
        kind=kind,
        current_state=f"{current.name} leads {challenger.name} by {margin:,} votes.",
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
        decisions: list[DecisionAnalysis] = []
        if rule == "winner" and len(ranked) >= 2:
            decisions.append(
                _pair_decision(
                    "winner", ranked[0], ranked[1], contest, source, latest, history
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
                )
            )
        if rule == "wa_judicial" and ranked:
            decisions.append(
                _majority_decision(ranked[0], contest, source, latest, history)
            )

        controlling, scope_note = _scope_status(source, contest)
        races.append(
            RaceAnalysis(
                contest_id=contest.contest_id,
                contest=contest.name,
                contest_order=contest.order,
                source_id=source.source_id,
                jurisdiction=source.jurisdiction,
                snapshot=latest.timestamp.isoformat(timespec="minutes"),
                ballots=contest.ballots,
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
