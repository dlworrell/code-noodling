# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:3bbd3a68-3e92-4ea0-b90b-a7a250ab0a41

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone

from .versions import load_versions
from .numerics import ERROR_POLICY, beta_binomial_tail, posterior, threshold
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
from .stages import annotate_snapshots, forecast_eligibility, stage_identity
from .evidence import (hydrate_snapshots, identity, source_facts, reviewed_rule,
                       remaining_pool, scientific_gate, summarize_sources, timestamp,
                       load_geography, geographic_comparison)


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
PROBABILITY_MODEL = "disjoint_tempered_beta_binomial_v2"
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
RCV_PENDING_NOTE = (
    "Ranked-choice contest with no first-choice majority: later rounds decide it, "
    "so the first-choice leader is not necessarily the winner. Rounds are tabulated "
    "only after the final count."
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
    if config.default_rule is not None:
        return config.default_rule
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
    # EA-04/08: reaching a source-wide turnout target never proves zero outstanding.
    # Kept as a deprecated private compatibility entry point; use reviewed pools.
    return None


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
) -> tuple[float | None, float]:
    """Return the predictive tail probability and exact required future share."""
    if remaining_votes <= 0:
        return 0.0, 1.0
    first = threshold(remaining_votes, margin, tie_changes)
    alpha, beta = posterior(change_votes, current_votes, latest_batch_share, latest_batch_votes)
    return beta_binomial_tail(remaining_votes, alpha, beta, first), first / remaining_votes


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
    # Snapshot count supplies no independent validation: repeated observations
    # of a cumulative tally are correlated. This remains an unvalidated rubric.
    raw_score = min(50.0, source.forecast_reliability * 0.5)
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
        f"Unvalidated configured source rubric {source.forecast_reliability}/100; "
        f"{history_points} correlated snapshot(s), with no independent-validation bonus; {observed_votes:,} observed "
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
        rationale = "No valid contemporaneous remaining pool or explicit scenario is available."
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
        if probability is None:
            required_share = None
            rationale = "Numerical budget exceeded; probability withheld under the error policy."
        elif remaining_votes <= 0:
            rationale = "The reviewed remaining pool explicitly confirms zero future decision votes."
        else:
            rationale = (
                f"Experimental, uncalibrated conditional scenario: {probability:.1%} "
                f"selected-boundary change probability across {remaining_votes:,} estimated remaining "
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


def _sensitivity(pool, change, current, margin, batch_share, batch_votes, tie_changes=False, pair=True):
    if not pool or pool.get("kind") != "scenario":
        return []
    rows=[]
    fraction = pool.get("contest_fraction",1) * (pool.get("pair_fraction",1) if pair else 1)
    for label in ("low","base","high"):
        remaining = round(pool[label]*fraction)
        probability, required = _tempered_beta_binomial_probability(change,current,remaining,margin,batch_share,batch_votes,tie_changes)
        rows.append({"scenario":label,"remaining_ballots":pool[label],"remaining_decision_votes":remaining,
                     "conditional_probability":probability,"required_share":required,
                     "interpretation":"sensitivity scenario; uncalibrated; not a confidence interval"})
    return rows


def _pair_decision(
    kind: str,
    current: ChoiceResult,
    challenger: ChoiceResult,
    contest: ContestResult,
    source: SourceConfig,
    latest: Snapshot,
    history: tuple[Snapshot, ...],
    controlling: bool,
    pool: dict | None = None,
) -> DecisionAnalysis:
    margin = current.votes - challenger.votes
    remaining_pair = (round(pool["ballots"] * pool.get("contest_fraction", 1) * pool.get("pair_fraction", 1))
                      if pool and pool.get("ballots") is not None else None)

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
    if len(history) >= 2 and latest.source_metadata.get("delta_kind", contest.metadata.get("delta_kind")) == "new_ballots":
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
    if current.votes + challenger.votes == 0:
        current_state = "No reported votes for this selected pair; its boundary is unresolved."
    elif margin == 0:
        current_state = f"{current.name} and {challenger.name} are tied; the outcome is unresolved."
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
        remaining_pool=pool or {"status": "unknown"},
        event={"scope": "selected_pair", "definition": "selected challenger strictly overtakes selected current side",
               "whole_outcome_probability": False, "tie_policy": "unresolved; no sort-order adjudication",
               "threshold_formula": "floor((M+R)/2)+1", "tie_changes": False},
        numerical_evidence={**ERROR_POLICY,"remaining_pool_sensitivity":_sensitivity(pool,challenger.votes,current.votes,margin,latest_batch_share,latest_batch_votes)},
        batch_evidence={"classification": "new_ballots" if latest_batch_votes is not None else "unknown_or_not_new",
                        "likelihood": "disjoint_history_and_latest_v2", "cumulative_cap": CUMULATIVE_EFFECTIVE_SAMPLE_CAP,
                        "latest_cap": LATEST_BATCH_EFFECTIVE_SAMPLE_CAP, "calibration": "hand-set; not tuned or validated"},
    )


def _majority_decision(
    leader: ChoiceResult,
    contest: ContestResult,
    source: SourceConfig,
    latest: Snapshot,
    history: tuple[Snapshot, ...],
    controlling: bool,
    pool: dict | None = None,
) -> DecisionAnalysis:
    total_votes = sum(choice.votes for choice in contest.choices)
    signed_margin = 2 * leader.votes - total_votes
    has_majority = signed_margin > 0
    margin = abs(signed_margin)
    remaining_valid = round(pool["ballots"] * pool.get("contest_fraction", 1)) if pool and pool.get("ballots") is not None else None

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
    if len(history) >= 2 and latest.source_metadata.get("delta_kind", contest.metadata.get("delta_kind")) == "new_ballots":
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
        numerical_evidence={**ERROR_POLICY,"remaining_pool_sensitivity":_sensitivity(pool,total_votes-leader.votes if has_majority else leader.votes,leader.votes if has_majority else total_votes-leader.votes,margin,latest_batch_share,latest_batch_votes,has_majority,False)},
        batch_evidence={"classification": "new_ballots" if latest_batch_votes else "unknown_or_not_new",
                        "likelihood": "disjoint_history_and_latest_v2", "calibration": "unvalidated hand-set caps"},
        current_state=state,
        current_side=current_side,
        change_side=change_side,
        margin=margin,
        risk=risk,
        margin_history=tuple(history_values),
        remaining_pool=pool or {"status": "unknown"},
        event={"scope": "selected_leader_majority_status", "definition": "selected leader gains or loses strict majority of continuing modeled choices",
               "whole_outcome_probability": False, "tie_policy": "a tie loses a strict majority; final tie disposition unresolved",
               "threshold_formula": "ceil((M+R)/2)" if has_majority else "floor((M+R)/2)+1",
               "tie_changes": has_majority},
    )


def _scope_status(source: SourceConfig, contest: ContestResult) -> tuple[bool, str]:
    if source.scope == "county" and len(contest.counties) > 1:
        return (
            False,
            f"This contest spans {', '.join(contest.counties)}; this source reports "
            "only its own county's share. Add the controlling district or statewide "
            "results before treating the risk as an outcome forecast.",
        )
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
        snapshots = hydrate_snapshots(config, source, annotate_snapshots(
            source.source_id, load_source_snapshots(source), config.result_statuses
        ))
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
        # Last-known per-contest evidence survives a partial/missing office update.
        last_known = {}
        for snapshot in snapshots:
            for contest in snapshot.contests.values():
                last_known[(contest.contest_id, contest.key)] = (snapshot, contest)
        for latest, contest in last_known.values():
            ident = identity(config, source, contest)
            key = ident["canonical_id"] if ident["reviewed"] and ident["role"] == "controlling" else f"{source.source_id}:{contest.contest_id}:{contest.key}"
            candidate = (source, latest, contest)
            existing = selected.get(key)
            if existing is None or (source.priority, contest.ballots) > (
                existing[0].priority,
                existing[2].ballots,
            ):
                selected[key] = candidate

    races: list[RaceAnalysis] = []
    geography = load_geography(config)
    for source, latest, contest in selected.values():
        ranked = _ranked_choices(contest, config)
        ident = identity(config, source, contest)
        rule_spec = reviewed_rule(config, ident["canonical_id"], _infer_rule(source, contest, config))
        rule = rule_spec["rule"]
        source_history = history_map[source.source_id]
        # Preserve every source version, but never derive a batch by subtracting
        # first rankings from an RCV round, or one result version from another.
        history = tuple(
            snapshot for snapshot in source_history
            if (previous := snapshot.contests.get(contest.key)) is not None
            and previous.contest_id == contest.contest_id and previous.name == contest.name
            and (
                snapshot is latest
                or (contest.result_status.tabulation_stage != "unknown"
                    and stage_identity(previous.result_status) == stage_identity(contest.result_status))
            )
        )
        seen_hashes = set()
        history = tuple(s for s in history if not (s.sha256 in seen_hashes or seen_hashes.add(s.sha256)))
        excluded_history = tuple(
            f"{snapshot.path.name}: incompatible or unknown tabulation stage/result version"
            for snapshot in source_history if snapshot not in history
        )
        controlling, scope_note = _scope_status(source, contest)
        controlling = ident["reviewed"] and ident["complete"] and ident["role"] == "controlling"
        if ident["reviewed"]:
            scope_note = "Reviewed full-jurisdiction representation." if controlling else "Reviewed county-only slice or other partial representation."
        else:
            scope_note = scope_note + " Unreviewed identity and jurisdiction coverage; descriptive source record only."
        facts = source_facts(latest, contest, config.information_time)
        if latest is not source_history[-1]:
            facts["coverage_complete"] = False
            facts["coverage_findings"] = ["Contest omitted from the latest source; last-known descriptive version retained."]
        comparison = geographic_comparison(contest, ident, facts, geography)
        valid_votes = sum(c.votes for c in contest.choices)
        data_status = "reported" if valid_votes > 0 else "no_reported_results"
        if valid_votes + (contest.overvotes or 0) + (contest.undervotes or 0) > contest.reported_ballots:
            data_status = "inconsistent_ballot_accounting"
        if comparison["status"] in {"contradictory", "source_vintage_conflict"}:
            data_status = comparison["status"]
            controlling = False
        if (rule in {"winner", "wa_judicial"} and len(ranked) >= 2 and ranked[0].votes == ranked[1].votes
            or rule in {"top_two", "wa_judicial"} and len(ranked) >= 3 and ranked[1].votes == ranked[2].votes
            or rule == "top_four" and len(ranked) >= 5 and ranked[3].votes == ranked[4].votes):
            if data_status == "reported": data_status = "unresolved_tie"
        pool = remaining_pool(config, ident["canonical_id"], ident.get("jurisdictions", []), timestamp(facts.get("count_updated_at") or facts.get("source_published_at")))
        context = {"source": facts, "identity": ident, "rule": rule_spec, "remaining_pool": pool,
                   "geography_comparison": comparison, "data_status": data_status,
                   "quarantined": data_status in {"contradictory", "source_vintage_conflict", "inconsistent_ballot_accounting"},
                   "calibration_status": "unvalidated", "rubric_status": "unvalidated_heuristic"}
        lifecycle_ok, lifecycle_reason = forecast_eligibility(config, contest.result_status, rule)
        gate = scientific_gate(config, context, lifecycle_ok, lifecycle_reason)
        eligible, reason = gate["eligible"], "; ".join(gate["reasons"]) or "Experimental conditional selected-boundary scenario."
        if facts.get("geography", {}).get("approximate"):
            controlling = False
            eligible, reason = False, "Election-day precinct subset / approximate borough assignment; incomplete whole-contest geography."
            gate["eligible"] = False
            gate["reasons"].append(reason)
        scope_note += " " + ("Reviewed canonical identity: " + ident["canonical_id"] if ident["reviewed"] else ident["finding"])
        # Suppress the calculation itself, not just its display. Retain descriptive
        # boundaries and margins with an explicit unestimated result.
        forecast_source = source if eligible else replace(source, expected_final_ballots=None)
        decisions: list[DecisionAnalysis] = []
        if rule == "winner" and len(ranked) >= 2:
            decisions.append(
                _pair_decision(
                    "winner",
                    ranked[0],
                    ranked[1],
                    contest,
                    forecast_source,
                    latest,
                    history,
                    controlling,
                    pool if eligible else None,
                )
            )
        elif rule == "top_four" and len(ranked) >= 5:
            decisions.append(
                _pair_decision(
                    "top_four_cutoff",
                    ranked[3],
                    ranked[4],
                    contest,
                    forecast_source,
                    latest,
                    history,
                    controlling,
                    pool if eligible else None,
                )
            )
        elif rule in {"top_two", "wa_judicial"} and len(ranked) >= 3:
            decisions.append(
                _pair_decision(
                    "top_two_cutoff",
                    ranked[1],
                    ranked[2],
                    contest,
                    forecast_source,
                    latest,
                    history,
                    controlling,
                    pool if eligible else None,
                )
            )
        if rule in {"wa_judicial", "rcv"} and ranked:
            decisions.append(
                _majority_decision(
                    ranked[0], contest, forecast_source, latest, history, controlling, pool if eligible else None
                )
            )
        if rule == "rcv" and ranked:
            # This ratio is a diagnostic of selected raw nonblank rankings, never
            # a statutory Round-1 majority or a final result.
            total = sum(choice.votes for choice in contest.choices)
            diagnostic = _majority_decision(ranked[0], contest, forecast_source, latest, history, False)
            diagnostic = replace(diagnostic, kind="raw_ranking_diagnostic",
                current_state=f"{ranked[0].name} has {ranked[0].votes / total:.2%} of reported nonblank raw first rankings; this is not an interpreted Round-1 majority or final winner." if total else "No raw rankings reported.",
                event={"scope": "raw_first_ranking_diagnostic", "whole_outcome_probability": False,
                       "denominator": "reported nonblank raw first rankings; includes generic write-in",
                       "stopping_rule": False})
            decisions = [diagnostic]
            scope_note += (
                " RCV stage is explicit; a raw first-ranking share is not a final winner "
                "or a tabulation stopping rule. Final allocation to supplied official finalists is separately reproducible; independent candidate eligibility remains unsupported."
            )
            if 2 * ranked[0].votes <= sum(choice.votes for choice in contest.choices):
                scope_note = f"{scope_note} {RCV_PENDING_NOTE}"
        if eligible and any(d.risk.estimated_remaining_votes and d.risk.change_probability is None for d in decisions):
            eligible = False
            reason = "Numerical budget/error policy cannot support this calculation."
            gate["eligible"] = False
            gate["checks"]["numerical_evaluation"] = False
            gate["reasons"].append(reason)
        if not eligible:
            decisions = [replace(decision, risk=replace(
                decision.risk, level="unknown", flip_band=FLIP_BANDS["unknown"],
                change_probability=None, probability_model="not_available",
                reliability_score=0, reliability_grade="Insufficient",
                reliability_rationale="Forecast withheld: " + reason,
                required_share=None, estimated_remaining_votes=None,
                latest_batch_share=None, latest_batch_votes=None,
                rationale="Forecast withheld: " + reason,
            )) for decision in decisions]

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
                overall_risk=_overall_risk(tuple(decisions)) if decisions or rule_spec.get("supported") else "unknown",
                result_status=contest.result_status,
                forecast_eligible=eligible,
                forecast_reason=reason,
                excluded_history=excluded_history,
                canonical_id=ident["canonical_id"], data_status=data_status,
                evidence=context, scientific_eligibility=gate,
            )
        )

    races.sort(key=lambda race: (race.contest_order, race.contest.casefold()))
    as_of = max(snapshot.timestamp for snapshot in all_snapshots).isoformat(
        timespec="minutes"
    )
    from .areas import analyze_areas

    race_tuple = tuple(races)
    return ElectionAnalysis(
        official_versions=load_versions(config),
        title=config.title,
        as_of=as_of,
        races=race_tuple,
        areas=analyze_areas(config, race_tuple),
        snapshots=tuple(
            sorted(all_snapshots, key=lambda item: (item.timestamp, item.source_id))
        ),
        report_mode=config.report_mode,
        forecast_target=config.forecast_target,
        information_time=config.information_time.isoformat() if config.information_time else None,
        election_status=config.election_status,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_summary=summarize_sources(race_tuple),
        coverage_summary={"configured_sources_without_snapshots": [s.source_id for s in config.sources if not history_map[s.source_id]],
                          "watcher_coverage": __import__("json").loads((config.directory/"SOURCES.json").read_text()).get("coverage", {}) if config.directory and (config.directory/"SOURCES.json").exists() else {},
                          "missing_or_incomplete": sum(not r.evidence["source"].get("coverage_complete") for r in race_tuple),
                          "status": "explicit_per_contest; missing sources never imply zero risk"},
    )


def risk_counts(analysis: ElectionAnalysis) -> Counter[str]:
    return Counter(
        race.overall_risk for race in analysis.races if race.controlling
    )
