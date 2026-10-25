# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:6b2bb0e3-2d47-4831-af04-dc015f347f33

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from . import __version__
from .analysis import FLIP_BANDS, PROBABILITY_MODEL, risk_counts
from .numerics import ERROR_POLICY
from .model import DecisionAnalysis, ElectionAnalysis, ElectionReportError, RaceAnalysis, ResultStatus


PUBLIC_REPORT_BASE = "https://github.com/dlworrell/code-noodling/blob/main/"
PUBLIC_REPORT_URL = "https://github.com/dlworrell/code-noodling/tree/main/election-analysis/reports"


def public_report_url(output_dir: Path) -> str:
    """The published Markdown report's URL for a repository-relative output directory."""
    if output_dir.is_absolute() or ".." in output_dir.parts:
        return PUBLIC_REPORT_URL
    return f"{PUBLIC_REPORT_BASE}{output_dir.as_posix().strip('/')}/election-analysis.md"


def _risk_label(value: str) -> str:
    return value.replace("_", " ").title()


def _decision_payload(decision: object) -> dict[str, object]:
    risk = decision.risk
    return {
        "kind": decision.kind,
        "event": decision.event,
        "remaining_pool": decision.remaining_pool,
        "batch_evidence": decision.batch_evidence,
        "numerical_evidence": decision.numerical_evidence,
        "current_state": decision.current_state,
        "current_side": decision.current_side,
        "change_side": decision.change_side,
        "margin": decision.margin,
        "risk": {
            "level": risk.level,
            "flip_band": risk.flip_band,
            "change_probability": risk.change_probability,
            "probability_model": risk.probability_model,
            "reliability_score": risk.reliability_score,
            "reliability_grade": risk.reliability_grade,
            "reliability_rationale": risk.reliability_rationale,
            "required_share": risk.required_share,
            "estimated_remaining_votes": risk.estimated_remaining_votes,
            "latest_batch_share": risk.latest_batch_share,
            "latest_batch_votes": risk.latest_batch_votes,
            "lead_changed": risk.lead_changed,
            "rationale": risk.rationale,
        },
        "margin_history": [
            {"snapshot": stamp, "margin": margin}
            for stamp, margin in decision.margin_history
        ],
    }


def analysis_payload(analysis: ElectionAnalysis) -> dict[str, object]:
    payload = _base_payload(analysis)
    if analysis.areas is not None:
        payload["areas"] = _areas_payload(analysis.areas)
    return payload


def _status_payload(status: ResultStatus) -> dict[str, object]:
    return {
        "lifecycle": status.lifecycle,
        "tabulation_stage": status.tabulation_stage,
        "authority_url": status.authority_url or None,
        "evidence_as_of": status.evidence_as_of.isoformat() if status.evidence_as_of else None,
        "result_version": status.result_version or None,
        "round_id": status.round_id or None,
    }


def _report_status(analysis: ElectionAnalysis) -> str:
    return (
        f"Report mode: {analysis.report_mode}. Forecast target: {analysis.forecast_target}. "
        f"Information time: {analysis.information_time or 'unknown'}. "
        f"Election lifecycle: {analysis.election_status.lifecycle}. "
        "Experimental and uncalibrated. A versioned evidence gate checks lifecycle, source freshness, coverage, provenance, contest identity, rules and remaining-pool basis. No held-out calibration has validated predictive accuracy. "
        + _source_coverage_text(analysis)
    )


def _source_coverage_text(analysis: ElectionAnalysis) -> str:
    facts = [r.evidence.get("source", {}) for r in analysis.races]
    publications = sorted({f["source_published_at"] for f in facts if f.get("source_published_at")})
    ages = [f["source_age_hours"] for f in facts if f.get("source_age_hours") is not None]
    publication_range = f"{publications[0]} to {publications[-1]}" if publications else "unknown"
    age_range = f"{min(ages):.1f} to {max(ages):.1f} hours at information time" if ages else "unknown"
    missing_sources = analysis.coverage_summary.get("configured_sources_without_snapshots", [])
    return (f"Source publication range: {publication_range}; known source-age range: {age_range}. "
            f"Unknown source age: {sum(f.get('source_age_hours') is None for f in facts)} contest record(s); "
            f"unknown count-update time: {sum(not f.get('count_updated_at') for f in facts)}; "
            f"incomplete/unverified coverage: {sum(f.get('coverage_complete') is not True for f in facts)}. "
            f"Configured sources without snapshots: {', '.join(missing_sources) or 'none'}. "
            "Retrieval and report generation times do not advance a source's count clock.")


def _source_context(race: RaceAnalysis) -> str:
    source = race.evidence.get("source", {})
    age = source.get("source_age_hours")
    return (f"Source published {source.get('source_published_at') or 'unknown'}; "
            f"count updated {source.get('count_updated_at') or 'unknown'}; "
            f"age {f'{age:.1f} hours' if age is not None else 'unknown'}; "
            f"coverage {'complete' if source.get('coverage_complete') is True else 'incomplete/unverified'}; "
            f"data {race.data_status}; identity {'reviewed' if race.evidence.get('identity', {}).get('reviewed') else 'unreviewed'}; "
            f"office/election rule {'supported' if race.evidence.get('rule', {}).get('supported') else 'unreviewed/unsupported'}; "
            f"provenance {source.get('source_url') or 'unknown'}; "
            f"input SHA-256 {source.get('snapshot_sha256') or 'unknown'}.")


def _pool_context(decision: DecisionAnalysis) -> str:
    pool = decision.remaining_pool
    counts = (f"low/base/high {pool['low']:,}/{pool['base']:,}/{pool['high']:,} ballots"
              if pool.get("kind") == "scenario" and all(k in pool for k in ("low", "base", "high"))
              else f"{pool['ballots']:,} {pool.get('ballot_unit', 'ballots')}" if pool.get("ballots") is not None
              else "unknown count")
    sensitivity = decision.numerical_evidence.get("remaining_pool_sensitivity", [])
    conditional = "; ".join(f"R={r['remaining_ballots']:,}: {_probability_label(r['conditional_probability'])}" for r in sensitivity)
    return (f"Remaining pool: {pool.get('status', 'unknown')}, {counts}, as of {pool.get('as_of') or 'unknown'}. "
            f"Jurisdictions: {', '.join(pool.get('jurisdictions', [])) or 'unknown'}; "
            f"ballot categories: {json.dumps(pool['categories'], sort_keys=True) if pool.get('categories') else pool.get('category_basis') or 'unreviewed'}. "
            f"Basis: {pool.get('basis') or pool.get('authority_url') or pool.get('reason') or 'unknown'}. "
            f"Contest/pair participation: {pool.get('contest_fraction', 'unknown')}/{pool.get('pair_fraction', 'unknown')}; "
            f"{pool.get('participation_basis') or 'mapping unreviewed'}. "
            + (f"Acceptance/cure assumption: {pool['acceptance_assumption']}. " if pool.get('acceptance_assumption') else "")
            + (f"Sensitivity: {conditional}. " if conditional else "")
            + "Sensitivity is conditional and uncalibrated; it is not a confidence interval.")


def _event_label(decision: DecisionAnalysis) -> str:
    if decision.event.get("scope") == "selected_pair":
        return f"Selected pair: {decision.change_side} strictly overtakes {decision.current_side}"
    return str(decision.event.get("definition") or decision.event.get("scope") or "Unresolved event")


def _status_text(status: ResultStatus) -> str:
    text = f"Lifecycle: {status.lifecycle}; tabulation stage: {status.tabulation_stage}."
    if status.result_version:
        text += f" Result version: {status.result_version}; round: {status.round_id or 'not applicable'}."
    if status.authority_url:
        stamp = status.evidence_as_of.isoformat() if status.evidence_as_of else "unknown"
        text += f" Authority: {status.authority_url}; evidence as of {stamp}."
    return text


def _areas_payload(areas) -> dict[str, object]:
    return {
        "source_id": areas.source_id,
        "snapshots": areas.snapshots,
        "reconciliation": areas.reconciliation,
        "county_breakdown": areas.breakdown,
        "county_changes": areas.county_changes,
        "precinct_changes": areas.precinct_changes,
        "remaining_ballot_flags": areas.remaining,
        "notes": areas.notes,
    }


def county_breakdown_csv(analysis: ElectionAnalysis) -> str | None:
    """Every race's per-county votes: one row per race, county and choice."""
    if analysis.areas is None or not analysis.areas.breakdown:
        return None
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Contest", "Contest ID", "Snapshot", "County", "Ballots", "Choice", "Votes", "County leader"])
    for race in analysis.areas.breakdown:
        for row in race["counties"]:
            for choice in race["choices"]:
                writer.writerow([race["contest"], race["contest_id"], race["snapshot"], row["county"],
                                 row["ballots"], choice, row["votes"][choice], row["leader"]])
    return buffer.getvalue()


def _area_table(header: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_table_text(str(cell)) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _signed(value: int) -> str:
    return f"{value:+,}"


def _area_nodes(areas) -> list[dict[str, object]]:
    nodes = [_node("areas-heading", "heading2", "Geography: county and precinct results", level=2)]
    snapshots = areas.snapshots
    nodes.append(_node(
        "areas-status", "paragraph",
        f"County snapshots: {len(snapshots.get('county', []))}; precinct snapshots: "
        f"{len(snapshots.get('precinct', []))}. Every race's per-county votes are in county-breakdown.csv; "
        "precinct rows are in the election's areas/ directory.",
    ))
    rec = areas.reconciliation
    if rec:
        county = rec.get("county_vs_statewide", {})
        precinct = rec.get("precinct_vs_county", {})
        text = (
            f"Reconciliation of the {rec.get('county_snapshot')} county snapshot: "
            f"{county.get('checked', 0):,} contest-choice totals compared with the statewide Summary Results, "
            f"{len(county.get('mismatches', [])):,} mismatches."
        )
        if "checked" in precinct:
            text += (
                f" Precincts against counties: {precinct['checked']:,} comparisons, "
                f"{len(precinct['mismatches']):,} mismatches; "
                f"{len(precinct.get('precinct_detail_missing', [])):,} contest-county pairs have county votes but no "
                "precinct rows in the state file."
            )
        else:
            text += f" Precincts: {precinct.get('status', 'not checked')}."
        nodes.append(_node("areas-reconciliation", "paragraph", text))
        mismatches = county.get("mismatches", []) + precinct.get("mismatches", [])
        if mismatches:
            nodes.append(_node("areas-mismatches", "paragraph", _area_table(
                ("Contest", "Area", "Choice", "Difference"),
                [(m["contest"], m.get("area", ""), m.get("choice", ""), _signed(m["difference"] or 0))
                 for m in mismatches[:25]],
            )))
        missing = precinct.get("precinct_detail_missing", [])
        if missing:
            counties = sorted({m["county"] for m in missing})
            nodes.append(_node(
                "areas-precinct-missing", "paragraph",
                f"Data-quality finding: the state precinct file has no precinct rows for {len(missing)} "
                f"contest-county pairs in {', '.join(counties)}; their precinct-level changes "
                "cannot be shown from the state file.",
            ))
    for level, changes in (("county", areas.county_changes), ("precinct", areas.precinct_changes)):
        title = "County" if level == "county" else "Precinct"
        if "status" in changes:
            nodes.append(_node(f"areas-{level}-changes", "paragraph", f"{title} changes: {changes['status']}."))
            continue
        nodes.append(_node(
            f"areas-{level}-changes-heading", "heading3",
            f"{title}s with the largest change, {changes['from']} to {changes['to']}", level=3,
        ))
        nodes.append(_node(f"areas-{level}-new-ballots", "paragraph", _area_table(
            (title, "Reported tally delta (classification unknown)"),
            [(item["area"], f"{item['new_ballots']:,}") for item in changes["largest_new_ballots"]],
        )))
        rows = []
        for race in changes["close_races"]:
            for swing in race["swings"]:
                rows.append((race["contest"], f"{race['current_side']} vs {race['change_side']}",
                             swing["area"], f"{swing['new_ballots']:,}", _signed(swing["margin_change"])))
        if rows:
            nodes.append(_node(f"areas-{level}-swings", "paragraph", _area_table(
                ("Contest", "Decision", title, "Reported tally delta (classification unknown)", "Margin change"), rows,
            )))
    if areas.remaining:
        nodes.append(_node("areas-remaining-heading", "heading3",
                           "Could the county with the most remaining ballots change the outcome?", level=3))
        nodes.append(_node("areas-remaining", "paragraph", _area_table(
            ("Contest", "County", "Remaining ballots", "Needed share", "County's current share", "Assessment"),
            [(f["contest"], f.get("county", ""), f"{f['remaining_ballots']:,}" if "remaining_ballots" in f else "",
              f"{f['needed_share']:.1%}" if "needed_share" in f else "",
              f"{f['county_change_side_share']:.1%}" if "county_change_side_share" in f else "",
              f["status"]) for f in areas.remaining],
        )))
    for index, note in enumerate(areas.notes, start=1):
        nodes.append(_node(f"areas-note-{index}", "paragraph", note))
    return nodes


def _base_payload(analysis: ElectionAnalysis) -> dict[str, object]:
    return {
        "schema_version": 4,
        "generated_at": analysis.generated_at,
        "source_summary": analysis.source_summary,
        "coverage_summary": analysis.coverage_summary,
        "official_result_versions": list(analysis.official_versions),
        "tool_version": __version__,
        "title": analysis.title,
        "as_of": analysis.as_of,
        "report_mode": analysis.report_mode,
        "forecast_target": analysis.forecast_target,
        "information_time": analysis.information_time,
        "election_status": _status_payload(analysis.election_status),
        "methodology": {
            "probability_status": (
                "experimental, uncalibrated conditional selected-boundary scenario"
            ),
            "probability_model": PROBABILITY_MODEL,
            "tail_evaluation": (
                "full-support normalized log recurrence; withhold above numerical budget"
            ),
            "numerical_error_policy": ERROR_POLICY,
            "likelihood": "preceding history excludes the separately weighted latest verified new-ballot batch",
            "calibration": "none; caps and rubric are hand-set and unvalidated",
            "overall_risk": "maximum selected-boundary triage band, not P(any outcome changes)",
            "effective_sample_caps": {
                "cumulative_votes": 300,
                "latest_batch_votes": 200,
            },
            "risk_bands": FLIP_BANDS,
            "remaining_votes": (
                "contemporaneous official category counts or declared low/base/high sensitivity scenarios with explicit contest/pair participation; no final-total backfill"
            ),
            "stage_policy": "ADR-003/004; versioned scientific evidence gate withholds unsupported calculations",
            "validation_status": "experimental_uncalibrated",
            "xlsx_ballot_denominator": (
                "maximum of reported Ballots Cast and valid votes plus overvotes "
                "plus undervotes"
            ),
        },
        "inputs": [
            {
                "source_id": snapshot.source_id,
                "path": snapshot.path.as_posix(),
                "timestamp": snapshot.timestamp.isoformat(timespec="minutes"),
                "timestamp_interpretation": "legacy filename label; timezone may be unknown",
                "source_evidence": snapshot.source_metadata,
                "sha256": snapshot.sha256,
                "contests": len(snapshot.contests),
                "result_versions": [
                    {"contest_id": contest.contest_id, **_status_payload(contest.result_status)}
                    for contest in snapshot.contests.values()
                ],
            }
            for snapshot in analysis.snapshots
        ],
        "races": [
            {
                "contest_id": race.contest_id,
                "canonical_id": race.canonical_id,
                "data_status": race.data_status,
                "evidence": race.evidence,
                "scientific_eligibility": race.scientific_eligibility,
                "contest": race.contest,
                "contest_order": race.contest_order,
                "source_id": race.source_id,
                "jurisdiction": race.jurisdiction,
                "snapshot": race.snapshot,
                "ballots": race.ballots,
                "reported_ballots": race.reported_ballots,
                "overvotes": race.overvotes,
                "undervotes": race.undervotes,
                "valid_votes": race.valid_votes,
                "rule": race.rule,
                "controlling": race.controlling,
                "scope_note": race.scope_note,
                "overall_risk": race.overall_risk,
                "result_status": _status_payload(race.result_status),
                "forecast_eligibility": {
                    "eligible": race.forecast_eligible,
                    "scope": "experimental_scenario_scientific_gate_v1",
                    "reason": race.forecast_reason,
                },
                "excluded_history": list(race.excluded_history),
                "choices": [
                    {"name": choice.name, "votes": choice.votes, "order": choice.order}
                    for choice in race.choices
                ],
                "decisions": [_decision_payload(item) for item in race.decisions],
            }
            for race in analysis.races
        ],
    }


def _slug(value: str) -> str:
    slug = "-".join(re.findall(r"[a-z0-9]+", value.casefold()))
    return slug[:72] or "race"


def _probability_label(value: float | None) -> str:
    if value is None:
        return "Not estimable"
    if value < 0.001:
        return "<0.1%"
    return f"{value:.1%}"


def _table_text(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _at_glance_table(analysis: ElectionAnalysis) -> str:
    rows: list[tuple[tuple[object, ...], str]] = []
    for race in analysis.races:
        if not race.decisions:
            scope = "Controlling" if race.controlling else "Partial/unreviewed"
            order = (
                not race.controlling,
                True,
                0.0,
                race.contest.casefold(),
                "not_applicable",
            )
            row = (
                f"| Not estimable | Unvalidated | {_table_text(race.contest)} | "
                f"No supported modeled boundary | — | {scope} | {_table_text(_source_context(race))} |"
            )
            rows.append((order, row))
        for decision in race.decisions:
            probability = decision.risk.change_probability
            order = (
                not race.controlling,
                probability is None,
                -(probability or 0.0),
                race.contest.casefold(),
                decision.kind,
            )
            reliability = (
                f"{decision.risk.reliability_score}/100 "
                f"{decision.risk.reliability_grade}"
            )
            boundary = _event_label(decision)
            current = (
                f"{decision.current_side} +{decision.margin:,} over "
                f"{decision.change_side}"
            )
            scope = "Controlling" if race.controlling else "Partial/unreviewed"
            row = (
                f"| {_probability_label(probability)} | {reliability} | "
                f"{_table_text(race.contest)} | {_table_text(boundary)} | "
                f"{_table_text(current)} | {scope} | "
                f"{_table_text(_source_context(race) + ' ' + _pool_context(decision))} |"
            )
            rows.append((order, row))
    rows.sort(key=lambda item: item[0])
    header = [
        "| Experimental conditional probability (uncalibrated) | Unvalidated rubric | Race | Selected boundary | Current margin | Scope | Evidence basis |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    return "\n".join(header + [row for _order, row in rows])


def _facebook_decisions(
    analysis: ElectionAnalysis,
    *,
    precinct_committee_officer: bool,
) -> list[tuple[RaceAnalysis, DecisionAnalysis]]:
    marker = "precinct committee officer"
    items: list[tuple[RaceAnalysis, DecisionAnalysis]] = []
    for race in analysis.races:
        is_precinct_race = marker in race.contest.casefold()
        if not race.controlling or is_precinct_race != precinct_committee_officer:
            continue
        for decision in race.decisions:
            if decision.risk.change_probability is not None:
                items.append((race, decision))
    return sorted(
        items,
        key=lambda item: (
            -(item[1].risk.change_probability or 0.0),
            item[0].contest.casefold(),
            item[1].kind,
        ),
    )


def _facebook_decision_item(
    race: RaceAnalysis,
    decision: DecisionAnalysis,
) -> str:
    boundary = _event_label(decision)
    current_state = decision.current_state
    if decision.kind in {"winner", "top_two_cutoff", "top_four_cutoff"}:
        if decision.margin == 0:
            current_state = (
                f"{decision.current_side} and {decision.change_side} are tied at "
                "this boundary."
            )
        else:
            vote_word = "vote" if decision.margin == 1 else "votes"
            current_state = (
                f"{decision.current_side} leads {decision.change_side} by "
                f"{decision.margin:,} {vote_word}."
            )
    return (
        f"• {race.contest}\n"
        f"  {boundary}: {current_state} "
        f"Experimental, uncalibrated conditional probability: "
        f"{_probability_label(decision.risk.change_probability)}. "
        f"Unvalidated evidence rubric: {decision.risk.reliability_score}/100 "
        f"({decision.risk.reliability_grade}). Event scope: {decision.event.get('scope', 'unresolved')}; selected boundary only, uncalibrated.\n"
        f"  {_source_context(race)}\n  {_pool_context(decision)}"
    )


def facebook_summary(analysis: ElectionAnalysis, report_url: str = PUBLIC_REPORT_URL) -> str:
    """Build a compact, plain-text report suitable for a Facebook post."""
    counts = risk_counts(analysis)
    controlling_races = [race for race in analysis.races if race.controlling]
    modeled_races = [race for race in controlling_races
                     if any(d.risk.change_probability is not None for d in race.decisions)]
    modeled_decisions = [
        decision
        for race in controlling_races
        for decision in race.decisions
        if decision.risk.change_probability is not None
    ]
    if analysis.report_mode != "current" or not modeled_decisions:
        # EA-01: a historical correction must not retain a prospective headline or
        # reinterpret withheld/null probabilities as a table of low-risk races.
        heading = "HISTORICAL ELECTION RESULTS" if analysis.report_mode == "historical" else "ELECTION RESULTS"
        boundaries = sorted(
            ((race, decision) for race in controlling_races for decision in race.decisions
             if race.valid_votes > 0 and race.result_status.lifecycle != "unknown"
             and "precinct committee officer" not in race.contest.casefold()),
            key=lambda item: (item[0].result_status.lifecycle == "unknown", item[1].margin, item[0].contest),
        )[:6]
        lines = [
            f"🗳️ {heading} — FORECASTS WITHHELD", "", analysis.title,
            _report_status(analysis),
            f"Latest input label: {analysis.as_of}; individual sources may be older.",
            _status_text(analysis.election_status), "",
            f"Reviewed {len(analysis.races)} source representations of contests; overlapping jurisdictions may appear separately. Unestimated probabilities are not numerical zero; "
            "no numerical risk ranking is published.", "", "DESCRIPTIVE MARGINS WITH DATED STATUS EVIDENCE",
        ]
        for race, decision in boundaries:
            lines.extend([f"• {race.contest}: {decision.current_state}", _status_text(race.result_status), _source_context(race)])
        lines += ["", "WITHHOLDING REASONS"]
        reasons = Counter(race.forecast_reason for race in controlling_races if race.decisions)
        lines.extend(f"• {count} contest(s): {reason}" for reason, count in sorted(reasons.items()))
        if analysis.report_mode == "historical":
            lines += ["", "This audit correction supersedes earlier prospective summaries. "
                      "Original inputs and the audited publication remain in repository history (audit #45)."]
        lines += ["", "Descriptive counts retain their source limits; this is not an official result, a race call, "
                  "or a validated forecast. Recount and litigation probabilities are unestimated.",
                  "", "Full race-by-race report and methodology:", report_url]
        return "\n".join(lines) + "\n"
    reliability_counts = Counter(
        decision.risk.reliability_grade for decision in modeled_decisions
    )
    reconciled_races = [
        race for race in controlling_races if race.ballots != race.reported_ballots
    ]
    other_decisions = _facebook_decisions(
        analysis,
        precinct_committee_officer=False,
    )[:6]
    precinct_decisions = [
        item
        for item in _facebook_decisions(
            analysis,
            precinct_committee_officer=True,
        )
        if (item[1].risk.change_probability or 0.0) >= 0.10
    ][:3]

    lines = [
        "🗳️ ELECTION RESULTS: WHAT COULD STILL CHANGE",
        "",
        analysis.title,
        _report_status(analysis),
        f"Latest filename label: {analysis.as_of.replace('T', ' ')}; individual source times and coverage vary",
        "",
        (
            f"The analysis reviewed {len(analysis.races)} source representations of contests; overlapping jurisdictions may appear separately. "
            f"It modeled {len(modeled_decisions)} controlling decision boundaries "
            f"across {len(modeled_races)} races."
        ),
        "",
        "HIGHEST-PROBABILITY CONTROLLING DECISIONS (EXCLUDING PCO RACES)",
        "",
    ]
    if other_decisions:
        lines.extend(
            _facebook_decision_item(race, decision)
            for race, decision in other_decisions
        )
    else:
        lines.append("• No controlling decision has an estimable probability.")

    if precinct_decisions:
        lines.extend(
            [
                "",
                "PRECINCT COMMITTEE OFFICER RACES",
                "",
                (
                    "These small-vote contests have some of the highest numerical "
                    "probabilities, but their reliability is limited."
                ),
            ]
        )
        lines.extend(
            _facebook_decision_item(race, decision)
            for race, decision in precinct_decisions
        )

    lines.extend(
        [
            "",
            "OVERALL EXPOSURE",
            "",
            (
                "Using each controlling race's highest-risk boundary: "
                f"Toss-up {counts.get('toss_up', 0)}, "
                f"High {counts.get('high', 0)}, "
                f"Meaningful {counts.get('meaningful', 0)}, "
                f"Low {counts.get('low', 0)}, and "
                f"Very low {counts.get('very_low', 0)}. "
                f"{counts.get('unknown', 0)} controlling races have withheld probabilities. "
                f"Another {counts.get('not_applicable', 0)} controlling races "
                "have no modeled change boundary."
            ),
            "",
            "UNVALIDATED EVIDENCE RUBRIC",
            "",
            (
                f"High {reliability_counts.get('High', 0)}, "
                f"Moderate {reliability_counts.get('Moderate', 0)}, "
                f"Low {reliability_counts.get('Low', 0)}, and "
                f"Insufficient {reliability_counts.get('Insufficient', 0)}."
            ),
            (
                "This hand-set rubric is unvalidated and is not a correctness probability; repeated snapshots do not establish calibration."
            ),
        ]
    )
    if reconciled_races:
        lines.extend(
            [
                "",
                "DATA QUALITY NOTE",
                "",
                (
                    f"The parser reconciled {len(reconciled_races)} contest ballot "
                    "denominators because valid votes plus overvotes and undervotes "
                    "exceeded the reported Ballots Cast value. Each adjustment is "
                    "disclosed in the full report and lowers its reliability score."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "IMPORTANT",
            "",
            (
                "These are conditional model estimates based on the remaining-ballot "
                "forecast and observed vote mix. Ballot batches are not random "
                "samples. This is not an official result, a race call, or a substitute "
                "for certification."
            ),
            "",
            "Full race-by-race report and methodology:",
            report_url,
            "",
            "#WashingtonElections #ElectionResults #ElectionData",
        ]
    )
    return "\n".join(lines) + "\n"


def _node(node_id: str, kind: str, text: str, **metadata: object) -> dict[str, object]:
    return {
        "id": node_id,
        "kind": kind,
        "text": text,
        "metadata": metadata,
        "source_regions": [],
        "children": [],
    }


def _race_nodes(race: RaceAnalysis, index: int) -> list[dict[str, object]]:
    prefix = f"race-{index:03d}-{_slug(race.contest)}"
    ballot_detail = ""
    if race.ballots != race.reported_ballots:
        ballot_detail = (
            " The workbook's Ballots Cast value was "
            f"{race.reported_ballots:,}; the displayed denominator is reconciled from "
            f"{race.valid_votes:,} valid votes, {race.overvotes or 0:,} overvotes, and "
            f"{race.undervotes or 0:,} undervotes."
        )
    nodes = [
        _node(f"{prefix}-heading", "heading2", race.contest, level=2),
        _node(
            f"{prefix}-result-status", "paragraph", _status_text(race.result_status),
        ),
        _node(
            f"{prefix}-forecast-status", "paragraph", f"Forecast eligibility: {race.forecast_reason}",
        ),
        _node(
            f"{prefix}-source",
            "paragraph",
            f"Source: {race.jurisdiction} ({race.source_id}), snapshot {race.snapshot}. "
            f"Ballots with contest: {race.ballots:,}. Valid votes: {race.valid_votes:,}."
            f"{ballot_detail}",
        ),
        _node(
            f"{prefix}-scope",
            "paragraph",
            f"Scope: {race.scope_note}",
        ),
        _node(
            f"{prefix}-standings",
            "paragraph",
            "Standings: "
            + "; ".join(f"{choice.name} {choice.votes:,}" for choice in race.choices)
            + ".",
        ),
    ]
    nodes.append(_node(f"{prefix}-evidence", "paragraph",
        "Data status: " + race.data_status + ". Source published: " + str(race.evidence.get("source", {}).get("source_published_at"))
        + "; count updated: " + str(race.evidence.get("source", {}).get("count_updated_at"))
        + "; retrieved: " + str(race.evidence.get("source", {}).get("retrieved_at"))
        + "; source age hours: " + str(race.evidence.get("source", {}).get("source_age_hours"))
        + ". Coverage complete: " + str(race.evidence.get("source", {}).get("coverage_complete"))
        + "; geography: " + str(race.evidence.get("geography_comparison", {}).get("status"))
        + "; geographic qualification: " + str(race.evidence.get("source", {}).get("geography", {}).get("label", "reviewed aliases required"))
        + ("; geography details: " + json.dumps(race.evidence["source"]["geography"], sort_keys=True) if race.evidence.get("source", {}).get("geography") else "")
        + "; remaining pool: " + str(race.evidence.get("remaining_pool", {}).get("status"))
        + "; rule supported: " + str(race.evidence.get("rule", {}).get("supported"))
        + "; calibration: unvalidated; evidence rubric: unvalidated heuristic."
        + " Findings: " + "; ".join(race.evidence.get("source", {}).get("coverage_findings", []))
        + ("; headline/detail differences: " + json.dumps(race.evidence["geography_comparison"]["differences"],sort_keys=True) if race.evidence.get("geography_comparison", {}).get("differences") else "")))
    if race.excluded_history:
        nodes.append(_node(f"{prefix}-excluded-history", "paragraph",
                           "Excluded from comparable history: " + "; ".join(race.excluded_history) + "."))
    if not race.decisions:
        message = (
            "No supported numeric decision boundary is published for this rule, stage or candidate count. Descriptive tallies do not establish a zero chance of outcome change."
        )
        nodes.append(_node(f"{prefix}-no-decision", "paragraph", message))
        return nodes
    for decision_index, decision in enumerate(race.decisions, start=1):
        decision_prefix = f"{prefix}-decision-{decision_index}"
        nodes.append(
            _node(
                f"{decision_prefix}-heading",
                "heading3",
                decision.kind.replace("_", " ").title(),
                level=3,
            )
        )
        nodes.append(_node(f"{decision_prefix}-event", "paragraph",
            "Event: " + json.dumps(decision.event, sort_keys=True) + "; remaining-pool basis: " + json.dumps(decision.remaining_pool, sort_keys=True)))
        nodes.append(
            _node(
                f"{decision_prefix}-state",
                "paragraph",
                decision.current_state,
            )
        )
        nodes.append(
            _node(
                f"{decision_prefix}-risk",
                "paragraph",
                f"Experimental, uncalibrated conditional probability: "
                f"{_probability_label(decision.risk.change_probability)}. "
                f"Unvalidated evidence rubric: {decision.risk.reliability_score}/100 "
                f"({decision.risk.reliability_grade}; not validated). Selected-boundary triage band: "
                f"{_risk_label(decision.risk.level)}. {decision.risk.rationale}",
            )
        )
        nodes.append(
            _node(
                f"{decision_prefix}-reliability",
                "paragraph",
                f"Evidence rubric basis: {decision.risk.reliability_rationale}",
            )
        )
        if decision.risk.latest_batch_share is not None:
            nodes.append(
                _node(
                    f"{decision_prefix}-batch",
                    "paragraph",
                    f"Latest comparable batch: {decision.change_side} received "
                    f"{decision.risk.latest_batch_share:.2%} of "
                    f"{decision.risk.latest_batch_votes:,} decision votes.",
                )
            )
        if decision.margin_history:
            nodes.append(
                _node(
                    f"{decision_prefix}-history",
                    "paragraph",
                    "Margin history: "
                    + "; ".join(
                        f"{stamp} {margin:+,}" for stamp, margin in decision.margin_history
                    )
                    + ".",
                )
            )
    return nodes


def build_edom(analysis: ElectionAnalysis) -> dict[str, object]:
    counts = risk_counts(analysis)
    children: list[dict[str, object]] = [
        _node("report-title", "title", analysis.title),
        _node("report-status", "paragraph", _report_status(analysis)),
        _node("election-status", "paragraph", _status_text(analysis.election_status)),
        _node(
            "report-as-of",
            "paragraph",
            f"Latest input label {analysis.as_of}; {len(analysis.races)} source representations of contests; overlapping jurisdictions may appear separately. "
            "Individual source dates may be older; this label is not an authoritative current-count time.",
        ),
        _node("source-coverage", "paragraph", "Generated: " + analysis.generated_at + "; mixed source vintages: " + json.dumps(analysis.source_summary, sort_keys=True) + "; coverage findings: " + json.dumps(analysis.coverage_summary, sort_keys=True)),
        _node("summary-heading", "heading2", "Summary", level=2),
        _node(
            "summary-counts",
            "paragraph",
            "Controlling-contest risk counts: "
            + "; ".join(
                f"{_risk_label(level)} {counts.get(level, 0)}"
                for level in (
                    "toss_up",
                    "high",
                    "meaningful",
                    "low",
                    "very_low",
                    "unknown",
                    "not_applicable",
                )
            )
            + ".",
        ),
        _node("at-glance-heading", "heading2", "At-a-glance decisions", level=2),
        _node(
            "at-glance-guide",
            "paragraph",
            "Experimental scenario numbers are conditional and uncalibrated. Unknown results are excluded from numeric rankings. Every event names its selected pair or majority boundary; a maximum boundary band is triage, not a joint probability that any election outcome changes.",
        ),
        _node("at-glance-table", "paragraph", _at_glance_table(analysis)),
        _node(
            "method-heading",
            "heading2",
            "Method and limits",
            level=2,
        ),
        _node(
            "method-risk",
            "paragraph",
            "Experimental conditional probabilities use disjoint preceding history and a separately weighted verified new-ballot batch. Finite beta-binomial tails use normalized log recurrence and an explicit numerical error policy; oversized calculations are withheld. Hand-set sample caps are unvalidated modeling choices.",
        ),
        _node(
            "method-remaining",
            "paragraph",
            "Remaining pools require contemporaneous official category counts or declared low/base/high sensitivity scenarios, compatible dates and jurisdictions, and explicit contest/pair participation assumptions. Unknown pools remain unestimated; final totals and turnout overruns cannot establish zero outstanding.",
        ),
        _node(
            "method-reliability",
            "paragraph",
            "The optional 0–100 evidence rubric is a hand-set, unvalidated heuristic. Separate fields expose lifecycle, source age, coverage, provenance, rule support, remaining-pool basis and calibration. Snapshot count is not independent validation, and rubric grades are not correctness probabilities.",
        ),
        _node(
            "method-scope",
            "paragraph",
            "When the available file is only a county slice of a multicounty or statewide "
            "race, the report marks it noncontrolling and does not treat it as the official "
            "outcome forecast.",
        ),
        _node(
            "method-ballot-reconciliation",
            "paragraph",
            "For statewide XLSX inputs, the ballot denominator is the larger of the "
            "reported Ballots Cast value and the auditable sum of valid votes, "
            "overvotes, and undervotes. Any adjustment is disclosed on that race.",
        ),
    ]
    if analysis.official_versions:
        children.append(_node("official-versions", "paragraph", "Dated official result versions (raw, final and recount kept separate): " + json.dumps(analysis.official_versions, sort_keys=True)))
    if analysis.areas is not None:
        children.extend(_area_nodes(analysis.areas))
    children += [
        _node("races-heading", "heading2", "Race-by-race analysis", level=2),
    ]
    for index, race in enumerate(analysis.races, start=1):
        children.extend(_race_nodes(race, index))
    return {
        "schema_version": 1,
        "root": {
            "id": "election-report",
            "kind": "document",
            "text": analysis.title,
            "metadata": {
                "profile": "election-result-analysis",
                "tool_version": __version__,
                "as_of": analysis.as_of,
            },
            "source_regions": [],
            "children": children,
        },
    }


def publish(analysis: ElectionAnalysis, output_dir: Path) -> None:
    try:
        from edt.document_reports import generate_document_reports
        from edt.edom_markdown import write_edom_markdown
        from edt.html import write_edom_html
    except ImportError as exc:
        raise ElectionReportError(
            "EDT is required for publication. Install dlworrell/engineering-docs-toolkit "
            "or add its checkout to PYTHONPATH."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = analysis_payload(analysis)
    edom = build_edom(analysis)
    analysis_path = output_dir / "analysis.json"
    edom_path = output_dir / "canonical-document.edom.json"
    facebook_path = output_dir / "facebook-summary.txt"
    analysis_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    edom_path.write_text(
        json.dumps(edom, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    facebook_path.write_text(
        facebook_summary(analysis, public_report_url(output_dir)), encoding="utf-8"
    )
    breakdown = county_breakdown_csv(analysis)
    if breakdown is not None:
        (output_dir / "county-breakdown.csv").write_text(breakdown, encoding="utf-8")
    generate_document_reports(edom, output_dir / "edt" / "document")
    write_edom_markdown(edom_path, output_dir / "election-analysis.md")
    write_edom_html(
        edom_path,
        output_dir / "election-analysis.html",
        title=analysis.title,
    )
