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
from .analysis import FLIP_BANDS, risk_counts
from .model import DecisionAnalysis, ElectionAnalysis, ElectionReportError, RaceAnalysis


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
            (title, "New ballots"),
            [(item["area"], f"{item['new_ballots']:,}") for item in changes["largest_new_ballots"]],
        )))
        rows = []
        for race in changes["close_races"]:
            for swing in race["swings"]:
                rows.append((race["contest"], f"{race['current_side']} vs {race['change_side']}",
                             swing["area"], f"{swing['new_ballots']:,}", _signed(swing["margin_change"])))
        if rows:
            nodes.append(_node(f"areas-{level}-swings", "paragraph", _area_table(
                ("Contest", "Decision", title, "New ballots", "Margin change"), rows,
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
        "schema_version": 2,
        "tool_version": __version__,
        "title": analysis.title,
        "as_of": analysis.as_of,
        "methodology": {
            "probability_status": (
                "conditional tempered beta-binomial predictive probability; not an "
                "election call"
            ),
            "probability_model": "tempered_beta_binomial_v1",
            "tail_evaluation": (
                "exact through 5,000 remaining decision votes; moment-matched normal "
                "approximation above 5,000"
            ),
            "effective_sample_caps": {
                "cumulative_votes": 300,
                "latest_batch_votes": 200,
            },
            "risk_bands": FLIP_BANDS,
            "remaining_votes": (
                "contest ballots scaled proportionally from the configured "
                "source-level expected final ballot count"
            ),
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
                "sha256": snapshot.sha256,
                "contests": len(snapshot.contests),
            }
            for snapshot in analysis.snapshots
        ],
        "races": [
            {
                "contest_id": race.contest_id,
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
            scope = "Controlling" if race.controlling else "County slice"
            order = (
                not race.controlling,
                True,
                0.0,
                race.contest.casefold(),
                "not_applicable",
            )
            row = (
                f"| Not applicable | N/A | {_table_text(race.contest)} | "
                f"No modeled boundary | — | {scope} |"
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
            boundary = decision.kind.replace("_", " ").title()
            current = (
                f"{decision.current_side} +{decision.margin:,} over "
                f"{decision.change_side}"
            )
            scope = "Controlling" if race.controlling else "County slice"
            row = (
                f"| {_probability_label(probability)} | {reliability} | "
                f"{_table_text(race.contest)} | {_table_text(boundary)} | "
                f"{_table_text(current)} | {scope} |"
            )
            rows.append((order, row))
    rows.sort(key=lambda item: item[0])
    header = [
        "| Change probability | Reliability | Race | Boundary | Current margin | Scope |",
        "| ---: | --- | --- | --- | --- | --- |",
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
    boundary = decision.kind.replace("_", " ").title()
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
        f"Modeled change probability: "
        f"{_probability_label(decision.risk.change_probability)}. "
        f"Reliability: {decision.risk.reliability_score}/100 "
        f"({decision.risk.reliability_grade})."
    )


def facebook_summary(analysis: ElectionAnalysis, report_url: str = PUBLIC_REPORT_URL) -> str:
    """Build a compact, plain-text report suitable for a Facebook post."""
    counts = risk_counts(analysis)
    controlling_races = [race for race in analysis.races if race.controlling]
    modeled_races = [race for race in controlling_races if race.decisions]
    modeled_decisions = [
        decision
        for race in controlling_races
        for decision in race.decisions
        if decision.risk.change_probability is not None
    ]
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
        f"Results snapshot: {analysis.as_of.replace('T', ' ')}",
        "",
        (
            f"The analysis reviewed {len(analysis.races)} reported contests. "
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
                f"Another {counts.get('not_applicable', 0)} controlling races "
                "have no modeled change boundary."
            ),
            "",
            "EVIDENCE RELIABILITY",
            "",
            (
                f"High {reliability_counts.get('High', 0)}, "
                f"Moderate {reliability_counts.get('Moderate', 0)}, "
                f"Low {reliability_counts.get('Low', 0)}, and "
                f"Insufficient {reliability_counts.get('Insufficient', 0)}."
            ),
            (
                "Reliability measures the strength of the available evidence. It is "
                "not the probability that the current result is correct."
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
    if not race.decisions:
        message = (
            "No modeled decision boundary: the race is uncontested, ignored by "
            "configuration, or has no more candidates than available top-two positions."
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
                f"Modeled change probability: "
                f"{_probability_label(decision.risk.change_probability)}. "
                f"Reliability: {decision.risk.reliability_score}/100 "
                f"({decision.risk.reliability_grade}). Exposure band: "
                f"{_risk_label(decision.risk.level)}. {decision.risk.rationale}",
            )
        )
        nodes.append(
            _node(
                f"{decision_prefix}-reliability",
                "paragraph",
                f"Reliability basis: {decision.risk.reliability_rationale}",
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
        _node(
            "report-as-of",
            "paragraph",
            f"Analysis as of {analysis.as_of}; {len(analysis.races)} contests reported.",
        ),
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
            "Sorted by modeled change probability, with controlling results before "
            "county-only slices. Reliability measures evidence quality, not the chance "
            "that the current result is correct.",
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
            "Change probabilities use a tempered beta-binomial posterior predictive "
            "model. Tails are exact through 5,000 remaining decision votes and use a "
            "moment-matched normal approximation above that. Effective sample caps "
            "prevent every counted ballot from being treated as an independent draw.",
        ),
        _node(
            "method-remaining",
            "paragraph",
            "Remaining contest votes are estimated by scaling current contest ballots "
            "to each source's configured expected final ballot count. Missing forecasts "
            "produce an Unknown assessment.",
        ),
        _node(
            "method-reliability",
            "paragraph",
            "Reliability is a 0–100 evidence score combining remaining-ballot forecast "
            "quality, compatible snapshot history, and observed decision-vote volume. "
            "Reconciled denominators reduce it; noncontrolling slices are capped at 25.",
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
