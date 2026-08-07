from __future__ import annotations

import json
import re
from pathlib import Path

from . import __version__
from .analysis import FLIP_BANDS, risk_counts
from .model import ElectionAnalysis, ElectionReportError, RaceAnalysis


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
    analysis_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    edom_path.write_text(
        json.dumps(edom, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generate_document_reports(edom, output_dir / "edt" / "document")
    write_edom_markdown(edom_path, output_dir / "election-analysis.md")
    write_edom_html(
        edom_path,
        output_dir / "election-analysis.html",
        title=analysis.title,
    )
