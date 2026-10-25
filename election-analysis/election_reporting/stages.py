# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:74b8cc36-b861-4798-b9cf-97a9f4c85274

"""EA-01 (#46): explicit result stages and a conservative forecast boundary.

Schema/design: docs/engineering/ADR-003-result-lifecycle-and-stages.md.
This gate does not establish source freshness, geographic coverage or calibration.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from urllib.parse import urlsplit

from .model import ElectionConfig, ElectionReportError, ResultStatus, ResultStatusRecord, Snapshot


LIFECYCLES = {
    "unknown", "awaiting_results", "counting", "final_tabulation", "certified",
    "recount_pending", "recount_completed",
}
TABULATION_STAGES = {
    "unknown", "plurality_count", "raw_first_rankings", "rcv_round", "final_round", "recount",
}


def aware_time(value: object, location: str) -> datetime | None:
    if value is None:
        return None
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ElectionReportError(f"{location} must be an ISO datetime with a time zone") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ElectionReportError(f"{location} must include a time zone")
    return stamp


def parse_status(row: object, location: str) -> ResultStatus:
    if not isinstance(row, dict):
        raise ElectionReportError(f"{location} must be a TOML table")
    lifecycle = row.get("lifecycle", "unknown")
    stage = row.get("tabulation_stage", "unknown")
    if not isinstance(lifecycle, str) or lifecycle not in LIFECYCLES:
        raise ElectionReportError(f"{location}.lifecycle is unsupported")
    if not isinstance(stage, str) or stage not in TABULATION_STAGES:
        raise ElectionReportError(f"{location}.tabulation_stage is unsupported")
    strings = {}
    for name in ("authority_url", "result_version", "round_id"):
        value = row.get(name, "")
        if not isinstance(value, str):
            raise ElectionReportError(f"{location}.{name} must be a string")
        strings[name] = value.strip()
    stamp = aware_time(row.get("evidence_as_of"), f"{location}.evidence_as_of")
    if lifecycle != "unknown" or stage != "unknown":
        try:
            url = urlsplit(strings["authority_url"])
        except ValueError as exc:
            raise ElectionReportError(f"{location}.authority_url is invalid") from exc
        if url.scheme != "https" or not url.hostname or url.username or url.password or stamp is None:
            raise ElectionReportError(f"{location} needs an HTTPS authority_url and dated evidence_as_of")
    if stage != "unknown" and not strings["result_version"]:
        raise ElectionReportError(f"{location}.result_version is required for a known stage")
    if stage == "rcv_round" and not strings["round_id"]:
        raise ElectionReportError(f"{location}.round_id is required for an RCV round")
    return ResultStatus(lifecycle=lifecycle, tabulation_stage=stage, evidence_as_of=stamp, **strings)


def parse_status_records(rows: object, source_ids: set[str]) -> tuple[ResultStatusRecord, ...]:
    if not isinstance(rows, list):
        raise ElectionReportError("[[result_status]] must be an array of tables")
    records = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        location = f"result_status[{index}]"
        if not isinstance(row, dict):
            raise ElectionReportError(f"{location} must be a TOML table")
        source = row.get("source_id")
        digest = row.get("snapshot_sha256")
        contest = row.get("contest_id", "*")
        if not isinstance(source, str) or source not in source_ids:
            raise ElectionReportError(f"{location}.source_id must name a configured source")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ElectionReportError(f"{location}.snapshot_sha256 must be a lowercase SHA-256")
        if not isinstance(contest, str) or not contest.strip():
            raise ElectionReportError(f"{location}.contest_id must be a non-empty string")
        key = (source, digest, contest)
        if key in seen:
            raise ElectionReportError(f"{location} duplicates a result-status selector")
        seen.add(key)
        records.append(ResultStatusRecord(source, digest, contest, parse_status(row, location)))
    return tuple(records)


def annotate_snapshots(
    source_id: str, snapshots: tuple[Snapshot, ...], records: tuple[ResultStatusRecord, ...]
) -> tuple[Snapshot, ...]:
    """Bind review metadata to immutable bytes; never apply today's status to every past file."""
    selected = [record for record in records if record.source_id == source_id]
    for record in selected:
        if not any(
            snapshot.sha256 == record.snapshot_sha256
            and (record.contest_id == "*" or any(
                contest.contest_id == record.contest_id for contest in snapshot.contests.values()
            ))
            for snapshot in snapshots
        ):
            raise ElectionReportError(f"result_status for {source_id}:{record.contest_id} matches no input")
    result = []
    for snapshot in snapshots:
        by_contest = {
            record.contest_id: record.status for record in selected
            if record.snapshot_sha256 == snapshot.sha256
        }
        contests = {
            key: replace(contest, result_status=by_contest.get(
                contest.contest_id, by_contest.get("*", ResultStatus())
            ))
            for key, contest in snapshot.contests.items()
        }
        result.append(replace(snapshot, contests=contests))
    return tuple(result)


def stage_identity(status: ResultStatus) -> tuple[str, str, str, str]:
    return status.lifecycle, status.tabulation_stage, status.round_id, status.result_version


def forecast_eligibility(config: ElectionConfig, status: ResultStatus, rule: str) -> tuple[bool, str]:
    """Do not turn a deadline, first-ranking share or completed count into a live forecast."""
    if config.report_mode != "current":
        return False, f"{config.report_mode.title()} report: outstanding-ballot forecasts are withheld."
    if config.forecast_target != "outstanding_ballots":
        return False, "No supported outstanding-ballot target; recount/litigation models are unavailable."
    if config.information_time is None:
        return False, "An explicit information time is required."
    for label, evidence in (("Election", config.election_status), ("Contest", status)):
        if evidence.lifecycle != "counting":
            return False, f"{label} lifecycle is {evidence.lifecycle}; no active count forecast."
        if not evidence.authority_url or evidence.evidence_as_of is None:
            return False, f"{label} counting status lacks dated authority evidence."
        if evidence.evidence_as_of > config.information_time:
            return False, f"{label} status evidence postdates the information time."
    if status.tabulation_stage != "plurality_count" or rule == "rcv":
        return False, "This tabulation stage has no supported outstanding-ballot model."
    if not status.result_version:
        return False, "A reviewed result version is required."
    return True, "Lifecycle/stage prerequisites satisfied; experimental conditional scenario, uncalibrated."
