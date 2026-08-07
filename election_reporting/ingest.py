from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .model import (
    ChoiceResult,
    ContestResult,
    ElectionReportError,
    Snapshot,
    SourceConfig,
)


DATE_PATTERN = re.compile(
    r"(?P<date>20\d{2}-\d{2}-\d{2})(?:[T_-]?(?P<time>\d{2}(?:\d{2})?))?"
)
VOTE_FOR_PATTERN = re.compile(r"\s*\(vote\s+for\s+\d+\)\s*$", re.IGNORECASE)


def normalize_contest_key(value: str) -> str:
    without_rule = VOTE_FOR_PATTERN.sub("", value)
    return " ".join(re.findall(r"[a-z0-9]+", without_rule.casefold()))


def _header_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _parse_int(value: str, location: str) -> int:
    normalized = value.replace(",", "").strip()
    if not normalized:
        return 0
    try:
        result = int(normalized)
    except ValueError as exc:
        raise ElectionReportError(f"{location} must be an integer; got {value!r}") from exc
    if result < 0:
        raise ElectionReportError(f"{location} must not be negative")
    return result


def _parse_float(value: str, location: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ElectionReportError(f"{location} must be numeric; got {value!r}") from exc


def _snapshot_timestamp(path: Path) -> datetime:
    match = DATE_PATTERN.search(path.name)
    if match is None:
        raise ElectionReportError(
            f"snapshot filename must contain YYYY-MM-DD: {path.name}"
        )
    date_text = match.group("date")
    time_text = match.group("time") or "0000"
    if len(time_text) == 2:
        time_text += "00"
    try:
        return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H%M")
    except ValueError as exc:
        raise ElectionReportError(f"invalid snapshot date/time in {path.name}") from exc


def _is_text_export(path: Path) -> bool:
    prefix = path.read_bytes()[:8]
    return not (prefix.startswith(b"PK") or prefix.startswith(b"\xd0\xcf\x11\xe0"))


def read_snapshot(path: Path, source_id: str) -> Snapshot:
    if not _is_text_export(path):
        raise ElectionReportError(
            f"{path} is a binary spreadsheet; export it as CSV before analysis"
        )
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ElectionReportError(f"{path} must be UTF-8 CSV text") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ElectionReportError(f"{path} has no CSV header")
    header_map = {_header_key(name): name for name in reader.fieldnames}
    required = {
        "contestorder",
        "contest",
        "contestid",
        "choiceorder",
        "choice",
        "ballotswithcontest",
        "votes",
    }
    missing = sorted(required - header_map.keys())
    if missing:
        raise ElectionReportError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    grouped: dict[tuple[str, str], list[ChoiceResult]] = defaultdict(list)
    metadata: dict[tuple[str, str], tuple[float, int]] = {}
    for line_number, row in enumerate(reader, start=2):
        contest_id = _clean_text(row[header_map["contestid"]])
        contest_name = _clean_text(row[header_map["contest"]])
        choice_name = _clean_text(row[header_map["choice"]])
        if not contest_id or not contest_name or not choice_name:
            raise ElectionReportError(
                f"{path}:{line_number} has an empty contest id, contest, or choice"
            )
        location = f"{path}:{line_number}"
        contest_order = _parse_float(
            row[header_map["contestorder"]], f"{location} ContestOrder"
        )
        choice_order = _parse_int(
            row[header_map["choiceorder"]], f"{location} Choice Order"
        )
        ballots = _parse_int(
            row[header_map["ballotswithcontest"]],
            f"{location} BallotsWith Contest",
        )
        votes = _parse_int(row[header_map["votes"]], f"{location} Votes")
        group_key = (contest_id, contest_name)
        previous = metadata.setdefault(group_key, (contest_order, ballots))
        if previous != (contest_order, ballots):
            raise ElectionReportError(
                f"{location} disagrees with another row for contest {contest_id}"
            )
        grouped[group_key].append(ChoiceResult(choice_name, votes, choice_order))

    if not grouped:
        raise ElectionReportError(f"{path} contains no result rows")

    contests: dict[str, ContestResult] = {}
    for (contest_id, contest_name), choices in grouped.items():
        contest_order, ballots = metadata[(contest_id, contest_name)]
        key = normalize_contest_key(contest_name)
        if key in contests:
            raise ElectionReportError(
                f"{path} contains duplicate normalized contest name: {contest_name}"
            )
        contests[key] = ContestResult(
            contest_id=contest_id,
            name=contest_name,
            order=contest_order,
            ballots=ballots,
            choices=tuple(sorted(choices, key=lambda item: item.order)),
            key=key,
        )

    return Snapshot(
        source_id=source_id,
        path=path,
        timestamp=_snapshot_timestamp(path),
        sha256=digest,
        contests=contests,
    )


def load_source_snapshots(source: SourceConfig) -> tuple[Snapshot, ...]:
    if not source.path.exists():
        return ()
    candidates = [
        path
        for path in source.path.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.name.casefold().endswith((".csv", ".csv.xls", ".xls", ".txt"))
    ]
    snapshots = [read_snapshot(path, source.source_id) for path in candidates]
    snapshots.sort(key=lambda item: (item.timestamp, item.path.name))
    timestamps: set[datetime] = set()
    for snapshot in snapshots:
        if snapshot.timestamp in timestamps:
            raise ElectionReportError(
                f"source {source.source_id} has duplicate snapshot timestamp "
                f"{snapshot.timestamp.isoformat()}"
            )
        timestamps.add(snapshot.timestamp)
    return tuple(snapshots)
