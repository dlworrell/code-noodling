from __future__ import annotations

import csv
import hashlib
import io
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
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
XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_DOCUMENT_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
XLSX_PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
MAX_XLSX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


def normalize_contest_key(value: str) -> str:
    without_rule = VOTE_FOR_PATTERN.sub("", value)
    normalized = " ".join(re.findall(r"[a-z0-9]+", without_rule.casefold()))

    house = re.search(
        r"\bu s representative\b.*\bcongressional district (?:no )?(\d+)\b",
        normalized,
    )
    if house is not None:
        return f"us-house:{int(house.group(1))}"

    legislative = re.search(
        r"\blegislative district (?:no )?(\d+)\b",
        normalized,
    )
    if legislative is not None:
        district = int(legislative.group(1))
        if "state senator" in normalized:
            return f"wa-leg:{district}:senate"
        position = re.search(
            r"\brepresentative (?:position|pos) (?:no )?(\d+)\b",
            normalized,
        )
        if position is not None:
            return f"wa-leg:{district}:rep:{int(position.group(1))}"

    justice = re.search(
        r"\bjustice position (?:no )?(\d+)\b",
        normalized,
    )
    if justice is not None:
        return f"wa-supreme:{int(justice.group(1))}"

    return normalized


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


def _xlsx_name(tag: str) -> str:
    return f"{{{XLSX_MAIN_NS}}}{tag}"


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    return [
        "".join(node.text or "" for node in item.iter(_xlsx_name("t")))
        for item in root.iter(_xlsx_name("si"))
    ]


def _xlsx_summary_path(archive: zipfile.ZipFile, path: Path) -> str:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
    except (KeyError, ET.ParseError) as exc:
        raise ElectionReportError(f"{path} is missing valid XLSX workbook metadata") from exc

    relation_targets = {
        relation.get("Id", ""): relation.get("Target", "")
        for relation in relationships.iter(
            f"{{{XLSX_PACKAGE_REL_NS}}}Relationship"
        )
    }
    relation_id = ""
    for sheet in workbook.iter(_xlsx_name("sheet")):
        if sheet.get("name") == "Summary Results":
            relation_id = sheet.get(f"{{{XLSX_DOCUMENT_REL_NS}}}id", "")
            break
    target = relation_targets.get(relation_id, "")
    if not target:
        raise ElectionReportError(f"{path} has no Summary Results worksheet")
    if target.startswith("/"):
        worksheet_path = target.lstrip("/")
    else:
        worksheet_path = posixpath.normpath(posixpath.join("xl", target))
    if not worksheet_path.startswith("xl/"):
        raise ElectionReportError(f"{path} has an unsafe worksheet relationship")
    return worksheet_path


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t", "")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(_xlsx_name("t"))
        )
    value = cell.find(_xlsx_name("v"))
    raw = "" if value is None or value.text is None else value.text
    if cell_type != "s" or not raw:
        return raw
    try:
        return shared_strings[int(raw)]
    except (ValueError, IndexError) as exc:
        raise ElectionReportError("XLSX shared-string index is invalid") from exc


def _xlsx_column(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference.upper())
    if match is None:
        return 0
    column = 0
    for character in match.group(1):
        column = column * 26 + ord(character) - ord("A") + 1
    return column


def _xlsx_rows(
    archive: zipfile.ZipFile,
    worksheet_path: str,
    shared_strings: list[str],
) -> list[list[str]]:
    try:
        root = ET.fromstring(archive.read(worksheet_path))
    except (KeyError, ET.ParseError) as exc:
        raise ElectionReportError("Summary Results worksheet is missing or invalid") from exc
    rows: list[list[str]] = []
    for row in root.iter(_xlsx_name("row")):
        values = [""] * 6
        for cell in row.iter(_xlsx_name("c")):
            column = _xlsx_column(cell.get("r", ""))
            if 1 <= column <= len(values):
                values[column - 1] = _xlsx_cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def _read_statewide_xlsx(
    path: Path,
    source_id: str,
    raw_bytes: bytes,
    digest: str,
) -> Snapshot:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            uncompressed_size = sum(item.file_size for item in archive.infolist())
            if uncompressed_size > MAX_XLSX_UNCOMPRESSED_BYTES:
                raise ElectionReportError(
                    f"{path} exceeds the {MAX_XLSX_UNCOMPRESSED_BYTES:,}-byte "
                    "uncompressed XLSX safety limit"
                )
            worksheet_path = _xlsx_summary_path(archive, path)
            rows = _xlsx_rows(
                archive,
                worksheet_path,
                _xlsx_shared_strings(archive),
            )
    except zipfile.BadZipFile as exc:
        raise ElectionReportError(f"{path} is not a valid XLSX archive") from exc
    if not rows:
        raise ElectionReportError(f"{path} Summary Results worksheet is empty")

    header_map = {_header_key(value): index for index, value in enumerate(rows[0])}
    required = {"officename", "contestid", "ballotname", "choiceid", "total"}
    missing = sorted(required - header_map.keys())
    if missing:
        raise ElectionReportError(
            f"{path} Summary Results is missing columns: {', '.join(missing)}"
        )

    grouped: dict[tuple[str, str], list[ChoiceResult]] = defaultdict(list)
    ballots: dict[tuple[str, str], int] = {}
    overvotes: dict[tuple[str, str], int] = defaultdict(int)
    undervotes: dict[tuple[str, str], int] = defaultdict(int)
    contest_order: dict[tuple[str, str], float] = {}
    next_order = 1
    for row_number, row in enumerate(rows[1:], start=2):
        office_name = _clean_text(str(row[header_map["officename"]]))
        contest_id = _clean_text(str(row[header_map["contestid"]]))
        ballot_name = _clean_text(str(row[header_map["ballotname"]]))
        choice_id = _clean_text(str(row[header_map["choiceid"]]))
        if not office_name or office_name == "None":
            continue
        group_key = (contest_id, office_name)
        if group_key not in contest_order:
            contest_order[group_key] = float(next_order)
            next_order += 1
        location = f"{path}:Summary Results:{row_number}"
        total = _parse_int(str(row[header_map["total"]]), f"{location} Total")
        if ballot_name.casefold() == "ballots cast":
            ballots[group_key] = total
        elif choice_id:
            grouped[group_key].append(
                ChoiceResult(ballot_name, total, len(grouped[group_key]) + 1)
            )
        elif ballot_name.casefold() == "over votes":
            overvotes[group_key] += total
        elif ballot_name.casefold() == "under votes":
            undervotes[group_key] += total

    contests: dict[str, ContestResult] = {}
    for (contest_id, contest_name), choices in grouped.items():
        group_key = (contest_id, contest_name)
        if group_key not in ballots:
            raise ElectionReportError(
                f"{path} has no Ballots Cast row for contest {contest_id}"
            )
        key = normalize_contest_key(contest_name)
        if key in contests:
            raise ElectionReportError(
                f"{path} contains duplicate normalized contest name: {contest_name}"
            )
        reported_ballots = ballots[group_key]
        tally_balance = (
            sum(choice.votes for choice in choices)
            + overvotes[group_key]
            + undervotes[group_key]
        )
        contests[key] = ContestResult(
            contest_id=contest_id,
            name=contest_name,
            order=contest_order[group_key],
            ballots=max(reported_ballots, tally_balance),
            reported_ballots=reported_ballots,
            overvotes=overvotes[group_key],
            undervotes=undervotes[group_key],
            choices=tuple(choices),
            key=key,
        )
    if not contests:
        raise ElectionReportError(f"{path} contains no statewide result contests")
    return Snapshot(
        source_id=source_id,
        path=path,
        timestamp=_snapshot_timestamp(path),
        sha256=digest,
        contests=contests,
    )


def read_snapshot(path: Path, source_id: str) -> Snapshot:
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if raw_bytes.startswith(b"PK"):
        return _read_statewide_xlsx(path, source_id, raw_bytes, digest)
    if raw_bytes.startswith(b"\xd0\xcf\x11\xe0"):
        raise ElectionReportError(
            f"{path} is a legacy binary spreadsheet; export it as CSV or XLSX"
        )
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
            reported_ballots=ballots,
            overvotes=None,
            undervotes=None,
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
        and path.name.casefold().endswith(
            (".csv", ".csv.xls", ".xls", ".txt", ".xlsx")
        )
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
