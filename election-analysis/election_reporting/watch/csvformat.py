# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:4d8b27d1-c3d3-41dc-9571-8010dc5390e5

"""Normalized election-result CSV written by the source watcher.

Every source the watcher acquires is converted to this one plain-text layout so
the analyzer reads every place the same way. The first seven columns are the
King County export columns that ``election_reporting.ingest`` already requires;
the rest are optional and either read by ingest (contest overvotes/undervotes)
or retained for people and later analysis (party, jurisdiction, county,
counties spanned, write-in flag).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..ingest import (
    _clean_text,
    _header_key,
    normalize_contest_key,
    summary_rows_to_contests,
    xlsx_summary_rows,
)
from ..model import ContestResult, ElectionReportError

COLUMNS = (
    "ContestOrder",
    "Contest",
    "Contest ID",
    "Choice Order",
    "Choice",
    "BallotsWith Contest",
    "Votes",
    "Contest Overvotes",
    "Contest Undervotes",
    "Party",
    "Jurisdiction",
    "County",
    "Counties",
    "Write-in",
)


@dataclass(frozen=True)
class Row:
    contest_order: float
    contest: str
    contest_id: str
    choice_order: int
    choice: str
    ballots: int
    votes: int
    overvotes: int | None
    undervotes: int | None
    party: str
    jurisdiction: str
    county: str
    counties: tuple[str, ...]
    write_in: bool


def _number(value: int | None) -> str:
    return "" if value is None else str(value)


def render(rows: Iterable[Row]) -> str:
    """Render rows as UTF-8 CSV text with LF line endings and a fixed column order."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(COLUMNS)
    count = 0
    for row in rows:
        writer.writerow(
            (
                f"{row.contest_order:.2f}",
                row.contest,
                row.contest_id,
                row.choice_order,
                row.choice,
                row.ballots,
                row.votes,
                _number(row.overvotes),
                _number(row.undervotes),
                row.party,
                row.jurisdiction,
                row.county,
                "; ".join(row.counties),
                "yes" if row.write_in else "",
            )
        )
        count += 1
    if not count:
        raise ElectionReportError("no result rows to write")
    return buffer.getvalue()


def _contest_rows(
    contest: ContestResult,
    *,
    jurisdiction: str,
    county: str,
    parties: dict[tuple[str, str], str],
) -> list[Row]:
    return [
        Row(
            contest_order=contest.order,
            contest=contest.name,
            contest_id=contest.contest_id,
            choice_order=choice.order,
            choice=choice.name,
            ballots=contest.reported_ballots,
            votes=choice.votes,
            overvotes=contest.overvotes,
            undervotes=contest.undervotes,
            party=parties.get((contest.contest_id, choice.name), ""),
            jurisdiction=jurisdiction,
            county=county,
            counties=(),
            write_in=False,
        )
        for choice in contest.choices
    ]


def from_summary_xlsx(raw_bytes: bytes, *, label: str, jurisdiction: str, county: str = "") -> str:
    """Convert a Washington results workbook's Summary Results sheet to CSV.

    The contests come from the same parser the analyzer uses for XLSX input,
    so the CSV and the workbook produce identical analysis.
    """
    path = Path(label)
    rows = xlsx_summary_rows(path, raw_bytes)
    contests = summary_rows_to_contests(path, rows)
    header = {_header_key(value): index for index, value in enumerate(rows[0])}
    parties: dict[tuple[str, str], str] = {}
    if "party" in header:
        for row in rows[1:]:
            if str(row[header["choiceid"]]).strip():
                parties[
                    (
                        _clean_text(str(row[header["contestid"]])),
                        _clean_text(str(row[header["ballotname"]])),
                    )
                ] = _clean_text(str(row[header["party"]]))
    ordered = sorted(contests.values(), key=lambda item: item.order)
    return render(
        row
        for contest in ordered
        for row in _contest_rows(
            contest, jurisdiction=jurisdiction, county=county, parties=parties
        )
    )


def _english(texts: object) -> str:
    if isinstance(texts, list):
        for item in texts:
            if isinstance(item, dict) and item.get("languageId") in ("en", None):
                text = item.get("text")
                if isinstance(text, str):
                    return _clean_text(text)
    return ""


def _count(value: object, location: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ElectionReportError(f"{location} must be a non-negative integer")
    return value


def from_votewa_data(data: dict, *, jurisdiction: str, county: str = "") -> str:
    """Convert a results.votewa.gov ``/data`` document to CSV.

    The API reports valid votes but not ballots cast, so ``BallotsWith Contest``
    is the contest's vote total plus any reported overvotes and undervotes.
    """
    items = data.get("ballotItems")
    if not isinstance(items, list):
        raise ElectionReportError("results data has no ballotItems list")
    name_counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            key = normalize_contest_key(_english(item.get("name")))
            name_counts[key] = name_counts.get(key, 0) + 1
    rows: list[Row] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ElectionReportError(f"ballotItems[{index}] is not an object")
        location = f"ballotItems[{index}]"
        name = _english(item.get("name"))
        options = (item.get("summaryResults") or {}).get("ballotOptions") or []
        if not name or not options:
            continue
        contest_id = str(item.get("id") or "").strip()
        if not contest_id:
            raise ElectionReportError(f"{location} has no id")
        if name_counts.get(normalize_contest_key(name), 0) > 1:
            # Two contests with one display name (seen for precinct committee
            # officer races). The id suffix keeps them distinct and is stable
            # across snapshots.
            name = f"{name} [{contest_id}]"
        overvotes = _count(item.get("overVoteCount"), f"{location}.overVoteCount")
        undervotes = _count(item.get("underVoteCount"), f"{location}.underVoteCount")
        choices: list[tuple[str, int, str, bool]] = []
        for option_index, option in enumerate(options, start=1):
            option_location = f"{location}.ballotOptions[{option_index}]"
            choice = _english(option.get("name"))
            votes = _count(option.get("voteCount"), f"{option_location}.voteCount")
            if not choice or votes is None:
                raise ElectionReportError(f"{option_location} has no name or vote count")
            party = option.get("party") or {}
            choices.append(
                (
                    choice,
                    votes,
                    _clean_text(str(party.get("abbreviation") or _english(party.get("name")) or "")),
                    bool(option.get("isWriteIn")),
                )
            )
        valid_votes = sum(votes for _, votes, _, _ in choices)
        ballots = valid_votes + (overvotes or 0) + (undervotes or 0)
        counties = tuple(
            _clean_text(str(value)) for value in (item.get("crossCounties") or []) if value
        )
        for choice_order, (choice, votes, party, write_in) in enumerate(choices, start=1):
            rows.append(
                Row(
                    contest_order=float(index),
                    contest=name,
                    contest_id=contest_id,
                    choice_order=choice_order,
                    choice=choice,
                    ballots=ballots,
                    votes=votes,
                    overvotes=overvotes,
                    undervotes=undervotes,
                    party=party,
                    jurisdiction=jurisdiction,
                    county=county,
                    counties=counties,
                    write_in=write_in,
                )
            )
    return render(rows)


def trim_votewa_data(data: dict) -> bytes:
    """Keep the results-bearing parts of a ``/data`` document as canonical JSON.

    Drops the jurisdiction's display text and the precinct list (names and
    map ids only), which together are most of the document and carry no
    results. Per-precinct turnout is kept.
    """
    keep = ("election", "ballotItems", "voterTurnout")
    trimmed = {key: data.get(key) for key in keep if key in data}
    return json.dumps(trimmed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
