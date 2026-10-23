# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:cfff55a1-ac9d-49b7-bc3c-2a85a33e2d4a

"""County and precinct results: conversion, loading, reconciliation and analysis.

Washington publishes two workbooks per results update (issue #14):

- "All Results": a statewide Summary Results sheet, and a "Precinct Results"
  sheet that is in fact a county breakdown of multi-county contests;
- "Summary and Precinct Results": a Summary Results sheet that is the county
  breakdown of every contest (columns County, Office Name, Contest ID, ...),
  and a Precinct Results sheet with every precinct (about 740,000 rows,
  240 MB unpacked), which is streamed rather than loaded.

`convert_area_workbook` turns either workbook into two plain files per
snapshot, kept apart from the analyzer's `input/` snapshots:

- county CSV: Area, Office Name, Contest ID, Choice, Votes. Rows whose Area is
  "Summary Results" carry the statewide totals from the All Results workbook
  of the same update, so reconciliation can always be re-run from the file;
- precinct CSV (gzipped; non-zero rows only): Precinct, County, Office Name,
  Contest ID, Choice, Votes.

Reconciliation checks that precincts add up to their county's row and that
counties add up to the statewide totals, for every contest and choice, and
reports any difference as a data-quality finding instead of accepting it.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .ingest import (
    _snapshot_timestamp,
    _xlsx_sheet_paths,
    iter_xlsx_sheet_rows,
    normalize_contest_key,
    xlsx_summary_rows,
)
from .model import AreaConfig, ElectionConfig, ElectionReportError, RaceAnalysis

SUMMARY_AREA = "Summary Results"
BALLOTS = "ballots cast"  # choice_key("Ballots Cast")
COUNTY_COLUMNS = ("Area", "Office Name", "Contest ID", "Choice", "Votes")
PRECINCT_COLUMNS = ("Precinct", "County", "Office Name", "Contest ID", "Choice", "Votes")
LEVELS = ("county", "precinct")
PAIR_DECISIONS = {"winner", "top_two_cutoff", "top_four_cutoff"}
TOP_AREAS = 10
CLOSE_RACES = 10


# --- conversion ------------------------------------------------------------


@dataclass(frozen=True)
class AreaConversion:
    county_csv: str
    precinct_csv_gz: bytes | None
    counties: int
    precincts: int
    contests: int
    statewide_contests: int


def _header(row: list[str]) -> dict[str, int]:
    return {cell.strip().casefold(): index for index, cell in enumerate(row) if cell.strip()}


def _cell(row: list[str], columns: dict[str, int], name: str) -> str:
    index = columns.get(name)
    return " ".join(row[index].split()) if index is not None and index < len(row) else ""


def choice_key(label: str) -> str:
    """Workbooks differ in label case ("Write-In", "Write-in"); compare without it."""
    return " ".join(label.split()).casefold()


def _votes(value: str, where: str) -> int:
    text = value.strip().replace(",", "")
    if not text:
        return 0
    try:
        number = float(text)
    except ValueError as exc:
        raise ElectionReportError(f"{where}: {value!r} is not a vote count") from exc
    if number < 0 or number != int(number):
        raise ElectionReportError(f"{where}: {value!r} is not a non-negative whole number")
    return int(number)


def _sheet_names(path: Path, raw: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return [name for name, _member in _xlsx_sheet_paths(archive, path)]
    except zipfile.BadZipFile as exc:
        raise ElectionReportError(f"{path} is not a valid XLSX archive") from exc


def _require(columns: dict[str, int], names: tuple[str, ...], where: str) -> None:
    missing = [name for name in names if name not in columns]
    if missing:
        raise ElectionReportError(f"{where} lacks columns: {', '.join(missing)}")


def convert_area_workbook(raw: bytes, *, label: str, statewide: bytes | None = None) -> AreaConversion:
    """Convert a county-breakdown or precinct workbook to area files."""
    path = Path(label)
    sheets = _sheet_names(path, raw)
    county: dict[tuple[str, str, str, str], int] = {}
    precinct_rows: list[tuple[str, str, str, str, str, int]] = []

    if "Summary Results" in sheets:
        rows = iter_xlsx_sheet_rows(path, raw, "Summary Results")
        columns = _header(next(rows, []))
        if "county" in columns:  # the county breakdown of every contest
            where = f"{label} Summary Results"
            _require(columns, ("county", "office name", "contest id", "ballot name", "total"), where)
            for number, row in enumerate(rows, start=2):
                area = _cell(row, columns, "county")
                if not area:
                    continue
                key = (area, _cell(row, columns, "office name"), _cell(row, columns, "contest id"), _cell(row, columns, "ballot name"))
                county[key] = county.get(key, 0) + _votes(_cell(row, columns, "total"), f"{where} row {number}")

    if "Precinct Results" in sheets:
        rows = iter_xlsx_sheet_rows(path, raw, "Precinct Results")
        columns = _header(next(rows, []))
        where = f"{label} Precinct Results"
        if "precinct" in columns and "county" in columns:
            _require(columns, ("office name", "contest id", "ballot name", "total"), where)
            derive = not county
            for number, row in enumerate(rows, start=2):
                name = _cell(row, columns, "precinct")
                area = _cell(row, columns, "county")
                if not name or not area:
                    continue
                office = _cell(row, columns, "office name")
                contest_id = _cell(row, columns, "contest id")
                choice = _cell(row, columns, "ballot name")
                votes = _votes(_cell(row, columns, "total"), f"{where} row {number}")
                if votes:
                    precinct_rows.append((name, area, office, contest_id, choice, votes))
                if derive:
                    key = (area, office, contest_id, choice)
                    county[key] = county.get(key, 0) + votes
        elif "precinct" in columns and not county:
            # All Results workbook: "Precinct" holds county names.
            _require(columns, ("office name", "contest id", "ballot name", "total"), where)
            for number, row in enumerate(rows, start=2):
                area = _cell(row, columns, "precinct")
                if not area:
                    continue
                key = (area, _cell(row, columns, "office name"), _cell(row, columns, "contest id"), _cell(row, columns, "ballot name"))
                county[key] = county.get(key, 0) + _votes(_cell(row, columns, "total"), f"{where} row {number}")

    if not county:
        raise ElectionReportError(f"{label} has no county or precinct breakdown")

    statewide_rows: list[tuple[str, str, str, str, int]] = []
    if statewide is not None:
        rows = xlsx_summary_rows(Path(f"{label} (statewide)"), statewide)
        columns = _header(rows[0])
        where = f"{label} statewide Summary Results"
        _require(columns, ("office name", "contest id", "ballot name", "total"), where)
        for number, row in enumerate(rows[1:], start=2):
            office = _cell(row, columns, "office name")
            if office:
                statewide_rows.append(
                    (SUMMARY_AREA, office, _cell(row, columns, "contest id"), _cell(row, columns, "ballot name"),
                     _votes(_cell(row, columns, "total"), f"{where} row {number}"))
                )

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COUNTY_COLUMNS)
    writer.writerows(statewide_rows)
    writer.writerows((area, office, contest_id, choice, votes) for (area, office, contest_id, choice), votes in county.items())
    precinct_gz = None
    if precinct_rows:
        text = io.StringIO()
        precinct_writer = csv.writer(text, lineterminator="\n")
        precinct_writer.writerow(PRECINCT_COLUMNS)
        precinct_writer.writerows(precinct_rows)
        precinct_gz = gzip.compress(text.getvalue().encode("utf-8"), compresslevel=9, mtime=0)
    return AreaConversion(
        county_csv=buffer.getvalue(),
        precinct_csv_gz=precinct_gz,
        counties=len({key[0] for key in county}),
        precincts=len({(row[0], row[1]) for row in precinct_rows}),
        contests=len({key[2] for key in county}),
        statewide_contests=len({row[2] for row in statewide_rows}),
    )


# --- loading ----------------------------------------------------------------


@dataclass
class AreaContest:
    contest_id: str
    name: str
    areas: dict[str, dict[str, int]] = field(default_factory=dict)  # area -> choice -> votes
    summary: dict[str, int] = field(default_factory=dict)  # statewide totals, when recorded


@dataclass
class AreaSnapshot:
    source_id: str
    level: str
    path: Path
    timestamp: object  # datetime
    # Keyed by (contest ID, office name): local contest IDs such as "14" repeat
    # between counties, and local contest names repeat too; the pair does not.
    contests: dict[tuple[str, str], AreaContest]
    county_of: dict[str, str] = field(default_factory=dict)  # precinct area -> county

    @property
    def stamp(self) -> str:
        return self.timestamp.isoformat(timespec="minutes")

    def find(self, contest_id: str, name: str) -> AreaContest | None:
        """By contest ID and name; else a unique ID; else a unique canonical name."""
        if (contest_id, name) in self.contests:
            return self.contests[(contest_id, name)]
        same_id = [c for (cid, _), c in self.contests.items() if cid == contest_id]
        if len(same_id) == 1:
            return same_id[0]
        key = normalize_contest_key(name)
        matches = [c for c in self.contests.values() if normalize_contest_key(c.name) == key]
        return matches[0] if len(matches) == 1 else None


def _words(name: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", name.casefold()))


def precinct_area(precinct: str, county: str) -> str:
    """Precinct names repeat between counties, so the county is part of the name."""
    return f"{precinct} ({county})"


def read_area_file(path: Path, source_id: str, level: str) -> AreaSnapshot:
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = COUNTY_COLUMNS if level == "county" else PRECINCT_COLUMNS
            if tuple(reader.fieldnames or ()) != expected:
                raise ElectionReportError(f"{path} must have columns {', '.join(expected)}")
            contests: dict[str, AreaContest] = {}
            county_of: dict[str, str] = {}
            for number, row in enumerate(reader, start=2):
                name, contest_id = row["Office Name"], row["Contest ID"]
                contest = contests.setdefault((contest_id, name), AreaContest(contest_id, name))
                votes = _votes(row["Votes"], f"{path} row {number}")
                if level == "county":
                    area = row["Area"]
                    target = contest.summary if area == SUMMARY_AREA else contest.areas.setdefault(area, {})
                else:
                    area = precinct_area(row["Precinct"], row["County"])
                    county_of[area] = row["County"]
                    target = contest.areas.setdefault(area, {})
                choice = choice_key(row["Choice"])
                target[choice] = target.get(choice, 0) + votes
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
        raise ElectionReportError(f"could not read {path}: {exc}") from exc
    # The statewide sheet and the breakdown word some names differently
    # ("State Representative Pos. 1 - Legislative District 3" against
    # "Legislative District 3 - State Representative Pos. 1"). Statewide
    # contest IDs are unique, but small local IDs repeat (7209 is both a
    # statewide EMS levy and a Mason County fire authority), so attach
    # statewide totals by ID only to a breakdown contest with the same words.
    for key in [k for k, c in contests.items() if c.summary and not c.areas]:
        matches = [c for k, c in contests.items()
                   if k[0] == key[0] and c.areas and _words(k[1]) == _words(key[1])]
        if len(matches) == 1 and not matches[0].summary:
            matches[0].summary = contests.pop(key).summary
    return AreaSnapshot(source_id, level, path, _snapshot_timestamp(path), contests, county_of)


def area_files(area: AreaConfig) -> list[Path]:
    if not area.path.is_dir():
        return []
    suffixes = (".csv",) if area.level == "county" else (".csv.gz", ".csv")
    files = [p for p in area.path.iterdir() if p.is_file() and p.name.endswith(suffixes)]
    return sorted(files, key=lambda p: (_snapshot_timestamp(p), p.name))


# --- reconciliation ---------------------------------------------------------


def _sum_areas(contest: AreaContest, areas=None) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for area, choices in contest.areas.items():
        if areas is None or area in areas:
            for choice, votes in choices.items():
                totals[choice] += votes
    return dict(totals)


def reconcile(county: AreaSnapshot, precinct: AreaSnapshot | None) -> dict:
    """Counties against the statewide totals; precincts against their county."""
    result: dict = {"county_snapshot": county.stamp}
    checked = 0
    mismatches = []
    without_statewide = 0
    for contest in county.contests.values():
        if not contest.summary:
            without_statewide += 1
            continue
        sums = _sum_areas(contest)
        for choice in sorted(set(sums) | set(contest.summary)):
            checked += 1
            expected, found = contest.summary.get(choice, 0), sums.get(choice, 0)
            if expected != found:
                mismatches.append({"contest": contest.name, "contest_id": contest.contest_id, "area": "all counties",
                                   "choice": choice, "statewide": expected, "counties": found, "difference": found - expected})
    result["county_vs_statewide"] = {"checked": checked, "mismatches": mismatches,
                                     "contests_without_statewide_total": without_statewide}
    if precinct is None:
        result["precinct_vs_county"] = {"status": "no precinct snapshot at this time"}
        return result
    result["precinct_snapshot"] = precinct.stamp
    checked = 0
    mismatches = []
    missing: list[dict] = []
    for key, contest in precinct.contests.items():
        reference = county.contests.get(key)  # same contest ID
        if reference is None:
            mismatches.append({"contest": contest.name, "contest_id": contest.contest_id, "area": "",
                               "choice": "", "difference": None, "problem": "contest missing from the county file"})
            continue
        by_county: dict[str, set[str]] = defaultdict(set)
        for area in contest.areas:
            by_county[precinct.county_of[area]].add(area)
        for county_name in sorted(set(by_county) | set(reference.areas)):
            sums = _sum_areas(contest, by_county.get(county_name, set()))
            row = reference.areas.get(county_name, {})
            if not sums and any(row.values()):
                missing.append({"contest": contest.name, "contest_id": contest.contest_id, "county": county_name,
                                "county_ballots": row.get(BALLOTS, 0)})
                continue
            for choice in sorted(set(sums) | set(row)):
                checked += 1
                expected, found = row.get(choice, 0), sums.get(choice, 0)
                if expected != found:
                    mismatches.append({"contest": contest.name, "contest_id": contest.contest_id, "area": county_name,
                                       "choice": choice, "county": expected, "precincts": found,
                                       "difference": found - expected})
    # Contests with county votes but no precinct rows at all for that county:
    # the precinct file omits them (possibly withheld); not an arithmetic error.
    for key, reference in county.contests.items():
        if key in precinct.contests:
            continue
        for county_name, row in reference.areas.items():
            if any(row.values()):
                missing.append({"contest": reference.name, "contest_id": reference.contest_id,
                                "county": county_name, "county_ballots": row.get(BALLOTS, 0)})
    result["precinct_vs_county"] = {"checked": checked, "mismatches": mismatches,
                                    "precinct_detail_missing": missing}
    return result


# --- analysis ---------------------------------------------------------------


@dataclass(frozen=True)
class AreaAnalysis:
    source_id: str
    snapshots: dict[str, list[str]]
    reconciliation: dict
    breakdown: list[dict]
    county_changes: dict
    precinct_changes: dict
    remaining: list[dict]
    notes: list[str]


def _close_races(races: tuple[RaceAnalysis, ...]) -> list[tuple[RaceAnalysis, object]]:
    pairs = []
    for race in races:
        if not race.controlling:
            continue
        for decision in race.decisions:
            if decision.kind in PAIR_DECISIONS:
                pairs.append((race, decision))
                break
    pairs.sort(key=lambda item: (-(item[1].risk.change_probability or -1.0), abs(item[1].margin), item[0].contest))
    return pairs[:CLOSE_RACES]


def _changes(latest: AreaSnapshot, previous: AreaSnapshot | None, close, level: str) -> dict:
    if previous is None:
        return {"status": f"needs two {level} snapshots; only {latest.stamp} is available"}
    new_ballots: dict[str, int] = {}
    for key, contest in latest.contests.items():
        before = previous.contests.get(key)
        for area, choices in contest.areas.items():
            old = before.areas.get(area, {}) if before else {}
            delta = choices.get(BALLOTS, 0) - old.get(BALLOTS, 0)
            new_ballots[area] = max(new_ballots.get(area, delta), delta)
    top = sorted(new_ballots.items(), key=lambda item: (-item[1], item[0]))[:TOP_AREAS]
    races = []
    for race, decision in close:
        now = latest.find(race.contest_id, race.contest)
        before = previous.find(race.contest_id, race.contest)
        if now is None or before is None:
            continue
        current, change = choice_key(decision.current_side), choice_key(decision.change_side)
        swings = []
        for area, choices in now.areas.items():
            old = before.areas.get(area, {})
            margin_now = choices.get(current, 0) - choices.get(change, 0)
            margin_before = old.get(current, 0) - old.get(change, 0)
            shift = margin_now - margin_before
            added = choices.get(BALLOTS, 0) - old.get(BALLOTS, 0)
            if shift or added:
                swings.append({"area": area, "new_ballots": added, "margin_change": shift})
        swings.sort(key=lambda item: (-abs(item["margin_change"]), item["area"]))
        races.append({"contest": race.contest, "current_side": decision.current_side,
                      "change_side": decision.change_side, "swings": swings[:5]})
    return {"from": previous.stamp, "to": latest.stamp,
            "largest_new_ballots": [{"area": area, "new_ballots": count} for area, count in top],
            "close_races": races}


def _remaining_flags(config: ElectionConfig, county: AreaSnapshot, close) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    if not config.remaining_ballots:
        return [], ["No county remaining-ballot counts are configured ([[remaining_ballots]] in election.toml), "
                    "so no county can be assessed as able to change an outcome."]
    latest_date = county.timestamp.date()
    stale = [r.county for r in config.remaining_ballots if r.as_of < latest_date]
    if stale:
        notes.append(f"Remaining-ballot counts older than the {latest_date} county snapshot: {', '.join(sorted(stale))}.")
    flags = []
    for race, decision in close:
        contest = county.find(race.contest_id, race.contest)
        if contest is None:
            continue
        declared = [r for r in config.remaining_ballots if r.county in contest.areas]
        if not declared:
            flags.append({"contest": race.contest, "status": "no remaining-ballot count for this race's counties"})
            continue
        top = max(declared, key=lambda r: (r.ballots, r.county))
        choices = contest.areas[top.county]
        current = choices.get(choice_key(decision.current_side), 0)
        change = choices.get(choice_key(decision.change_side), 0)
        ballots = choices.get(BALLOTS, 0)
        entry = {"contest": race.contest, "county": top.county, "remaining_ballots": top.ballots,
                 "as_of": top.as_of.isoformat(), "current_side": decision.current_side,
                 "change_side": decision.change_side, "margin": decision.margin}
        if not ballots or not current + change:
            entry["status"] = "the county has no counted votes for this pair yet"
            flags.append(entry)
            continue
        remaining_pair = round(top.ballots * (current + change) / ballots)
        if remaining_pair <= 0:
            entry["status"] = "no remaining decision votes"
            flags.append(entry)
            continue
        needed = (decision.margin + remaining_pair) / (2 * remaining_pair)
        share = change / (current + change)
        entry.update({"remaining_decision_votes": remaining_pair, "needed_share": round(needed, 4),
                      "county_change_side_share": round(share, 4)})
        if needed > 1:
            entry["status"] = "cannot change the outcome alone"
        elif share >= needed:
            entry["status"] = "would change the outcome at the county's current split"
        else:
            entry["status"] = "could change the outcome, but only at a larger share than its current split"
        flags.append(entry)
    return flags, notes


def analyze_areas(config: ElectionConfig, races: tuple[RaceAnalysis, ...]) -> AreaAnalysis | None:
    if not config.areas:
        return None
    by_level: dict[str, AreaConfig] = {}
    for area in config.areas:
        by_level.setdefault(area.level, area)
    source_id = next(iter(by_level.values())).source_id
    county_files = area_files(by_level["county"]) if "county" in by_level else []
    precinct_files = area_files(by_level["precinct"]) if "precinct" in by_level else []
    snapshots = {"county": [str(p.name) for p in county_files], "precinct": [str(p.name) for p in precinct_files]}
    notes: list[str] = []
    if not county_files:
        notes.append("No county snapshots yet.")
        return AreaAnalysis(source_id, snapshots, {}, [], {"status": "no county snapshots"},
                            {"status": "no precinct snapshots"}, [], notes)

    county = [read_area_file(p, source_id, "county") for p in county_files[-2:]]
    precinct = [read_area_file(p, source_id, "precinct") for p in precinct_files[-2:]]
    latest_county = county[-1]
    paired = next((p for p in precinct if p.timestamp == latest_county.timestamp), None)
    reconciliation = reconcile(latest_county, paired)

    breakdown = []
    for race in races:
        if not race.controlling:
            continue
        contest = latest_county.find(race.contest_id, race.contest)
        if contest is None:
            continue
        names = [choice.name for choice in race.choices]
        rows = []
        for area, choices in sorted(contest.areas.items(), key=lambda item: (-item[1].get(BALLOTS, 0), item[0])):
            votes = {name: choices.get(choice_key(name), 0) for name in names}
            leader = max(names, key=lambda name: (votes[name], -names.index(name))) if names else ""
            rows.append({"county": area, "ballots": choices.get(BALLOTS, 0), "votes": votes,
                         "leader": leader if votes.get(leader) else ""})
        breakdown.append({"contest": race.contest, "contest_id": race.contest_id,
                          "snapshot": latest_county.stamp, "choices": names, "counties": rows})

    close = _close_races(races)
    county_changes = _changes(county[-1], county[-2] if len(county) > 1 else None, close, "county")
    if precinct:
        precinct_changes = _changes(precinct[-1], precinct[-2] if len(precinct) > 1 else None, close, "precinct")
    else:
        precinct_changes = {"status": "no precinct snapshots"}
    remaining, remaining_notes = _remaining_flags(config, latest_county, close)
    notes.extend(remaining_notes)
    return AreaAnalysis(source_id, snapshots, reconciliation, breakdown, county_changes,
                        precinct_changes, remaining, notes)
