# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:c69ecc04-eaca-416b-8e66-14a62e5e38e2

"""Alaska Division of Elections results: elections, places and conversion.

Alaska has no counties. Its county equivalents are 19 boroughs and 11 census
areas, and the Division reports results by precinct and House district only.
``election-analysis/data/crosswalks/alaska/precinct-borough.csv`` maps each precinct to
the borough or census area its voters live in, so the watcher can write:

- ``alaska``: statewide totals, from every row of the precinct file;
- one source per selected borough or census area, from its election-day
  precincts. Absentee, early, questioned and remote ballots are reported per
  House district, not per precinct, so they are in the statewide totals only.

Primaries are nonpartisan and advance the top four candidates; general
elections use ranked-choice voting for President, Congress, Governor and the
Legislature (Division of Elections, "General Election Information"). The
precinct file holds first choices only.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from ..model import ElectionReportError
from . import csvformat
from .http import Fetch
from .sources import ElectionRef, Place

SITE = "https://www.elections.alaska.gov"
RESULTS_PAGE = SITE + "/election-results/e/?id={election_id}"
CROSSWALK = Path("crosswalks") / "alaska" / "precinct-borough.csv"
# Rule for candidate contests by election kind; measures and judicial
# retention questions (yes/no) are always "winner".
DEFAULT_RULES = {"primary": "top_four", "general": "rcv", "special": "rcv"}
NAME_SUFFIXES = (" city and borough", " census area", " municipality", " borough")
PRECINCT_ID = re.compile(r"^\d\d-\d\d\d$")
HOUSE_DISTRICTS = 40
CSV_LINK = re.compile(r"""href=["']([^"']+\.csv)["']""", re.IGNORECASE)
REQUIRED_COLUMNS = (
    "Precinct_name",
    "Pct_Id",
    "Contest_Id",
    "Contest_seq_nbr",
    "Contest_title",
    "candidate_name",
    "cand_seq_nbr",
    "Party_Code",
    "total_ballots",
    "total_votes",
    "total_under_votes",
    "total_over_votes",
)
DISTRICT_LEVEL_NOTE = (
    "Election-day precinct votes only. Alaska reports absentee, early, questioned "
    "and remote ballots per House district, and federal-only overseas ballots "
    "statewide, so they are in the statewide totals only."
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def list_elections(today: date) -> list[ElectionRef]:
    """State primary and general elections around ``today``.

    Primary: the third Tuesday in August (AS 15.25.020). General: the Tuesday
    after the first Monday in November of even years (AS 15.15.020).
    """
    elections = []
    for year in range(today.year - 2, today.year + 3):
        if year % 2:
            continue
        short = f"{year % 100:02d}"
        primary = _nth_weekday(year, 8, 1, 3)
        general = _nth_weekday(year, 11, 0, 1) + timedelta(days=1)
        elections.append(ElectionRef(primary, f"{short}prim", f"{year} Primary Election"))
        elections.append(ElectionRef(general, f"{short}genr", f"{year} General Election"))
    return sorted(elections, key=lambda item: item.day)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def _short_name(name: str) -> str:
    """``"Kenai Peninsula Borough"`` -> ``"Kenai Peninsula"``, keeping case."""
    for suffix in NAME_SUFFIXES:
        if name.casefold().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _short(name: str) -> str:
    folded = name.casefold()
    for suffix in NAME_SUFFIXES:
        if folded.endswith(suffix):
            return folded.removesuffix(suffix)
    return folded


@dataclass(frozen=True)
class Crosswalk:
    """Each precinct's borough or census area, by precinct ID (``DD-NNN``)."""

    precincts: dict[str, str]  # precinct ID -> borough name
    boroughs: tuple[str, ...]
    split_precincts: tuple[str, ...] = ()
    vintage: dict = field(default_factory=lambda: {"geometry_year": 2023, "census_year": 2020, "assignment_method": "largest resident-population share; not measured votes"})


def load_crosswalk(data_root: Path) -> Crosswalk:
    """Read the committed crosswalk; a split precinct goes to its main borough.

    Five precincts cross a borough line, four of them by under 2% of residents;
    assigning each wholly to its main borough is approximate even with integer counts.
    """
    path = data_root / CROSSWALK
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise ElectionReportError(f"Alaska precinct crosswalk not found: {path}") from exc
    best: dict[str, tuple[float, str]] = {}
    for row in rows:
        share = float(row["population_share"])
        precinct = row["precinct_id"]
        if precinct not in best or share > best[precinct][0]:
            best[precinct] = (share, row["borough"])
    return Crosswalk(
        precincts={precinct: borough for precinct, (_, borough) in best.items()},
        boroughs=tuple(sorted({row["borough"] for row in rows})),
        split_precincts=tuple(sorted(p for p in best if sum(r["precinct_id"] == p for r in rows) > 1)),
    )


def resolve_places(crosswalk: Crosswalk, names: tuple[str, ...], all_boroughs: bool) -> list[Place]:
    """Statewide plus the named boroughs or census areas ("Bethel" is enough)."""
    known: dict[str, str] = {}
    for borough in crosswalk.boroughs:
        known[borough.casefold()] = borough
        known[_short(borough)] = borough
    wanted = crosswalk.boroughs if all_boroughs else names
    places = [Place("alaska", "Alaska", "state", "")]
    unknown = []
    for name in wanted:
        borough = known.get(name.casefold()) or known.get(_short(name))
        if borough is None:
            unknown.append(name)
        elif all(place.name != borough for place in places):
            places.append(Place(_slug(borough), borough, "county", ""))
    if unknown:
        raise ElectionReportError(
            f"unknown Alaska borough or census area name(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(crosswalk.boroughs)}"
        )
    return places


def results_file_url(fetch: Fetch, election: ElectionRef) -> str | None:
    """The precinct results CSV linked from the election's results page, if any."""
    page_url = RESULTS_PAGE.format(election_id=election.public_id)
    page = fetch(page_url).body.decode("utf-8", errors="replace")
    for link in CSV_LINK.findall(page):
        url = urllib.parse.urljoin(page_url, link.replace("&amp;", "&"))
        if "precinct" in url.casefold():
            return url
    return None


def _count(row: dict[str, str], column: str, line: int) -> int:
    value = (row.get(column) or "").strip()
    if not value:
        return 0
    if not value.isdigit():
        raise ElectionReportError(f"line {line}: {column} is not a non-negative whole number: {value!r}")
    return int(value)


@dataclass
class _Contest:
    order: float
    title: str
    ballots: int = 0
    undervotes: int = 0
    overvotes: int = 0


def convert(raw: bytes, crosswalk: Crosswalk, places: list[Place]) -> dict[str, str]:
    """Convert the precinct results file to one normalized CSV per place.

    Contest ballots, undervotes and overvotes repeat on every candidate row of
    a precinct, so they are counted once per precinct and contest.
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ElectionReportError("Alaska precinct results are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
    if missing:
        raise ElectionReportError(f"Alaska precinct results lack columns: {', '.join(missing)}")
    borough_place = {place.name: place for place in places if place.scope == "county"}
    state_place = next(place for place in places if place.scope == "state")
    areas: dict[str, dict[str, _Contest]] = {place.place_id: {} for place in places}
    votes: dict[str, dict[tuple[str, str], list]] = {place.place_id: {} for place in places}
    spans: dict[str, set[str]] = {}
    seen: set[tuple[str, str]] = set()
    unknown_precincts: set[str] = set()
    seen_choices = set()
    for line, row in enumerate(reader, start=2):
        precinct_id = (row["Pct_Id"] or "").strip()
        contest_id = row["Contest_Id"].strip()
        title = " ".join(row["Contest_title"].split())
        if not contest_id or not title:
            raise ElectionReportError(f"line {line}: contest ID and title are required")
        borough = None
        # House districts are 01-40. Other IDs, such as 99-999 "HD99 Fed
        # Overseas Absentee" (federal-only overseas voters), have no borough.
        if PRECINCT_ID.match(precinct_id) and 1 <= int(precinct_id[:2]) <= HOUSE_DISTRICTS:
            borough = crosswalk.precincts.get(precinct_id)
            if borough is None:
                unknown_precincts.add(precinct_id)
            else:
                spans.setdefault(contest_id, set()).add(borough)
        targets = [state_place.place_id]
        if borough in borough_place:
            targets.append(borough_place[borough].place_id)
        ballot_key = (precinct_id, row["Precinct_name"], contest_id)
        choice_key = (*ballot_key, row["cand_seq_nbr"], row["candidate_name"])
        if choice_key in seen_choices:
            raise ElectionReportError(f"line {line}: duplicate Alaska precinct/category candidate row")
        seen_choices.add(choice_key)
        first_row = ballot_key not in seen
        seen.add(ballot_key)
        for target in targets:
            contest = areas[target].setdefault(
                contest_id, _Contest(order=float(_count(row, "Contest_seq_nbr", line)), title=title)
            )
            if first_row:
                contest.ballots += _count(row, "total_ballots", line)
                contest.undervotes += _count(row, "total_under_votes", line)
                contest.overvotes += _count(row, "total_over_votes", line)
            name = " ".join(row["candidate_name"].split())
            entry = votes[target].setdefault(
                (contest_id, name),
                [_count(row, "cand_seq_nbr", line), (row["Party_Code"] or "").strip(), 0],
            )
            entry[2] += _count(row, "total_votes", line)
    if unknown_precincts:
        raise ElectionReportError(
            "precincts missing from the Alaska crosswalk (precincts redrawn? rebuild it with "
            f"election-analysis/tools/alaska_crosswalk.py): {', '.join(sorted(unknown_precincts))}"
        )

    outputs: dict[str, str] = {}
    for place in places:
        rows = []
        for (contest_id, name), (order, party, count) in votes[place.place_id].items():
            contest = areas[place.place_id][contest_id]
            rows.append(
                csvformat.Row(
                    contest_order=contest.order,
                    contest=contest.title,
                    contest_id=contest_id,
                    choice_order=order,
                    choice=name,
                    ballots=contest.ballots,
                    votes=count,
                    overvotes=contest.overvotes,
                    undervotes=contest.undervotes,
                    party=party,
                    jurisdiction=place.name,
                    county=place.name if place.scope == "county" else "",
                    # Read only for borough sources (a borough's share of a
                    # contest spanning several is not controlling). Short names
                    # keep the files small; most contests span many boroughs.
                    counties=()
                    if place.scope == "state"
                    else tuple(sorted(_short_name(b) for b in spans.get(contest_id, ()))),
                    write_in="write-in" in name.casefold() or "write in" in name.casefold(),
                    metadata={"geography": {**crosswalk.vintage,
                        "label": "all supplied statewide categories" if place.scope == "state" else "election-day precinct subset / approximate borough assignment",
                        "approximate": place.scope != "state", "whole_contest_coverage": place.scope == "state",
                        "split_precincts": list(crosswalk.split_precincts),
                        "included_categories": ["all supplied election-day, absentee, early, questioned, remote and federal-only overseas categories"] if place.scope == "state" else ["mapped election-day precincts"],
                        "excluded_categories": [] if place.scope == "state" else ["district absentee", "district early", "district questioned", "district remote", "federal-only overseas"],
                        "counting_policy": "each precinct/category candidate row once; repeated ballot denominator once"}},
                )
            )
        if rows:
            rows.sort(key=lambda r: (r.contest_order, r.contest_id, r.choice_order, r.choice))
            outputs[place.place_id] = csvformat.render(rows)
    return outputs
