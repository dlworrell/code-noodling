"""Maine Bureau of Corporations, Elections and Commissions results.

Maine posts no election-night state results. Official results arrive as one
workbook per office on the results page, typically 3 to 4 weeks after the
election (the 2026 primary's final files are dated 21 days after it, the 2024
general's corrected files 30 days after). Each workbook lists votes by
municipality with a county code; offices with districts or counties stack one
block per district or county, each with its own header rows.

Workbook layouts vary between offices and years (candidate names above or
below the column-header row, ``BLANK`` and ``TBC`` columns, several kinds of
total row), so every parsed block is checked against the workbook's own total
row and a block that does not reconcile is refused rather than guessed at.

``STATE UOCAVA`` rows (military and overseas voters) carry no municipality, so
they count in statewide totals only, never in a county's.

www.maine.gov sits behind a firewall that refuses every non-browser client,
including honestly identified ones, so requests to that host send a browser
User-Agent (``http.USER_AGENT_OVERRIDES``); each snapshot records this.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from ..ingest import xlsx_worksheets
from ..model import ElectionReportError
from . import csvformat
from .http import Fetch
from .sources import ElectionRef, Place

SITE = "https://www.maine.gov"
RESULTS_PAGE = SITE + "/sos/elections-voting/election-results-data"
YEAR_PAGE = RESULTS_PAGE + "/election-results-{year}"
COUNTIES = {
    "AND": "Androscoggin",
    "ARO": "Aroostook",
    "CUM": "Cumberland",
    "FRA": "Franklin",
    "HAN": "Hancock",
    "KEN": "Kennebec",
    "KNO": "Knox",
    "LIN": "Lincoln",
    "OXF": "Oxford",
    "PEN": "Penobscot",
    "PIS": "Piscataquis",
    "SAG": "Sagadahoc",
    "SOM": "Somerset",
    "WAL": "Waldo",
    "WAS": "Washington",
    "YOR": "York",
}
PARTIES = {
    "democratic",
    "republican",
    "independent",
    "green independent",
    "libertarian",
    "unenrolled",
    "green",
    "no party",
}
NOTE = (
    "Fetched with a browser User-Agent: www.maine.gov's firewall refuses non-browser "
    "clients. Municipal votes only; STATE UOCAVA (military and overseas) ballots are "
    "in the statewide totals only."
)
STATE_NOTE = (
    "Fetched with a browser User-Agent: www.maine.gov's firewall refuses non-browser "
    "clients. Combined from the office workbooks listed in the raw ZIP's SOURCES.txt."
)
# Column headers that identify rows rather than hold votes. Spellings vary
# ("DIS", "DIST", "District"; "Municipality", "Municipalitly", "Municipaity", "TOWN").
_KEY_HEADERS = (
    (re.compile(r"^(dis|dist|district)$", re.IGNORECASE), "district"),
    (re.compile(r"^(cty|county)$", re.IGNORECASE), "county"),
    (re.compile(r"^(municip\w*|town)$", re.IGNORECASE), "municipality"),
    (re.compile(r"^office$", re.IGNORECASE), "office"),
    (re.compile(r"^ward[- ]?precinct$", re.IGNORECASE), "precinct"),
)
_BALLOTS = re.compile(r"^(tbc|tbd|total ballots cast|total ballots)$", re.IGNORECASE)
_BLANK = re.compile(r"^blanks?$", re.IGNORECASE)
_OTHERS = re.compile(r"^(others?|write-?ins?|scattering)$", re.IGNORECASE)
_QUESTION = re.compile(r"\bquestion\s*(\d+)\b", re.IGNORECASE)
_DECLARED = re.compile(r"\s*\(declared write-?in\)", re.IGNORECASE)
_NUMBER = re.compile(r"^\d+(?:\.0+)?$")
_DECIMAL = re.compile(r"^\d*\.\d+(?:E-?\d+)?$", re.IGNORECASE)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def list_elections(today: date) -> list[ElectionRef]:
    """June primaries (second Tuesday, even years) and November elections.

    November elections are held every year: general elections in even years,
    referendum elections in odd years.
    """
    elections = []
    for year in range(today.year - 2, today.year + 3):
        november = _nth_weekday(year, 11, 0, 1) + timedelta(days=1)
        if year % 2 == 0:
            primary = _nth_weekday(year, 6, 1, 2)
            elections.append(ElectionRef(primary, primary.strftime("%Y%m%d"), f"{year} Primary Election"))
            elections.append(ElectionRef(november, november.strftime("%Y%m%d"), f"{year} General Election"))
        else:
            elections.append(ElectionRef(november, november.strftime("%Y%m%d"), f"{year} Referendum Election"))
    return sorted(elections, key=lambda item: item.day)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def resolve_places(names: tuple[str, ...], all_counties: bool) -> list[Place]:
    """Statewide plus the named counties ("York" or "York County")."""
    known = {name.casefold(): name for name in COUNTIES.values()}
    wanted = tuple(COUNTIES.values()) if all_counties else names
    places = [Place("maine", "Maine", "state", "")]
    unknown = []
    for item in wanted:
        name = known.get(item.casefold().removesuffix(" county").strip())
        if name is None:
            unknown.append(item)
        elif all(place.name != f"{name} County" for place in places):
            places.append(Place(f"{_slug(name)}-county", f"{name} County", "county", ""))
    if unknown:
        raise ElectionReportError(
            f"unknown Maine county name(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(COUNTIES.values())}"
        )
    return places


@dataclass(frozen=True)
class Link:
    url: str
    label: str  # office label, e.g. "United States Senator" or "Governor - Democratic"
    rcv: bool  # listed in a ranked-choice section


def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def _date_label(day: date) -> str:
    return f"{day:%B} {day.day}, {day.year}"


def _office_label(heading: str, link_text: str) -> str:
    label = re.sub(r"\s*\((?:excel|xlsx)\)\s*$", "", link_text, flags=re.IGNORECASE).strip()
    label = re.sub(r"\s+by\s+County/Town$", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+Results$", "", label, flags=re.IGNORECASE)
    if label.casefold() in PARTIES:
        return f"{heading} - {label}"
    generic = re.search(r"(?i)^(tabulation of votes|rcv\b)|first choice|1st choice", label)
    if generic or not label:
        return heading
    return label


def links_on_page(page: str, page_url: str, day: date) -> list[Link]:
    """Result workbooks listed under the election's headings on a results page.

    Cast vote record exports, candidate lists and the President-by-district
    workbook (a second view of the same votes) are skipped.
    """
    wanted = _date_label(day).casefold()
    links: list[Link] = []
    in_election = False
    rcv = False
    heading = ""
    skip_section = False
    token = re.compile(r"<(h[2-4])\b[^>]*>(.*?)</\1>|<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.S | re.I)
    for match in token.finditer(page):
        level, heading_html, href, link_html = match.groups()
        if level:
            text = _text(heading_html)
            if level.lower() == "h2":
                in_election = wanted in text.casefold()
                folded = text.casefold()
                rcv = "ranked choice" in folded and "non-ranked" not in folded
                heading = text
                skip_section = False
            elif in_election:
                skip_section = "cast vote record" in text.casefold()
                if level.lower() == "h3" and not skip_section:
                    heading = text
            continue
        if not in_election or skip_section:
            continue
        url = urllib.parse.urljoin(page_url, html.unescape(href))
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1]).casefold()
        text = _text(link_html)
        if not name.endswith(".xlsx") or "cvr" in name or "candidate list" in name:
            continue
        if "by congressional district" in text.casefold():
            continue
        links.append(Link(url=url, label=_office_label(heading, text), rcv=rcv))
    return links


def result_links(fetch: Fetch, election: ElectionRef) -> list[Link]:
    """The election's workbooks: on the current results page, else its year's page."""
    for url in (RESULTS_PAGE, YEAR_PAGE.format(year=election.day.year)):
        try:
            page = fetch(url).body.decode("utf-8", errors="replace")
        except ElectionReportError:
            continue
        found = links_on_page(page, url, election.day)
        if found:
            return found
    return []


@dataclass
class Contest:
    name: str
    contest_id: str
    rcv: bool
    choices: list[tuple[str, str, bool]]  # (name, party, write-in)
    # (county code or "" for UOCAVA, votes per choice, blank, ballots)
    rows: list[tuple[str, list[int], int, int]] = field(default_factory=list)


def _number(value: str, where: str) -> int:
    text = value.strip().replace(",", "")
    if not text:
        return 0
    if not _NUMBER.match(text):
        raise ElectionReportError(f"{where}: {value!r} is not a whole number")
    return int(float(text))


def _row_kind(row: list[str], keys: dict[str, int]) -> str:
    """'data', 'uocava', 'total', 'percent', 'blank' or 'header'."""
    cells = [cell.strip() for cell in row]
    if not any(cells):
        return "blank"
    # An OFFICE column repeats the office name on every row, totals included.
    office = keys.get("office")
    filled = [cell.replace(",", "") for i, cell in enumerate(cells) if cell and i != office]
    if all(_NUMBER.match(cell) for cell in filled):
        return "total"  # an unlabeled total row, as in the RCV first-choice workbooks
    if all(_NUMBER.match(cell) or _DECIMAL.match(cell) for cell in filled):
        return "percent"
    joined = " ".join(cells).casefold()
    county = cells[keys["county"]] if "county" in keys and keys["county"] < len(cells) else ""
    place = cells[keys["municipality"]] if "municipality" in keys and keys["municipality"] < len(cells) else ""
    numbers = [cell for cell in cells if _NUMBER.match(cell.replace(",", ""))]
    # "STATE UOCAVA" in most workbooks, plain "UOCAVA" in some referendum ones.
    if "uocava" in joined and numbers:
        return "uocava"
    if "total" in joined and numbers:
        return "total"
    if county.upper() in COUNTIES and place and numbers:
        return "data"
    return "header"


def _key_columns(header_rows: list[list[str]]) -> dict[str, int]:
    keys: dict[str, int] = {}
    for row in header_rows:
        for index, cell in enumerate(row):
            key = next((name for pattern, name in _KEY_HEADERS if pattern.match(cell.strip())), None)
            if key and key not in keys:
                keys[key] = index
    return keys


def _blocks(rows: list[list[str]], where: str):
    """Split a worksheet into (header rows, body rows) blocks."""
    keys: dict[str, int] = {}
    header: list[list[str]] = []
    body: list[list[str]] = []
    blocks = []
    for row in rows:
        kind = _row_kind(row, keys or {"county": 0, "municipality": 1})
        if kind in {"blank", "percent"}:
            continue
        if kind == "header":
            if body:
                blocks.append((header, body, keys))
                header, body = [], []
            header.append(row)
            keys = _key_columns(header) or keys
            continue
        if not header:
            raise ElectionReportError(f"{where}: results before any column headers")
        body.append(row)
    if body:
        blocks.append((header, body, keys))
    return blocks


def _columns(header: list[list[str]], keys: dict[str, int]):
    """Classify value columns: choices, blank, ballots, and referendum questions."""
    width = max(len(row) for row in header)
    texts: dict[int, list[str]] = {}
    for row in header:
        for index, cell in enumerate(row):
            if index not in keys.values() and cell.strip():
                texts.setdefault(index, []).append(" ".join(cell.split()))
    questions: dict[int, str] = {}
    current = ""
    for index in range(width):
        for text in texts.get(index, []):
            match = _QUESTION.search(text)
            if match:
                current = f"Question {int(match.group(1))}"
        if current:
            questions[index] = current
    return texts, questions


def parse_workbook(raw: bytes, *, label: str, rcv: bool, url: str) -> tuple[list[Contest], list[str]]:
    """Parse one office workbook into contests, each reconciled to its total row.

    Returns the contests and a list of problems: a block that cannot be read or
    does not reconcile is left out and reported, and the other blocks are kept.
    """
    contests: list[Contest] = []
    problems: list[str] = []
    for sheet, rows in xlsx_worksheets(Path(url.rsplit("/", 1)[-1]), raw):
        where = f"{label} ({sheet})"
        for number, (header, body, keys) in enumerate(_blocks(rows, where), start=1):
            try:
                if "county" not in keys:
                    raise ElectionReportError(f"{where} block {number}: no CTY column")
                contests.extend(_parse_block(header, body, keys, label, rcv, f"{where} block {number}"))
            except ElectionReportError as error:
                problems.append(str(error))
    names = [contest.name for contest in contests]
    for name in sorted(set(names)):
        if names.count(name) > 1:
            problems.append(f"{label}: contest {name!r} appears in more than one block; left out")
            contests = [contest for contest in contests if contest.name != name]
    return contests, problems


def _parse_block(header, body, keys, label, rcv, where) -> list[Contest]:
    texts, questions = _columns(header, keys)
    ballots_col = next((i for i, t in texts.items() if any(_BALLOTS.match(x) for x in t)), None)
    groups: dict[str, dict[str, object]] = {}
    for index in sorted(texts):
        if index == ballots_col:
            continue
        values = texts[index]
        group = questions.get(index, "")
        entry = groups.setdefault(group, {"choices": [], "blank": None})
        if any(_BLANK.match(x) for x in values):
            entry["blank"] = index
            continue
        if any(_QUESTION.search(x) for x in values) and len(values) == 1:
            continue  # a question label alone above a YES column is handled by YES
        names = [x for x in values if not _QUESTION.search(x)]
        party = next((x for x in names if x.casefold() in PARTIES or "/" in x and x.isupper()), "")
        others = any(_OTHERS.match(x) for x in names)
        declared = any("write-in" in x.casefold() for x in names)
        candidates = [_DECLARED.sub("", x) for x in names if x != party and not x.startswith("(")]
        name = next((x for x in candidates if "," in x), candidates[0] if candidates else "")
        if others:
            name = "Write-in"
        if not name:
            raise ElectionReportError(f"{where}: column {index + 1} has no choice name")
        entry["choices"].append((index, name, party, others or declared))

    district = ""
    counties_in_block = set()
    data = []
    total = None
    for row in body:
        kind = _row_kind(row, keys)
        cell = lambda i: row[i].strip() if i is not None and i < len(row) else ""  # noqa: E731
        if kind == "total":
            total = row
            continue
        if kind == "percent":
            continue
        if kind == "data":
            counties_in_block.add(cell(keys["county"]).upper())
        if "district" in keys and cell(keys["district"]):
            district = district or cell(keys["district"])
        data.append((kind, row))
    if total is None:
        raise ElectionReportError(f"{where}: no total row to check the block against")

    contests = []
    for group, entry in groups.items():
        choices = entry["choices"]
        if not choices:
            continue
        blank_col = entry["blank"]
        value_cols = [index for index, *_ in choices] + ([blank_col] if blank_col is not None else [])
        if ballots_col is not None:
            value_cols.append(ballots_col)
        sums = {index: 0 for index in value_cols}
        rows_out = []
        for kind, row in data:
            values = {i: _number(row[i] if i < len(row) else "", where) for i in value_cols}
            for i, v in values.items():
                sums[i] += v
            votes = [values[index] for index, *_ in choices]
            blank = values[blank_col] if blank_col is not None else 0
            ballots = values[ballots_col] if ballots_col is not None else sum(votes) + blank
            county = row[keys["county"]].strip().upper() if kind == "data" else ""
            rows_out.append((county, votes, blank, ballots))
        for index in value_cols:
            expected = _number(total[index] if index < len(total) else "", where)
            if expected != sums[index]:
                raise ElectionReportError(
                    f"{where}: column {index + 1} sums to {sums[index]:,} but the total row says {expected:,}"
                )
        name = label
        if group:
            name = f"{label} {group}"
        elif _is_county_office(label) and counties_in_block:
            # County offices repeat district numbers from county to county. A
            # block is one county's contest; a stray row coded to another
            # county (one in the 2026 Hancock sheriff primary) does not rename it.
            main = max(sorted(counties_in_block), key=lambda code: sum(r[0] == code for r in rows_out))
            name = f"{label} - {COUNTIES[main]}"
            if district:
                name = f"{name} District {district}"
        elif district and not re.search(r"\bdistrict\s+\d+\b", label, re.IGNORECASE):
            name = f"{label} District {district}"
        contests.append(
            Contest(
                name=name,
                contest_id=_slug(name),
                rcv=rcv,
                choices=[(n, p, w) for _, n, p, w in choices],
                rows=rows_out,
            )
        )
    return contests


COUNTY_OFFICES = (
    "judge of probate",
    "register of probate",
    "county treasurer",
    "register of deeds",
    "sheriff",
    "district attorney",
    "county commissioner",
    "county referendum",
    "county budget",
)


def _is_county_office(label: str) -> bool:
    folded = label.casefold()
    return any(office in folded for office in COUNTY_OFFICES)


def convert(contests: list[Contest], places: list[Place]) -> dict[str, str]:
    """One normalized CSV per place from the parsed contests."""
    by_county = {f"{name} County": code for code, name in COUNTIES.items()}
    outputs: dict[str, str] = {}
    for place in places:
        code = by_county.get(place.name) if place.scope == "county" else None
        rows = []
        for order, contest in enumerate(contests, start=1):
            selected = [r for r in contest.rows if code is None or r[0] == code]
            if not selected:
                continue
            spans = sorted({COUNTIES[r[0]] for r in contest.rows if r[0]})
            ballots = sum(r[3] for r in selected)
            blank = sum(r[2] for r in selected)
            for position, (name, party, write_in) in enumerate(contest.choices):
                rows.append(
                    csvformat.Row(
                        contest_order=float(order),
                        contest=contest.name,
                        contest_id=contest.contest_id,
                        choice_order=position + 1,
                        choice=name,
                        ballots=ballots,
                        votes=sum(r[1][position] for r in selected),
                        overvotes=None,
                        undervotes=blank,
                        party=party,
                        jurisdiction=place.name,
                        county=place.name if place.scope == "county" else "",
                        counties=() if place.scope == "state" else tuple(spans),
                        write_in=write_in,
                    )
                )
        if rows:
            outputs[place.place_id] = csvformat.render(rows)
    return outputs
