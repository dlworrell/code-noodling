"""Official result sources: the Washington results site and King County Elections.

``results.votewa.gov`` (Washington Secretary of State) serves the state and all
39 counties through one public API:

- ``/results/public/api/jurisdictions/washington``: counties and elections;
- ``/results/public/api/elections/<jurisdiction>/<YYYYMMDD>``: an election's
  metadata, including downloadable report files;
- ``/results/public/api/elections/<jurisdiction>/<YYYYMMDD>/data``: full
  results for that jurisdiction;
- ``/cdn/results/<jurisdiction id>/<report file>``: report downloads.

King County Elections also posts its own nightly CSV export
(``webresults-YYYYMMDD.csv``, ``-final`` after certification) on the county
results page for each election. It carries ballots, overvotes and undervotes,
which the state API does not, so it is preferred for King County.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import date

from ..model import ElectionReportError
from .http import Fetch

VOTEWA_API = "https://results.votewa.gov/results/public/api"
VOTEWA_CDN = "https://results.votewa.gov/cdn/results"
STATE_REPORT_NAME = "All Results Excel"
# County breakdown of every contest plus every precinct (issue #14).
PRECINCT_REPORT_NAME = "Summary and Precinct Results Excel"
KC_RESULTS = "https://kingcounty.gov/en/dept/elections/results"
KC_FILE = re.compile(
    r'href="(?P<url>https://cdn\.kingcounty\.gov/[^"]*/webresults-(?P<date>\d{8})(?P<final>-final)?\.csv[^"]*)"',
    re.IGNORECASE,
)
MONTHS = (
    "january february march april may june july august september october november december"
).split()


@dataclass(frozen=True)
class Place:
    """One watched jurisdiction."""

    place_id: str  # directory name, e.g. "washington" or "king-county"
    name: str  # display name, e.g. "Washington" or "King County"
    scope: str  # "state" or "county"
    votewa: str  # results.votewa.gov short name
    county_files: bool = False  # King County's own CSV export is used


def _json(fetch: Fetch, url: str) -> dict:
    response = fetch(url)
    try:
        value = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ElectionReportError(f"{url} did not return JSON") from exc
    if not isinstance(value, dict):
        raise ElectionReportError(f"{url} did not return a JSON object")
    return value


def _english(texts: object) -> str:
    if isinstance(texts, list):
        for item in texts:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                if item.get("languageId") in ("en", None):
                    return " ".join(item["text"].split())
    return ""


def state_jurisdiction(fetch: Fetch) -> dict:
    return _json(fetch, f"{VOTEWA_API}/jurisdictions/washington")


def resolve_places(
    jurisdiction: dict,
    state: str,
    counties: tuple[str, ...],
    prefer_county_files: bool,
    all_counties: bool = False,
) -> list[Place]:
    """Map county names such as ``"King"`` to results-site jurisdictions."""
    known: dict[str, tuple[str, str]] = {}
    for child in jurisdiction.get("childLocalities") or []:
        name = _english(child.get("name"))
        short = str(child.get("shortName") or "")
        if name and short:
            known[name.casefold()] = (name, short)
            known[name.casefold().removesuffix(" county")] = (name, short)
    places = [Place("washington", state, "state", "washington")]
    unknown: list[str] = []
    if all_counties:
        counties = tuple(sorted({name for name, _ in known.values()}))
    for county in counties:
        match = known.get(county.casefold()) or known.get(county.casefold().removesuffix(" county"))
        if match is None:
            unknown.append(county)
            continue
        name, short = match
        place_id = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
        places.append(
            Place(
                place_id,
                name,
                "county",
                short,
                county_files=prefer_county_files and short == "king-county-wa",
            )
        )
    if unknown:
        valid = sorted({name for name, _ in known.values()})
        raise ElectionReportError(
            f"unknown {state} county name(s): {', '.join(unknown)}. Valid names: {', '.join(valid)}"
        )
    return places


@dataclass(frozen=True)
class ElectionRef:
    day: date
    public_id: str  # YYYYMMDD
    name: str


def list_elections(jurisdiction: dict) -> list[ElectionRef]:
    elections = []
    for item in jurisdiction.get("elections") or []:
        try:
            day = date.fromisoformat(str(item.get("electionDate")))
        except ValueError:
            continue
        public_id = str(item.get("publicElectionId") or day.strftime("%Y%m%d"))
        elections.append(ElectionRef(day, public_id, _english(item.get("name"))))
    return sorted(elections, key=lambda item: item.day)


def election_metadata(fetch: Fetch, place: Place, election: ElectionRef) -> dict:
    return _json(fetch, f"{VOTEWA_API}/elections/{place.votewa}/{election.public_id}")


def results_data(fetch: Fetch, place: Place, election: ElectionRef) -> dict:
    return _json(fetch, f"{VOTEWA_API}/elections/{place.votewa}/{election.public_id}/data")


def report_url(metadata: dict, report_name: str = STATE_REPORT_NAME) -> str | None:
    """The download URL of a named report, or ``None`` when it is not published yet."""
    jurisdiction_id = str(metadata.get("jurisdictionId") or "")
    for category in metadata.get("publicReportCategories") or []:
        for report in category.get("reports") or []:
            if report.get("reportName") == report_name and report.get("blobName"):
                blob = urllib.parse.quote(str(report["blobName"]))
                if not jurisdiction_id:
                    return None
                return f"{VOTEWA_CDN}/{jurisdiction_id}/{blob}"
    return None


def king_county_page(fetch: Fetch, election_day: date) -> str | None:
    """Find King County's results page for an election from its year index."""
    year_url = f"{KC_RESULTS}/{election_day.year}"
    page = fetch(year_url).body.decode("utf-8", "replace")
    month = MONTHS[election_day.month - 1]
    pattern = re.compile(rf'href="(/en/dept/elections/results/{election_day.year}/{month}[^"#?]*)"')
    match = pattern.search(page)
    if match is None:
        return None
    return urllib.parse.urljoin(year_url, match.group(1))


@dataclass(frozen=True)
class CountyFile:
    day: date
    final: bool
    url: str


def king_county_files(fetch: Fetch, page_url: str) -> list[CountyFile]:
    page = fetch(page_url).body.decode("utf-8", "replace")
    files: dict[str, CountyFile] = {}
    for match in KC_FILE.finditer(page):
        url = html.unescape(match.group("url"))
        stamp = match.group("date")
        day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
        item = CountyFile(day, bool(match.group("final")), url)
        files[f"{stamp}{'-final' if item.final else ''}"] = item
    return sorted(files.values(), key=lambda item: (item.day, item.final))
