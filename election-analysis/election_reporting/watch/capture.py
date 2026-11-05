# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:6581e6f1-a246-47e0-a895-f5ff3f0a5dba

"""Capture-only collection: keep official result files from more states, unparsed.

Purpose. Owner decision (2026-11-03): beyond the three parsed states
(Washington, Alaska, Maine), keep copies of the official result files of the
states whose results sites share a platform we can read, plus North Carolina.
Election-night files are overwritten as counts update, so a copy not taken at
the time cannot be downloaded later. Nothing here parses results or writes to
the repository: every body is saved unmodified by ``OriginalsKeeper`` into the
run's originals directory, which the workflow keeps as a 90-day artifact.

Inputs:
- ``data/capture.toml``: one ``[[capture]]`` table per state with its platform,
  host, jurisdiction, election date, hard ``not_before`` floor (the close of
  the state's last polls), ``end`` and cadence.
- ``directory``: the originals directory (``ORIGINALS_DIR``).
- ``now``: the current time (injectable for tests).

What each platform captures (statewide files first, for every due state, then
county files while the time budget lasts):
- ``enhanced-voting`` (Georgia, Idaho, Rhode Island, Utah, Virginia; the same
  public API as results.votewa.gov): the jurisdiction list, the election's
  metadata, every published report file, the statewide results JSON, and the
  results JSON of every county or town (their local contests are not in the
  statewide data).
- ``clarity`` (Colorado, Iowa): the election list, current version, settings,
  the statewide ``detailxml.zip`` and ``summary.zip``, and each participating
  county's ``detailxml.zip`` (precinct level, including local contests).
- ``ncsbe`` (North Carolina): the statewide precinct results zip, which already
  holds every county and every contest, local ones included.

Guarantees and limits:
- No request for a state before its ``not_before`` or after its ``end``; there
  is no force option. A held state makes no request at all.
- Cadence: every run (every 30 minutes) for ``night_hours`` after the floor,
  then once per ``every_minutes``. The decision uses clock slots, not the time
  of the last fetch, so two runs close together can both fetch; bodies are
  stored once per run by SHA-256, but the requests are still made.
- Hosts and URL prefixes are limited to ``http.CAPTURE_PREFIXES``.
- A time budget (default 240 s) bounds the whole capture so the parsed
  watcher that runs after it keeps its share of the job's 15 minutes; the
  states start in a rotating order so no state is always last.
- Every failure is reported (``::warning::``) and never stops other files,
  other states or the watcher. Returns 0 unless the configuration is invalid.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import tomllib
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from ..model import ElectionReportError
from .http import LARGE_MAX_BYTES, fetch as http_fetch
from .originals import OriginalsKeeper

PLATFORMS = ("enhanced-voting", "clarity", "ncsbe")
CLARITY_HOST = "results.enr.clarityelections.com"
NCSBE_BASE = "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS"
DEFAULT_BUDGET_SECONDS = 240.0


@dataclass(frozen=True)
class Target:
    """One state to capture."""

    state: str  # directory name, e.g. "georgia"
    platform: str
    election: date
    not_before: datetime
    end: datetime
    host: str = ""
    jurisdiction: str = ""  # enhanced-voting short name, or the Clarity state code
    counties: bool = True
    night_hours: float = 8.0
    every_minutes: int = 180


def load_targets(path: Path) -> list[Target]:
    """Read ``capture.toml``. Raises ElectionReportError on any invalid entry."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ElectionReportError(f"{path}: schema_version must be 1")
    defaults = document.get("defaults", {})
    targets = []
    for index, table in enumerate(document.get("capture", []), start=1):
        entry = {**defaults, **table}
        where = f"{path}: capture #{index}"
        try:
            target = Target(
                state=str(entry["state"]),
                platform=str(entry["platform"]),
                election=entry["election"],
                not_before=entry["not_before"],
                end=entry["end"],
                host=str(entry.get("host", "")),
                jurisdiction=str(entry.get("jurisdiction", "")),
                counties=bool(entry.get("counties", True)),
                night_hours=float(entry.get("night_hours", 8)),
                every_minutes=int(entry.get("every_minutes", 180)),
            )
        except KeyError as missing:
            raise ElectionReportError(f"{where}: missing {missing}") from None
        if target.platform not in PLATFORMS:
            raise ElectionReportError(f"{where}: platform must be one of {', '.join(PLATFORMS)}")
        if not isinstance(target.election, date) or isinstance(target.election, datetime):
            raise ElectionReportError(f"{where}: election must be a date")
        for name in ("not_before", "end"):
            moment = getattr(target, name)
            if not isinstance(moment, datetime) or moment.tzinfo is None:
                raise ElectionReportError(f"{where}: {name} must be a date-time with a UTC offset")
        if target.end <= target.not_before:
            raise ElectionReportError(f"{where}: end must be after not_before")
        if target.every_minutes < 30:
            raise ElectionReportError(f"{where}: every_minutes must be at least 30")
        if target.platform in ("enhanced-voting", "clarity") and not target.jurisdiction:
            raise ElectionReportError(f"{where}: jurisdiction is required for {target.platform}")
        if target.platform == "enhanced-voting" and not target.host:
            raise ElectionReportError(f"{where}: host is required for enhanced-voting")
        targets.append(target)
    states = [t.state for t in targets]
    if len(states) != len(set(states)):
        raise ElectionReportError(f"{path}: a state is listed twice")
    return targets


def due(target: Target, now: datetime) -> tuple[bool, str]:
    """Whether ``target`` is collected now, with the reason."""
    if now < target.not_before:
        return False, f"held until {target.not_before.isoformat()} (hard floor)"
    if now >= target.end:
        return False, f"ended {target.end.isoformat()}"
    elapsed = now - target.not_before
    if elapsed < timedelta(hours=target.night_hours):
        return True, "election night: every run"
    minutes = int(elapsed.total_seconds() // 60)
    if minutes % target.every_minutes < 30:
        return True, f"every {target.every_minutes} minutes"
    return False, f"not due (every {target.every_minutes} minutes)"


def _decoded(body: bytes) -> bytes:
    """Some hosts send gzip bodies unasked; the kept original stays as served."""
    return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body


@dataclass
class Capture:
    fetch: Callable  # fetch(url, max_bytes=...) -> Response; records originals
    log: Callable[[str], None] = print
    deadline: float = float("inf")
    clock: Callable[[], float] = time.monotonic
    fetched: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    out_of_time: list[str] = field(default_factory=list)

    def get(self, url: str, large: bool = False):
        """Fetch one URL; a failure is recorded and returns None."""
        try:
            response = self.fetch(url, max_bytes=LARGE_MAX_BYTES) if large else self.fetch(url)
        except ElectionReportError as error:
            self.failures.append(f"{url}: {error}")
            self.log(f"::warning::capture: {error}")
            return None
        self.fetched.append(url)
        return response

    def json(self, url: str):
        response = self.get(url)
        if response is None:
            return None
        try:
            return json.loads(_decoded(response.body))
        except (ValueError, OSError) as error:
            self.failures.append(f"{url}: not JSON ({error})")
            self.log(f"::warning::capture: {url} is not JSON")
            return None

    def time_left(self) -> bool:
        return self.clock() < self.deadline


# Each platform has a statewide step, returning the county URLs (fetched later).


def _enhanced_statewide(cap: Capture, t: Target) -> list[str]:
    api = f"https://{t.host}/results/public/api"
    jurisdiction = cap.json(f"{api}/jurisdictions/{t.jurisdiction}")
    if not isinstance(jurisdiction, dict):
        return []
    wanted = t.election.isoformat()
    # Every election listed for the day: a state can list several (a recount,
    # one primary per party, a special election alongside the general).
    elections = [e for e in jurisdiction.get("elections") or []
                 if str(e.get("electionDate", ""))[:10] == wanted and e.get("publicElectionId")]
    if not elections:
        cap.log(f"capture {t.state}: no {wanted} election listed yet")
        return []
    urls: set[str] = set()
    for election in elections:
        public_id = urllib.parse.quote(str(election["publicElectionId"]), safe="")
        urls.update(_enhanced_election(cap, t, api, jurisdiction, public_id))
    return sorted(urls)


def _enhanced_election(cap: Capture, t: Target, api: str, jurisdiction: dict, public_id: str) -> list[str]:
    metadata = cap.json(f"{api}/elections/{t.jurisdiction}/{public_id}")
    if isinstance(metadata, dict):
        jurisdiction_id = urllib.parse.quote(str(metadata.get("jurisdictionId") or ""), safe="")
        for category in metadata.get("publicReportCategories") or []:
            for report in category.get("reports") or []:
                blob = report.get("blobName")
                if blob and jurisdiction_id:
                    cap.get(f"https://{t.host}/cdn/results/{jurisdiction_id}/"
                            f"{urllib.parse.quote(str(blob))}", large=True)
    data = cap.json(f"{api}/elections/{t.jurisdiction}/{public_id}/data")
    if not t.counties or not isinstance(data, dict):
        return []
    with_election = {str(e.get("jurisdictionId", "")).casefold()
                     for e in data.get("localityElections") or []}
    urls = []
    for child in jurisdiction.get("childLocalities") or []:
        short = str(child.get("shortName") or "")
        if short and str(child.get("id", "")).casefold() in with_election:
            urls.append(f"{api}/elections/{urllib.parse.quote(short, safe='')}/{public_id}/data")
    return sorted(urls)


def _clarity_statewide(cap: Capture, t: Target) -> list[str]:
    base = f"https://{CLARITY_HOST}/{t.jurisdiction}"
    elections = cap.json(f"{base}/elections.json")
    if not isinstance(elections, list):
        return []
    wanted = f"{t.election.month}/{t.election.day}/{t.election.year}"
    matches = [e for e in elections if str(e.get("Date", "")).split(" ")[0] == wanted
               and str(e.get("EID", "")).isdigit()]
    if not matches:
        cap.log(f"capture {t.state}: no {t.election.isoformat()} election listed yet")
        return []
    urls: set[str] = set()
    for match in matches:
        urls.update(_clarity_election(cap, t, base, str(match["EID"])))
    return sorted(urls)


def _clarity_election(cap: Capture, t: Target, base: str, eid: str) -> list[str]:
    version = cap.get(f"{base}/{eid}/current_ver.txt")
    if version is None:
        return []
    ver = urllib.parse.quote(version.body.decode("ascii", "replace").strip(), safe="")
    if not ver.isdigit():
        cap.failures.append(f"{base}/{eid}/current_ver.txt: unexpected version {ver!r}")
        return []
    settings = cap.json(f"{base}/{eid}/{ver}/json/en/electionsettings.json")
    for name in ("detailxml.zip", "summary.zip"):
        cap.get(f"{base}/{eid}/{ver}/reports/{name}", large=True)
    if not t.counties or not isinstance(settings, dict):
        return []
    details = (settings.get("settings") or {}).get("electiondetails") or {}
    urls = []
    for item in details.get("participatingcounties") or []:
        parts = str(item).split("|")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            county = urllib.parse.quote(parts[0], safe="_")
            urls.append(f"{base}/{county}/{parts[1]}/{parts[2]}/reports/detailxml.zip")
    return sorted(urls)


def _ncsbe_statewide(cap: Capture, t: Target) -> list[str]:
    day = t.election
    cap.get(f"{NCSBE_BASE}/{day:%Y_%m_%d}/results_pct_{day:%Y%m%d}.zip", large=True)
    return []  # the statewide file already holds every county and contest


STATEWIDE = {"enhanced-voting": _enhanced_statewide, "clarity": _clarity_statewide, "ncsbe": _ncsbe_statewide}


def capture(targets: list[Target], keeper: OriginalsKeeper, now: datetime,
            fetch: Callable = http_fetch, log: Callable[[str], None] = print,
            budget_seconds: float = DEFAULT_BUDGET_SECONDS,
            clock: Callable[[], float] = time.monotonic) -> Capture:
    """Capture every due target's files into ``keeper``; see the module docstring."""

    cap = Capture(fetch=keeper.recording(fetch), log=log, deadline=clock() + budget_seconds, clock=clock)
    active = []
    for target in targets:
        ok, reason = due(target, now)
        log(f"capture {target.state}: {reason}")
        if ok:
            active.append(target)
    if not active:
        return cap
    start = int(now.timestamp() // 1800) % len(active)  # rotate who goes first each half hour
    active = active[start:] + active[:start]
    county_urls: dict[str, list[str]] = {}
    for target in active:
        if not cap.time_left():
            cap.out_of_time.append(target.state)
            continue
        keeper.state = target.state
        county_urls[target.state] = STATEWIDE[target.platform](cap, target)
    for target in active:
        for url in county_urls.get(target.state, []):
            if not cap.time_left():
                cap.out_of_time.append(f"{target.state} counties")
                break
            keeper.state = target.state
            cap.get(url, large=target.platform == "clarity")
    if cap.out_of_time:
        log(f"::warning::capture time budget reached; not captured this run: {', '.join(cap.out_of_time)}")
    log(f"capture: {len(cap.fetched)} file(s) fetched, {len(cap.failures)} failure(s)")
    return cap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m election_reporting.watch.capture")
    parser.add_argument("directory", type=Path, help="originals directory (ORIGINALS_DIR)")
    parser.add_argument("--config", type=Path, default=Path("election-analysis/data/capture.toml"))
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS, help="seconds")
    parser.add_argument("--now", help="ISO time, for rehearsal only (default: the real clock)")
    args = parser.parse_args(argv)
    try:
        targets = load_targets(args.config)
    except (ElectionReportError, OSError, tomllib.TOMLDecodeError) as error:
        print(f"::error::capture configuration: {error}")
        return 2
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    keeper = OriginalsKeeper(args.directory)
    capture(targets, keeper, now, budget_seconds=args.budget)
    try:
        keeper.save()
    except OSError as error:
        print(f"::warning::could not write the originals manifest: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
