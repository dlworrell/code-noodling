"""Load ``election-data/watch.toml``: which places to watch and when."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..model import ElectionReportError

SUPPORTED_STATES = {"washington": "Washington"}
# Per-state defaults; [[watch]] tables can override timezone and schedule.
STATE_DEFAULTS = {
    "washington": {"timezone": "America/Los_Angeles", "polls_close": "20:00"},
}


@dataclass(frozen=True)
class WatchConfig:
    state: str
    state_id: str  # directory name under the elections root, e.g. "washington"
    counties: tuple[str, ...]
    election: str  # "next" or an ISO date
    timezone: ZoneInfo
    polls_close: time
    election_night_until: time
    election_night_every_minutes: int
    every_minutes: int
    start: datetime | None
    end: datetime | None
    prefer_county_files: bool
    elections_root: Path


def _time(value: object, location: str, default: str) -> time:
    text = default if value is None else value
    if isinstance(text, time):
        return text
    if not isinstance(text, str):
        raise ElectionReportError(f"{location} must be HH:MM")
    try:
        return time.fromisoformat(text)
    except ValueError as exc:
        raise ElectionReportError(f"{location} must be HH:MM") from exc


def _minutes(value: object, location: str, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not 5 <= value <= 24 * 60:
        raise ElectionReportError(f"{location} must be whole minutes from 5 through 1440")
    return value


def _moment(value: object, location: str, zone: ZoneInfo) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ElectionReportError(f"{location} must be an ISO date-time") from exc
    if not isinstance(value, datetime):
        raise ElectionReportError(f"{location} must be an ISO date-time")
    return value if value.tzinfo else value.replace(tzinfo=zone)


@dataclass(frozen=True)
class WatchSettings:
    """Every watched state from one ``watch.toml``."""

    elections_root: Path  # election-data/elections; each state has a subdirectory
    watches: tuple[WatchConfig, ...]


def _parse_watch(watch: dict, location: str, elections_root: Path) -> WatchConfig:
    state_name = watch.get("state")
    if not isinstance(state_name, str) or state_name.strip().casefold() not in SUPPORTED_STATES:
        raise ElectionReportError(
            f"{location}.state must be one of: " + ", ".join(sorted(SUPPORTED_STATES.values()))
        )
    state_id = state_name.strip().casefold()
    defaults = STATE_DEFAULTS[state_id]
    counties = watch.get("counties", [])
    if not isinstance(counties, list) or not all(
        isinstance(item, str) and item.strip() for item in counties
    ):
        raise ElectionReportError(f"{location}.counties must be a list of county names")
    election = watch.get("election", "next")
    if isinstance(election, date):
        election = election.isoformat()
    if not isinstance(election, str) or (
        election != "next" and not _is_iso_date(election)
    ):
        raise ElectionReportError(f'{location}.election must be "next" or a date like 2026-11-03')
    try:
        zone = ZoneInfo(str(watch.get("timezone", defaults["timezone"])))
    except ZoneInfoNotFoundError as exc:
        raise ElectionReportError(f"{location}.timezone is not a known time zone") from exc
    prefer = watch.get("prefer_county_files", True)
    if type(prefer) is not bool:
        raise ElectionReportError(f"{location}.prefer_county_files must be true or false")
    schedule = watch.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ElectionReportError(f"{location}.schedule must be a TOML table")
    where = f"{location}.schedule"

    seen: set[str] = set()
    names: list[str] = []
    for item in counties:
        key = item.strip().casefold()
        if key not in seen:
            seen.add(key)
            names.append(item.strip())

    return WatchConfig(
        state=SUPPORTED_STATES[state_id],
        state_id=state_id,
        counties=tuple(names),
        election=election,
        timezone=zone,
        polls_close=_time(schedule.get("polls_close"), f"{where}.polls_close", defaults["polls_close"]),
        election_night_until=_time(
            schedule.get("election_night_until"), f"{where}.election_night_until", "02:00"
        ),
        election_night_every_minutes=_minutes(
            schedule.get("election_night_every_minutes"),
            f"{where}.election_night_every_minutes",
            30,
        ),
        every_minutes=_minutes(schedule.get("every_minutes"), f"{where}.every_minutes", 60),
        start=_moment(schedule.get("start"), f"{where}.start", zone),
        end=_moment(schedule.get("end"), f"{where}.end", zone),
        prefer_county_files=prefer,
        elections_root=elections_root / state_id,
    )


def load_watch_settings(path: Path) -> WatchSettings:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ElectionReportError(f"watch configuration not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ElectionReportError(f"invalid TOML in {path}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise ElectionReportError("schema_version must be 1")
    root = raw.get("paths", {}).get("elections", "elections")
    if not isinstance(root, str) or not root.strip():
        raise ElectionReportError("paths.elections must be a directory name")
    elections_root = path.parent / root.strip()

    tables = raw.get("watch")
    if not isinstance(tables, list) or not tables or not all(isinstance(t, dict) for t in tables):
        raise ElectionReportError("at least one [[watch]] table is required")
    watches = tuple(
        _parse_watch(table, f"watch[{index}]", elections_root)
        for index, table in enumerate(tables, start=1)
    )
    states = [w.state_id for w in watches]
    duplicates = sorted({s for s in states if states.count(s) > 1})
    if duplicates:
        raise ElectionReportError(f"each state may be watched once; repeated: {', '.join(duplicates)}")
    return WatchSettings(elections_root=elections_root, watches=watches)


def _is_iso_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True
