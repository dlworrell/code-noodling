"""Load ``election-data/watch.toml``: which places to watch and when."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..model import ElectionReportError

SUPPORTED_STATES = {"washington": "Washington"}


@dataclass(frozen=True)
class WatchConfig:
    state: str
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


def load_watch_config(path: Path) -> WatchConfig:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ElectionReportError(f"watch configuration not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ElectionReportError(f"invalid TOML in {path}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise ElectionReportError("schema_version must be 1")

    watch = raw.get("watch")
    if not isinstance(watch, dict):
        raise ElectionReportError("[watch] is required")
    state_name = watch.get("state")
    if not isinstance(state_name, str) or state_name.strip().casefold() not in SUPPORTED_STATES:
        raise ElectionReportError(
            "watch.state must be one of: " + ", ".join(sorted(SUPPORTED_STATES.values()))
        )
    counties = watch.get("counties", [])
    if not isinstance(counties, list) or not all(
        isinstance(item, str) and item.strip() for item in counties
    ):
        raise ElectionReportError("watch.counties must be a list of county names")
    election = watch.get("election", "next")
    if isinstance(election, date):
        election = election.isoformat()
    if not isinstance(election, str) or (
        election != "next" and not _is_iso_date(election)
    ):
        raise ElectionReportError('watch.election must be "next" or a date like 2026-11-03')
    try:
        zone = ZoneInfo(str(watch.get("timezone", "America/Los_Angeles")))
    except ZoneInfoNotFoundError as exc:
        raise ElectionReportError("watch.timezone is not a known time zone") from exc
    prefer = watch.get("prefer_county_files", True)
    if type(prefer) is not bool:
        raise ElectionReportError("watch.prefer_county_files must be true or false")

    schedule = raw.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ElectionReportError("[schedule] must be a TOML table")
    root = raw.get("paths", {}).get("elections", "elections")
    if not isinstance(root, str) or not root.strip():
        raise ElectionReportError("paths.elections must be a directory name")

    seen: set[str] = set()
    names: list[str] = []
    for item in counties:
        key = item.strip().casefold()
        if key not in seen:
            seen.add(key)
            names.append(item.strip())

    return WatchConfig(
        state=SUPPORTED_STATES[state_name.strip().casefold()],
        counties=tuple(names),
        election=election,
        timezone=zone,
        polls_close=_time(schedule.get("polls_close"), "schedule.polls_close", "20:00"),
        election_night_until=_time(
            schedule.get("election_night_until"), "schedule.election_night_until", "02:00"
        ),
        election_night_every_minutes=_minutes(
            schedule.get("election_night_every_minutes"),
            "schedule.election_night_every_minutes",
            30,
        ),
        every_minutes=_minutes(schedule.get("every_minutes"), "schedule.every_minutes", 60),
        start=_moment(schedule.get("start"), "schedule.start", zone),
        end=_moment(schedule.get("end"), "schedule.end", zone),
        prefer_county_files=prefer,
        elections_root=path.parent / root.strip(),
    )


def _is_iso_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True
