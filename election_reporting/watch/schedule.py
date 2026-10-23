"""When to check: from polls closing until certification.

Washington certification deadlines (RCW 29A.60.190 for counties and
RCW 29A.60.250 for the Secretary of State), counted from election day:

| Election | Counties | State |
|---|---|---|
| General  | 21 days  | 30 days |
| Primary  | 14 days  | 17 days |
| Special  | 10 days  | 10 days (the state has no role; the county date is used) |
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import WatchConfig

CERTIFICATION_DAYS = {
    "general": {"county": 21, "state": 30},
    "primary": {"county": 14, "state": 17},
    "special": {"county": 10, "state": 10},
}


def election_kind(name: str) -> str:
    folded = name.casefold()
    if "general" in folded:
        return "general"
    if "primary" in folded:
        return "primary"
    return "special"


def certification_date(election_day: date, kind: str, scope: str) -> date:
    return election_day + timedelta(days=CERTIFICATION_DAYS[kind][scope])


@dataclass(frozen=True)
class Window:
    start: datetime
    election_night_end: datetime
    end: datetime

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end


def watch_window(config: WatchConfig, election_day: date, kind: str, has_state: bool) -> Window:
    """The checking window: polls close until the last certification day ends."""
    zone = config.timezone
    start = config.start or datetime.combine(election_day, config.polls_close, zone)
    night_end = datetime.combine(
        election_day + timedelta(days=1), config.election_night_until, zone
    )
    last = certification_date(election_day, kind, "state" if has_state else "county")
    end = config.end or datetime.combine(last + timedelta(days=1), datetime.min.time(), zone)
    return Window(start=start, election_night_end=max(start, night_end), end=end)


def is_due(config: WatchConfig, window: Window, now: datetime) -> bool:
    """Whether a run at ``now`` should check the sources.

    The workflow is scheduled every 30 minutes. On election night every run
    checks; afterwards only runs in the first ``every_minutes`` slot of the
    period do, measured from the window start, so hourly is one run in two.
    """
    if not window.contains(now):
        return False
    if now <= window.election_night_end:
        step = config.election_night_every_minutes
    else:
        step = config.every_minutes
    elapsed = int((now - window.start).total_seconds() // 60)
    slot = 30  # the workflow's cron interval
    return elapsed % step < slot
