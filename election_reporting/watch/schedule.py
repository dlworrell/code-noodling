"""When to check: from polls closing until certification.

Washington certification deadlines (RCW 29A.60.190 for counties and
RCW 29A.60.250 for the Secretary of State), counted from election day:

| Election | Counties | State |
|---|---|---|
| General  | 21 days  | 30 days |
| Primary  | 14 days  | 17 days |
| Special  | 10 days  | 10 days (the state has no role; the county date is used) |

Alaska has no counties and no county certification. The State Review Board
certifies; the Division of Elections' 2026 calendar targets 13 days after the
primary (2026-08-31) and 22 days after the general (2026-11-25), AS 15.15.450.
These are target dates, not statutory deadlines; `[watch.schedule] end`
extends the window if certification slips.

Maine posts official results only, as office workbooks weeks after the
election, and publishes no certification date. Its window is 35 days, a
margin over what the posted files show (the 2026 primary's final files are
dated 21 days after it, the 2024 general's corrected ones 30 days after).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import WatchConfig

CERTIFICATION_DAYS = {
    "washington": {
        "general": {"county": 21, "state": 30},
        "primary": {"county": 14, "state": 17},
        "special": {"county": 10, "state": 10},
    },
    "alaska": {
        "general": {"county": 22, "state": 22},
        "primary": {"county": 13, "state": 13},
        "special": {"county": 22, "state": 22},
    },
    "maine": {
        "general": {"county": 35, "state": 35},
        "primary": {"county": 35, "state": 35},
        "special": {"county": 35, "state": 35},
    },
}
# States whose window end is not a published certification date; their
# election.toml gets no certification_date.
NO_CERTIFICATION_DATE = frozenset({"maine"})


def election_kind(name: str) -> str:
    folded = name.casefold()
    if "general" in folded:
        return "general"
    if "primary" in folded:
        return "primary"
    return "special"


def certification_date(
    election_day: date, kind: str, scope: str, state_id: str = "washington"
) -> date:
    return election_day + timedelta(days=CERTIFICATION_DAYS[state_id][kind][scope])


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
    last = certification_date(
        election_day, kind, "state" if has_state else "county", config.state_id
    )
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
