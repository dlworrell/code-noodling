# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:7ab9e687-a49b-4516-b5a0-c3f0756da0fa

"""Daily cadence gate for the Election Source Watch workflow.

Owner decision (2026-11-06): after the first 24 hours of November 11 (end of
Nov 11 Pacific, 2026-11-12T08:00Z) checks move to once every 24 hours. The
workflow keeps its hourly :07 schedule; this gate decides whether a scheduled
fire runs the watcher.

A cadence day starts at 02:00 UTC (6 PM Pacific, after Washington's afternoon
county releases). The first scheduled fire of a cadence day runs the watcher,
however late GitHub starts it, unless a watcher run (scheduled or dispatched)
already succeeded in that cadence day. A failed run does not count, so the
next hourly fire retries. Manual dispatches always run.

Usage (the workflow pipes prior runs as JSON lines on stdin):
    python -m election_reporting.watch.cadence --event schedule --now 2026-11-13T03:07:00Z
Each input line is {"created_at": ISO time, "watch": job conclusion or null}.
Prints run=true or run=false.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable

CUTOFF = datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc)
DAY_START_HOUR = 2  # UTC


def _time(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError(f"time without a zone: {value!r}")
    return stamp.astimezone(timezone.utc)


def day_start(now: datetime) -> datetime:
    """The 02:00 UTC that begins the cadence day containing now."""
    start = now.astimezone(timezone.utc).replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
    return start if now >= start else start - timedelta(days=1)


def should_run(event: str, now: datetime, prior: Iterable[tuple[datetime, str | None]]) -> tuple[bool, str]:
    if event != "schedule":
        return True, f"{event}: manual and other events always run"
    if now < CUTOFF:
        return True, "hourly cadence until 2026-11-12T08:00Z"
    start = day_start(now)
    for created, watch in prior:
        if created >= start and watch == "success":
            return False, f"the watcher already ran successfully at {created.isoformat()} in the cadence day from {start.isoformat()}"
    return True, f"first successful run of the cadence day from {start.isoformat()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--now", required=True, help="ISO time with a zone")
    args = parser.parse_args(argv)
    prior = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            row = json.loads(line)
            prior.append((_time(row["created_at"]), row.get("watch")))
    run, reason = should_run(args.event, _time(args.now), prior)
    print(f"run={'true' if run else 'false'}")
    print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
