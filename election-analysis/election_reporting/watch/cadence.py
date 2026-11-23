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

Usage in the workflow (reads the run history itself with ``gh api``):
    python -m election_reporting.watch.cadence --event schedule --now 2026-11-13T03:07:00Z \
        --repo owner/name --run-id 123
Without --repo, prior runs are read as JSON lines on stdin, each
{"created_at": ISO time, "watch": job conclusion or null}.
Prints run=true or run=false on stdout and exits 0.

Fails closed (fitment #221, #229): if any ``gh api`` call fails or returns
something unreadable, it prints run=false, so this fire does not run the watcher
and the next hourly fire tries again; an API error never turns into an extra
visit to the state sites. Before the cutoff, and for events other than
schedule, the history is not read at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

CUTOFF = datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc)
DAY_START_HOUR = 2  # UTC
WORKFLOW = "election-source-watch.yml"
HISTORY_DAYS = 2

Gh = Callable[[str], Any]


class HistoryUnavailable(RuntimeError):
    """The run history could not be read; the gate must not run the watcher."""


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


def gh_api(path: str) -> Any:
    """``gh api <path>`` parsed as JSON; HistoryUnavailable on any failure."""
    try:
        result = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HistoryUnavailable(f"gh api {path}: {exc}") from exc
    if result.returncode != 0:
        raise HistoryUnavailable(f"gh api {path}: exit {result.returncode}: {result.stderr.strip()[:200]}")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise HistoryUnavailable(f"gh api {path}: unreadable JSON") from exc


def collect_prior(repo: str, run_id: int, now: datetime, gh: Gh = gh_api) -> list[tuple[datetime, str | None]]:
    """Earlier runs of the workflow from the last HISTORY_DAYS days with their watch job conclusion."""
    since = (now - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        listing = gh(f"repos/{repo}/actions/workflows/{WORKFLOW}/runs?created=>={since}&per_page=100")
        prior = []
        for run in listing["workflow_runs"]:
            if int(run["id"]) == run_id:
                continue
            jobs = gh(f"repos/{repo}/actions/runs/{int(run['id'])}/jobs")
            watch = next((job.get("conclusion") for job in jobs["jobs"] if job.get("name") == "watch"), None)
            prior.append((_time(run["created_at"]), watch))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise HistoryUnavailable(f"unexpected run history shape: {exc!r}") from exc
    return prior


def decide(event: str, now: datetime, repo: str, run_id: int, gh: Gh = gh_api) -> tuple[bool, str]:
    """The gate as the workflow runs it: read the history only when it matters, fail closed."""
    if event != "schedule" or now < CUTOFF:
        return should_run(event, now, [])
    try:
        prior = collect_prior(repo, run_id, now, gh)
    except HistoryUnavailable as exc:
        return False, f"run history unavailable, skipping this fire (the next hourly fire retries): {exc}"
    return should_run(event, now, prior)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--now", required=True, help="ISO time with a zone")
    parser.add_argument("--repo", help="owner/name; read the run history with gh api instead of stdin")
    parser.add_argument("--run-id", type=int, default=0, help="this run, left out of the history")
    args = parser.parse_args(argv)
    now = _time(args.now)
    if args.repo:
        run, reason = decide(args.event, now, args.repo, args.run_id)
        if not run and reason.startswith("run history unavailable"):
            print(f"::warning::{reason}", file=sys.stderr)
    else:
        prior = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                row = json.loads(line)
                prior.append((_time(row["created_at"]), row.get("watch")))
        run, reason = should_run(args.event, now, prior)
    print(f"run={'true' if run else 'false'}")
    print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
