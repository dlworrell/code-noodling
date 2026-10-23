"""Command line: ``python -m election_reporting.watch {plan,fetch} --config ...``."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..model import ElectionReportError
from .config import load_watch_config
from .http import fetch as http_fetch
from .run import carry_forward, fetch_snapshots, plan, write_election_config


def _now(text: str | None) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m election_reporting.watch")
    parser.add_argument("command", choices=("plan", "fetch", "carry"))
    parser.add_argument(
        "--from",
        dest="carry_from",
        type=Path,
        help="carry: an elections directory from an unmerged automation branch",
    )
    parser.add_argument("--config", type=Path, default=Path("election-data/watch.toml"))
    parser.add_argument("--now", help="ISO date-time to use as the current time (testing)")
    parser.add_argument("--force", action="store_true", help="fetch even when not due")
    args = parser.parse_args(argv)
    try:
        config = load_watch_config(args.config)
        now = _now(args.now)
        if args.command == "carry":
            if args.carry_from is None:
                parser.error("carry needs --from")
            count = carry_forward(args.carry_from, config.elections_root)
            print(f"carried forward {count} snapshot(s)")
            return 0
        if args.command == "plan":
            election_plan = plan(config, http_fetch, now)
            print(f"Election: {election_plan.election.day} {election_plan.election.name} ({election_plan.kind})")
            for place in election_plan.places:
                source = "King County export" if place.county_files else "results.votewa.gov"
                print(f"  {place.place_id:<18} {place.name:<18} {place.scope:<6} {source}")
            window = election_plan.window
            print(
                f"Window: {window.start:%Y-%m-%d %H:%M} to {window.end:%Y-%m-%d %H:%M}; "
                f"every {config.election_night_every_minutes} min until "
                f"{window.election_night_end:%Y-%m-%d %H:%M}, then every {config.every_minutes} min"
            )
            added = write_election_config(election_plan)
            target = election_plan.directory / "election.toml"
            print(f"{target}: " + (f"added {', '.join(added)}" if added else "up to date"))
            return 0
        result = fetch_snapshots(config, http_fetch, now, force=args.force)
    except ElectionReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"new_snapshots={len(result.added)}\n")
    # Partial failures still publish what was fetched; fail only when nothing worked.
    return 1 if result.errors and not result.added and not result.skipped else 0


if __name__ == "__main__":
    sys.exit(main())
