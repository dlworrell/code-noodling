# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:b3a2dd04-0435-4a05-8c13-54f98c980e21

"""Command line: ``python -m election_reporting.watch {plan,fetch} --config ...``."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..model import ElectionReportError
from .config import load_watch_settings
from .http import fetch as http_fetch
from .run import carry_forward, fetch_snapshots, plan, write_election_config
from .schedule import before_start, hold_message


def _now(text: str | None) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m election_reporting.watch")
    parser.add_argument("command", choices=("status", "plan", "fetch", "carry"))
    parser.add_argument(
        "--from",
        dest="carry_from",
        type=Path,
        help="carry: an elections directory from an unmerged automation branch",
    )
    parser.add_argument("--config", type=Path, default=Path("election-analysis/data/watch.toml"))
    parser.add_argument("--now", help="status only: simulate a time without making source requests")
    parser.add_argument("--force", action="store_true", help="ignore cadence/end, never the hard start gate")
    parser.add_argument("--require-eligible", action="store_true", help="status: exit 3 if all states are held or ended")
    args = parser.parse_args(argv)
    if args.now and args.command != "status":
        parser.error("--now is only allowed for offline status; live commands use the real clock")
    try:
        settings = load_watch_settings(args.config)
        now = _now(args.now)
        if args.command == "status":
            eligible = False
            for config in settings.watches:
                held = before_start(config, now)
                ended = config.end is not None and now > config.end and not args.force
                if held:
                    print(hold_message(config))
                elif ended:
                    print(f"{config.state}: checking window ended")
                else:
                    print(f"{config.state}: start gate open")
                eligible = eligible or not (held or ended)
            output = os.environ.get("GITHUB_OUTPUT")
            if output:
                with open(output, "a", encoding="utf-8") as handle:
                    handle.write(f"eligible={str(eligible).lower()}\n")
            return 3 if args.require_eligible and not eligible else 0
        if args.command == "carry":
            if args.carry_from is None:
                parser.error("carry needs --from")
            count = carry_forward(args.carry_from, settings.elections_root)
            print(f"carried forward {count} snapshot(s)")
            return 0
        if args.command == "plan":
            for config in settings.watches:
                if before_start(config, now):
                    print(hold_message(config))
                else:
                    _print_plan(config, plan(config, http_fetch, now))
            return 0
    except ElectionReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    added = 0
    failed = False
    for config in settings.watches:
        print(f"== {config.state}")
        try:
            result = fetch_snapshots(config, http_fetch, now, force=args.force)
        except ElectionReportError as error:
            # One state's outage must not stop the others.
            print(f"error: {config.state}: {error}", file=sys.stderr)
            failed = True
            continue
        added += len(result.added)
        if result.errors and not result.added and not result.skipped:
            failed = True
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"new_snapshots={added}\n")
    # Partial failures still publish what was fetched; fail only when a state
    # could not be checked at all.
    return 1 if failed else 0


def _print_plan(config, election_plan) -> None:
    print(f"{config.state}: {election_plan.election.day} {election_plan.election.name} ({election_plan.kind})")
    for place in election_plan.places:
        if config.state_id == "alaska":
            source = "precinct file" if place.scope == "state" else "precinct file, election-day precincts"
        elif config.state_id == "maine":
            source = "office workbooks" if place.scope == "state" else "office workbooks, municipal rows"
        else:
            source = "county export" if place.county_files else "results site"
        print(f"  {place.place_id:<18} {place.name:<18} {place.scope:<6} {source}")
    window = election_plan.window
    print(
        f"  Window: {window.start:%Y-%m-%d %H:%M} to {window.end:%Y-%m-%d %H:%M} {config.timezone.key}; "
        f"every {config.election_night_every_minutes} min until "
        f"{window.election_night_end:%Y-%m-%d %H:%M}, then every {config.every_minutes} min"
    )
    added = write_election_config(election_plan)
    target = election_plan.directory / "election.toml"
    print(f"  {target}: " + (f"added {', '.join(added)}" if added else "up to date"))


if __name__ == "__main__":
    sys.exit(main())
