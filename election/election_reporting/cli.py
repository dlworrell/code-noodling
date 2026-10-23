from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import analyze_election
from .config import load_config
from .model import ElectionReportError
from .publish import publish


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m election_reporting",
        description="Analyze dated election-result snapshots and publish with EDT.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="election configuration, e.g. election/data/elections/washington/2026-08-04/election.toml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="generated report directory, e.g. election/reports/washington/2026-08-04",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        config = load_config(args.config)
        analysis = analyze_election(config)
        publish(analysis, args.output)
    except ElectionReportError as exc:
        print(f"election-report: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        f"wrote {len(analysis.races)} race analyses to {args.output} "
        f"from {len(analysis.snapshots)} snapshot(s)"
    )
