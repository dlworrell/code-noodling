# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:1e75638f-5922-4cd1-9d63-16e72885dc59

"""Hash-bound, stage-aware descriptive replay. This is not a forecast backtest."""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from .analysis import analyze_election
from .ingest import read_snapshot
from .model import ElectionConfig, ElectionReportError, ResultStatus, ResultStatusRecord, SourceConfig
from .versions import validate_versions
from .watch.maine import governing_rcv

AUTHORITIES = {
    "washington": "https://results.vote.wa.gov/results/20241105/export.html",
    "alaska": "https://www.elections.alaska.gov/election-results/e/?id=24genr",
    "maine": "https://www1.maine.gov/sos/elections-voting/election-results-data/election-results-2024",
}


def run(fixtures: Path):
    manifest = json.loads((fixtures/"replay-2024/manifest.json").read_text())
    official = validate_versions(json.loads((fixtures/"official-result-versions.json").read_text()))
    output = []
    with tempfile.TemporaryDirectory() as directory:
        for spec in manifest["inputs"]:
            state = spec["state"]
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ElectionReportError("unsafe frozen replay path")
            raw = (fixtures/"replay-2024"/relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                raise ElectionReportError("frozen replay input hash mismatch")
            path = Path(directory)/"2026-10-23-0000.csv"
            path.write_bytes(raw)  # original acquisition label; not a count-update time
            snapshot = read_snapshot(path,state)
            if len(snapshot.contests) != spec["records"]:
                raise ElectionReportError("frozen replay record count mismatch")
            statuses, rules = [], {}
            for contest in snapshot.contests.values():
                choices = {c.name.casefold() for c in contest.choices}
                binary = choices <= {"yes","no"}
                rcv = not binary and (state == "alaska" or state == "maine" and governing_rcv(contest.name,date(2024,11,5)))
                rule = "rcv" if rcv else "winner"
                rules[f"{state}:{contest.contest_id}"] = rule
                status = ResultStatus("final_tabulation", "raw_first_rankings" if rcv else "plurality_count",
                    AUTHORITIES[state], datetime.fromisoformat(manifest["evidence_observed_at"]),
                    "frozen-2024-summary-"+snapshot.sha256[:12])
                statuses.append(ResultStatusRecord(state,snapshot.sha256,contest.contest_id,status))
            config = ElectionConfig(f"2024 {state} frozen descriptive replay",
                (SourceConfig(state,Path(directory),state,"state",100),),frozenset({"write-in","write in"}),
                rule_overrides=rules,default_rule="winner",report_mode="historical",result_statuses=tuple(statuses))
            analysis = analyze_election(config)
            output.append(replace(analysis,snapshots=tuple(replace(s,path=fixtures/"replay-2024"/relative) for s in analysis.snapshots),official_versions=tuple(r for r in official if r["canonical_id"].startswith("ak:") == (state == "alaska") and state in {"alaska","maine"})))
    return tuple(output)


def main():
    import argparse
    from .publish import publish
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    for state,analysis in zip(("washington","alaska","maine"),run(args.fixtures)):
        publish(analysis,args.output/state)


if __name__ == "__main__":
    main()
