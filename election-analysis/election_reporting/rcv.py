# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:bf804f60-4343-4b06-84d1-09daa523b34d

"""EA-05: anonymous rank-pattern allocation to officially supplied finalists.

This is not an independent candidate-eligibility or elimination engine. The
rule/version and official finalists must be supplied, not inferred from raw
first-choice percentages. Maine's official RCV resources distinguish Round 0
from interpreted Round 1; 21-A section 723-A defines exhaustion. Historical
rule applicability is recorded in the source manifest, not backdated from a
current statute automatically.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from .ingest import _xlsx_sheet_paths, iter_xlsx_sheet_rows
from .model import ElectionReportError

BLANK = "undervote"
OVERVOTE = "overvote"


def allocate_patterns(patterns, finalists, noncontinuing, *, rule_version):
    if not rule_version or len(set(finalists)) != 2 or set(finalists) & set(noncontinuing):
        raise ElectionReportError("Final allocation requires two distinct official finalists and an explicit rule version")
    known = set(finalists) | set(noncontinuing) | {BLANK, OVERVOTE}
    totals = Counter({candidate: 0 for candidate in finalists})
    exhausted = Counter({"skipped_ranks": 0, "no_continuing_choice": 0, "overvote": 0})
    ballots = 0
    for item in patterns:
        count, ranks = item.get("count"), item.get("ranks")
        if type(count) is not int or count <= 0 or not isinstance(ranks, list) or not 1 <= len(ranks) <= 100:
            raise ElectionReportError("Invalid anonymous rank pattern/count")
        if any(not isinstance(rank, str) or rank not in known for rank in ranks):
            raise ElectionReportError("Unknown candidate/write-in eligibility; supply official finalist/noncontinuing evidence")
        blanks = 0
        destination = None
        reason = "no_continuing_choice"
        for rank in ranks:
            if rank == BLANK:
                blanks += 1
                if blanks >= 2:
                    reason = "skipped_ranks"
                    break
                continue
            if rank == OVERVOTE:
                reason = "overvote"
                break
            blanks = 0
            if rank in finalists:
                destination = rank
                break  # exactly one vote, regardless of duplicate later rankings
        if destination:
            totals[destination] += count
        else:
            exhausted[reason] += count
        ballots += count
    continuing = sum(totals.values())
    if continuing + sum(exhausted.values()) != ballots:
        raise ElectionReportError("RCV ballot conservation failed")
    return {"schema_version": 1, "stage": "final_allocation_to_official_finalists",
            "rule_version": rule_version, "totals": dict(totals), "continuing": continuing,
            "exhausted": dict(exhausted), "ballots": ballots,
            "margin": abs(totals[finalists[0]] - totals[finalists[1]]),
            "finalists_source": "supplied official summary", "independent_eligibility_engine": False}


def rebuild_patterns(root: Path, manifest: dict):
    """Read all required original exports, retain only anonymous pattern counts.

    Source hashes, six-export coverage, per-file counts and duplicate machine
    keys are checked before a final-allocation regression can be called complete.
    Ballot IDs are used transiently for duplicates and never returned or stored.
    """
    files = manifest.get("files", [])
    if len(files) != 6 or len({f["path"] for f in files}) != 6:
        raise ElectionReportError("Complete Maine CD2 validation requires all six distinct exports")
    patterns = Counter()
    duplicate_keys = set()
    accounting = []
    for spec in files:
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ElectionReportError("CVR source path must stay inside the supplied source root")
        path = root / relative
        if not path.is_file() or path.stat().st_size > 40_000_000:
            raise ElectionReportError(f"Missing or oversized CVR export: {relative}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ElectionReportError(f"CVR source hash mismatch: {relative}")
        with zipfile.ZipFile(path) as archive:
            sheets = _xlsx_sheet_paths(archive, path)
        count = 0
        for name, _ in sheets:
            rows = iter_xlsx_sheet_rows(path, raw, name)
            header = next(rows, [])
            if not header or header[0] != "Cast Vote Record":
                raise ElectionReportError(f"Unsupported CVR header: {relative}")
            for row in rows:
                if not row or not row[0]:
                    if any(row):
                        raise ElectionReportError("Nonempty CVR row has no record key")
                    continue
                if len(row) < 6:
                    row += [""] * (6-len(row))
                if spec.get("kind") == "machine":
                    key = tuple(row[:3])
                    if key in duplicate_keys:
                        raise ElectionReportError("Duplicate machine CVR key")
                    duplicate_keys.add(key)
                ranks = tuple(cell.strip() if cell else BLANK for cell in row[3:6])
                patterns[ranks] += 1
                count += 1
        if count != spec["ballots"]:
            raise ElectionReportError(f"CVR export count mismatch: {relative}")
        accounting.append({"path": str(relative), "ballots": count, "sha256": spec["sha256"]})
    ballots = sum(patterns.values())
    if ballots != manifest["ballots"]:
        raise ElectionReportError("Complete CVR ballot count mismatch")
    return {"schema_version": 1, "files": accounting, "ballots": ballots,
            "duplicate_machine_keys": 0,
            "patterns": [{"ranks": list(ranks), "count": count} for ranks,count in sorted(patterns.items())]}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = rebuild_patterns(args.source_root, manifest)
    result["allocation"] = allocate_patterns(result["patterns"], manifest["finalists"], manifest["noncontinuing"], rule_version=manifest["rule_version"])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
