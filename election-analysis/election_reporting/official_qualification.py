# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:555c1746-af75-4b66-b6b9-6ff3666a5a26

"""Independent extraction/count reconciliation, separate from scoring.

Requires Poppler and Pandas for independent oracles, not production use.
Measure-title abbreviations and damaged source accents are reported, not merged.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .model import ElectionReportError
from .official_archives import read_count_pdf, read_final_summary, read_pool_pdf
from .retrospective import digest


def qualify(fixtures):
    import pandas as pd
    root = Path(fixtures)
    manifest_path = root/"oc-official-2024-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = root/"oc-official-2024.zip"
    if digest(source) != manifest["archive_sha256"]:
        raise ElectionReportError("qualification archive hash differs")
    with zipfile.ZipFile(source) as z:
        files = {s["file"]: z.read(s["file"]) for s in manifest["files"]}
    if any(hashlib.sha256(files[s["file"]]).hexdigest() != s["sha256"] for s in manifest["files"]):
        raise ElectionReportError("qualification member hash differs")
    reports, oracles = {}, []
    # Poppler is an independent PDF implementation; the flat terminal-column
    # oracle does not use the pypdf layout state machine or title segmentation.
    suffix = re.compile(r"(?:^|\s)([\d,]+)\s+[\d.]+%\s+([\d,]+)\s+[\d.]+%\s+([\d,]+)\s+[\d.]+%\s*$")
    for name in ("gen2024-run12.pdf", "gen2024-run12-canonical.pdf", "gen2024-run13.pdf"):
        report = read_count_pdf(files[name]); reports[name] = report
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"source.pdf";path.write_bytes(files[name])
            out = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, check=True, timeout=60).stdout.decode()
        independent = [tuple(int(n.replace(",", "")) for n in m.groups()) for line in out.splitlines() if (m := suffix.search(line))]
        expected = []
        for c in report["contests"].values():
            expected.extend((v["vote_center"], v["mail"], v["votes"]) for v in c["choices"].values())
            expected.append((sum(v["vote_center"] for v in c["choices"].values()), sum(v["mail"] for v in c["choices"].values()), c["cast_votes"]))
        if independent != expected:
            raise ElectionReportError(f"independent Poppler numeric rows diverge: {name}: {len(independent)} vs {len(expected)}")
        oracles.append({"file": name, "candidate_and_cast_rows": len(expected), "contests": len(report["contests"]), "poppler_agrees": True})
    equivalent = reports["gen2024-run12.pdf"]["contests"] == reports["gen2024-run12-canonical.pdf"]["contests"]
    if not equivalent:
        raise ElectionReportError("November 8 copies have different count semantics")
    final = read_final_summary(files["gen2024-final-summary.csv"])
    df = pd.read_csv(io.BytesIO(files["gen2024-final-summary.csv"]), skiprows=[0, 1, 3], dtype=str, keep_default_na=False)
    for k in ("Vote-by-Mail Ballots Votes", "Vote Center Ballots Votes", "Total Votes"):
        if not df[k].str.fullmatch(r"\d+").all():
            raise ElectionReportError("independent summary integer format differs")
    if not (df["Vote-by-Mail Ballots Votes"].astype(int)+df["Vote Center Ballots Votes"].astype(int) == df["Total Votes"].astype(int)).all():
        raise ElectionReportError("independent summary method totals disagree")
    with zipfile.ZipFile(io.BytesIO(files["gen2024-media.zip"])) as media:
        if media.namelist() != ["results-final.txt"] or media.getinfo("results-final.txt").file_size > 20_000_000:
            raise ElectionReportError("unsupported precinct export inventory/size")
        raw = media.read("results-final.txt")
    precinct = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, keep_default_na=False)
    columns = {"Total Votes": "votes", "Vote Center Votes": "vote_center", "Vote by Mail Votes": "mail"}
    for k in columns:
        if not precinct[k].str.fullmatch(r"\d+").all():
            raise ElectionReportError("independent precinct integer invalid")
        precinct[k] = precinct[k].astype(int)
    if not (precinct["Vote Center Votes"]+precinct["Vote by Mail Votes"] == precinct["Total Votes"]).all():
        raise ElectionReportError("precinct source method accounting differs")
    grouped = precinct.groupby(["Contest Title", "Choice Name1"])[list(columns)].sum()
    titles = set(precinct["Contest Title"]); exact_titles = titles & set(final["contests"])
    checked, missing, extras = 0, [], []
    for title in sorted(exact_titles):
        c = final["contests"][title]
        for name, v in c["choices"].items():
            if (title, name) not in grouped.index:
                missing.append({"title": title, "choice": name, "reason": "exact source name unavailable; no fuzzy/accent repair"})
                continue
            if any(int(grouped.loc[(title, name), source_key]) != v[target] for source_key, target in columns.items()):
                raise ElectionReportError("summary/precinct exact choice counts disagree")
            checked += 1
        for name in grouped.loc[title].index:
            if name not in c["choices"]:
                extras.append({"title": title, "source_name": name, "votes": int(grouped.loc[(title, name), "Total Votes"])})
    return {"schema_version": 1, "archive_sha256": digest(source), "manifest_sha256": digest(manifest_path),
            "qualification_only_not_scoring": True, "original_files": len(files), "pdf_numeric_oracles": oracles,
            "nov8_semantically_equal_despite_different_bytes": equivalent,
            "final_summary_contests": len(final["contests"]), "final_summary_choice_rows": len(df),
            "precinct_export_rows": len(precinct), "precinct_contests": len(titles), "exact_title_matches": len(exact_titles),
            "exact_choices_reconciled_all_three_vote_columns": checked, "unmatched_summary_names": missing,
            "unmatched_summary_titles": sorted(set(final["contests"])-titles), "unmatched_precinct_titles": sorted(titles-set(final["contests"])),
            "extra_precinct_names": extras, "pool_forms": [read_pool_pdf(files[name]) for name in ("unprocessed-20241108.pdf", "unprocessed-20241109.pdf")],
            "independent_oracles": {"pdf": "pdftotext -layout terminal-column triples in original source order", "csv_and_tsv": "Pandas typed exact-name group sums; no fuzzy joining"},
            "qualification": "Verifies extraction/accounting, not historical-byte authenticity, correction-free increments, countable pool assumptions or forecast accuracy."}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = qualify(args.fixtures)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("original_files", "exact_title_matches", "exact_choices_reconciled_all_three_vote_columns")}))


if __name__ == "__main__":
    main()
