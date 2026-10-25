# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:1e402e05-c146-4ff7-82d1-b33a17abc0c5

"""Independent Pandas/SciPy checks of sealed OC scores; no parameter fitting."""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path


def main():
    import argparse
    import pandas as pd
    from scipy.stats import betabinom, binom
    import scipy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.scope)
    docs = root/"docs/engineering"
    report = json.loads((docs/"oc-official-scores-v1.json").read_text())
    seal = json.loads((docs/"oc-official-seal-v1.json").read_text())
    predictions = docs/"oc-official-predictions-v1.json"
    assert hashlib.sha256(predictions.read_bytes()).hexdigest() == seal["predictions_sha256"] == report["sealed_predictions_sha256"]
    with zipfile.ZipFile(root/"tests/fixtures/oc-official-2024.zip") as z:
        with zipfile.ZipFile(io.BytesIO(z.read("gen2024-media.zip"))) as media:
            raw = media.read("results-final.txt")
    df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, keep_default_na=False)
    df["Total Votes"] = df["Total Votes"].astype(int)
    totals = df.groupby(["Contest Title", "Choice Name1"])["Total Votes"].sum()
    base = [r for r in report["scored_rows"] if r["scenario"] == "base"]
    differences = []
    for row in report["scored_rows"]:
        current = int(totals.loc[(row["title"], row["current_side"])])
        change = int(totals.loc[(row["title"], row["change_side"])])
        assert (current, change) == (row["final_current_votes"], row["final_change_votes"])
        assert int(change > current) == row["outcome"]
        n, margin = row["remaining_pair_votes"], row["margin"]
        first = (n+margin)//2+1
        primary = row["posterior"]
        conditional = row["conditional_increment_posterior"]
        expected = {"model": float(betabinom.sf(first-1, n, primary["alpha"], primary["beta"])),
                    "cumulative_share": float(binom.sf(first-1, n, row["change_votes"]/(row["change_votes"]+row["current_votes"]))),
                    "untempered_beta_binomial": float(betabinom.sf(first-1, n, 1+row["change_votes"], 1+row["current_votes"]))}
        if conditional:
            expected["disjoint_increment_conditional"] = float(betabinom.sf(first-1, n, conditional["alpha"], conditional["beta"]))
        for method, value in expected.items():
            actual = row["probability"] if method == "model" else row["baselines"][method]
            error = abs(value-actual)
            assert math.isfinite(value) and error <= 1e-8, (row["title"], method, error)
            differences.append(error)
    independent = {}
    for method, metrics in report["scores"].items():
        ps = [r["probability"] if method == "model" else r["baselines"][method] for r in base]
        ys = [r["outcome"] for r in base]
        brier = math.fsum((p-y)**2 for p,y in zip(ps,ys))/len(base)
        clipped = [min(1-1e-12,max(1e-12,p)) for p in ps]
        loss = -math.fsum(y*math.log(p)+(1-y)*math.log1p(-p) for p,y in zip(clipped,ys))/len(base)
        assert abs(brier-metrics["brier"]) < 1e-14 and abs(loss-metrics["log_loss"]) < 1e-14
        independent[method] = {"brier": brier, "log_loss": loss}
    result = {"schema_version": 1, "status": "pass", "independent_exact_precinct_outcomes": len(report["scored_rows"]),
              "unique_base_contests": len(base), "losses": independent, "maximum_scipy_absolute_error": max(differences),
              "scipy_comparisons": len(differences), "pandas": pd.__version__, "scipy": scipy.__version__,
              "scope": "Independent extraction/event aggregation and numerical distribution checks only. Not independent final scientific re-audit or calibration."}
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","unique_base_contests","scipy_comparisons","maximum_scipy_absolute_error")}))


if __name__ == "__main__":
    main()
