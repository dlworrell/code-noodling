# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:99702333-eb22-4e8a-b355-ff631693f8bb

"""Frozen, two-phase Orange County official-queue reconstruction.

A publisher's unprocessed estimate is evidence, not an accepted/countable pool.
The separate addition sensitivity assumes the cumulative change has no tally
corrections. It never enables the production latest-batch likelihood.
"""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .evidence import timestamp
from .model import ElectionReportError
from .numerics import MAX_REMAINING, beta_binomial_tail, posterior_details, threshold
from .official_archives import read_count_pdf, read_final_summary, read_pool_pdf
from .retrospective import _metrics, binomial_tail, digest
from .validation import protocol_hash

SCOPE = Path(__file__).resolve().parents[1]
MODE = "retrospective_official_queue_scenarios"
BEFORE = "gen2024-run12-canonical.pdf"
CURRENT = "gen2024-run13.pdf"
POOL_BEFORE = "unprocessed-20241108.pdf"
POOL = "unprocessed-20241109.pdf"
FINAL = "gen2024-final-summary.csv"
INPUTS = {BEFORE, CURRENT, POOL_BEFORE, POOL}
SCENARIOS = {"base": [1.0, False], "lower_acceptance": [.8, False], "separate_challenged": [1.0, True]}
BASELINES = ("fixed_half", "cumulative_share", "untempered_beta_binomial",
             "official_queue_margin_rule", "reported_increment_only_conditional", "disjoint_increment_conditional")
QUALIFICATION = ("Official dated queue estimates, with assumed acceptance and geographic/pair participation; "
                 "not official contest-specific countable remaining votes. No future-arrival upper bound. "
                 "Reported-increment models additionally assume no corrections within the tally change. "
                 "Current archive bytes are not authenticated historical captures. Reconstructed after public outcomes; "
                 "not a blind experiment, published historical forecast, calibration or production eligibility.")


def load_protocol(path, scope=SCOPE):
    p = json.loads(Path(path).read_text())
    if (p.get("schema_version") != 3 or p.get("mode") != MODE or not timestamp(p.get("frozen_at"))
            or p.get("used_for_tuning") is not False or p.get("training_groups") != []
            or p.get("held_out_groups") != ["ca:orange:2024-general"]
            or p.get("scenarios") != SCENARIOS or p.get("baselines") != list(BASELINES)
            or p.get("model_id") != "disjoint_tempered_beta_binomial_v2"
            or p.get("information_time") != "2024-11-09T19:05:00-08:00"
            or p.get("event_scope") != "selected_pair_strict_reversal"
            or p.get("verified_new_ballot_batch") is not False):
        raise ElectionReportError("unsupported frozen official-queue protocol")
    certification = timestamp(p.get("certification_lower_bound"))
    if not certification or not timestamp(p["information_time"]) < certification < timestamp(p["frozen_at"]):
        raise ElectionReportError("historical reconstruction/certification/freeze chronology invalid")
    files = p.get("model_files", {})
    if not files:
        raise ElectionReportError("frozen model/harness hashes required")
    for name, sha in files.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or digest(Path(scope)/rel) != sha:
            raise ElectionReportError("frozen implementation bytes changed")
    return p


class Corpus:
    def __init__(self, fixtures, protocol):
        root = Path(fixtures)
        manifest = root/"oc-official-2024-manifest.json"
        if digest(manifest) != protocol["corpus_manifest_sha256"]:
            raise ElectionReportError("official corpus manifest changed")
        self.manifest = json.loads(manifest.read_text())
        self.specs = {s["file"]: s for s in self.manifest["files"]}
        if len(self.specs) != 8 or len(self.manifest["files"]) != 8:
            raise ElectionReportError("official corpus inventory invalid")
        path = root/"oc-official-2024.zip"
        if digest(path) != self.manifest["archive_sha256"]:
            raise ElectionReportError("official source archive changed")
        self.archive = zipfile.ZipFile(path)
        infos = self.archive.infolist()
        if (len(infos) != 8 or set(self.archive.namelist()) != set(self.specs)
                or sum(i.file_size for i in infos) > 20_000_000
                or any(i.flag_bits & 1 or i.file_size > 12_000_000 for i in infos)):
            self.archive.close()
            raise ElectionReportError("official archive members/encryption/size invalid")
        self.reads = []

    def read(self, name, allowed):
        if name not in allowed or name not in self.specs:
            raise ElectionReportError("reserved evidence forbidden in this phase")
        raw = self.archive.read(name)
        if len(raw) != self.specs[name]["bytes"] or hashlib.sha256(raw).hexdigest() != self.specs[name]["sha256"]:
            raise ElectionReportError("original official file hash/size mismatch")
        self.reads.append(name)
        return raw

    def close(self):
        self.archive.close()


def transition(before, current, old_pool, pool, manifest):
    """Reconcile the source statement without claiming a pure new-ballot batch."""
    a, b = before["source_ballots"], current["source_ballots"]
    facts = manifest["publisher_facts"]
    if (set(before["contests"]) != set(current["contests"])
            or a != old_pool["processed_ballots"] or b != pool["processed_ballots"]
            or b-a != facts["ballots_added_on_nov9"]
            or pool["source_field_2"] != facts["ballots_added_since_election_night"]
            or a-old_pool["source_field_2"] != b-pool["source_field_2"]
            or not timestamp(before["count_updated_at"]) < timestamp(current["count_updated_at"])):
        raise ElectionReportError("count/form/release clocks or processed-ballot accounting disagree")
    records = []
    for title, c in sorted(current["contests"].items()):
        prior = before["contests"][title]
        compatible = (set(c["choices"]) == set(prior["choices"]) and c["vote_for"] == prior["vote_for"]
                      and c["precincts"] == prior["precincts"] and c["ballots"] >= prior["ballots"])
        deltas = {n: {k: v[k]-prior["choices"][n][k] for k in ("votes", "mail", "vote_center")}
                  for n, v in c["choices"].items() if n in prior["choices"]}
        nonnegative = compatible and all(x >= 0 for v in deltas.values() for x in v.values())
        records.append({"title": title, "classification": "reported_additions_without_correction_partition" if nonnegative else "incompatible_or_revision_unknown",
                        "source_ballot_delta": c["ballots"]-prior["ballots"], "choice_deltas": deltas,
                        "conditional_increment_usable": nonnegative, "verified_new_ballots": False})
    return {"processed_ballots_added": b-a, "source_field_2_is_latest_batch": False,
            "election_night_processed_baseline": b-pool["source_field_2"],
            "unprocessed_estimate_change": pool["estimated_unprocessed"]-old_pool["estimated_unprocessed"],
            "processed_plus_queue_change": b+pool["estimated_unprocessed"]-a-old_pool["estimated_unprocessed"],
            "challenged_overlap": pool["challenged_overlap_with_total"], "contests": records,
            "operational_delta_kind": "unknown", "qualification": QUALIFICATION}


def in_scope(title, contest, protocol):
    if title == "County Supervisor 1st District":
        return contest["vote_for"] == 1
    return contest["vote_for"] == 1 and any(title.startswith(f"CITY OF {city} ") for city in protocol["whole_county_cities"])


def build_predictions(protocol, fixtures):
    generated = datetime.now(timezone.utc).isoformat()
    if timestamp(generated) < timestamp(protocol["frozen_at"]):
        raise ElectionReportError("computation precedes freeze")
    corpus = Corpus(fixtures, protocol)
    try:
        before = read_count_pdf(corpus.read(BEFORE, INPUTS))
        current = read_count_pdf(corpus.read(CURRENT, INPUTS))
        old_pool = read_pool_pdf(corpus.read(POOL_BEFORE, INPUTS))
        pool = read_pool_pdf(corpus.read(POOL, INPUTS))
        information = timestamp(protocol["information_time"])
        if any(timestamp(t) > information for t in (before["count_updated_at"], current["count_updated_at"], old_pool["reported_at"], pool["reported_at"])):
            raise ElectionReportError("input source clock is after reconstruction time")
        audit = transition(before, current, old_pool, pool, corpus.manifest)
        changes = {r["title"]: r for r in audit["contests"]}
        rows, ledger = [], []
        for title, contest in sorted(current["contests"].items()):
            record = {"title": title, "reason": None, "scenario_exclusions": {}}
            ledger.append(record)
            if not in_scope(title, contest, protocol):
                record["reason"] = "unreviewed_whole_jurisdiction_or_not_single_seat"
                continue
            choices = sorted(contest["choices"].values(), key=lambda v: (-v["votes"], v["name"]))
            if len(choices) < 2 or choices[0]["votes"] == choices[1]["votes"]:
                record["reason"] = "uncontested_or_tied_as_of"
                continue
            leader, challenger = choices[:2]
            total = leader["votes"]+challenger["votes"]
            share = total/current["source_ballots"]
            if not 0 < share <= 1:
                raise ElectionReportError("pair participation exceeds county ballot denominator")
            previous = before["contests"][title]["choices"]
            dc = challenger["votes"]-previous[challenger["name"]]["votes"]
            dl = leader["votes"]-previous[leader["name"]]["votes"]
            batch_n = dc+dl
            usable = changes[title]["conditional_increment_usable"] and dc >= 0 and dl >= 0 and batch_n > 0
            for scenario, (acceptance, add_challenged) in protocol["scenarios"].items():
                queue = round(pool["estimated_unprocessed"]*acceptance)+(pool["challenged_left_to_cure"] if add_challenged else 0)
                n = round(queue*share)
                if not 0 < n <= MAX_REMAINING:
                    record["scenario_exclusions"][scenario] = "nonpositive_or_unsupported_numerical_pool"
                    continue
                margin = leader["votes"]-challenger["votes"]
                first = threshold(n, margin, False)
                primary = posterior_details(challenger["votes"], leader["votes"])
                conditional = posterior_details(challenger["votes"], leader["votes"], dc/batch_n, batch_n) if usable else None
                if conditional:
                    conditional["source_reported_increment_pair_votes"] = conditional.pop("latest_verified_pair_votes")
                    conditional["qualification"] = "Conditional on every observed tally change being a newly counted vote; not verified batch evidence."
                rows.append({"election_id": "ca:orange:2024-general", "jurisdiction_cluster": "Orange County, California",
                    "canonical_id": "ca:orange:2024-general:"+hashlib.sha256(title.encode()).hexdigest()[:16],
                    "title": title, "information_time": protocol["information_time"], "count_updated_at": current["count_updated_at"],
                    "historical_availability": "Qualified current publisher archive reconstruction, not authenticated historical capture",
                    "prediction_generated_at": generated, "input_files": sorted(INPUTS),
                    "current_side": leader["name"], "change_side": challenger["name"], "current_votes": leader["votes"], "change_votes": challenger["votes"],
                    "margin": margin, "scenario": scenario, "official_estimated_unprocessed": pool["estimated_unprocessed"],
                    "assumed_countable_county_queue": queue, "pair_participation": share, "remaining_pair_votes": n,
                    "pool_assumptions": {"acceptance_fraction": acceptance, "challenged_fully_separate": add_challenged,
                                         "future_arrivals": "excluded from this scenario, not asserted zero or bounded", "geographic_and_pair_share": "same as cumulative counted votes"},
                    "probability": beta_binomial_tail(n, primary["alpha"], primary["beta"], first),
                    "baselines": {"fixed_half": .5, "cumulative_share": binomial_tail(n, challenger["votes"]/total, first),
                        "untempered_beta_binomial": beta_binomial_tail(n, 1+challenger["votes"], 1+leader["votes"], first),
                        "official_queue_margin_rule": 0.0 if first > n else .5,
                        "reported_increment_only_conditional": binomial_tail(n, dc/batch_n, first) if usable else None,
                        "disjoint_increment_conditional": beta_binomial_tail(n, conditional["alpha"], conditional["beta"], first) if usable else None},
                    "posterior": primary, "conditional_increment_posterior": conditional,
                    "verified_new_ballot_pair_votes": 0, "operational_delta_kind": "unknown",
                    "state": "California", "contest_type": "County" if title.startswith("County Supervisor") else "City",
                    "counting_stage": "middle", "missing_data_stratum": "official_queue_estimate; acceptance/geography/future arrivals assumed; correction partition unavailable",
                    "event_scope": "selected_pair_strict_reversal", "operationally_eligible": False, "qualification": QUALIFICATION})
        return {"schema_version": 3, "mode": MODE, "protocol_sha256": protocol_hash(protocol), "generated_at": generated,
                "rows": rows, "inclusion_exclusion_ledger": ledger, "transition_ledger": audit,
                "read_artifacts": corpus.reads, "reserved_outcomes_read": [], "qualification": QUALIFICATION}
    finally:
        corpus.close()


def score_predictions(protocol, predictions_path, fixtures, expected_sha):
    raw = Path(predictions_path).read_bytes()
    if len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ElectionReportError("sealed prediction bytes differ")
    predictions = json.loads(raw)
    generated = timestamp(predictions.get("generated_at"))
    if (predictions.get("mode") != MODE or predictions.get("protocol_sha256") != protocol_hash(protocol)
            or not generated or generated < timestamp(protocol["frozen_at"])
            or predictions.get("reserved_outcomes_read") != [] or set(predictions.get("read_artifacts", [])) != INPUTS):
        raise ElectionReportError("prediction phase/lineage/freeze invalid")
    # Recompute solely from the allowed as-of evidence; this also verifies the
    # complete identity/parameter/exclusion ledger instead of trusting rows alone.
    expected = build_predictions(protocol, fixtures)
    def semantic(value):
        if isinstance(value, dict):
            return {k: semantic(v) for k, v in value.items() if k not in {"generated_at", "prediction_generated_at"}}
        return [semantic(v) for v in value] if isinstance(value, list) else value
    if semantic(predictions) != semantic(expected):
        raise ElectionReportError("sealed rows differ from the frozen as-of recipe")
    corpus = Corpus(fixtures, protocol)
    try:
        final = read_final_summary(corpus.read(FINAL, {FINAL}))
        rows = []
        for row in predictions["rows"]:
            contest = final["contests"].get(row["title"])
            if contest is None or any(n not in contest["choices"] for n in (row["current_side"], row["change_side"])):
                raise ElectionReportError("exact certified selected-pair identity missing")
            choices = contest["choices"]
            rows.append({**row, "outcome": int(choices[row["change_side"]]["votes"] > choices[row["current_side"]]["votes"]),
                         "final_current_votes": choices[row["current_side"]]["votes"], "final_change_votes": choices[row["change_side"]]["votes"],
                         "outcome_file": FINAL, "outcome_sha256": corpus.specs[FINAL]["sha256"]})
        base = [r for r in rows if r["scenario"] == "base"]
        scores = {name: _metrics(base, name) for name in ("model", *BASELINES)}
        strata = {field: {v: _metrics([r for r in base if r[field] == v], "model") for v in sorted({r[field] for r in base})}
                  for field in ("state", "contest_type", "counting_stage", "missing_data_stratum")}
        return {"schema_version": 3, "mode": MODE, "status": "exploratory; audit EA-13 validation remains incomplete",
            "protocol_sha256": protocol_hash(protocol), "sealed_predictions_sha256": expected_sha,
            "scored_base_rows": len(base), "all_scenario_rows": len(rows), "unique_contests": len({r["canonical_id"] for r in base}),
            "election_groups": 1, "jurisdiction_clusters": 1, "reversals": sum(r["outcome"] for r in base), "scores": scores, "strata": strata,
            "scenario_sensitivity": {label: {name: _metrics([r for r in rows if r["scenario"] == label], name) for name in ("model", *BASELINES)} for label in SCENARIOS},
            "uncertainty": {"interval": None, "reason": "One election in one county; no independent geographic or election replication. City contests are dependent, not resampling clusters."},
            "coverage": {"predictive_intervals": "NOT RUN: only binary selected-pair event probabilities are supplied", "source_contests": len(predictions["inclusion_exclusion_ledger"]),
                         "exclusions": dict(Counter(r["reason"] for r in predictions["inclusion_exclusion_ledger"] if r["reason"]))},
            "verified_latest_batch_baseline": "NOT RUN: source does not partition corrections; reported-increment sensitivity is separate",
            "official_countable_contest_pool_baseline": "NOT RUN: queue estimate requires declared acceptance/geography/arrival assumptions",
            "calibration_established": False, "production_eligible": False, "qualification": QUALIFICATION,
            "inclusion_exclusion_ledger": predictions["inclusion_exclusion_ledger"], "transition_ledger": predictions["transition_ledger"],
            "scored_rows": rows, "outcome_artifacts_read": corpus.reads}
    finally:
        corpus.close()


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    bins = [b for b in report["scores"]["model"].get("calibration_bins", []) if b["rows"]]
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
    axes[0].scatter([b["predicted"] for b in bins], [b["observed"] for b in bins], color="#2563eb")
    axes[0].set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="Conditional queue-scenario probability", ylabel="Observed selected-pair reversal fraction", title="Calibration diagnostic; one county")
    available = [(k, v["brier"]) for k, v in report["scores"].items() if v["status"] != "NOT RUN"]
    axes[1].barh([k.replace("_", " ") for k, _ in available], [v for _, v in available], color="#2563eb")
    axes[1].set(xlabel="Brier score (lower is better)", title=f"Same {report['scored_base_rows']} as-of selected-pair contests")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Orange County 2024 · official queue scenarios · unvalidated", fontweight="bold")
    fig.text(.5, -.02, "Acceptance and participation assumed · increment comparisons assume no corrections · no independent confidence interval", ha="center", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["predict", "score"])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--predictions-sha256")
    parser.add_argument("--plot")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    result = build_predictions(protocol, args.fixtures) if args.phase == "predict" else score_predictions(protocol, args.predictions, args.fixtures, args.predictions_sha256)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if args.plot and args.phase == "score":
        plot(result, args.plot)
    print(json.dumps({"phase": args.phase, "rows": len(result.get("rows", result.get("scored_rows", []))), "output_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
