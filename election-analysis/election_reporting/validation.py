# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:2837ab21-dd90-44bd-9d6a-43e418209eff

"""Prospective validation contract and grouped proper-score evaluation.

This evaluates supplied forecasts; it never reconstructs a prediction from a
final tally. Frozen descriptive replays are explicitly ineligible.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

from .evidence import timestamp
from .model import ElectionReportError


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluate(rows, protocol):
    frozen = timestamp(protocol.get("frozen_at"))
    if not frozen or protocol.get("schema_version") != 1 or not protocol.get("model_id"):
        raise ElectionReportError("a dated frozen protocol and model are required")
    if not rows:
        return {"status": "unvalidated", "reason": "No eligible held-out forecasts supplied; final-only replay is not validation.",
                "protocol_sha256": protocol_hash(protocol), "scored_rows": 0, "independent_groups": 0}
    digest = protocol_hash(protocol)
    groups = {}
    keys = set()
    for row in rows:
        information = timestamp(row.get("information_time"))
        outcome = timestamp(row.get("outcome_published_at"))
        if not information or not outcome or not frozen <= information < outcome:
            raise ElectionReportError("forecast must follow protocol freeze and precede its certified outcome")
        if row.get("protocol_sha256") != digest or row.get("model_id") != protocol["model_id"]:
            raise ElectionReportError("forecast does not match frozen protocol/model")
        if row.get("split") != "held_out" or row.get("used_for_tuning") or row.get("remaining_pool_status") not in {"official_outstanding", "conditional_scenario"}:
            raise ElectionReportError("final-only, tuned, or unestimated row is ineligible for validation")
        if row.get("event_scope") != protocol.get("event_scope") or not row.get("input_sha256"):
            raise ElectionReportError("forecast target or input lineage is missing/incompatible")
        identity = (row.get("election_id"), row.get("canonical_id"), row.get("jurisdiction_id"))
        if (not all(identity) or row["election_id"] in protocol.get("training_elections", [])
                or row["jurisdiction_id"] in protocol.get("training_jurisdictions", [])
                or list(identity[:2]) in protocol.get("training_groups", [])):
            raise ElectionReportError("related election/contest group leaks into training and held-out sets")
        group = row["election_id"]  # entire elections cluster related races/jurisdictions/snapshots
        key = (*identity[:2], information.isoformat())
        if key in keys:
            raise ElectionReportError("duplicate forecast snapshot")
        keys.add(key)
        p, y = row.get("probability"), row.get("outcome")
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 or y not in (0, 1):
            raise ElectionReportError("invalid forecast probability or binary event outcome")
        baselines = row.get("baselines", {})
        if set(baselines) != set(protocol["baselines"]) or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in baselines.values()):
            raise ElectionReportError("all preregistered baseline probabilities must be supplied")
        groups.setdefault(group, []).append(row)
    epsilon = protocol.get("log_loss_clip", 1e-12)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ElectionReportError("invalid preregistered log-loss clipping policy")
    def score(p, y):
        q = min(1-epsilon, max(epsilon, p))
        return (p-y)**2, -(y*math.log(q)+(1-y)*math.log1p(-q))
    scores = {}
    for name in ["model", *protocol["baselines"]]:
        group_scores = []
        for records in groups.values():
            values = [score(r["probability"] if name == "model" else r["baselines"][name], r["outcome"]) for r in records]
            group_scores.append(tuple(sum(v[i] for v in values)/len(values) for i in (0,1)))
        averages = [sum(v[i] for v in group_scores)/len(group_scores) for i in (0,1)]
        rng = random.Random(20261025)
        bootstraps = sorted(sum(rng.choice(group_scores)[0] for _ in group_scores)/len(group_scores) for _ in range(1000))
        scores[name] = {"group_weighted_brier": averages[0], "group_weighted_log_loss": averages[1],
                        "group_bootstrap_brier_95pct": [bootstraps[24], bootstraps[974]]}
    bins = []
    for low in range(10):
        # Weight each election/contest group equally, including repeated snapshots.
        selected = [(r,1/len(rs)) for rs in groups.values() for r in rs if low/10 <= r["probability"] <= 1 and (r["probability"] < (low+1)/10 or low == 9)]
        weight = sum(w for r,w in selected)
        bins.append({"low": low/10, "high": (low+1)/10, "rows": len(selected),
                     "predicted": sum(r["probability"]*w for r,w in selected)/weight if weight else None,
                     "observed": sum(r["outcome"]*w for r,w in selected)/weight if weight else None})
    return {"status": "evaluation_only; independent review required before promotion", "protocol_sha256": digest,
            "scored_rows": len(rows), "independent_groups": len(groups), "group_definition": "whole election", "scores": scores, "calibration_bins": bins,
            "coverage": {"interval_coverage": "not applicable: binary-event probabilities, no outcome intervals supplied"},
            "uncertainty": "whole-election group bootstrap; small group counts do not establish calibration"}
