# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:f4bef2b4-113a-4af9-9a03-0912ba27df49

"""Normalized beta-binomial recurrence; no normal approximation switch.

The method sums the finite distribution, normalizing in log space. Its numerical
error is distinct from model calibration. The supported budget is deliberately
bounded; larger problems withhold the number rather than silently approximate.
"""
from __future__ import annotations

import math

MAX_REMAINING = 200_000
ERROR_POLICY = {"absolute_tolerance": 1e-8, "relative_tolerance": 1e-6,
                "relative_tolerance_floor": 1e-10, "max_remaining": MAX_REMAINING,
                "method": "normalized_full_support_log_recurrence_v1",
                "validation": "independent SciPy beta-binomial and rational small-case oracles; numerical only"}


def threshold(remaining: int, margin: int, tie_changes: bool) -> int:
    """Strict reversal vs loss of strict majority, using integer arithmetic."""
    return (remaining + margin + 1) // 2 if tie_changes else (remaining + margin) // 2 + 1


def posterior(change, current, batch_share=None, batch_votes=None):
    """Separate the preceding history from the latest verified new-ballot batch."""
    total = change + current
    batch_change = round(batch_share * batch_votes) if batch_share is not None and batch_votes else 0
    if (not batch_votes or batch_votes > total or batch_change > change
            or batch_votes - batch_change > current):
        batch_votes = batch_change = 0
    historical_total = total - batch_votes
    historical_change = change - batch_change
    historical_weight = min(historical_total, 300)
    share = historical_change / historical_total if historical_total else 0.5
    alpha = 1 + share * historical_weight
    beta = 1 + (1-share) * historical_weight
    if batch_votes:
        weight = min(batch_votes, 200)
        alpha += batch_change / batch_votes * weight
        beta += (batch_votes-batch_change) / batch_votes * weight
    return alpha, beta


def beta_binomial_tail(n, alpha, beta, first):
    if n < 0 or alpha <= 0 or beta <= 0:
        raise ValueError("invalid beta-binomial parameters")
    if first <= 0:
        return 1.0
    if first > n:
        return 0.0
    if n > MAX_REMAINING:
        return None
    # An arbitrary log mass at zero is sufficient because we normalize all mass.
    # Sum separately on each side with stable, short-block log-sum-exp reductions.
    logs = [0.0]
    value = 0.0
    for k in range(n):
        value += (math.log(n-k) - math.log(k+1)
                  + math.log(k+alpha) - math.log(n-k-1+beta))
        logs.append(value)
    maximum = max(logs)
    denominator = math.fsum(math.exp(v-maximum) for v in logs)
    numerator = math.fsum(math.exp(v-maximum) for v in logs[first:])
    return min(1.0, max(0.0, numerator / denominator))
