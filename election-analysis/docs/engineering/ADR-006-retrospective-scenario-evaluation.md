<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:b58839c5-3175-4f8a-b316-b8e5ef00f908
-->

# ADR-006: Frozen retrospective turnout scenarios

Status: Proposed for review; experimental, no forecast promotion
Date: 2026-10-25 UTC
Issues: EA-10 #55, EA-13 #58; parent #45
Extends: ADR-004/005; independent offline artifact schema 2

## Decision and evidence

PR #63 merged at `df8499f1bb336c64967d506953b7532d3bac61da`.
Twelve of fourteen audit children are now accepted. The remaining empirical
questions require an honest distinction between a computable retrospective
scenario and a historically published, eligible prediction.

Three dated government bulletins supply pre-election turnout assumptions:
45% (2023), 82% (2024) and 45% (2025). Exact authority URLs, publication and
actual retrieval times, and raw-page hashes are in
`king-historical-forecast-sources.json`. They do not supply outstanding,
accepted, cured, rejected or geographic/category batch counts. A 2022 bulletin
revises an earlier projection downward; migrated 2022 archive bytes remain
excluded. Final turnout is never substituted for a missing assumption.

`election_reporting.retrospective` is a separate two-phase, bounded evaluator.
It does not change `validation.evaluate`, which still requires a prospective
freeze before information time and certified outcome. It does not alter live
forecast eligibility, publication or the production model.

The schema-2 protocol binds all 33 Python source files in `election_reporting`,
the frozen archive manifest, source-fact file, recipe, split and named outcomes.
It was frozen at `2026-10-25T06:40:41.360400+00:00`, before the first actual
prediction/scoring run. A filename-order correction and synthetic regression
occurred before that run and are retained in `pre_run_revisions`. No parameter
was fitted or selected using these outcomes. Historical outcomes were already
public and have previously been inspected: phase separation is computational,
not a claim of blind preregistration or forecasts published in 2023–2025.

## Information, event and pool semantics

The reviewed research scope includes whole King County/County Council and
Seattle/Seattle Council candidate tallies in the county's archive. State,
federal, other jurisdictions, measures, unopposed and tied as-of pairs are
excluded explicitly. Composite source identities are not merged by name.
The event is strict reversal of the as-of top two named candidates in a named
original certified CSV; ties are not reversals. It is not a legal-winner claim,
write-in eligibility adjudication or latest-recount selection.

The information clock is the later of the dated report's next day at 16:30
Pacific and its HTTP Last-Modified proxy. The official archive pages state that
daily reports are made available by that next-day deadline. This reconstructed
clock is not authenticated then-captured bytes or the count-update clock, which
remains null. Snapshots older than 48 hours by this policy, post-certification
proxies, final files and duplicate bytes are excluded. Certification-day
midnight is a conservative lower bound, not an invented exact publication time.

Each low/base/high hypothetical final source pool uses the as-of maximum district
registration count times the dated turnout projection minus/plus five percentage
points, then subtracts as-of maximum district ballots. The positive source pool
maps to pair votes using observed pair participation in those source ballots.
It assumes future participation like observed participation and countable ballot
categories; acceptance/category evidence is unavailable. A reached or exceeded
turnout target withholds the estimate and never becomes official zero remaining.
The offsets are hand-set sensitivity inputs, not confidence limits. The scenarios
can have different eligible rows; their scores cannot establish a superior offset.

Every tally delta stays unknown. A monotone increase does not prove a new-ballot
batch. The existing 300-history/200-latest power-likelihood parameters remain
unchanged, with no latest votes used in this experiment. Cumulative-share
binomial, untempered beta-binomial, fixed-half and scenario margin-lock baselines
are computable. Verified-latest-batch and official-outstanding-pool baselines
are explicitly NOT RUN. This experiment cannot validate the 200 latest cap.

## Execution, scoring and limits

`predict` may parse only nonreserved as-of artifacts. It emits all rows, read
artifacts and complete file/contest/scenario inclusion-exclusion records.
The 493 scenario rows were hashed and sealed in a separate command before
`score` read the three reserved certified files. The prediction seal is
`9be7e233195698250928284909e070e96f4cd1b4de2f519820af7de99af72426`.
Scoring verifies that seal, protocol lineage, clocks, input hashes, identities,
event scope, probability bounds, missing baselines and duplicate keys.

Base scoring retains 147 observations across 21 contests: 98/11/38 observations
and 10/1/10 contests for 2023/2024/2025. Three contests have at least one observed
selected-pair reversal; the repeated snapshots are not independent events.
Weights give equal mass to elections, then contests, then snapshots. Thus the
single 2024 contest receives one third of aggregate election weight. The
diagnostic reports this limited design, not a representative accuracy rate.

| Base construction | Brier | Log loss |
| --- | ---: | ---: |
| Frozen cumulative fallback model | 0.03755657 | 0.21044475 |
| Cumulative-share binomial | 0.03837115 | 0.95873236 |
| Untempered beta-binomial | 0.03815923 | 0.92030408 |
| Scenario margin lock | 0.15147727 | 0.41998418 |
| Fixed half | 0.25000000 | 0.69314718 |

These are descriptive losses against realized tallies under hypothetical pools,
not calibration conditional on those pools actually being realized. The small
Brier differences do not establish model superiority. Log loss clips only its
logarithm at `1e-12`; no probability or Brier value is clipped. Counting-stage,
contest-type, state and missing-data strata and calibration bins are retained.
There is only one geographic dependence cluster. No defensible geographic
confidence interval is reported; binary probabilities supply no outcome
intervals to test. Missing baselines, archive authenticity, broader coverage
and independent review keep #55/#58 and the parent open.

## Reproduction

Run from `election-analysis/` with Python 3.12; plotting additionally needs
Matplotlib. The rational regression oracle uses only the standard library.
Numerical SciPy checks and full tool versions are in the validation record.

```bash
PYTHONPATH=. python -m election_reporting.retrospective predict --protocol docs/engineering/king-retrospective-protocol-v2.json --fixtures tests/fixtures --forecast-sources docs/engineering/king-historical-forecast-sources.json --output /tmp/king-predictions.json
sha256sum /tmp/king-predictions.json
# Record that new hash BEFORE running the separate scoring phase.
PYTHONPATH=. python -m election_reporting.retrospective score --protocol docs/engineering/king-retrospective-protocol-v2.json --fixtures tests/fixtures --predictions /tmp/king-predictions.json --predictions-sha256 RECORDED_NEW_HASH --output /tmp/king-scores.json --plot /tmp/king-diagnostic.png
```

Actual regenerated timestamps and therefore prediction hashes differ on rerun;
model probabilities, membership and scores must agree. Keep the committed first
seal/predictions intact. A changed source/model requires a new protocol. See
`audit-acceptance/RECONSTRUCTION-VALIDATION.md` and `INVESTIGATION.md` for the
comparison, actual checks and concrete remaining evidence request.
