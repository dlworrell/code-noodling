<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:694e3d97-3c23-49f3-91b4-bf856f1a223e
-->

# ADR-004: Reviewed evidence and experimental scenario boundaries

Status: Proposed for review; no scientific promotion
Date: 2026-10-24 (Pacific)
Issues: EA-02–EA-14, parent [#45](https://github.com/dlworrell/code-noodling/issues/45)
Supersedes: numerical, identity, timing and remaining-pool assumptions in ADR-001/002; extends ADR-003

## Contract

Config schema 3 retains schemas 1/2 as descriptive compatibility inputs. An
optional `[audit]` table uses `schema_version = 1`, a scoped `election_id`, and a
positive finite `max_source_age_hours` (default 48). Missing evidence withholds
numeric scenarios; it never creates an election rule, timestamp or zero pool.

| Table | Reviewed key and required evidence |
| --- | --- |
| `identities` | `source_id`, `contest_id`, exact `source_contest_name`; canonical election/jurisdiction/office, HTTPS authority, explicit jurisdiction members, coverage flag and controlling/partial role |
| `source_versions` | source ID and immutable normalized snapshot SHA-256; separate publication, count-update and retrieval times; coverage findings and publication identity |
| `area_versions` | source ID and immutable area-file SHA-256; publication identity/time for compatibility |
| `rules` | canonical ID, supported rule, positions and authority; passage numerator/denominator and validation requirements; unsupported cases withhold numbers |
| `remaining_pools` | canonical ID, offset-aware as-of time, exact jurisdiction/category coverage; official count/authority or ordered low/base/high sensitivity scenarios and basis; explicit contest/pair participation fractions |
| `official_versions` | relative JSON path and SHA-256; individual version/round records with authority, evidence observation time, accounting and unresolved-coverage findings |

Duplicate selectors fail. Names and local IDs alone never merge source records.
An exact ID/name/jurisdiction area alias is required for geographic joins. Missing
offices retain last-known descriptive versions with coverage findings. Unreviewed
records remain separate rather than silently disappearing behind a statewide name.
The primary changes from 244 guessed canonical records to 298 explicit source
representations (147 statewide and 151 King County); six identities have reviewed
crosswalks. Only one full Snoqualmie representation controls that canonical contest.

## Time and coverage

Filename dates remain legacy labels with unknown timezone unless independently
reviewed. Publication (including an HTTP Last-Modified proxy), count-update,
retrieval and report generation are distinct fields. A repeated retrieval of the
same bytes records an observation without moving the count clock. Publication
freshness is explicitly a proxy when the count-update time is unavailable.

`SOURCES.json` schema 2 adds observations and coverage, retaining schema-1 reads.
Maine adds per-workbook/block publication and source hashes, discovered/failed
workbooks, a minimal expected-office inventory and unresolved full-coverage status.
A newest workbook never refreshes every other office. Full district/municipality
coverage still requires a reviewed inventory. Optional normalized CSV `Result
Metadata` carries geography, raw-stage and source evidence; evidence sections must
be objects and metadata is size bounded. Raw metadata cannot upgrade an unreviewed
source to complete coverage.

## Scientific gate and states

Analysis schema 4 exposes generated time, source vintages, coverage, canonical
identity, data state, official versions, decision events, pool sensitivity,
numerical policy and `scientific_eligibility` schema 1. Checks cover lifecycle/
stage, source age, provenance, complete identity/scope/coverage, compatible geography,
noncontradictory reported data, supported rules and a valid remaining pool. Numeric
budget failures also withhold calculations. Descriptive tallies remain visible.

No-results, inconsistent ballot accounting, contradictory headline/detail,
source-vintage conflict, unresolved tie, unknown pool and confirmed-zero outstanding
are distinct states. A known zero outstanding count does not adjudicate a tie.
Date-only legacy county remaining counts cannot support numeric county scenarios.

`eligible` means eligible for an **experimental conditional scenario** only.
`validated_forecast_eligible` is false: no real held-out calibration has been
completed. Calibration is an explicit unresolved gate for promotion, not a string
that passing lifecycle or numerical tests converts to validated status. Human,
JSON, HTML and social outputs retain the experimental/uncalibrated designation.

## Events and numerical model

Every probability names a selected strict pair reversal or selected-leader
majority change. A pair event is not a full top-two set probability. Maximum
boundary bands are triage, not a joint probability that anything changes.
Unsupported supermajority, quorum, RCV, recount and litigation scenarios withhold
numbers. The exact integer thresholds are `floor((M+R)/2)+1` for strict reversal
and `ceil((M+R)/2)` for loss of a strict majority.

`disjoint_tempered_beta_binomial_v2` separates preceding history from the latest
explicitly verified, compatible nonnegative new-ballot batch. Duplicate bytes,
corrections, transfers, recounts and unknown deltas cannot become a new likelihood
increment. The caps of 300/200 are hand-set and unvalidated. Low/base/high remaining
scenarios propagate separately, with explicit participation assumptions and no
confidence-interval claim. No final-total backfill or turnout-overrun zero is used.

[ADR-009](ADR-009-consecutive-whole-contest-batches.md) repairs the independent
review's counterexamples: qualification uses the complete immediate source
transition before history de-duplication, for both pair and majority likelihoods.

Full-support normalized log recurrence replaces the normal approximation switch.
The supported numerical budget is 200,000 remaining decision votes; larger problems
return no number. The checked policy is 1e-8 absolute / 1e-6 relative tolerance,
relative floor 1e-10. Rational finite-distribution and independently generated SciPy
references cover integer thresholds, the 5,000/5,001 boundary and skewed cases.
Numerical validation does not calibrate an election model.

The optional evidence rubric is explicitly unvalidated, capped at 50 and gives no
independent-validation bonus to repeated cumulative snapshots. Separate evidence
dimensions carry the actual scientific limits.

## RCV and replay

Maine RCV applicability follows office/election type, not webpage grouping. Raw
nonblank first-rank percentages are diagnostics, never interpreted Round-1 majority
or final-winner/stopping rules. The six original CD2 exports reproduce 402,936
ballots, official finalists 197,151/194,445 and 11,340 noncontinuing ballots. The
allocator handles skipped ranks, overvotes and duplicate later rankings, but accepts
official finalists/noncontinuing eligibility as inputs. It is not an independent
eligibility or elimination engine; unknown marks are refused. No ballot IDs persist.

Alaska borough CSV and reports retain included/excluded modes, approximate largest
resident-population assignment, five split precincts, 2023 geometry and 2020 Census
vintages. These precinct subsets never control full contests. Duplicate category/
candidate rows are refused; statewide denominators count each supplied category
once. Integer output is not evidence of geographic completeness.

Official round/final/recount records require conservation and remain separate from
raw rankings and transfers. Nine Alaska RCV contests are individually represented;
unretrieved reports have explicit unresolved findings. Indexed official PDF text is
labeled as such, with no invented binary hash. The derivative MEDSL archive remains
identified. Measure 2's 737 archive margin and later 743 recount margin are separate.

The frozen 539-record 2024 replay verifies source ingestion and stage separation.
It is never a chronological forecast backtest. Prospective `validation.py` enforces
frozen protocol/model, pre-outcome times, group isolation, no tuning leakage and
preregistered baselines, then computes grouped proper scores, calibration bins and
bootstrap uncertainty. Actual eligible held-out predictions are still missing.

## Review and recovery

Existing PR, verified-commit, provenance, license, source-size and numeric controls
remain in force. Tests do not grant authority to merge or promote a forecast.
Correct an evidence annotation through a new reviewable version; keep originals in
Git and their immutable archive hashes. The parent stays open until its independent
re-audit and any unresolved validation findings have an honest disposition.
