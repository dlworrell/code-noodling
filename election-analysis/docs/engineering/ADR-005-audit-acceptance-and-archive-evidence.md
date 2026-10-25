<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:41db8d37-2ec2-405d-b276-e6c73c5c4064
-->

# ADR-005: Audit acceptance, result lineage and archival evidence

Status: Accepted by merged PR #63; experimental scope, no scientific promotion
Date: 2026-10-25 UTC
Issues: parent #45; EA-02/06/07/08/09/10/11/13/14
Extends: ADR-003/004

## Problem and decision

PR #61 merged at `6f9fd06aaf5b6fb825ece8f3757e869c30f0bca4`. Acceptance
against that exact tree closes EA-03/04/05/12, alongside the earlier EA-01.
Further comparison exposed missing short-summary context, unchecked rule scope,
missing recount selection lineage and unreproduced sensitivity tables. Newly
acquired archives also use formats different from the live watcher. This change
finishes those implementation checks while keeping missing scientific evidence
visible.

Tool version becomes 0.5.1. Config schema 3, audit contract 1 and analysis JSON 4
remain compatible; the additional evidence fields are additive. Incomplete older
rule/scenario records remain readable but cannot authorize numeric scenarios.

## Scenario and publication contract

Reviewed rules require exact `office` agreement with the unique office in the
canonical identity and `election_type` agreement with `audit.election_type`.
An authority URL alone cannot make a general-election rule apply to a primary.
Unsupported supermajority, quorum and RCV rules continue to withhold numbers.

Scenario pools require an explicit `category_basis` as well as ordered
`low/base/high`, `basis`, dated compatible jurisdiction scope and participation
mapping. The category basis states which potentially countable ballots the
assumptions cover; it is not evidence that those ballots actually exist or will
be accepted. Official pools retain structured categories and acceptance/cure
assumptions. Missing data does not imply zero.

Every report status, including the short summary, states the known publication
and age ranges, unknown source/count clocks, incomplete coverage and missing
configured sources. Every displayed numeric decision carries its selected event,
source context, dated pool, geographic/category basis, participation assumptions
and sensitivity qualification. Tables name the selected challenger/current pair
instead of labeling that number simply “Winner” or “Top Two.” A third or fourth
candidate can change the winner/advancers without the selected pair reversing.
The aggregate evidence rubric remains an unvalidated heuristic.

`posterior_details` exposes preceding-history and latest-batch raw counts,
effective weights, beta shapes and zero reused overlap. These are power-likelihood
weights, not measured independent sample sizes. The 300/200 caps are unchanged,
hand-set and uncalibrated. Only a compatible, explicitly classified new-ballot
increment supplies a latest-batch likelihood; corrections, identical totals,
negative deltas, transfers and recount revisions cannot supply it.

## Official result version schema 2

`versions.py` reads schema 1 and 2. Schema 2 records explicitly include nullable,
offset-aware `source_published_at` and `count_updated_at`, separately from the
actual `evidence_as_of` observation. A date or local report time with unknown
timezone stays descriptive. `certification_status` reports the evidence actually
available, including “unknown”; an official label is not silently promoted to a
verified certification event.

Full accounting still requires nonnegative integer candidate, continuing, blank,
exhausted, overvote and ballot totals with exact conservation. A
`partial_accounting` record must conserve its known candidate/combined
noncontinuing totals, retain a finding and use null for unprovided categories.
For Measure 2, the official recount has 321,203 valid votes and 335,767 Times Cast:
the derived combined noncontinuing total is 14,564. The source does not partition
that number into blanks/exhausted/overvotes. The 743 recount margin and 737
derivative-archive margin remain separate.

Explicit reviewed `selections` name an existing canonical version/round, HTTPS
authority, reason and distinct superseded versions. An unresolved, raw-first or
intermediate round cannot become the current final reference. Selection is a
retrospective descriptive reference, never permission to place that result into
an earlier forecast. HD28 selects the later recount while retaining the
unresolved original final report; Maine selects the official final allocation
while retaining raw-first-ranking diagnostics. Indexed official text remains
labeled, without a fabricated binary hash.

## Independent archival readers

`archives.py` is separate from live acquisition and numeric forecast eligibility.
It uses bounded bytes, rows, fields and integer counts; duplicate/missing headers,
invalid counts, unknown modes and ambiguous positive repetitions are errors.

| Format | Preserved interpretation |
| --- | --- |
| King GEMS 2022–2025 | Exact contest/district/title identity tuple; UTF-8 or explicitly reported Windows-1252; district ballots stay district ballots; missing contest under/overvotes stay null; no automatic complete/controlling status |
| Alaska MEDSL 2024 derivative | All supplied modes once statewide; zero-filled repetitions collapsed without losing positive votes; padded precinct IDs mapped explicitly; judicial retention questions separated by judge; district rows cannot enter borough slices |

The Alaska source has 2,788 zero-filled repeated keys and four district early-vote
rows labeled `ELECTION DAY`. Those rows remain statewide, outside every borough.
All 73 statewide contests match the frozen replay's full candidate/ballot
accounting. Thirty borough/census-area slices retain exclusions, five split
precincts, 2023 geometry and 2020 population vintages; they remain approximate and
noncontrolling. This does not independently authenticate the derivative's
original official ENR bytes.

The King archive preserves 63 unique CSVs from four official election pages
(64 acquisition references, one duplicate). A 2022 page's November 22 link
actually targets a 2023 CSV; those bytes stay in the 2023 group. The 2023 final
filename and certification-day labels also differ. Provider paths, hashes,
retrieval times and HTTP Last-Modified proxies are retained without inventing
count-update clocks.

## Validation and unresolved evidence

The fixed Assessor/LD42 tables in `audit_sensitivity.py` independently reconstruct
the statically inspected old formula and compare with the October 23 audit.
Legacy overlap/normal-tail calculations are diagnostic subjects only and are
never imported by production publication. The post-event audit information time
and source hashes are fixed. No table is a confidence interval or an operational
forecast, and no construction is chosen by its fit to these examples.

The 63 King files supply 13,191 descriptive contest observations, but no eligible
held-out forecast record: contemporaneous outstanding/category evidence and
reviewed rule/scope records are still missing. Final outcomes/turnout cannot
backfill those inputs. The existing prospective validation contract remains
strict; a later retrospective design must separately establish historical
availability and must not claim reconstructed predictions were published then.
No Brier score, log-loss, calibration plot or historical accuracy is invented.
EA-13 remains open. Parent closure also requires independent final re-audit;
self-comparison and passing CI do not satisfy that requirement.
