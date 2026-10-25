<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:e46861a6-711f-4015-9f30-5682271d399f
-->

# Audit acceptance and next iteration

Date: 2026-10-25 UTC. Parent: [#45](https://github.com/dlworrell/code-noodling/issues/45).
Base: merged PR #61, `6f9fd06aaf5b6fb825ece8f3757e869c30f0bca4`.
Scope: `election-analysis/`; experimental, no forecast promotion.
This directory is the selected prompt's `OUTPUT` directory.

## Prompt selection and filled execution scope

| Dropbox prompt | Applied role | Server modification (UTC) |
| --- | --- | --- |
| `/prompt/PRM-0030.txt` | Primary: compare merged implementation with independent, evidenced acceptance oracles; classify matches, intended divergences and missing expectations | 2026-10-24 03:40:56 |
| `/prompt/PRM-0029.txt` | Implement the remaining bounded corrections in the existing project; preserve inputs and keep a reviewable branch | 2026-10-24 03:40:52 |
| `/prompt/PRM-0026.txt` | Recover the newly encountered King GEMS and Alaska MEDSL formats; bounded independent readers and malformed-input checks | 2026-10-24 03:40:40 |
| `/prompt/PRM-0068.txt` | Reconcile unknowns statically; identify the smallest missing evidence without fabricating data or completeness | 2026-10-24 03:43:39 |

PRM-0030 is the most operable starting prompt because PR #61 is already merged.
PRM-0026 is the more qualified prompt for the actual newly discovered format gap.
PRM-0068 supplies the missing-evidence discipline, rather than a generic request
for additional data. The machine verification records retrieved-text hashes;
the templates themselves remain in Dropbox. Tools used are Git, Python/unittest,
EDT, the connected GitHub/Dropbox/Notion services and primary-source retrieval.
No REA engine is exposed or claimed. No original replay script was executed.

Authorized independent implementation execution includes local tests, source
reconstruction and report generation. Original input and crosswalk bytes are
preserved. Legacy source code and audit scripts were read statically for formula
and format recovery. The prompt's branch/review boundary is retained: no merge,
deployment or production/account-setting change is performed by this iteration.

## Oracle provenance and comparison matrix

The authoritative audit is the version-6 narrative linked in #45. The frozen
package SHA-256 is
`06d150c885e9f95aa3235dc4ceaab09f34379bc27f8de66864c00e85af67f73e`.
Independent numerical oracles are rational finite-distribution calculations and
the committed SciPy fixture; source/count oracles are the frozen files and
explicit official indexed references. New implementation output is not used as
its own expected value.

| Case | Evidence / expected behavior | Disposition |
| --- | --- | --- |
| Merged PR #61 tree | Exact election-analysis tree equals tested PR head; clean rerun of 136 tests | MATCH; EA-03/04/05/12 accepted and closed, alongside EA-01 |
| Snoqualmie | One full 161/45 controlling result; King slice remains separate | MATCH |
| Generic repeated measure IDs/titles | Two counties remain distinct and noncontrolling without reviewed scope | MATCH in separate merged-tree adversarial check |
| South Beach / unknown / tie | 982/613 detail contradicts zero headline; unknown differs from confirmed zero; tie remains unresolved | MATCH; no unsupported numeric forecast |
| Maine six-export allocation | 402,936 = 391,596 continuing + 11,340 noncontinuing; finalists 197,151/194,445; missing supplemental file rejected | MATCH; official finalists supplied, independent eligibility engine unsupported |
| Numerical tails | 1,080 rational/integer checks, 16 independent reference cases, skewed R=5,001 near 0.24985007 | MATCH within ADR-004 error policy; unsupported budgets withheld |
| Mixed Washington vintage | Old guessed 147/97 split must not acquire one headline clock; reviewed identity repair retains 147 state + 151 county representations | Intended DIVERGENCE from old heuristic count; 151 unknown publication clocks remain explicit |
| Short summaries | Source-age/coverage ranges and dated category/participation pool basis were absent | Intended DIVERGENCE: include them beside numeric decisions and in report status |
| Wrong office/election rule | An HTTPS rule from another scope must not authorize a scenario | Intended DIVERGENCE: exact office/election agreement now required |
| Multi-candidate outcomes | A third/fourth candidate can change the winner/advancers while the selected pair does not reverse | MATCH to independent counterexamples; tables/social text name the selected pair |
| Alaska derivative full source | All modes once statewide; 73 contest candidate/ballot totals equal frozen replay | MATCH; 31 outputs, including 30 qualified noncontrolling geographic slices |
| Alaska mode/precinct conflicts | Four district early-vote rows labeled Election Day must stay outside boroughs; 2,788 zero-filled repetitions must not lose positive votes | MATCH with source findings retained |
| Recount selection | Preserve archive margin 737, select official later 743 with lineage; missing partition remains null | Intended DIVERGENCE: schema-2 reviewed selection and partial-accounting support |
| HD40 final | 1,008/1,541, continuing 2,549 + blanks 117 + exhausted 489 + overvotes 15 = 3,170 | MATCH to official indexed report; binary bytes/hash NOT VERIFIED |
| Assessor fixed sensitivity | Remaining pair counts 0/44,515/87,279/121,278 and legacy rounded 0/26.9/35.0/37.5% | MATCH; zero row is a hypothetical legacy assumption, not official zero |
| LD42 fixed sensitivity | Legacy cumulative-only 48.7%, share −2 points 43.2%, overlap 56.3%, +2 points 68.7% | MATCH; v2, latest-only and untempered alternatives disclosed without selecting by these outcomes |
| Actual historical forecast validation | Then-available outstanding/category evidence, frozen predictions and held-out targets required | EXPECTATION/DATA INCOMPLETE; no eligible scoring, no accuracy claim |
| Independent final re-audit | A separate qualified review of repaired outputs required by #45 | NOT RUN; this self-comparison is not an independent final audit |

Incidental temporary paths and reconstruction observation times are normalized
for semantic comparison. Source clock, stage, jurisdiction, vote count, ballot
category, result version and probability-event differences are never normalized
away. Invalid/missing headers, counts, rule scope, category assumptions, lineage,
naive clocks, ambiguous positive source repetitions and unsupported geography
have explicit negative tests.

## Reproduction

All 146 offline tests pass. The deliberately failing two-test sample inside the
provenance harness prints an expected inner failure banner; the outer suite and
recorded test results pass. REUSE/File-ID/manifest and patch checks are recorded
in `../audit-acceptance-validation.json`. CI and remote verification must bind to
the uploaded branch head, rather than assuming the local worktree is published.

```bash
EDT_DIR=/path/to/engineering-docs-toolkit python election-analysis/ci/test_results.py --output /tmp/election-checks
PYTHONPATH=election-analysis python -m election_reporting.audit_sensitivity --election election-analysis/data/elections/washington/2026-08-04 --output /tmp/sensitivity.json
PYTHONPATH=election-analysis python -m election_reporting.archives --medsl-alaska /path/to/frozen/historical-2024/alaska-medsl.csv --data-root election-analysis/data --output /tmp/alaska-reconstruction
```

The second and third commands execute newly written implementation. The Alaska
source's hash, all output hashes, the crosswalk and exact 73-contest comparison
are in `../alaska-archive-reproduction-20261025.json`. The immutable King CSV
bytes are in `../../../tests/fixtures/king-archives.zip`; the neighboring manifest
has exact URLs/revisions, hashes, encoding, actual retrieval times, HTTP proxy
clocks and per-file exclusions. Its 13,191 contest observations are descriptive.

The primary regenerated publication retains 298 records and zero numeric
forecasts. Known counts, the completed LD42 recount and source conflicts remain
visible. The six-export Maine reconstruction and 539-record descriptive replay
retain their scope and are not relabeled forecast validation.

## Remaining acceptance

At this checkpoint **5/14 child issues are closed**: #46/#48/#49/#50/#57. The
follow-up branch supplies remaining implementation evidence for #47/#51/#52/#53/
#54/#55/#56 and the gate matrix in #59; they remain open until merge acceptance.
EA-13 #58 remains blocked on actual eligible historical evidence. A 2026
retrieval timestamp, final turnout, or final ballot status cannot stand in for
then-known outstanding counts. A later retrospective protocol must document its
distinct availability/reconstruction semantics; the current prospective
evaluator is not loosened to accept backdated forecasts. Caps/weights remain
hand-set and unvalidated, with unresolved empirical comparison carried to #58.

Direct Alaska PDF downloads still return 5xx errors. Indexed official text
provides scoped references, not verified binary authenticity. Original HD28
final accounting and the Measure 2 recount partition remain explicitly
unresolved; the acceptance contract permits visible unresolved coverage rather
than fabricated totals. The independent final review required by #45/#59 is
still outstanding. These limits prevent a truthful “no issues remain” claim.

## Continuation: accepted main and frozen scenario experiment

PR #63 is now merged at `df8499f1bb336c64967d506953b7532d3bac61da`.
Its exact tree matches the tested head/CI tree; all eight workflows, 146 tests
and actual provenance overall 0 passed. Seven further scoped acceptances bring
the ledger to **12/14 children closed**. The earlier 5/14 checkpoint and its
PRM-0029 build-only merge boundary are historical. Under the renewed resolution
instruction, PRM-0030 governs acceptance through the repository's normal signed
commit/PR controls; no deployment or scientific promotion is inferred.

Filled current inputs: implementation `election_reporting.retrospective` and
`tests/test_election_reporting_retrospective.py`; original/specification evidence
is the audit's frozen formula, rational distribution oracles, recorded source
archives, dated government bulletins and explicit EA-10/13 contracts. Comparison
scope is captured originals plus execution of the new implementation only.
Output remains this directory, linked to `../king-retrospective-*-v2.*` artifacts.
PRM-0068 statically reconciles missing evidence in [INVESTIGATION.md](INVESTIGATION.md).
The current source/recipe/split and limitations are specified in [ADR-006](../ADR-006-retrospective-scenario-evaluation.md).

| Case | Independent expectation / actual observation | Disposition |
| --- | --- | --- |
| Frozen model and whole groups | All 33 package files hash-bound; no fitting; training membership or altered source rejected by synthetic adversarial cases | MATCH |
| Reserved outcome separation | Predict succeeds even when the reserved outcome is invalid CSV; actual prediction reads 32 as-of files and no reserved files | MATCH |
| Prediction seal and lineage | Changed bytes, duplicate scenarios, altered clocks, eligibility promotion or fabricated/missing baseline values rejected | MATCH |
| Historical proxy limits | Independent calendar cases preserve next-day 16:30 Pacific, delayed modification, stale/migrated exclusion and earliest duplicate availability | MATCH, qualified proxy only |
| Nonpositive turnout remainder | A reached turnout projection is unavailable pool evidence, never official zero | MATCH |
| Binomial numerical baseline | Small exact rational enumeration and nine independent SciPy cases through n=200,000; max absolute difference 2.52e-11 | MATCH; numerical only |
| Dependence-aware loss weights | Independent grouped aggregation and synthetic repeated-snapshot invariance; 147 base rows, 21 contests, three elections, one county | MATCH; no geographic confidence interval claimed |
| Named certified event | Strict selected-pair reversal; certified tie is not reversal; original certified CSV identity retained, no legal-winner/latest-recount assertion | MATCH to synthetic event oracle |
| Verified latest-batch baseline | Official new-ballot/correction partition is absent; positive tally deltas cannot manufacture it | NOT RUN |
| Official outstanding baseline | Dated accepted/category/outstanding counts absent; turnout scenarios do not substitute for this baseline | NOT RUN |
| Historical-byte authenticity | HTTP metadata and current archive labels do not authenticate unchanged bytes at historical publication | EXPECTATION UNKNOWN |
| Independent final re-audit | Separate qualified scientific review required by parent #45 | NOT RUN |

Current matrix: **8 MATCH, 0 observed DIVERGENCE, 3 NOT RUN, 1 EXPECTATION UNKNOWN**.
The largest unresolved gap is batch/pool evidence and independent review, not a
tested equivalence claim. The 493 hypothetical scenario rows were generated
after the protocol freeze and sealed before scoring; 147 base rows received
Brier/log-loss diagnostics. Scores compare realized outcomes under hypothetical
pools and cannot validate calibration conditional on those pools being realized.
The model's base Brier is 0.03755657 and log loss 0.21044475; cumulative-share
values are 0.03837115/0.95873236. Small differences and one county do not establish
superiority. Low/high cases have different coverage and are sensitivity only.

Nine new offline tests cover the phase boundary and numerical/control cases.
The final structured suite passes all **155 tests**, including the chronological
duplicate regression. Local provenance is **overall 0**: 122/122 files licensed,
71 authored File-IDs and 62 exemptions, with no File-ID violations. Actual
versions, hashes, aggregate check and commands are in
`../king-retrospective-validation-v2.json`. The original 539-record replay stays
descriptive, the prospective evaluator stays unchanged, and every actual primary
forecast remains withheld. #55/#58 and parent #45 remain open for the named
scientific and independent-review criteria.


## Current continuation — official Orange County queue evidence

Applied the current Dropbox PRM-0030 (primary acceptance), PRM-0026 (new PDF,
AcroForm and export formats) and PRM-0068 (static residual evidence). The new
implementation was executed under the user's renewed resolution instruction.
Original bundled replay scripts and a REA engine were not executed. Source bytes,
clocks, qualification, versioned schema and reproduction are in ADR-007 and the
neighboring `oc-official-*-v1` records. PR #64 remains the merged base.

| Case | Evidence / actual result | Disposition |
| --- | --- | --- |
| Original count extraction | Three PDFs, all candidate/cast row triples agree between Poppler and pypdf; 212 contests each | MATCH |
| Nov8 copies | Different raw hashes/metadata, identical 212-contest parsed values | MATCH; bytes never normalized away |
| Nov9 press count link | Both dated labels actually link to the Nov8 report; canonical Run_13 supplies Nov9 | Source DIVERGENCE retained |
| Last-report form label | 140894/205077 are election-night additions; latest increase is 64183 | Source-semantic DIVERGENCE reconciled to dated registrar release |
| Pool forms | AMENDED retained; category sums 275420/212152; challenged overlap unknown, not added in base | MATCH |
| Final source comparison | 172 exact titles and 412 choices agree in total/mail/center; 40 abbreviated titles and 4 damaged names remain unmatched | MATCH within explicit coverage |
| Scored pairs | All 159 scenario events independently match exact precinct aggregates for 53 distinct contests | MATCH |
| Frozen groups and phase boundary | 37 model/dependency hashes; no fitting; one held-out county/election; four as-of reads, zero reserved outcome reads; full recipe replay matches sealed rows | MATCH |
| Proper scores and numerics | All seven methods share 53 base cases; direct losses agree within 1e-14; 636 SciPy comparisons within 9.65e-11 | MATCH, arithmetic only |
| Real new-ballot/correction partition | Explicit official additions exist, correction partition does not | NOT RUN for verified latest likelihood; conditional sensitivity labeled |
| Official countable contest pool | Official unprocessed estimate acquired; acceptance/geography/arrivals remain assumptions | NOT RUN for guaranteed countable-pool baseline; qualified queue comparison run |
| Historical original versions | Present bytes and producer clocks cannot authenticate historical content identity | EXPECTATION UNKNOWN |
| Independent final re-audit | Separate qualified scientific reviewer still required by #45 | NOT RUN |

The frozen recipe produced 159 rows, scored 53 base contests and observed two
selected-pair reversals. Primary Brier/log loss is 0.03368266/0.09501048 versus
cumulative-share 0.03777127/0.13645187. Conditional disjoint increments produce
0.02887058/0.08424815, but their no-corrections premise is unverified. None of
these differences establishes superiority or calibration. The diagnostic labels
its one-county and conditional assumptions. Full source/format checks and nine
new regression tests pass; the complete structured suite passes **164 tests**.
Final local provenance and exact remote/CI evidence accompany merge acceptance.

The data is sufficient to verify extraction, accounting and these qualified
comparisons. It is insufficient to clear every remaining audit requirement.
#55/#58 and parent #45 remain open; 12/14 child issues remain resolved. The
current next-data contract in INVESTIGATION.md replaces the earlier blanket
statement that no official queue source had been acquired.


## Continuation: dependency drift and a publisher-hash-verified inventory

| Case | Captured evidence and actual comparison | Disposition |
| --- | --- | --- |
| Current dependency drift | PR #67 changes pypdf 6.10.0 to 6.21.2; current-source old-protocol execution rejects the changed manifest | Intended rejection; no first protocol rewritten |
| Exact-source/runtime runner | Original 37 files and package inventory verified; Python 3.12.14 / pypdf 6.10.0 / reuse 6.2.0; 159 predictions and complete scores/exclusions match | MATCH after computation clocks and the new score seal only |
| Real negative controls | Changed source pin and mismatching installed interpreter both exit 1 before prediction | MATCH |
| Current parser qualification | pypdf 6.21.2 regenerates the original PDF/CSV/TSV qualification byte-for-byte | MATCH, extraction/accounting only |
| Official RLA manifest | Original bytes equal the publisher's displayed SHA-256; 10,208 composite batch keys and source total 2,757,630 | MATCH for final inventory; units not equated to voter ballots |
| Dated batch likelihood | Manifest lacks clocks, candidate totals and correction/recount/transfer partitions | NOT RUN; explicitly ineligible, not an inferred zero |
| SF CVR archive | First browser request and one reload return HTTP 403 | NOT ACQUIRED; source absence not asserted |
| Independent final review | Review packet prepared with original criteria and closure questions | NOT RUN |

Nine new regression tests cover runtime/source drift, package inventory, safe
extraction, incidental normalization and malformed/duplicate batch inventory.
The complete modern-runtime suite passes 173 tests. Final exact-tree provenance,
source/blob/signature and CI verification are required at acceptance. Full new
reproduction outputs, seal and logs are preserved in the hash-bound runtime ZIP;
they do not replace the original first run. The original missing-science limits
remain explicit and no implementing-agent self-review substitutes for #45.
