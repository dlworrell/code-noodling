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
