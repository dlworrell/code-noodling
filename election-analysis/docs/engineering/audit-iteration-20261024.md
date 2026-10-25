<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:6fc5f0e3-5ac9-49f0-b54b-e95e0c31d730
-->

# Election audit implementation and remaining-evidence ledger

Date: October 24, 2026 (Pacific; execution continues October 25 UTC)
Authoritative issue ledger: [EA-AUD-2026-10-23 #45](https://github.com/dlworrell/code-noodling/issues/45)
Base: merged PR #60, `9d34651be0f07785793caedee6c05a8246e7aef5`
Branch: `fix/election-audit-iteration`
Design: [ADR-004](ADR-004-scientific-evidence-contract.md)
Machine evidence: [audit-iteration-validation.json](audit-iteration-validation.json)

This record supports the GitHub ledger; it does not replace the original audit,
reclassify experimental work as validated, or close unmerged implementation issues.
EA-01 is merged and closed. All subsequent rows require review/merge acceptance;
the rows with scientific/data blockers also retain those blockers after code review.

## Applied prompts and authorized scope

Selected from Dropbox `/prompt/REA-PROMPTS-INDEX.txt`:

- **PRM-0029**, rebuild one observed feature in an existing project, revision
  `65e8ddd0d60b201b90224`: implementation/testing in this governed Python scope,
  in bounded passes for time/identity, RCV/geography, pool/events/numerics, publication
  and validation. Preserve originals; work on a feature branch; do not merge/deploy.
- **PRM-0068**, resume an investigation and audit remaining unknowns, modified
  `2026-10-24T03:43:39Z`: static reconciliation of the existing audit/package,
  source coverage, alternative explanations and next distinguishing checks.
  RECORD is parent #45 plus this ledger; TARGET is the base and current branch;
  QUESTION is which missing evidence prevents each acceptance criterion; OUTPUT is
  this ledger and the existing parent record. Runtime reconstruction/testing belongs
  to the separately authorized PRM-0029 implementation scope.

No REA analysis engine is exposed. Actual tools are Git, Python/unittest, the pinned
EDT checkout, official-source web retrieval and connected Dropbox/GitHub/Notion.
No original archived replay scripts were executed. The newly authored CVR reader
was run against preserved source data. No user desktop or terminal was required.

## Issue-by-issue disposition

| Item | Implementation and verification | Remaining acceptance |
| --- | --- | --- |
| EA-01 #46 | Merged PR #60; exact tested head preserved on main, 119 earlier tests and successful CI; closed after parent evidence update | No calibration inference; parent independent re-audit remains |
| EA-02 #47 | Four clocks, per-contest source evidence, hash-bound publication identities, observation ledger, per-workbook Maine dates/coverage; actual Assessor/Skykomish conflicts retained | Review/merge; full Maine office/district/municipality coverage remains explicitly unreviewed |
| EA-03 #48 | Reviewed exact source-ID/name/canonical/jurisdiction aliases; no heuristic controlling collapse; one full Snoqualmie plus noncontrolling King slice; unrelated source records retained | Review/merge; remaining identities stay unreviewed and cannot forecast |
| EA-04 #49 | Empty/contradictory/reconciled/tie/unknown-pool/confirmed-zero states, South Beach quarantine, explicit zero only from reviewed counts; no turnout-overrun zero | Review/merge; scientific forecasts still unvalidated |
| EA-05 #50 | Office/election RCV scope; raw first-rank diagnostic; all six CVR files rebuilt; exact finalist and exhaustion accounting, duplicate/skip/overvote tests | Review/merge; official finalists supplied, independent candidate eligibility/elimination unsupported |
| EA-06 #51 | Hash-bound version reader; official round/final/recount conservation; nine Alaska contests individually covered; Maine raw/final separate; Measure 2 archive 737/recount 743 distinct | Direct official binaries still unavailable; HD28 original and HD40 full final accounting unresolved; recount Measure 2 complete accounting unresolved; MEDSL remains derivative |
| EA-07 #52 | CSV-to-analysis-to-human geography exclusions, approximate split-precinct assignment and vintages; statewide categories counted once, duplicate row rejection | Review/merge; borough slices remain noncontrolling |
| EA-08 #53 | Dated official category counts or low/base/high sensitivity, scope/freshness/acceptance/participation checks; no final leakage; legacy county scenarios withheld | Review/merge; no real contemporaneous 2024 outstanding series acquired; sensitivity is not a confidence interval |
| EA-09 #54 | Explicit selected pair/majority event and integer tie policy, reviewed positions/rules; unsupported passage/quorum rules suppressed; maximum band is triage only | Review/merge; no joint top-two/RCV/unsupported measure probability claimed |
| EA-10 #55 | Disjoint historical/latest likelihood, explicit new-ballot classification, no correction/transfer/recount/duplicate increment; caps remain hand-set | Actual training/held-out evidence needed to validate weights/compositional assumptions; stays unvalidated |
| EA-11 #56 | Separate evidence dimensions, conspicuous experimental numbers, unvalidated rubric with no repeated-snapshot validation bonus | Review/merge; no rubric correctness-probability claim |
| EA-12 #57 | 1,080 rational/integer oracles, 16 independent SciPy reference cases, 5,000/5,001 skew regression; stable recurrence and explicit budget/error policy | Review/merge; numerical accuracy does not validate the election model |
| EA-13 #58 | Frozen prospective protocol and scoring scaffold; time/lineage/tuning checks, whole-election uncertainty clusters, isolation from training elections/jurisdictions, baselines and proper scores | **BLOCKED: no eligible real chronological held-out forecast dataset**; no historical accuracy result is reported |
| EA-14 #59 | Versioned experimental-scenario evidence gate, validated-forecast gate false; all named regressions plus stage-aware 539-record replay, consistent outputs | Independent re-audit and held-out validation required for parent acceptance/promotion |

## Preserved evidence and successful reconstructions

The original archive is `Election_Analysis_2024_Replay_Package.zip`, SHA-256
`06d150c885e9f95aa3235dc4ceaab09f34379bc27f8de66864c00e85af67f73e`.
The fixture manifest preserves exact normalized input hashes for 204 Washington,
73 Alaska and 262 Maine records. The replay explicitly stages all 539 and withholds
every forecast. It is an ingestion regression, not an out-of-sample backtest.

The six original Maine exports contain 398,666 machine, 2,786 UOCAVA and 1,484
auxiliary ballots: 402,936 total, zero duplicate machine keys. The rebuilt anonymous
patterns reproduce Golden 197,151, Theriault 194,445, 391,596 continuing, margin
2,706. Noncontinuing: 10,673 skipped-rank, 388 no-continuing-choice and 279 overvote
ballots, 11,340 total. The 196,388 raw Golden rankings are not a final-round result.

The actual Washington primary retains the four-vote LD42 descriptive margin with
no prospective probability. Assessor and Skykomish keep conflicting vintages;
South Beach retains zero headline versus 982/613 detail as a contradiction. Snoqualmie
161/45 is the full reviewed representation; the old King 26/2 slice is separate.
The 298 source representations intentionally replace 244 guessed merged records.

## Residual unknown audit (PRM-0068)

| Unknown | Evidence sought and actual result | Affected conclusion and next distinguishing check |
| --- | --- | --- |
| F6 / EA-13 dated forecast history | Original package, repository snapshot history, connected-file discovery and official result pages reviewed. Package contains final-only counts and no eligible pre-outcome outstanding series; general file searches yielded no additional eligible series. These searches do not prove nonexistence. | Cannot estimate real forecast calibration. Acquire source-published pre-outcome snapshots, then-available category/pool evidence, frozen predictions and later certified event outcomes. Exclude already inspected 2024 outcomes from prospective held-out claims. |
| EA-06 Alaska official binary coverage | Official 2024 index identifies all nine RCV reports and HD28 recount. Direct downloads return HTTP 503/timeouts. Official indexed text supplies complete final accounting for US House, Senate D/F/L, House 6/36/38 and HD28 recount. | Indexed references lack verified binary hashes. Retrieve official binaries and independently verify text/accounting. Original HD28 and complete HD40 final accounting remain unresolved rather than filled from partial snippets. |
| EA-05 independent eligibility | All six original CVR hashes/counts and finalist allocation reproduced; official final summary supplies finalists | Full independent RCV elimination/write-in eligibility needs separately sourced historical rules and adjudication data. Current allocator refuses unknown eligibility and makes no independent-engine claim. |
| EA-02 Maine complete expected inventory | Per-workbook/block times, failures and minimal expected offices are now recorded; no reviewed complete district/municipality inventory acquired | Keep coverage false. Obtain a dated full office/district/municipality inventory and reconcile each expected block before enabling any current scenario. |
| EA-09 measure-specific requirements | Selected measures lack reviewed passage/quorum evidence in this correction | Keep unsupported rules unestimated; source election-specific legal requirements and update reviewed rule records. No generic majority assumption substitutes for them. |
| EA-10 predictive weighting | Nonoverlap arithmetic repaired and independently checked; no empirical model tuning performed | Compare preregistered baselines on isolated training/held-out election/jurisdiction groups before claiming weights are validated. |

Highest-priority unknown: eligible pre-outcome forecast history and outstanding-pool
evidence. Its absence blocks predictive validation even though numerical tests and
descriptive reconstruction pass. The code gate preserves that distinction.

## Reproduction

From the repository root, with EDT at its pinned revision
`bca66c1e2d2a9bf9c05f9429e618299216a17f27`:

```bash
PYTHONPATH=election-analysis:/path/to/edt python3 -m unittest discover -s election-analysis/tests -q
EDT_DIR=/path/to/edt election-analysis/ci/reports.sh generate
PYTHONPATH=election-analysis:/path/to/edt python3 -m election_reporting.replay --fixtures election-analysis/tests/fixtures --output /tmp/repaired-replay
PYTHONPATH=election-analysis python3 -m election_reporting.rcv --source-root /path/to/unpacked-original-package --manifest election-analysis/tests/fixtures/maine-2024-cd2-manifest.json --output /tmp/maine-rebuilt.json
PATH=/path/to/reuse/bin:$PATH python3 election-analysis/ci/provenance.py all --output /tmp/election-provenance
git diff --check
```

The expected deliberately failing sample printed by the provenance-test harness is
a harness fixture; use the outer suite result. Passing document/provenance checks
is not an independent statistical re-audit. Machine evidence records actual tool
versions, hashes and test totals; PR CI and remote signed commit/blob checks are
linked in the authoritative parent ledger after upload.


## Final implementation review

The initial 134-test / 12-commit proposal passed all eight GitHub workflows at
`3f5b77d7906f9c76d26f4348384a0ae705cc5e70`. A further review added guards for
judicial leader ties, candidates present only in detail files, duplicate area
aliases and multiple controlling identities for one canonical contest. The final
offline suite passes **136 tests**. The actual primary payload remains identical
to the generated correction except for generation time. Updated file hashes and
actual checks are retained in the machine evidence; final PR CI binds the amended
head. No independent statistical re-audit is implied by this implementation review.
