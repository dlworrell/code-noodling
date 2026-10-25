<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell
SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:fb45edcc-a939-4cad-afde-a24445865176
-->

# Independent statistical review of EA-AUD-2026-10-23

**Reviewed snapshot:** `dlworrell/code-noodling@3e9ae3bee1c657c91629b133f379a5b73e41781a`  
**Repository tree:** `2ccd8e0a0a08d610e61e0e51938794d57dddecd8`  
**Review date:** October 25, 2026 UTC  
**Reviewer:** A separate agentic AI from the implementing agent. This is an independent AI review within the same authorized task and shared workspace. It is not a human expert review, external audit certification, ballot audit, or blind preregistration. The reviewer did not implement repairs, modify the implementation checkout, contact anyone, or mutate GitHub issues or PRs.

## Decision at this exact snapshot

**Do not close #55 or parent #45 at this snapshot.** The repaired publications and frozen computations are substantially improved and reproduce, but independent adversarial checks found two defects in latest-batch qualification. They violate an original mandatory #55 acceptance condition even though all 173 existing tests pass. They do not invalidate the separately reproduced zero-latest-vote King/Orange County experiments, and no affected numeric forecast was found in the actual Washington historical publication.

**#58 can be resolved only as a documented unvalidated result under its original permitted failure outcome.** Its intended chronological real-world forecast validation has not been accomplished. The original issue explicitly says: “If historical inputs are insufficient, publish a documented unvalidated result and acquisition gaps.” The available records do that, retain all missing-baseline/coverage limits, and support reproducible conditional experiments. Closing an acquisition/evaluation work item with that disposition must not say that historical forecast validity, calibration, trained weights, historical byte identity, or generalization was established.

Disposition vocabulary in this review: **PASS** means the reviewed implementation criterion is met within its stated scope; **PARTIAL** means useful evidence exists but the intended empirical claim remains unvalidated, including an expressly permitted failure outcome; **BLOCKED** means an original mandatory implementation condition fails or a required final acceptance action is still missing. Passing implementation criteria does not promote a forecast.

## Authority and scope

I fetched the current authoritative bodies of parent [#45](https://github.com/dlworrell/code-noodling/issues/45) and its actual fourteen original children [#46–#59](https://github.com/dlworrell/code-noodling/issues?q=is%3Aissue+%22Election+audit%22). The parent identifies #59 as EA-14; no child number was inferred. Full captured issue bodies are retained in the review work products. I read the original F1–F8/N1 methods audit, the current version-6 replay narrative, ADR-003 through ADR-008, current README/user guide/public outputs, the independent-review packet, frozen protocols, source/score/exclusion ledgers, reproduction ZIP, DATA-REQUEST and INVESTIGATION. The implementer's packet and green CI were treated as evidence inputs, not conclusions.

No AGENTS.md was found in the checkout or its applicable ancestor directories. The root README, `aes-manifest.yaml`, scope REUSE annotations, source-license texts and authored File-ID convention were reviewed. The scope is explicitly experimental. The requested PRM-0030 framework was read from its prepared prompt capture and applied as a case/evidence comparison; no REA engine was available or claimed. This review used Git, Python, independent CSV parsing, rational formulas, SciPy, OpenPyXL, Poppler and primary-source retrieval. Unrelated native/security pipelines were not reproduced.

The exact original 54-member replay ZIP was independently CRC checked and hashed: **15,956,607 bytes**, SHA-256 `06d150c885e9f95aa3235dc4ceaab09f34379bc27f8de66864c00e85af67f73e`. Its methods member matches `a99403241718c80db7a3e2f8e61cb6e85264d5528bdf3b1648f55805169a992a`. A different earlier local transfer was truncated; it was excluded, and the matching full package was used. This was a resolved local-transfer issue, not an alteration to the canonical audit evidence. The current standalone version-6 narrative is 20,008 bytes, SHA-256 `a766499d838a53b28024d2af0cc289cb48a483226d1d25b83cd2b764fbde7dab`; it was not confused with the older embedded narrative.

## Findings requiring action

### ISR-001 — Medium: a selected pair can bypass contradictory whole-contest changes

**CONFIRMED.** In [`analysis.py:363`](https://github.com/dlworrell/code-noodling/blob/3e9ae3bee1c657c91629b133f379a5b73e41781a/election-analysis/election_reporting/analysis.py#L363), a `new_ballots` annotation plus nonnegative deltas for only the selected two candidates authorizes a latest-batch likelihood. It does not check the whole choice set, a decrease in another candidate, a decrease in the ballot denominator, or compatibility/conservation of the complete transition. The analogous majority path at line 450 checks leader/aggregate deltas and can hide offsetting negative changes among other choices.

Using normal synthetic reviewed counting metadata, a current winner boundary, the same stage/version, and source final-ballot scenario 300:

| Case | Previous contest | Current contest | Expected latest votes | Actual latest votes |
|---|---|---|---:|---:|
| Negative third candidate | Ballots 150; A 80, B 60, C 10 | Ballots 160; A 90, B 65, C 5 | 0 | 15 selected-pair votes |
| Decreasing denominator | Ballots 150; A 80, B 60 | Ballots 145; A 85, B 60 | 0 | 5 selected-pair votes |
| Removed candidate | Ballots 150; A 80, B 60, C 10 | Ballots 155; A 90, B 65 | 0 | 15 selected-pair votes |
| Majority with negative third candidate | Same first case, majority event | Same first case | 0 | 10 total choice votes |

All four cases are marked scenario-eligible and receive numeric conditional probabilities. A `new_ballots` label cannot override an observed incompatible/revision transition. This is a counterexample to original #55 acceptance criterion 1, not a demand for new publisher cryptography or new empirical data.

**Minimum repair:** qualify one complete, actually adjacent transition before either likelihood path uses it: immutable snapshot/predecessor identity, stage/version, complete choice identity, nonnegative complete candidate changes, nondecreasing reported ballots and appropriate accounting. A contradictory annotation must visibly fail closed for latest evidence; quarantine the affected scenario where current transition/data validity is contradictory. Preserve a documented distinction between unverified/unknown deltas and verified new-ballot evidence. Add the counterexamples to both pair and majority regressions. Do not infer that a monotone transition proves new ballots.

### ISR-002 — Medium: deduplication can manufacture a new latest batch from repeated bytes

**CONFIRMED.** [`analysis.py:590`](https://github.com/dlworrell/code-noodling/blob/3e9ae3bee1c657c91629b133f379a5b73e41781a/election-analysis/election_reporting/analysis.py#L590) deduplicates the history, but `_pair_decision` and `_majority_decision` still subtract `history[-2]` from the separately retained latest snapshot.

Three dated files, same contest/stage/version: August 5 has A 50/B 30 (80 ballots); August 6 has A 60/B 40 (100 ballots); August 7 repeats August 6's bytes exactly. With an explicit `new_ballots` annotation, the August 7 record manufactures **20** latest pair votes. Its displayed history contains August 5 and August 6; the actual immediate transition August 6→7 is zero. Existing two-file identical-count tests do not cover this case.

**Minimum repair:** determine the actual immediate predecessor before display-history filtering/deduplication. A latest record with duplicate bytes or unchanged tally contributes no new batch. Stage-filtering or an omitted/incompatible intervening record must not silently change which transition an annotation describes. Keep the display/history and likelihood transition identities separately reviewable. Add a three-snapshot duplicate regression for both likelihood paths. This independently violates original #55 acceptance criterion 1.

### ISR-003 — Low: a user-guide result needs its archive/recount stage stated

**CONFIRMED.** [`docs/election-reporting.md:251`](https://github.com/dlworrell/code-noodling/blob/3e9ae3bee1c657c91629b133f379a5b73e41781a/election-analysis/docs/election-reporting.md#L251) describes the 2024 Measure 2 737-vote margin as a known result without identifying the original archive stage or the later 743-vote recount. The structured version records correctly preserve both. State both stages and margins together in this public guide. Also prefer “298 source representations” to “298 reported contests” in the historical summary where distinct canonical identities remain largely unreviewed. No unsupported numeric probability or calibration claim was found in the actual current README/primary report.

## Statistical assessment

### Event, likelihood and numerical accuracy

The repaired probability is a conditional selected-pair strict reversal, or a selected leader's gain/loss of strict majority. It is not the chance of any winner/advancer-set change. The as-of named-pair outcome remains stable in scoring, ties are not strict reversals, and raw RCV rankings do not adjudicate legal winners or stopping rules. Unsupported supermajorities, validation requirements, RCV, recount and litigation probabilities are withheld. The A/B/C/D issue is addressed by precise event labeling rather than an unsupported joint model.

For future pair votes `R`, current margin `M` and future challenger votes `X`, strict reversal is `2X − R > M`, hence `floor((M+R)/2)+1`; loss of strict majority includes equality and uses `ceil((M+R)/2)`. Independent direct-inequality/rational enumeration passed all **1,080** small cases, maximum absolute error **3.33e-16**. The repaired skewed `R=5001, alpha=1, beta=2` example is approximately **0.24985007**. The full-support log recurrence and 200,000-vote computation budget avoid the old unjustified normal-tail switch. Budget failures withhold a number.

The specified model subtracts a qualified latest batch from cumulative counts before applying powers capped at 300 history and 200 latest, with Beta(1,1) prior. This removes algebraic reuse when the input really is a compatible disjoint batch. The fractional powers are experimental tuning parameters, not measured independent sample sizes or a fitted sampling model. The latest component can still receive 40% of added weight despite less than 1% raw participation; the fixed LD42 sensitivity correctly makes that assumption visible. The two defects above concern admission of a batch to this construction, not the beta-binomial formula itself. The caps remain scientifically unvalidated.

Independent direct SciPy mass sums checked **1,479** King and **636** Orange County model/baseline values across all scenarios. Maximum absolute discrepancy was **7.82e-11**, below the stated numerical policy. These are arithmetic checks, not evidence of predictive calibration.

### Remaining pools and units

The live contract distinguishes missing pools, source contradictions, ties and official zero. Turnout-overrun does not prove zero outstanding. Remaining pools require dates, scope/categories and participation assumptions. A positive scenario is explicitly conditional, and low/base/high inputs are sensitivities, not confidence intervals.

King uses historical government turnout projections of 45%/82%/45%, not contemporaneous outstanding or accepted-ballot ledgers. The construction also assumes turnout-derived countable pools and future geographic/pair participation resembling counted ballots. The present source pages corroborate the historical projections and next-day 16:30 archive convention; that does not establish unchanged historical archive bytes. Final turnout was not backfilled into the prediction recipe.

Orange County supplies dated official **unprocessed estimates**, not a guaranteed countable contest-specific remaining pool. Visible forms reconcile 1,148,043→1,212,226 processed ballots, a **64,183** increase. Field 2 values 140,894/205,077 refer arithmetically to additions since the common 1,007,149 election-night baseline, despite the form's last-report wording. Queue estimates 275,420/212,152 change by −63,268; processed-plus-queue increases by 915. Challenged counts 9,462/9,512 have unknown overlap. Base does not silently add them. The 80% acceptance and separate-challenged cases explicitly impose different assumptions; future arrivals are excluded as a scenario condition, not asserted zero or bounded.

The final RLA inventory independently has **10,208 unique Container/Tabulator/Batch Name keys**, source-column sum **2,757,630** (578,577 in-person; 2,179,053 mail), and exact publisher-displayed SHA-256 `36b6c0079cc3882e601dc7ef7024d987973e3a17bceb9263dabbc433c6e78cff`. These sheet/card units cannot substitute for 1,417,397 counted voter ballots, remaining votes, or a dated candidate/correction ledger. The new reader correctly reports new-ballot-likelihood eligibility false. The unsent aggregate request targets the remaining useful fields without requiring personal voter data.

### Chronology, leakage and dependence

Prediction-phase code and ledger inspection, exact reruns and independent raw parsers support **computational phase isolation**: 32 King as-of files and four OC as-of files; zero reserved outcome reads in prediction. Scores subsequently read only the named outcome CSVs. OC also regenerates and checks the entire as-of recipe before scoring. The two first seals are preserved:

- King: `9be7e233195698250928284909e070e96f4cd1b4de2f519820af7de99af72426`.
- OC: `98b0437b03bb082dad87d3fa948c3f5fe70e4b0108ba2592432ca9d3ce3378eb`.

These are reconstructed-after-outcome experiments. Public final outcomes/identities were inspected before freezing; no blind preregistration is established. King archive labels/HTTP proxies and OC face/producer clocks do not prove historical content availability or absence of later amendment. The amended November 8 OC form is correctly not backdated; the common November 9 19:05 Pacific reconstruction time is a declared convention, not authenticated publication. No result should be relabeled as a forecast published in 2023–2025 or 2024.

Whole related elections, contests, repeated snapshots and sensitivities do not cross into training. Training groups are empty and no fitted weight/cap is claimed. This is a legitimate no-fitting design, but “held out” here describes the computational evaluation boundary, not blindness to public outcomes. King and OC have different pool evidence/assumptions and must not be pooled as interchangeable replications.

Independent raw CSV parsing verified all **493 King** as-of selected-pair counts/outcomes; separate OC final CSV and precinct TSV aggregation verified all **159 OC** scenario outcomes. Base King contains **147 observations, 21 contests, three elections, one county**: 98/11/38 rows and 10/1/10 contests in 2023/2024/2025. Three contests reverse at least once; seven reversing rows are not seven independent events. OC contains **53 base contests, two reversals, one county/election**. Its 159 sensitivity rows are not 159 independent observations.

King's prescribed equal-election, then equal-contest, then equal-snapshot weights reproduce exactly; its one 2024 contest receives one third of the total election weight. The naive row-mean Brier is 0.04485327, versus the prescribed 0.03755657. This difference illustrates estimand/weight sensitivity, not a scoring error. OC uses equal contest weights. I independently recomputed every available base-method Brier, clipped log loss and weighted calibration-bin value: maximum score difference **1.12e-16**. Calibration plots/bins are descriptive; neither within-county contest independence nor enough replicated county/election clusters is established for a defensible generalization interval. The experiments correctly leave intervals and predictive-interval coverage unavailable. No predictive margin/remaining-pool interval was supplied to test.

| Experiment / base construction | Brier | Log loss |
|---|---:|---:|
| King frozen model | 0.03755657 | 0.21044475 |
| King cumulative-share | 0.03837115 | 0.95873236 |
| OC frozen model | 0.03368266 | 0.09501048 |
| OC cumulative-share | 0.03777127 | 0.13645187 |
| OC reported increment only, conditional | 0.03869495 | 0.13811865 |
| OC disjoint increment, conditional | 0.02887058 | 0.08424815 |

The remaining fixed-half, untempered and margin-rule baselines also reproduce. Missing verified-latest/official-countable-contest baselines remain explicit NOT RUN. The conditional OC increment comparisons do not establish correction-free batches and must not be selected as a new model because their observed score is lower. Importantly, these losses compare actual outcomes with predictions under hypothetical pools that were not shown to have been realized. They cannot establish calibration conditional on those pools or model superiority.

### Descriptive repairs and reproducibility

The actual primary has **298 source representations** (147 state, 151 King), **187 decisions and zero numeric forecasts**. Independently regenerated race payloads and social text equal the committed outputs. LD42 retains its four-vote margin and recount-completed authority; mixed source clocks are visible. Snoqualmie has one complete controlling representation, with the King slice separate. South Beach and vintage/denominator conflicts are quarantined, and unestimated values are not zeros.

The stage-aware 2024 replay remains exactly **204/73/262** records and has zero probabilities. Its three normalized fixture files are byte-identical to their members in the exact original ZIP. Independent OpenPyXL reading of all six original-hash Maine exports reproduces all 108 anonymous patterns, **402,936 = 391,596 continuing + 11,340 exhausted**, finalists **197,151/194,445**, and exhaustion **10,673 skipped-rank + 388 no continuing choice + 279 overvote**. This verifies allocation to supplied official finalists; it does not adjudicate generic write-ins or independently recreate every elimination round.

Full-source Alaska reconstruction preserves all 73 frozen contest tallies and 31 outputs, with independent source-key/mode aggregation confirming 130,516 rows, 2,788 zero-filled repetitions and four district early-vote rows mislabeled Election Day. Thirty borough/census-area outputs remain approximate precinct subsets with excluded modes, five split precincts and 2023-geometry/2020-population provenance. Summed mode contributions are selections across contests, not distinct voter ballots. Official round/recount records preserve all nine relevant RCV contests, the unresolved original HD28 record, the later HD28 recount selection, Measure 2's 737/743 versions and incomplete recount partition. Indexed official text is not misrepresented as independently retrieved binary PDF evidence. All final RCV rounds were not independently recomputed.

The full current-runtime suite independently passes **173/173**. Exact OC source `09d825d81083049903bd544e550adacd915a2c4a` with Python 3.12.14, pypdf 6.10.0 and reuse 6.2.0 reproduces all 159 predictions and complete score/exclusion semantics after only computation-clock/new-score-seal normalization. Actual changed-source and wrong-runtime controls reject before prediction. Modern pypdf 6.21.2 independently regenerates the original qualification report byte-for-byte; this is kept separate from original scoring-runtime reproduction. King reproduces all 493 predictions and complete score/exclusion semantics. The original source/seals are untouched.

## Original child acceptance matrix

This table evaluates original criteria, not the number of checked boxes or later increasingly strong acquisition requests. The companion JSON contains each original acceptance text and criterion-level disposition. Common documentation/provenance/evidence-link criteria were checked against the cited ADRs, current authoritative issue ledgers and retained hashes. The historical integration PRs are #60, #61/#63, #64, #65 and #69; this review does not independently certify every governance/signature claim in their narratives.

| Child | Original criterion focus | Evidence and disposition at reviewed snapshot |
|---|---|---|
| EA-01 #46 | Completed count, deadlines, separate stages, consistent outputs | **PASS.** Actual LD42 null probability/four-vote result, explicit dated status, past-deadline/unknown and stage tests; regenerated JSON/social agree. PR #60/ADR-003. |
| EA-02 #47 | Mixed vintages, conflicts, repeated retrieval, missing Maine blocks | **PASS.** 147/151 source representations replace the old guessed 147/97 merge; unknown county publication clocks remain visible. Assessor/Skykomish conflicts and missing-office/block tests withhold forecasts. ADR-004/005, PR #63. |
| EA-03 #48 | One Snoqualmie controller, collision/membership safety | **PASS.** 161/45 full controller; county slice separate; reviewed crosswalk and collision/missing-membership tests. Unreviewed identities remain partial. PR #61/63. |
| EA-04 #49 | Zero/unknown/contradiction/tie distinction | **PASS.** Actual South Beach contradiction has no probability; official-zero/tie and denominator regressions pass. No turnout-overrun zero pool. PR #61. |
| EA-05 #50 | RCV scope, raw majority, complete six-export accounting | **PASS within supplied-finalist scope.** Independent six-export oracle and raw-ranking/eligibility/blank/overvote/missing-file tests. No independent eligibility engine claimed. PR #61. |
| EA-06 #51 | Official final/recount versions and explicit unresolved coverage | **PASS for structured implementation.** Nine-contest ledger, 737/743 lineage and Maine accounting; original HD28 and partial recount remain explicit. ISR-003 requests public-guide stage clarification. PR #63. |
| EA-07 #52 | Alaska exclusions, approximate splits, preserved markers | **PASS.** Full-source/statewide comparison and independent mode aggregation; approximate borough outputs remain noncontrolling. PR #63. |
| EA-08 #53 | Dated pools/scenarios, zero/unknown and participation | **PASS for conditional contract.** Assessor sensitivity, scope/category/time tests, original replay without invented probabilities; OC queue evidence retains conditions. Empirical pool validity unestablished. PR #63/65. |
| EA-09 #54 | Selected events, ties, rules and all-format labels | **PASS.** Explicit pair/majority events, A/B/C/D counterexamples, unsupported-rule gates and no legal RCV claims. PR #63. |
| EA-10 #55 | No manufactured new ballots; disjoint weights; sensitivity; grouping/no tuning | **BLOCKED.** Disjoint arithmetic/diagnostics, LD42 sensitivity, fixed grouping and conditional comparisons pass, but original criterion 1 fails ISR-001/002. Empirical cap validation separately remains unvalidated. |
| EA-11 #56 | Evidence dimensions and unvalidated rubric | **PASS.** Actual outputs withhold reassuring numeric risk, expose dimensions and experimental status; no correctness mapping. The cap-50 rubric is still an arbitrary heuristic. PR #63. |
| EA-12 #57 | 1,080 checks, skewed 5,001, numerical policy | **PASS.** Independent rational and 2,115 direct SciPy checks, edge/budget checks. Numerical results are not empirical validity. PR #61. |
| EA-13 #58 | Frozen chronology/groups, reproducible scores, missing inputs | **PARTIAL; permitted documented-unvalidated resolution available.** Phase isolation, frozen no-fitting grouping and conditional score/exclusion reproduction pass; demonstrably historical input versions, fully matched baselines and calibration/generalization remain unavailable. See original failure clause and closure wording below. |
| EA-14 #59 | Scientific gate, descriptive retention, full regression matrix | **PARTIAL.** Existing named-failure matrix, 539-record replay, Maine/Alaska lineage and real publication pass. The newly observed unsupported batch transitions require matrix extension with the #55 fix; no document/provenance pass can accept them. PR #63 and later extensions. |

## Closure recommendations and minimum remaining work

**#55 — remain open at this snapshot.** Fix ISR-001/002 with complete actual-predecessor qualification shared by pair/majority paths, visible fail-closed behavior and the five preserved counterexamples. Rerun the appropriate existing/new tests, real publication and the relevant immutable experiment reproduction checks. Obtain independent delta review of exact changed bytes. Current #55 criterion 1 must not stay checked as satisfied for this snapshot. The current evidence permits an explicitly experimental fixed-parameter construction and conditional baseline comparisons; it does not validate the 300/200 caps. Missing empirical evidence can carry a clearly unvalidated disposition into #58 rather than being concealed.

**#58 — resolution as documented unvalidated work is faithful to its original failure clause, once the record says so plainly.** It is not faithful to label the current experiment a completed leakage-free historical forecast validation. Original acceptance criteria 1 and 3 remain only partially met for that scientific goal: computational reserved-outcome isolation and conditional losses reproduce, but historical version/availability and a complete evidence-matched validation set do not. Retain those limits in the checklist text or an explicit disposition table. Suggested truthful resolution wording:

> Resolved as a documented unvalidated evaluation under the issue's original insufficient-input failure outcome. Frozen conditional experiments, source/score/exclusion ledgers and independent review are recorded. Computational outcome isolation was verified; historical publication/version identity, verified latest-batch/countable-pool baselines, calibration and generalization remain unestablished. No operational forecast or historical accuracy claim is accepted. The 2024 final-source replay remains descriptive.

Further acquisition of correction-separated aggregate batches, historical versions, countable/category/geographic pools and independent county/election replication is necessary before scientific promotion or weight selection. It is not an unstated requirement for a publisher digital signature or for perpetual work-item openness after an expressly permitted unvalidated result. Independent replication and better source lineage are appropriate future research requirements; their absence must remain visible if the task is closed under the failure branch.

**#45 — remain open at this snapshot.** The independent-review requirement is now substantively addressed by this separate AI report, subject to maintainer acceptance of that exact scope and attachment in the audit record. The parent cannot yet truthfully assert all mandatory child implementation criteria are met because of ISR-001/002. Correct them, independently review the exact delta, clarify ISR-003, link the resulting evidence and give #55/#58/#59 precise scoped dispositions. After those steps, administrative audit closure can be compatible with retaining the model/forecast-validation findings explicitly unvalidated; it must not imply calibrated probabilities, broad operational eligibility or complete source authentication.

## Reproduction record and limits

The companion `independent-statistical-review-record.json` records exact commands, exit codes, output hashes, original criterion texts, source identities and residual risks. The reusable independent oracle scripts are `independent_checks.py`, `independent_rcv_checks.py` and `independent_alaska_checks.py`; their JSON outputs preserve the calculations and five counterexamples. The new reproduction seals and full runtime outputs are review evidence, not replacements for first seals. The implementation checkout was clean at the end of review.

Primary-source spot checks corroborated the three turnout projections, the King's archive deadline, OC's 64,183-ballot release and today's displayed manifest hash. They do not establish historical bytes or physical ballot correctness. The complete SOV PDF and every original Alaska binary were not independently authenticated/recounted. The prospective `validation.evaluate` scaffold has no real eligible series in this evidence; its generic bootstrap is not new empirical evidence. No claim of test/formatting equivalence, complete software correctness, human statistical certification or operational predictive validity is made.
