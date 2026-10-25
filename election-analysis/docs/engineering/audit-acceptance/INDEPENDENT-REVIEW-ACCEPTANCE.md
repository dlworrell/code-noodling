<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:5917fc02-c8ec-4f3f-a5b2-7737982f40aa
-->

# Acceptance of the independent statistical review

Date: October 25, 2026 (America/Los_Angeles).
This is the implementing agent's acceptance/packaging record. The two linked
review reports and their conclusions were authored by a separate agentic AI,
with a fresh review context, after the user explicitly authorized delegation.
They are retained byte-for-byte. This is not human expert certification.
The authoritative merge, CI and issue dispositions are recorded in
[audit #45](https://github.com/dlworrell/code-noodling/issues/45).

## Review and repair

The [initial independent review](../INDEPENDENT-STATISTICAL-REVIEW-20261025.md)
evaluated commit `3e9ae3bee1c657c91629b133f379a5b73e41781a`, tree
`2ccd8e0a0a08d610e61e0e51938794d57dddecd8`. It found two mandatory batch
qualification defects and one public wording issue. Its original BLOCKED
disposition and counterexamples remain preserved.

The [independent repair review](../INDEPENDENT-STATISTICAL-DELTA-REVIEW-20261025.md)
accepts the exact thirteen source/publication files in the
[candidate manifest](../independent-statistical-repair-manifest.json), whose
SHA-256 is `b6c6d0be506c4c187fdb431cfd1e303a6365381580bf96834832eef199b63852`.
Both the manifest and review were completed before repository acceptance.
Remote Git blobs must match that inventory; later packaging files are identified
separately and do not change the reviewed implementation.
Two exact-file Git whitespace attributes preserve the reviewer's intentional
CommonMark hard breaks without changing either report hash. EOF/tab checks are
retained; other authored files keep their normal trailing-blank check.

| Finding | Accepted repair | Independent evidence |
|---|---|---|
| ISR-001: selected/aggregate deltas hide another candidate's correction or a falling denominator | One whole-contest transition check shared by pair and majority paths; contradictory annotations quarantine scenarios | Negative candidate/denominator, changed identity, accounting/counter and positive-weight controls |
| ISR-002: duplicate third snapshot reuses an older increase | Qualify the actual source predecessor before display-history filtering/deduplication | Duplicate, omitted and incompatible immediate-predecessor cases |
| ISR-003: ambiguous archive margin and contest count | State Measure 2 archive 737/recount 743 separately; label overlapping source representations | All regenerated human formats and unchanged core primary JSON |

[ADR-009](../ADR-009-consecutive-whole-contest-batches.md) records the design.
The final separate review passes **28 independent transition cases** and
**179/179 tests**, verifies 55 immutable scientific/input artifacts, and confirms
the primary retains 298 source representations, 187 decisions and zero numeric
forecasts. Removing only the computation clock and three newly added evidence
fields leaves its entire prior JSON payload unchanged. The initial review also
retains rational/SciPy checks, independent raw-parser/grouped scoring, full Maine
and Alaska source checks and exact historical-source/runtime reproduction.

## Accepted issue dispositions

| Issue | Accepted disposition |
|---|---|
| EA-10 #55 | Implementation PASS within the specified experimental fixed-cap construction. Revision/duplicate guards and disjoint arithmetic are accepted; empirical cap validity remains unvalidated. |
| EA-13 #58 | PARTIAL for the intended historical forecast backtest, resolved as the original explicitly permitted documented-unvalidated outcome. No missing historical input or calibration criterion is marked scientifically proven. |
| EA-14 #59 | Gate/regression implementation PASS, extended with the independent counterexamples. `validated_forecast_eligible` remains false. |
| Parent #45 | Scoped remediation and independent AI re-audit accepted after exact repair/review publication and child dispositions are linked. Scientific findings remain explicitly unvalidated. |

EA-13 originally permits: “If historical inputs are insufficient, publish a
documented unvalidated result and acquisition gaps.” Its original incomplete
historical-availability and complete-validation criteria retain that scientific
disposition. Acceptance does not invent a publisher digital-signature requirement
or reinterpret conditional comparisons as verified historical forecasts.

The remaining scientific gaps are correction-separated candidate batches,
then-available source versions, countable/category/geographic remaining pools,
independent county/election replication, empirical weight selection and
calibration. King turnout assumptions and OC unprocessed estimates remain
different conditional constructions. No operational model promotion is accepted.
The unsent [data request](DATA-REQUEST.md) and [investigation](INVESTIGATION.md)
remain useful requirements for future research, without being silently treated
as completed acquisition work.

## Retained review work products

- [Initial machine record](../independent-statistical-review-record.json) and
  [repair machine record](../independent-statistical-delta-review-record.json)
  contain original criterion texts, exact commands, hashes and dispositions.
- [Original independent checks](../independent-check-results.json),
  [Maine checks](../independent-rcv-results.json),
  [Alaska checks](../independent-alaska-results.json), and
  [repair checks](../independent-delta-check-results.json) preserve actual results.
- [Evidence bundle](../independent-statistical-review-evidence.json) packages
  captured original issues, test/command logs, test records and the exact
  candidate manifest as UTF-8 text with original byte sizes and SHA-256 hashes.
  The implementing agent packaged it without changing member bytes. Raw web
  retrieval dumps and redundant full prediction/score reproductions are omitted;
  the original frozen ledgers/seals and exact reproduction records remain linked.
- The unchanged reviewer scripts are in scope `ci/`:
  [original checks](../../../ci/independent_checks.py),
  [Maine oracle](../../../ci/independent_rcv_checks.py),
  [Alaska oracle](../../../ci/independent_alaska_checks.py), and
  [repair checks](../../../ci/independent_delta_checks.py).

The original oracle is specific to a clean checkout of the reviewed base commit;
its output's reviewed-commit field names that snapshot. The repair oracle takes
separate `--repo`, clean original `--base`, `--manifest`, `--output` and `--edt`
arguments and binds the thirteen changed files. Original-package identity,
Python/library versions, actual commands and expected first-seal identities are
in the machine records. Run outputs belong in a separate directory. Historical
protocols continue to require their original frozen source/runtime; no current
dependency downgrade or rewritten first seal is part of acceptance.
