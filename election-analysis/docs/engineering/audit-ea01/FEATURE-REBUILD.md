<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:06937853-84b8-4d6a-933a-048ed16ba529
-->

# EA-01 feature repair and verification record

Date: 2026-10-24 (America/Los_Angeles).
Disposition: implemented on a review branch; merge and audit acceptance pending.
Authoritative issue/closure ledger: [audit #45](https://github.com/dlworrell/code-noodling/issues/45).
Work item: [EA-01 #46](https://github.com/dlworrell/code-noodling/issues/46).

## Applied prompt and bounded scope

Selected Dropbox `/prompt/PRM-0029.txt`, revision `65e8ddd0d60b201b90224`,
modified `2026-10-24T03:40:52Z`: focused implementation plus meaningful comparison.
The directory index and earlier PRM-0001–0016 were reviewed; those older prompts
concern repository reorganization/governance or unrelated work. PRM-0068 is static
continuation only, so does not fit an implementation task as well.

Filled inputs:

- ORIGINAL: `dlworrell/code-noodling@99c03904a7258994bcbc25c2e73ede722d00c461`,
  `election-analysis/`; this is also the audited revision.
- FEATURE: lifecycle/stage prerequisites for ordinary outstanding-ballot forecasts.
- DESTINATION: the same Python governed scope, branch `fix/election-ea01-result-stages`.
- OUTPUT: this directory; machine evidence is `../ea01-validation-20261024.json`.
- COMPARISON_SCOPE: preserved audited analysis JSON and original input files;
  explicit synthetic counting cases and local offline tests of the changed code.
  The old application was not run as a new runtime oracle. No election endpoint is
  contacted during tests/builds.

No REA engine/tool is exposed in this session. Actual alternatives: static source
inspection, Git, standard-library Python/unittest, pinned EDT and REUSE. No REA
call, formal proof, independent re-audit or forecast-calibration result is claimed.

## Recovered behavior and repair

CONFIRMED (F1 / EA-01): the original `analysis.py` derives remaining votes from a
source-wide turnout target without requiring active counting evidence. Its
`certification_date` is configuration metadata, not an observed lifecycle. The
audited JSON includes LD42's four-vote boundary and an active modeled reversal.
This arithmetic is not the specification for a completed count.

The replacement specification is [ADR-003](../ADR-003-result-lifecycle-and-stages.md),
which amends ADR-001/002 and remains Proposed for review. Configuration schema 2
and analysis JSON schema 3 separate election/contest lifecycle, tabulation stage,
authority evidence, exact input hash, round/version, target and information time.
Legacy schema 1 becomes descriptive/unknown. Historical, completed, unknown and
unsupported stages withhold the calculation and preserve margins. Histories compare
only the same lifecycle/stage/round/version; original versions remain in lineage.

The actual Washington primary is historical. The election-level certification
announcement and the official completed-recount table are cited in configuration
and published output. LD42's summary stays a plurality-count source; the separate
recount-completed lifecycle does not relabel it as a downloaded recount artifact.
Other legacy contest versions remain unknown. Nothing changes the original raw,
normalized or geographic inputs. The original publication remains at the audited
Git revision, and the new social summary explicitly supersedes it.

## Comparison matrix

| Case | Expected behavior and oracle | Result |
| --- | --- | --- |
| Existing synthetic counting arithmetic | Original independently asserted numerical expectations, now with explicit synthetic counting evidence | MATCH; existing arithmetic assertions retained |
| LD42 completed count/recount | Official table: 13,125 and 13,121, unchanged four-vote margin; no prospective count scenario | MATCH; four-vote margin retained, probability null |
| Legacy config / passed deadline | No source confirms completion or current counting | MATCH; lifecycle unknown, no probability |
| Awaiting, final, certified, recount pending/completed, unknown | Lifecycle contract: only evidenced counting can run the model | MATCH; tests assert the tail function is never called |
| Target or information time missing; future status evidence | No supported information-time claim | MATCH; calculation withheld |
| Raw rankings / RCV rounds / final rounds / recounts / changed versions | Distinct phases cannot form new-ballot batches | MATCH; excluded history visible, no batch field |
| Maine CD2 raw 50.23% share | Audit counterexample: raw share is not a final winner/stopping rule | MATCH within EA-01 scope; raw descriptive share retained, forecast withheld |
| Hash mismatch or invalid metadata | Bound identity and actionable refusal | MATCH; errors surfaced |
| JSON, Markdown, HTML, Facebook | Historical mode, explicit target, completed lifecycle and unestimated probability agree | MATCH |

These are regression/specification matches, not equivalence to an election authority
or validation of all 2024 ballot interpretation. The official recount table and
certification announcement were read during this task; the linked scanned
certificate supplies lineage only, not an invented event timestamp. The evidence
observation time is explicit and is never backdated into an election-night forecast.

## Commands and results

EDT is pinned to the repository workflow revision
`bca66c1e2d2a9bf9c05f9429e618299216a17f27`.
From the repository root:

```bash
export PYTHONPATH="election-analysis:/path/to/pinned/engineering-docs-toolkit"
python3 -m unittest discover -s election-analysis/tests -p test_election_reporting.py -v
python3 -m unittest discover -s election-analysis/tests -q
EDT_DIR=/path/to/pinned/engineering-docs-toolkit election-analysis/ci/reports.sh generate
PATH="$HOME/.local/bin:$PATH" python3 election-analysis/ci/provenance.py all --output /tmp/ea01-provenance
git diff --check
```

- Focused reporting suite: 24 tests passed, including 10 new EA-01 regression tests.
- Full election suite: 119 tests passed. Its test-result harness deliberately runs
  a failing two-test child fixture; that expected child output is not a failed
  outer test suite. No tests were skipped in this checkout.
- Full generation: 244 Washington races, 166 decision boundaries, zero numerical
  probabilities. LD42 retains four votes and recount-completed evidence. The other
  three election directories contain no snapshots and were explicitly skipped.
- EDT quality marks the generated document publication-ready. That is a formatting
  result, not scientific certification.
- Provenance/File-ID/REUSE details and exact input/output hashes are in the adjacent
  machine evidence. SHA-256 establishes file identity, not authenticity/calibration.

## Integration and residual work

Changed integration points: `model.py`, `config.py`, new `stages.py`,
`analysis.py`, `publish.py`, package version 0.4.0, focused reporting tests,
the primary configuration, user/design documentation and regenerated reports.
The watcher and beta-binomial arithmetic are unchanged. Corrections reference
EA-01 in code comments and cite the governing design or official source.

EA-02 through EA-14 remain open. In particular, mixed source vintages, canonical
contest identity, South Beach's contradictory placeholder, complete RCV/CVR
interpretation, geography, remaining-pool uncertainty and held-out calibration
are not repaired by this lifecycle gate. They are not certified away by passing
tests or by setting a known election-level lifecycle.

The next dependent issue is EA-02 (#47), while independent EA-03 (#48) and the
numerical oracle EA-12 (#57) retain their original prerequisites. Do not close
the parent audit or promote the model on the strength of this change.
