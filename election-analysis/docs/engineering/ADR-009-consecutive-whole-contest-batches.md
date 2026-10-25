<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:68426fdb-d566-458a-af43-1ce152955f3c
-->

# ADR-009: Qualify the complete consecutive source transition before batch use

Status: Accepted for the experimental scenario implementation.

The independent statistical review of commit
`3e9ae3bee1c657c91629b133f379a5b73e41781a` found that the selected-pair and
majority paths checked only their selected or aggregate vote deltas. A decrease
for another candidate, a decreasing ballot denominator or a removed candidate
could therefore enter a likelihood annotated as new ballots. De-duplicating
history before selecting the predecessor also allowed an unchanged third
snapshot to reuse an older increase. These violate ADR-004 and EA-10 #55.

## Decision

Qualify one transition per contest against the immediately preceding **source**
snapshot before any history filtering or de-duplication. Do not skip a missing
contest or an incompatible stage to find a convenient older predecessor. Bind
the record to both source hashes and use it in both pair and majority paths.

A `new_ballots` annotation is necessary but insufficient. Require complete
coverage, unchanged contest and candidate identities, compatible known stages,
increasing dated source/count evidence, nondecreasing candidate tallies and both
ballot denominators, consistent accounting at both endpoints and a positive
valid-vote increment. Known overvote/undervote counters must not decrease or
change availability. The sum of all observed vote/counter increments cannot
exceed the reported ballot increment. Unknown counters remain unknown; a
nonnegative residual is not evidence of its composition.

Identical latest bytes supply no new likelihood, even when the annotation says
new ballots. Missing or incompatible predecessors also supply no batch. A
contradictory annotated transition (negative counts, changed candidate inventory
or inconsistent accounting) additionally withholds the conditional scenario
while preserving descriptive tallies and the explicit rejection reasons.
Unclassified changes and explicitly classified corrections continue to use no
separate latest-batch likelihood; this does not establish empirical calibration.

The decision and race evidence expose the qualification record. Existing model
caps, probability targets and numerical policy do not change. The analysis
schema's evidence additions are additive. Historical protocols and first seals
are not rewritten: reproduce them from their original frozen source/runtime.

## Verification

Retain the independent review's original failing cases. Add regressions for
third-candidate corrections, decreasing denominators, candidate removal/addition,
known counter revisions, duplicate third snapshots, missing/incompatible
immediate predecessors and valid all-candidate increases. Exercise pair and
majority paths and verify the rejected increments contribute zero latest votes.
The separate reviewer must assess the repair before the audit is closed.
