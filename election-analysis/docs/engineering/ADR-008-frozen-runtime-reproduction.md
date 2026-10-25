<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:b8cfacd7-cd9e-4d64-96f0-6e225b9b39ee
-->

# ADR-008: Reproduce a frozen experiment after dependency updates

Status: Proposed for review; experimental offline tooling
Date: 2026-10-25 (Pacific)
Issues: EA-10 #55, EA-13 #58; parent #45
Base: `7018e9d229a316687365fd1192d161ff82e9e283`

## Trigger and decision

Merged Dependabot PR #67 changed the current CI pin from pypdf 6.10.0 to 6.21.2.
OC v1 correctly refuses that changed dependency file: it is one of its 37 frozen
model/dependency hashes. The original reproduction instructions did not make
clear that the old source and runtime must be selected together. Do not rewrite
the original protocol/seal, ignore its hash check or revert the current upgrade.

Add `ci/reproduce_official_experiment.py`, a separate runner outside the frozen
package. It reads exact source commit `09d825d81083049903bd544e550adacd915a2c4a`
from local Git objects, verifies all 37 hashes and the complete Python package
inventory, extracts only regular scoped files into a temporary directory and
checks installed versions against the original pins. It does not fetch Git
objects, install packages or alter repository settings. It refuses missing or
changed runtime pins, unlisted package code, path traversal/links/duplicate
archive members and a nonempty output directory. Source refs are constrained to
40 hex characters before any Git invocation. Python runs in isolated mode.

An existing matching interpreter must be selected; a shallow checkout missing
the original commit fails with the Git diagnostic rather than silently using
current code. The executed target is our new offline implementation from PR #65,
not an original bundled audit/replay script. User-authorized new-implementation
execution remains distinct from PRM-0068's static residual analysis.

## Actual verification

The isolated historical runtime is Python 3.12.14, pypdf 6.10.0, reuse 6.2.0.
Its predict phase produces 159 rows with no reserved outcome reads. A new seal
is written before scoring. Predictions and the complete score/exclusion ledger
match the first run after normalizing only actual computation clocks and the
reproduction's new prediction hash in the score. Source/count clocks, input
hashes, event identity, probabilities and versions are never normalized away.
The first prediction seal remains
`98b0437b03bb082dad87d3fa948c3f5fe70e4b0108ba2592432ca9d3ce3378eb`.

Two actual negative controls reject a changed-source dependency manifest and
the modern interpreter's mismatching installed runtime. Separately, the current
pypdf 6.21.2 repeats the independent PDF/CSV/TSV qualification byte-for-byte
against the original report. That establishes bounded extraction/accounting
compatibility; it is not silently substituted as the original scoring runtime.
The complete current-runtime suite passes 173 tests, including nine new source,
archive, runtime, normalization and malformed-manifest checks.

Full regenerated predictions, seal, scores, logs and runtime verification are
in `oc-runtime-reproduction-v1.zip`. Exact hashes and negative-control results
are in `evidence-continuation-validation.json`. The archive stores new execution
clocks; it does not overwrite or relabel the original first run.

## New publisher-hash-verified data

Acquired the original official OC RLA manifest, SHA-256
`36b6c0079cc3882e601dc7ef7024d987973e3a17bceb9263dabbc433c6e78cff`, identical
to the publisher's displayed hash. The per-file Git attribute preserves original
CRLF bytes and accepts CR as an end-of-line marker while retaining ordinary
blank/tab checks; no source line endings are rewritten. It has 10,208 unique
composite identities:
Container, Tabulator, Batch Name. Batch Name alone is not globally unique.
The source count column sums to 2,757,630: in-person 578,577, mail 2,179,053.
The associated official RLA report calls this a ballot-sheet manifest and names
ballot cards audited; do not equate its row sum to 1,417,397 unique counted voter
ballots, remaining ballots or contest votes. Preserve the publisher's label and
this unreconciled unit distinction.

The file has no historical clock, candidate counts or correction partition.
The bounded reader therefore returns `new_ballot_likelihood_eligible: false`.
It narrows the request to concrete composite batch IDs but cannot supply the
November 8–9 new-ballot likelihood. No new forecast is computed from it. Source
record and raw CSV remain separate from the already-frozen eight-file archive.

The SF candidate archive returns HTTP 403 after one reload. Its raw records
were not acquired. Policy documents about CVR releases do not substitute for
actual files. These are scoped observations, not an exhaustive nonexistence
claim. An unsent aggregate request and an independent-review brief are prepared
in the existing acceptance directory; no other person was contacted.

## Reproduction and limits

From a checkout containing the runner and original Git commit:

```bash
python election-analysis/ci/reproduce_official_experiment.py --repo . --python /path/to/historical-python --output /tmp/oc-reproduction
```

The selected interpreter must have the original pinned packages. Use an isolated
historical environment for this fixed-input reproduction; current CI keeps its
modern dependency pin. The runner itself installs nothing. The evidence record
reports actual package and Python versions; it does not claim a completely
pinned operating system or transitive-tool environment.

This repair addresses reproducibility after dependency drift. It does not
validate the caps, authenticate historical archive versions, establish accepted
contest-specific remaining votes or supply independent scientific review.
Original acceptance conditions must be distinguished from later stronger data
requests when the independent reviewer assesses closure. No issue is closed by
reinterpreting unsupported data as proof.
