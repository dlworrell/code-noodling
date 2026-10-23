<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:114fcf55-39e1-4e17-b930-50b9c4913252
-->

# Crypto pipeline evidence

These records belong to the `crypto_pipeline` governed project scope
(Catylist ADR-0004). Until 2026-10-16 they lived in the repository-level
`docs/engineering/evidence/` directory. The crypto engineering notes moved
alongside them from `docs/engineering/` to `crypto_pipeline/docs/engineering/`.
Git history follows each file across the move (`git log --follow <file>`).

Each record describes the repository as it was when it was captured. Paths
inside a record, such as `proof/run_proofs.py` or
`docs/engineering/AES-SEC-001-waivers.md`, are the paths that existed at
capture time. The records are not rewritten to match later layouts.

| Former path | Current path |
|---|---|
| `proof/` (until 2026-10-14) | `crypto_pipeline/proof/` |
| `docs/engineering/evidence/<record>` | `crypto_pipeline/docs/engineering/evidence/<record>` |
| `docs/engineering/CRYPTO-*.md` and the dated crypto review notes | `crypto_pipeline/docs/engineering/` |
| `crypto_pipeline/test_*.c`, `crypto_pipeline/test_*.py` | `crypto_pipeline/tests/` |

## Records

| Record | Captured | Subject |
|---|---|---|
| `montgomery-local-20261013.json` | 2026-10-13 | Local constant-flow candidate run |
| `montgomery-gcc13-O2.asm` | 2026-10-13 | GCC 13 `-O2` assembly of the candidate |
| `montgomery-diagnosis-20261013.json` | 2026-10-13 | Timing-signal diagnosis with controls |
| `montgomery-formal-20261013.json` | 2026-10-13 | Scoped source checks and arithmetic lemmas |
| `montgomery-formal-phase2-20261013.json` | 2026-10-13 | Source-bound local refinements |
| `montgomery-formal-phase3-20261013.json` | 2026-10-13 | Loop composition and context field proofs |
| `revision-review-20261014.json` | 2026-10-14 | Review of the 2026-10-13 submitted revisions |
| `timing-runner-update-20261014.json` | 2026-10-14 | Timing runner and setup integration |
| `carry-refinement-20261014.json` | 2026-10-14 | REDC suffix carry refinement at 16 limbs |
