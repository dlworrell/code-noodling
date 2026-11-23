<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:8c770721-500e-4668-b558-1220a59e68a2
-->

# Lean proofs: GRH primality link

Lean 4 proofs, using Mathlib, about the GRH-conditional primality test in
`crypto_prime.c`. See [`CRYPTO-PRIMALITY-GRH.md`](../../docs/engineering/CRYPTO-PRIMALITY-GRH.md)
for the mathematics.

## What is proved

| Theorem | Statement | Conditional? |
|---|---|---|
| `cBound_numerator_lt` | The C bound's numerator is below 2^61 for every supported size (bit length ≤ 65,536), so it cannot overflow `uint64_t`. | No |
| `two_log_sq_le_cBound` | The C bound is at least 2 (ln N)², the bound in Bach's theorem. | No |
| `prime_millerAllBases` | Every prime passes every base the C loop tries, so the test never calls a prime composite. | No |
| `millerAllBases_prime` | An odd N ≥ 3 that passes every base up to min(bound, N − 2) is prime. | Yes: assumes `BachWitnessBound` |
| `millerAllBases_iff_prime` | The two together. | Yes: assumes `BachWitnessBound` |

`BachWitnessBound` states Bach's theorem: every odd composite N has a strong
witness below 2 (ln N)². It is a **hypothesis of the theorems**, not an
axiom. CI checks that every theorem uses only Lean's three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`) and no `sorry`.

## What is not proved

- **Bach's theorem itself**, which needs the extended Riemann hypothesis and
  substantial analytic number theory. Mathlib does not contain it.
- **That the C code implements this model.** `cBound` is the C expression
  transcribed by hand, and `StrongProbablePrime` describes the check the C
  loop performs. Agreement is tested (`tests/test_prime.c`, `tests/test_prime.py`
  check the same bound values), not proved; there is no Lean model of C here.

## Checking

CI runs `.github/workflows/crypto_lean.yml`:

- it installs a pinned elan release, verified by SHA-256;
- `lake exe cache get` fetches Mathlib `v4.15.0` build outputs;
- `lake build` checks the proofs;
- `lake env lean CheckAxioms.lean` runs the axiom audit.

Locally, with elan installed, run the same three commands from this directory.
