<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:8c770721-500e-4668-b558-1220a59e68a2
-->

# Lean proofs: GRH primality link and zeta-zeros Turing step

Lean 4 proofs, using Mathlib, about two parts of this scope:

- `CryptoPipelineGrh.lean`: the GRH-conditional primality test in
  `crypto_prime.c`. See [`CRYPTO-PRIMALITY-GRH.md`](../../docs/engineering/CRYPTO-PRIMALITY-GRH.md).
- `CryptoPipelineTuring.lean`: the Turing step of the zeta-zeros verification
  in `crypto_zeta.c`. See [`CRYPTO-ZETA-ZEROS.md`](../../docs/engineering/CRYPTO-ZETA-ZEROS.md), §6.

## What is proved: GRH primality link

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

## What is proved: Turing step

The C code computes an upper bound
`U = 1 + (A − Σᵢ (t₁ − hᵢ) + Iθ/π) / (t₁ − t₀)` for `N(t₀)`, with
`A = 2.067 + 0.059 log t₁`, and declares the verification complete when
`⌊U⌋` equals the number `c` of sign changes found up to `t₀`.

| Theorem | Statement | Conditional? |
|---|---|---|
| `integral_step` | The integral of the step `t ↦ [h < t]` over `[t₀, t₁]` is `t₁ − h` for `t₀ ≤ h ≤ t₁`. | No |
| `integral_stepCount` | The counting function `L` of the right ends `hᵢ` integrates to `Σᵢ (t₁ − hᵢ)`. | No |
| `turing_upper_bound` | `N(t₀) ≤ U`. | Yes: assumes `hTrud`, `hcount`, `hIθ` |
| `turing_count_exact` | If `N(t₀) = n₀ ≥ c` and `⌊U⌋₊ = c`, then `n₀ = c`. | No |
| `turing_verifies` | The two together: the check the C code performs. | Yes: as `turing_upper_bound` |

The hypotheses are:

- `hTrud`: `∫_{t₀}^{t₁} (N − θ/π − 1) ≤ A`. This is Trudgian's Theorem 2.2,
  `∫ S ≤ A` (Math. Comp. 80 (2011)), written with the identity
  `N = θ/π + 1 + S`.
- `hcount`: `N(t₀) + L(t) ≤ N(t)` on `[t₀, t₁]`. Each sign change of `Z`
  marks a zero on the critical line.
- `hIθ`: `∫_{t₀}^{t₁} θ ≤ Iθ`, which the C code bounds by evaluating θ on
  balls.

## What is not proved

- **Trudgian's theorem, the identity `N = θ/π + 1 + S`, and the
  intermediate-value step behind `hcount`.** They are hypotheses of the
  Turing theorems, never axioms. Mathlib does not contain the zeta function's
  argument or zero-counting theory needed to prove them.
- **That `crypto_zeta.c` computes `U` as modelled.** The C code evaluates the
  same formula with directed rounding; the correspondence is by inspection
  and tests, not proved.
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

## Pinned sources

`lake-manifest.json` pins every dependency (Mathlib and the packages it pulls
in) to an exact commit, and [`SOURCES.md`](SOURCES.md) lists those commits
and the Lean toolchain with their Software Heritage identifiers, so the build
can be reproduced from the archive if an upstream repository disappears.
