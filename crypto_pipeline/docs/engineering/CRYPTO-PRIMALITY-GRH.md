<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:cea4a8b5-da90-45c1-b012-1eec41e79357
-->

# GRH-conditional primality test and square roots

Status: Experimental (slices 1 and 2, with a Lean link)
Scope: `crypto_pipeline/` (governed project scope, Catylist ADR-0004)
Files: `crypto_prime.h`, `crypto_prime.c`, `tests/test_prime.c`, `tests/test_prime.py`,
`proof/lean/`

## What it does

`crypto_is_prime_grh` decides whether a number is prime by running the
Miller–Rabin (strong) test for **every** base from 2 up to an explicit bound,
instead of a few random bases. `crypto_mod_exp` is the modular exponentiation
it is built on, and is usable on its own.

The bound comes from Bach's theorem:

> Assuming the extended Riemann hypothesis (ERH), every odd composite N has a
> strong witness a with a < 2 (ln N)².
> E. Bach, *Explicit bounds for primality testing and related problems*,
> Mathematics of Computation 55 (1990), 355–380.

ERH here means the generalized Riemann hypothesis for Dirichlet L-functions.

## What is and is not proved

- **A "composite" answer is unconditional.** It is returned only when a base
  fails the strong test, which is a proof of compositeness.
- **A "prime" answer is proved only under ERH.** The code does not make it
  unconditional, and the header and this note say so.

### Status of the hypothesis (as of 2026-11-22)

A Lean 4 formalization of a claimed proof of the Riemann hypothesis and of the
generalized Riemann hypothesis for Dirichlet L-functions was announced on
2026-11-20 (verification files: <https://math.mit.edu/~drew/grh/>). The
verification page states that:

- the published Lean source compiles, and the main theorems depend only on
  Lean's three standard axioms;
- some kernel replays were still running when the page was last updated;
- Dedekind zeta, Hecke and automorphic L-functions were not checked.

The page does not claim peer review. Until the result is reviewed and accepted,
this code treats ERH as an **assumption**. If the result is accepted, the
"prime" answers become unconditional for the Dirichlet case Bach's theorem
uses, with no code change. This scope does not depend on that outcome: the
composite answers, the arithmetic and the tests stand either way.

## The bound in code

`crypto_grh_witness_bound` returns `ceil(2 · (L · 0.69315)²)`, where `L` is
the bit length of N. Since N < 2^L and 0.69315 > ln 2 = 0.693147…, this is at
least 2 (ln N)², so testing every base up to it covers Bach's bound. It is
computed in exact integer arithmetic (`13863/20000` for 0.69315); for
`L ≤ 65536` the numerator stays below 2^61. The test caps the last base at
N − 2, because N − 1 is never a witness.

## Cost

The number of bases grows as the square of the bit length. Measured locally
(single core, this container, `-O2`):

| N | Bit length | Bases | Time to prove prime |
|---|---:|---:|---:|
| 2^127 − 1 | 127 | 15,498 | 0.4 s |
| 2^255 − 19 | 255 | 62,483 | 4.6 s |
| 2^521 − 1 | 521 | 260,831 | 104 s |

A composite usually returns after the first few bases. At 2048 bits a prime
needs about 4 million bases, which is hours. The `max_bases` argument lets a
caller refuse that work up front: the call then fails with `ERANGE` and gives
no answer. It never falls back to a probabilistic result.

Practical use: a reference check for moderate sizes, for example to confirm
results from a faster probabilistic test. It is not a replacement for one at
cryptographic sizes.

## Implementation notes

- **Montgomery arithmetic.** Exponentiation runs in Montgomery form through
  the existing `crypto_multiply_reduce` (Comba below 64 limbs, Karatsuba above).
  `R mod N` and `R² mod N` are computed by repeated modular doubling, so no new
  division routine is needed. The result is converted back by a Montgomery
  multiplication by 1.
- **Variable-time.** Square-and-multiply branches on exponent bits, and the
  test returns at the first witness. Both are fine for public candidates and
  public moduli. Do not use these routines on secrets.
- **No review-required primitives.** The new file avoids `memcpy`, `memset`
  and `memmove`, so it adds no entries to the review-disposition ledger. The
  pinned AEMS AES-SEC-001 scan (`--strict --strict-review-ratchet`) reports no
  findings in the new files.
- **Separate files.** The code is in new files, so the line positions of
  existing ledger findings in `crypto_multiplier.c` do not change.

## Tests

- `tests/test_prime.c` (run by `make all`, under Valgrind in CI, and under
  ASan/UBSan in `make bounds-sanitize`):
  - every value below 3,000 against trial division;
  - Carmichael numbers and strong pseudoprimes;
  - known primes (2^61 − 1, 2^64 − 59, 2^89 − 1);
  - the bound value, the `max_bases` cap, and error contracts.
- `tests/test_prime.py` (CI `differential` step):
  - `crypto_mod_exp` against Python `pow` from 1 to 64 limbs;
  - every value below 20,000 against trial division;
  - random values up to 128 bits against an independent Python implementation
    of the same test;
  - strong pseudoprimes to all prime bases up to 37
    (318665857834031151167461 and 3317044064679887385961981);
  - known primes up to 2^127 − 1, the bound formula, the work cap, and errors.

## Square roots modulo a prime (slice 2)

`crypto_mod_sqrt_grh` computes a square root of `a` modulo a prime `p` by
Tonelli–Shanks. The one step of Tonelli–Shanks that is normally randomized is
finding a quadratic non-residue `z`. Here it is a deterministic search over
`z = 2, 3, …` up to the same bound, `crypto_grh_witness_bound(p)`:

> Assuming ERH, the least quadratic non-residue modulo a prime p is below
> 2 (ln p)². (Bach 1990, as above.)

What the routine guarantees:

- **Any root it returns is correct, unconditionally.** It squares the result
  and compares with `a mod p` before returning, so even a composite `p` cannot
  produce a wrong root. It returns the smaller root, `min(r, p − r)`.
- **ERH only bounds the search.** It guarantees the search ends within the
  bound for a prime `p`. If no non-residue is found below the bound, the call
  fails with `EDOM`: for a prime `p` that would be a counterexample to ERH, so
  in practice it means `p` is composite.
- **Composite moduli are rejected when detected.** Euler's criterion, the
  non-residue search and the Tonelli–Shanks loop each check an identity that
  holds for primes; a failure gives `EDOM`.
- **The search is short in practice.** The least non-residue is usually 2, 3
  or 5. `max_trials` caps it anyway (`ERANGE`, output unchanged).
- `p = 2` is accepted; other even moduli and `p ≤ 1` give `EDOM`.

Tests (added to `tests/test_prime.c` and `tests/test_prime.py`):

- every residue of small primes, against brute force, including that
  non-residues fail with `EDOM` and leave the output unchanged;
- random primes from 8 to 521 bits, in each class p ≡ 3 (mod 4) and
  p ≡ 1 (mod 8), with unreduced inputs;
- primes where p − 1 has a large power of two (998244353, 1004535809,
  2^64 − 2^32 + 1, 3·2^189 + 1), which exercise the Tonelli–Shanks loop;
- composite moduli, checking that any returned root is a true root;
- the trial cap and argument errors.

## Formal link (Lean)

`proof/lean/` (see its [README](../../proof/lean/README.md)) proves, with
Mathlib and CI-checked by `.github/workflows/crypto_lean.yml`:

- the C bound's numerator stays below 2^61 for every supported size, so the
  `uint64_t` arithmetic in `crypto_grh_witness_bound` cannot overflow;
- that bound is at least 2 (ln N)², the bound in Bach's theorem;
- every prime passes every base the C loop tries, so the test never calls a
  prime composite (unconditional);
- with Bach's theorem as an explicit hypothesis (`BachWitnessBound`), an odd
  N ≥ 3 that passes every base up to the bound is prime.

The axiom audit shows only Lean's three standard axioms (`propext`,
`Classical.choice`, `Quot.sound`) and no `sorry`. Two gaps remain: Bach's
theorem itself, which needs the extended Riemann hypothesis and analytic
number theory that Mathlib does not contain; and the correspondence between
the C code and the Lean model, which is tested, not proved.

For primality without GRH, see the unconditional certificate checkers:
[Pocklington](CRYPTO-POCKLINGTON.md) and [ECPP](CRYPTO-ECPP.md).
