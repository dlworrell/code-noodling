<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:cea4a8b5-da90-45c1-b012-1eec41e79357
-->

# GRH-conditional primality test

Status: Experimental (first slice)
Scope: `crypto_pipeline/` (governed project scope, Catylist ADR-0004)
Files: `crypto_prime.h`, `crypto_prime.c`, `tests/test_prime.c`, `tests/test_prime.py`

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

## Next slices (not in this change)

- A deterministic Tonelli–Shanks square root modulo a prime, using the same
  ERH bound for the least quadratic non-residue.
- A formal link: state Bach's bound in Lean and prove that the C loop computes
  it. This is a research-sized task; Mathlib does not yet contain Bach's theorem.
