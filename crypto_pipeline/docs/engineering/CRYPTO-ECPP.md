<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:705d3f55-92dc-4e75-adc2-6e49677682c9
-->

# ECPP primality certificates

Status: Experimental (slices 1 and 2 of 4)
Scope: `crypto_pipeline/` (governed project scope, Catylist ADR-0004)
Files: `crypto_ecpp.h`, `crypto_ecpp.c`, `crypto_bigint.h`, `crypto_bigint.c`,
`ecpp_cert.py`, `tests/test_ecpp.c`, `tests/test_ecpp.py`, `tests/fixtures/ecpp/`

## Why

A Pocklington certificate ([CRYPTO-POCKLINGTON.md](CRYPTO-POCKLINGTON.md))
needs more than √N of N − 1 factored, so it suits primes built for it. Elliptic
curves give every prime a certificate: the group E(F_p) has about p points, and
some curve's order has a large prime factor. Finding the certificate is the
hard part; checking it takes a few scalar multiplications per step.

## The theorem

**ECPP (Goldwasser and Kilian 1986; Atkin and Morain 1993).** Let
gcd(N, 6) = 1 and E: y² = x³ + ax + b over Z/N with gcd(4a³ + 27b², N) = 1.
Let P be a point of E, k ≥ 1, and q a prime with q > (N^¼ + 1)². If Q = kP is
not the identity modulo any prime factor of N, and qQ is the identity modulo
every prime factor of N, then N is prime.

*Proof.* Let p be a prime factor of N. E reduces to a non-singular curve over
F_p, and Q reduces to a point other than the identity with qQ = O, so Q has
order q there (q is prime). Hence q ≤ #E(F_p) ≤ (√p + 1)² by Hasse's theorem.
If p ≤ √N this gives q ≤ (N^¼ + 1)², a contradiction; so every prime factor of
N exceeds √N, and N is prime.

The checker uses q ≥ (⌊N^¼⌋ + 2)², which implies the strict bound and needs only
integer square roots: ⌊N^¼⌋ = ⌊√⌊√N⌋⌋.

## Working modulo an N that may be composite

The checker cannot work in E(F_p): it does not know p. It computes modulo N,
which is the same as computing modulo every prime factor of N at once (the
Chinese remainder theorem), and it must be sure the result is right modulo
each one.

Affine formulas would need an inverse at every step, each with a gcd check.
Instead the checker uses projective coordinates and a single addition law for
both adding and doubling: Renes, Costello and Batina (2016), Algorithm 1, which
is an addition law of Bosma and Lenstra (1995) for short Weierstrass curves.
Over a field, such a law returns the true sum P + Q or the invalid triple
(0:0:0), and it is (0:0:0) only when P − Q has order 2. The formulas are
homogeneous of positive degree in each input, so once a coordinate triple is
(0:0:0) modulo p it stays (0:0:0) through every later addition.

So, modulo each prime factor p of N, every intermediate point is either correct
or (0:0:0) from that point on. The checker requires:

1. gcd(Z_Q, N) = 1. Then modulo each p, Q is not (0:0:0) (so it is correct)
   and is not the identity (whose Z is 0).
2. Z_qQ ≡ 0 (mod N) and gcd(Y_qQ, N) = 1. Then modulo each p, qQ is not
   (0:0:0), so it is correct, and it has Z = 0, so it is the identity.

Condition 2 needs both halves. `test_collapse_forgery_for_a_composite` builds
composites N = p·s where modulo the small prime p the computation of qQ
collapses to (0:0:0): Q is proper and Z_qQ ≡ 0 (mod N), so a checker testing
only Z would accept N. gcd(Y_qQ, N) = p rejects it.

Other checks per step: N odd, not divisible by 3, and above 3; 0 ≤ a, x, y < N;
1 ≤ k < N and q < N (so the chain shrinks); and gcd(4a³ + 27b², N) = 1, so E is
non-singular modulo every factor. b is not stored but computed as
y² − x³ − ax, so P is on E by construction. Each step's q must equal the next
step's N, and the last q must be below 2^64 and pass `crypto_is_prime_u64`
(Miller–Rabin to bases 2..37, exact below 3.18 × 10²³).

Arithmetic is in Montgomery form through `crypto_multiply_reduce`, which works
for any odd modulus. Montgomery form multiplies by R = 2^(64n), a unit modulo
N, so the zero and gcd tests above are unaffected.

## Evidence

- The addition law, exhaustively over F_5, F_7, F_11 and F_13: every pair of
  points on every non-singular curve gives the true sum or (0:0:0), never a
  wrong point; (0:0:0) occurs, and only when P − Q has order 2. (0:0:0) is
  absorbing. During development the same check ran over F_5 to F_29.
- The C scalar multiplication against the Python reference: every point and
  several scalars over F_7, F_11 and F_13, and random odd moduli (prime or
  not) to 300 bits.
- Chains accepted: random provable primes of 80 to 384 bits (up to 16 steps),
  zero-padded limbs, and the work cap at its exact boundary. Each chain is
  re-checked with affine arithmetic, independent of the projective law.
- Rejected, one defect at a time: wrong cofactor, a different curve or point,
  values out of range, a singular curve, Q the identity, a genuine point whose
  prime order is below the bound, N even or divisible by 3, a broken link, a
  chain ending above 2^64, a composite final leaf, the collapse forgery above,
  and plain CRT forgeries for composites.
- `tests/test_ecpp.c` runs the C contract under Valgrind and ASan/UBSan.
- `crypto_bigint.c`: integer square root and N mod 3 against Python on values
  to 900 bits and at perfect squares.

## The generator is untrusted

`ecpp_cert.py` uses complex multiplication with the nine discriminants of class
number 1, whose j-invariants are integers, so no class polynomials are needed.
For each D with (D/N) = 1 it solves 4N = t² + |D|v² (Cornacchia), takes the
orders N + 1 − u over the twists, removes prime factors below 2^15, and keeps
orders whose remainder q is a probable prime above the bound. It searches
depth-first down to q < 2^64, then finds a curve of the right order by testing
random curves with that j-invariant. A 512-bit prime takes a few seconds and
about 20 steps. Class number 1 alone cannot certify every prime, so
`provable_prime` draws primes until one works and `certify` can return None.

## PARI/GP certificate chains for primes at least 2^64

`ecpp_cert.from_pari` converts a certificate from PARI/GP's `primecert(N)`
(checked against PARI 2.15.4's own help and output). Each entry is
`[N, t, s, a, [x, y]]`: the curve y² = x³ + ax + b passes through (x, y), its
order is m = N + 1 − t, s is the cofactor and q = m / s; PARI ends the chain
below 2^64. That maps field for field onto a checker step (k = s), and b is
again derived from the point.

`python3 ecpp_cert.py pari N` accepts integer inputs `N >= 2^64` and prints
the certificate chain in fixture format for a prime. Below this range, PARI
returns scalar certificates for primes; the fixture format represents chain
rows only. The CLI rejects inputs below `2^64` before invoking GP, reports the
supported range on stderr, exits with status 2 and prints no fixture output.
The Python helper `pari_certificate` raises `ValueError` for these inputs.
For in-range inputs, GP errors (including a composite input or stack overflow)
are reported as failures even when GP itself exits 0. CI does not install
PARI, so `tests/test_ecpp.py` checks this with a stub `gp` that exits 0: an
error line on stdout or stderr, or no output, raises `RuntimeError`, while a
`*** Warning` line alone does not.

`tests/fixtures/ecpp/` holds six committed certificates, each with the command,
PARI version and date in its header:

| Fixture | N | Bits | Steps |
|---|---|---|---|
| `p127` | 2^127 + 45 | 128 | 2 |
| `p255` | 2^255 − 19 (Curve25519) | 255 | 6 |
| `p256` | 2^256 − 2^224 + 2^192 + 2^96 − 1 (NIST P-256) | 256 | 9 |
| `p333` | first prime above 10^100 | 333 | 12 |
| `p513` | first prime above 3 · 2^511 | 513 | 17 |
| `p1024` | first prime above 2^1023 + 2^512 | 1024 | 41 |

The checker accepts all six unchanged; the 1024-bit chain takes about 0.4 s.
The class-number-1 generator finds no certificate for the last four within its
search budget, so these are primes this repository could not certify before.
CI checks only the committed fixtures and does not need PARI;
`test_live_pari_certificates` also certifies fresh random primes when gp is
installed. PARI is the prover only: a wrong certificate from it would be
rejected, never accepted.

## Limits and next slices

- Soundness rests on Hasse's theorem and the Bosma–Lenstra property of the
  addition law (both proven, not machine-checked here; the law's property is
  also tested exhaustively above) and on the published 64-bit Miller–Rabin
  computation.
- Slice 3: mixed certificates, where an ECPP step's q is proven by a
  Pocklington tree or the reverse.
- Slice 4: the theorem in Lean with Hasse's bound as a hypothesis.
- Variable-time, public values only; research prototype.

## References

- S. Goldwasser, J. Kilian, "Almost all primes can be quickly certified",
  STOC 1986.
- A. O. L. Atkin, F. Morain, "Elliptic curves and primality proving",
  Math. Comp. 61 (1993).
- W. Bosma, H. W. Lenstra, "Complete systems of two addition laws for elliptic
  curves", J. Number Theory 53 (1995).
- J. Renes, C. Costello, L. Batina, "Complete addition formulas for prime order
  elliptic curves", EUROCRYPT 2016.
- H. Cohen, *A Course in Computational Algebraic Number Theory*, Springer 1993
  (Cornacchia's algorithm, Algorithm 1.5.3).
