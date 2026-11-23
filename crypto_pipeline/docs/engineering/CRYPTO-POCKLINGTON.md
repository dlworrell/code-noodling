<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:bca5c6a8-9554-4cbc-bfa6-ed1a5f33c1d1
-->

# Pocklington primality certificates

Status: Experimental (first slice)
Scope: `crypto_pipeline/` (governed project scope, Catylist ADR-0004)
Files: `crypto_cert.h`, `crypto_cert.c`, `pocklington_cert.py`,
`tests/test_cert.c`, `tests/test_pocklington.py`

## Why

`crypto_is_prime_grh` (see [CRYPTO-PRIMALITY-GRH.md](CRYPTO-PRIMALITY-GRH.md))
proves primality only if the generalized Riemann hypothesis holds. Bach's
theorem, which turns GRH into the witness bound, is analytic number theory and
cannot be proven in C. This slice takes the other route: an unconditional
proof of primality that C can check exactly. A certificate costs work to find,
but checking it uses only integer arithmetic and one classical theorem.

## The theorem

**Pocklington (1914), in the form of Brillhart, Lehmer and Selfridge (1975).**
Let N > 2 and N − 1 = F · R with F > √N. Suppose that for every prime q
dividing F there is an integer a with

- a^(N−1) ≡ 1 (mod N), and
- gcd(a^((N−1)/q) − 1, N) = 1.

Then N is prime.

*Proof sketch.* Let p be any prime factor of N. For each q^e ∥ F, the order of
a mod p divides N − 1 but not (N − 1)/q, so q^e divides that order, which
divides p − 1. Hence F divides p − 1, so p > F > √N. A composite N has a prime
factor at most √N, so N is prime.

The primes q must themselves be proven. A certificate is therefore a tree:

- an internal node proves its q by another node (`child >= 0`);
- a leaf is q < 2^64, proven by deterministic Miller–Rabin with the twelve
  prime bases 2, 3, …, 37. This is exact for every n < 3.18 × 10^23
  (Jiang and Deng 2014; Sorenson and Webster 2017), established by computer
  search; it uses no hypothesis.

## What the checker does

`crypto_pocklington_verify(nodes, count, root, max_modexps, &failed)` checks,
for every node reachable from `root`:

1. N is odd and N > 2; every q satisfies 2 ≤ q < N and every a satisfies
   1 < a < N − 1.
2. F = ∏ q^e fits below N (checked as it is built), F divides N − 1, and
   F² > N.
3. For each q: x = a^((N−1)/q) mod N, then x^q ≡ 1 (so a^(N−1) ≡ 1 without a
   third exponentiation), then gcd(x − 1, N) = 1.
4. Leaves have q < 2^64 and pass `crypto_is_prime_u64`; a child's N equals its
   q exactly (zero-padded limb counts may differ).

Structure is checked too: indices in range, depth at most 64, no cycles, and a
shared child (a DAG) verified once. `max_modexps` caps the work at two
exponentiations per factor; 0 means no cap. Error codes are `EINVAL` for a
malformed certificate, `EDOM` for a failed mathematical condition and `ERANGE`
for the cap; a rejection says nothing about whether N is prime.

The checker reuses `crypto_mod_exp` from the primality slice. Its other
arithmetic (schoolbook product, binary long division, binary GCD) is in
`crypto_bigint.c`, shared with the ECPP checker.
Everything is variable-time on public values.

## The generator is untrusted

`pocklington_cert.py` builds certificates: from a known factorization of part of
N − 1 (`certify`), or for random primes bottom-up in the style of Maurer's
method (`provable_prime`, N = 2Rq + 1 with q itself certified and q > √N / 2).
A bug there can only make the checker reject. One was found while writing this
slice: the candidate filter rejected the primes 41…71 because it used them as
their own bases; `tests/test_pocklington.py` now checks the filter against trial
division.

## Evidence

- `tests/test_cert.c` (also under Valgrind and ASan/UBSan): 64-bit leaves
  against trial division and known strong pseudoprimes; 65537 with the F² > N
  boundary on both sides; each defect with its error code; argument checks;
  multi-node trees: a DAG whose shared node is counted once (exact modexp cap),
  an out-of-range child index, a two-node cycle, and the depth limit (a
  65-node chain accepted, 66 rejected). A cycle is always caught by the
  child-value check first, because every edge goes from N to some q < N; the
  in-progress guard is defence in depth.
- `tests/test_pocklington.py` (ctypes): Proth and Fermat primes, 2^127 − 1 with
  the full factorization of N − 1, random provable primes of 64 to 2048 bits
  with nested nodes, zero-padded limbs, a DAG sharing one grandchild, and
  rejections of a wrong base, F too small, F not dividing N − 1, composite and
  oversized leaves, a child proving a different number, a self-reference, an
  out-of-range index, forged certificates for Carmichael numbers and strong
  pseudoprimes, even and tiny N, and the work cap at its exact boundary. Every
  valid certificate is re-checked by an independent Python implementation.

## Limits and next steps

- N − 1 methods need F > √N, so N must be built for the method or have a
  partially known N − 1. Certifying an arbitrary given prime needs ECPP; its
  checker is in [CRYPTO-ECPP.md](CRYPTO-ECPP.md).
- Mathlib has the Lucas primality test (`lucas_primality`), the F = N − 1
  special case. Proving the general Pocklington statement in `proof/lean/` and
  matching it to the conditions above would tie the C checks to a checked
  theorem.
- The 64-bit leaf test rests on a published computation, not a proof in this
  repository.

## References

- H. C. Pocklington, "The determination of the prime or composite nature of
  large numbers by Fermat's theorem", Proc. Cambridge Philos. Soc. 18 (1914).
- J. Brillhart, D. H. Lehmer, J. L. Selfridge, "New primality criteria and
  factorizations of 2^m ± 1", Math. Comp. 29 (1975).
- U. Maurer, "Fast generation of prime numbers and secure public-key
  cryptographic parameters", J. Cryptology 8 (1995).
- Y. Jiang, Y. Deng, "Strong pseudoprimes to the first eight prime bases",
  Math. Comp. 83 (2014).
- J. Sorenson, J. Webster, "Strong pseudoprimes to twelve prime bases",
  Math. Comp. 86 (2017).
