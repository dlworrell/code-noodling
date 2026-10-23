<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:12da95c1-99a4-4f61-b062-624fd990dc24
-->

# Multiplication experiment: mathematical contracts and validation

## Motivation

The archived Scientific American article by Jack Murtagh motivated the comparison
of schoolbook/Comba, Karatsuba, and transform multiplication. Modular arithmetic
and blinding are separate experiments layered on that full-product engine.
The article does not establish the correctness or security of this implementation.

## Full products

Let B=2^64 and write a=a0+B^h*a1, b=b0+B^h*b1. Karatsuba evaluates

    z0 = a0*b0
    z2 = a1*b1
    cross = (a0+a1)*(b0+b1) - z0 - z2
    a*b = z0 + B^h*cross + B^(2*h)*z2.

The sums need h+1 limbs, not h. Two separate full-width subtractions retain
borrows correctly; reconstruction propagates the carry through the remaining
output. Odd lengths are padded by one zero limb before recursion. The recursion
has three subproducts of about half the length plus linear bookkeeping, giving
the Karatsuba exponent log2(3), approximately 1.585. Comba remains the base case
at 16 limbs. This is a mathematical scaling statement, not a measured speedup.

The original code discarded the midpoint carries. Multiplying two 32-limb
all-ones operands is a deterministic regression witness.

## Exact NTT and CRT

Split each n-limb number into d=4*n base-2^16 digits. Choose a power-of-two
transform length L>=2*d, at most 8192. Zero padding prevents cyclic wraparound.
Use p1=998244353 and p2=1004535809, each with primitive root 3. The tests verify
primality, root order, divisibility of p-1 by 8192, and the coefficient bound.

Every convolution coefficient c satisfies

    0 <= c <= d*(2^16-1)^2 <= 4096*65535^2 < p1*p2.

Thus the two residues uniquely determine the nonnegative integer coefficient:

    c = r1 + p1 * (((r2-r1) * inverse(p1 mod p2)) mod p2).

Propagating base-2^16 carries yields the exact 2*n-limb integer product.
All CRT and modular products fit uint64_t for these primes and this length cap.
This is a bounded practical transform implementation, not a general proof of
O(N log N) integer multiplication for unbounded inputs, nor an implementation of
the Harvey–van der Hoeven algorithm mentioned in the article.

## Montgomery versus ordinary arithmetic

For R=B^n and odd m>1, REDC requires 0<=T<m*R. At iteration i, choose
u=product[i]*(-m[0]^-1) mod B so adding u*m*B^i cancels the low limb.
After n cancellations, the integer quotient is below 2*m. Preserve its extra
high bit and conditionally subtract m once. Dropping that bit caused the former
one-limb case m=2^64-1, a=b=m-1 to return 0 instead of 1.

Subtraction must widen m[i]+borrow before comparing; otherwise UINT64_MAX+1
wraps to zero and loses a borrow. The regression test includes this exact shape.
All multiplication dispatch paths now apply the same REDC semantics. Calling
Montgomery multiplication with ordinary a,b yields a*b*R^-1 mod m, not a*b mod m.
The distinct ordinary API performs exact multiplication then binary long division.

## Inverse and blinding

The variable-time binary extended-GCD inverse maintains
u == input*x1 (mod m) and v == input*x2 (mod m). When an odd coefficient must be
halved, use (x+m)/2, retaining the addition carry. Modular coefficient subtraction
keeps residues in [0,m). A final gcd other than 1 is an EDOM error, not a fabricated
inverse. Inputs are first reduced; the supported modulus is odd and greater than 1.
The former placeholder returned 1 for 3 inverse mod 11; the result must be 4.

Blinding samples uniformly from a power-of-two interval and rejects candidates
outside 1..m-1 or with gcd(k,m)!=1. It returns (a*k mod m,k); unblinding computes
blinded*inverse(k) mod m in the ordinary domain. Sampling has a 128-attempt limit
and can report EAGAIN. Short entropy reads and EINTR are handled. This construction
is algebraically checked but its variable-time routines are not side-channel safe.

## Evidence

`test_arithmetic.py`: independent Python integer oracle, fixed random seeds,
more than 5000 inverse cases, adversarial carries/borrows, odd lengths, all three
full-product algorithms, both dispatch modes, the 1024-limb cap, ordinary even
moduli, invalid domains, preserved error outputs, blinding round-trips, and FFI.
`test_bounds.c`: ASan/UBSan, output sentinels, invalid lengths/pointers, and
explicit Karatsuba/NTT all-ones, Montgomery carry, inverse, and blinding assertions.
Neither suite constitutes a proof or a constant-time audit.

A sample local GCC -O2 benchmark, median of seven batches of ten calls, including
FFI overhead (microseconds per full product):

| Limbs | Comba | Karatsuba | NTT |
|---:|---:|---:|---:|
| 32 | 2.21 | 2.37 | 76.66 |
| 64 | 6.56 | 5.90 | 159.26 |
| 128 | 31.44 | 17.34 | 311.07 |
| 256 | 106.15 | 57.23 | 804.25 |
| 512 | 431.67 | 185.92 | 1637.54 |
| 1024 | 2014.50 | 689.56 | 3914.04 |

This shared-runtime sample motivates a provisional Comba/Karatsuba switch at
64 limbs and leaves NTT opt-in. Reproduce with `benchmark_multiply.py` on the
target hardware; it is not a portable crossover claim. Ordinary bitwise modular
reduction can dominate the full-product cost and is a future optimization target.

Remaining work includes a vetted constant-time design, compiler/assembly and
timing analysis, capacity-aware APIs, and broader platform validation. Secret
intermediates in multiplication/transform scratch are not comprehensively erased.
Do not use this experimental implementation to protect real secrets.
