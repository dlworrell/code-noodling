/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:0ed25988-4060-44a7-bf6a-0d6e57a8f5e1
 */

#ifndef CRYPTO_ECPP_H
#define CRYPTO_ECPP_H
#include "crypto_multiplier.h"

/* Unconditional primality certificates by elliptic curves (ECPP, in the form
 * of Goldwasser-Kilian and Atkin-Morain).
 *
 * Theorem. Let gcd(N, 6) = 1 and E: y^2 = x^3 + a x + b over Z/N with
 * gcd(4a^3 + 27b^2, N) = 1. If P is a point of E, k >= 1, q is prime with
 * q > (N^(1/4) + 1)^2, Q = kP is not the identity modulo any prime factor of
 * N, and qQ is the identity modulo every prime factor of N, then N is prime.
 * (For a prime p | N, Q has order q in E(F_p), so by Hasse's bound
 * q <= (sqrt(p) + 1)^2; hence p > sqrt(N).)
 *
 * A certificate is a chain of steps: each step's q is the next step's N, and
 * the last q is below 2^64 and proven by crypto_is_prime_u64 (crypto_cert.h).
 * b is not stored: it is y^2 - x^3 - a x, so P lies on E by construction.
 *
 * Arithmetic uses projective coordinates and one complete addition law
 * (Renes, Costello and Batina 2016, from Bosma and Lenstra 1995). Modulo each
 * prime factor of N that law returns either the true sum or (0:0:0), and
 * (0:0:0) is never changed by later additions. The checker therefore requires
 * gcd(Z_Q, N) = 1, Z_qQ = 0 (mod N) and gcd(Y_qQ, N) = 1, which rule out a
 * collapse modulo any factor. See docs/engineering/CRYPTO-ECPP.md.
 *
 * Within a step, every value uses the step's `limbs` count (little-endian
 * base-2^64, zero-padded). Research prototype, variable-time, public values
 * only. */
#ifdef __cplusplus
extern "C" {
#endif

#define CRYPTO_ECPP_MAX_STEPS 4096u

typedef struct crypto_ecpp_step {
    const limb_t *n;   /* N: gcd(N, 6) = 1, N > 3 */
    const limb_t *a;   /* curve coefficient, 0 <= a < N */
    const limb_t *x;   /* P = (x, y), 0 <= x, y < N; b = y^2 - x^3 - a x */
    const limb_t *y;
    const limb_t *k;   /* cofactor, 1 <= k < N */
    const limb_t *q;   /* (floor(N^(1/4)) + 2)^2 <= q < N */
    size_t limbs;      /* 1..CRYPTO_MAX_LIMBS */
} crypto_ecpp_step;

/* Verifies that steps[0] proves its N prime, through the chain steps[1..].
 *
 * Returns 0 when the chain proves steps[0].n prime. Otherwise returns -1 and
 * sets:
 *   EINVAL  malformed: null pointer, count out of range, steps[i].q differs
 *           from steps[i+1].n, or the last q is 2^64 or more;
 *   EDOM    a mathematical check failed: N even, divisible by 3 or <= 3, a
 *           value out of range, q below the bound, a singular curve, Q the
 *           identity or collapsed, qQ not the identity, or a final 64-bit q
 *           that is not prime;
 *   ERANGE  more than max_point_ops point additions would be needed (0 means
 *           no cap); each step needs 2 * (bits(k) + bits(q)).
 * A rejected certificate says nothing about whether N is prime.
 * If failed_step is not NULL, it receives the index of the step where checking
 * stopped (unchanged on success). Variable-time. */
int crypto_ecpp_verify(const crypto_ecpp_step *steps, size_t count, uint64_t max_point_ops,
                       size_t *failed_step);

/* Test hook: (X:Y:Z) = k * (x:y:1) on y^2 = x^3 + a x + b mod m with the
 * checker's addition law and double-and-add, starting from (0:1:0). Inputs and
 * outputs are ordinary residues (not Montgomery form); m odd > 1, all values
 * n limbs and below m. Returns 0, or -1 with errno = EINVAL. */
int crypto_ecpp_scalar_mul(limb_t *X, limb_t *Y, limb_t *Z, const limb_t *a, const limb_t *b,
                           const limb_t *x, const limb_t *y, const limb_t *k, const limb_t *m,
                           size_t n);

#ifdef __cplusplus
}
#endif
#endif
