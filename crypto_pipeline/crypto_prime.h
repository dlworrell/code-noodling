/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:14e7233e-d327-4be0-9df2-23758d2f6764
 */

#ifndef CRYPTO_PRIME_H
#define CRYPTO_PRIME_H
#include "crypto_multiplier.h"

/* Modular exponentiation and a GRH-conditional deterministic primality test.
 * Research prototype: every routine here is VARIABLE-TIME and is meant for
 * public values (candidate primes, public moduli), never for secrets.
 * Conventions follow crypto_multiplier.h: little-endian base-2^64 limbs,
 * 1 <= n <= CRYPTO_MAX_LIMBS, outputs must not overlap inputs, EINVAL for a bad
 * pointer or count, EDOM for a bad mathematical domain, outputs unchanged on
 * failure, errno cleared on success. See docs/engineering/CRYPTO-PRIMALITY-GRH.md. */
#ifdef __cplusplus
extern "C" {
#endif

/* Ordinary base^exponent mod mod. base, mod and result have n limbs;
 * exponent has exp_n limbs (1..CRYPTO_MAX_LIMBS). mod must be odd and > 1;
 * base need not be reduced. exponent 0 gives 1. Built on Montgomery
 * multiplication (crypto_multiply_reduce). Returns 0, or -1 with errno. */
int crypto_mod_exp(limb_t *result, const limb_t *base, const limb_t *exponent,
                   size_t exp_n, const limb_t *mod, size_t n);

/* Upper bound on the bases the GRH test tries for candidate (n limbs):
 * ceil(2 * (L * 0.69315)^2), where L is the bit length of candidate. Since
 * candidate < 2^L and 0.69315 > ln 2, it is at least 2 * (ln candidate)^2, the
 * bound in Bach's theorem. Returns 0 with EINVAL for a bad pointer or count,
 * and 0 with errno cleared for candidate 0. */
uint64_t crypto_grh_witness_bound(const limb_t *candidate, size_t n);

/* Deterministic primality test, CONDITIONAL on the generalized Riemann
 * hypothesis for Dirichlet L-functions (ERH). Bach (1990) proved that, under
 * ERH, every odd composite N has a strong (Miller-Rabin) witness below
 * 2 (ln N)^2. This runs Miller-Rabin for every base a with
 * 2 <= a <= min(crypto_grh_witness_bound(candidate), N - 2).
 *
 * Sets *is_prime to 1 for a prime and 0 for a composite (0, 1 and even numbers
 * other than 2 are composite or not prime). A "composite" answer is
 * unconditional: it comes with a witness. A "prime" answer is proved only
 * under ERH.
 *
 * The number of bases grows as the square of the bit length (about 4 million
 * at 2048 bits). max_bases caps the work: if more than max_bases bases would be
 * needed, the call fails with ERANGE and *is_prime is unchanged; it never falls
 * back to a probabilistic answer. max_bases 0 means no cap.
 * Returns 0, or -1 with errno. */
int crypto_is_prime_grh(int *is_prime, const limb_t *candidate, size_t n,
                        uint64_t max_bases);

#ifdef __cplusplus
}
#endif
#endif
