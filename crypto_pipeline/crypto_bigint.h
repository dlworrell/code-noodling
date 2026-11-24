/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:742bd50e-498b-40ea-851d-59780d6fedb1
 */

#ifndef CRYPTO_BIGINT_H
#define CRYPTO_BIGINT_H
#include "crypto_multiplier.h"

/* Internal multi-limb helpers shared by the certificate checkers
 * (crypto_cert.c, crypto_ecpp.c, crypto_mixed.c). Not a public API: no argument checks beyond
 * what each comment states. Little-endian base-2^64, n >= 1 limbs. All are
 * variable-time and meant for public values. */
#ifdef __cplusplus
extern "C" {
#endif

/* Number of limbs up to and including the top nonzero one (at least 1). */
size_t crypto_bn_significant(const limb_t *a, size_t n);
size_t crypto_bn_bit_length(const limb_t *a, size_t n);
void crypto_bn_copy(limb_t *dst, const limb_t *src, size_t n);
void crypto_bn_set_small(limb_t *dst, limb_t value, size_t n);
int crypto_bn_is_zero(const limb_t *a, size_t n);
/* -1, 0 or 1 as a <, =, > b. */
int crypto_bn_compare(const limb_t *a, const limb_t *b, size_t n);
/* a (an limbs) and b (bn limbs) are equal as integers. */
int crypto_bn_same_value(const limb_t *a, size_t an, const limb_t *b, size_t bn);
/* r = a + b mod 2^(64n), returns the carry. r may alias a or b. */
limb_t crypto_bn_add(limb_t *r, const limb_t *a, const limb_t *b, size_t n);
/* r = a - b mod 2^(64n), returns the borrow. r may alias a or b. */
limb_t crypto_bn_sub(limb_t *r, const limb_t *a, const limb_t *b, size_t n);
void crypto_bn_shift_right_one(limb_t *a, size_t n);
/* wide (2n limbs) = a * b, schoolbook. wide must not alias a or b. */
void crypto_bn_mul_full(limb_t *wide, const limb_t *a, const limb_t *b, size_t n);
/* quot = a / d, rem = a mod d (all n limbs, d nonzero, no aliasing). */
void crypto_bn_divide(limb_t *quot, limb_t *rem, const limb_t *a, const limb_t *d, size_t n);
/* g = gcd(a, b) for odd b (all n limbs). gcd(0, b) = b. */
void crypto_bn_gcd_odd(limb_t *g, const limb_t *a, const limb_t *b, size_t n);
/* gcd(a, m) == 1 for odd m. */
int crypto_bn_coprime_odd(const limb_t *a, const limb_t *m, size_t n);
/* r = floor(sqrt(a)) (n limbs each, no aliasing). */
void crypto_bn_isqrt(limb_t *r, const limb_t *a, size_t n);
/* a mod 3. */
unsigned crypto_bn_mod3(const limb_t *a, size_t n);

/* Modular arithmetic for odd m > 1 with inputs < m. r may alias inputs. */
void crypto_bn_add_mod(limb_t *r, const limb_t *a, const limb_t *b, const limb_t *m, size_t n);
void crypto_bn_sub_mod(limb_t *r, const limb_t *a, const limb_t *b, const limb_t *m, size_t n);
/* Montgomery constants for R = 2^(64n): one = R mod m, r2 = R^2 mod m. */
void crypto_bn_montgomery_constants(limb_t *one, limb_t *r2, const limb_t *m, size_t n);
/* out = a * b * R^-1 mod m via crypto_multiply_reduce. wide: 2n limbs of
 * scratch. out may alias a or b. Returns 0, or -1 with errno set. */
int crypto_bn_mont_mul(limb_t *out, const limb_t *a, const limb_t *b, const limb_t *m,
                       size_t n, limb_t *wide);

#ifdef __cplusplus
}
#endif
#endif
