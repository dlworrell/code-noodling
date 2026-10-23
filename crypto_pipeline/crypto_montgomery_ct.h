/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:958a37c9-4551-4452-9a41-9272cb852313
 */

#ifndef CRYPTO_MONTGOMERY_CT_H
#define CRYPTO_MONTGOMERY_CT_H
#include "crypto_multiplier.h"
#ifdef __cplusplus
extern "C" {
#endif
/** Montgomery context for one public modulus.
 *
 * Public, immutable after initialization. Not for secret RSA factors/moduli.
 * Do not construct or alter by hand; initialize with the function below.
 */
typedef struct {
    size_t limbs;   /**< Modulus length n in limbs; 1 <= n <= CRYPTO_MAX_LIMBS. */
    limb_t inverse; /**< -m^-1 mod 2^64, from the modulus's least significant limb. */
    limb_t modulus[CRYPTO_MAX_LIMBS]; /**< Odd modulus m > 1, least significant limb
                                           first; limbs n and above are zero. */
} crypto_montgomery_ctx;

/* Public setup: modulus is n limbs, odd and >1; 1<=n<=1024.
 * Overlap is rejected on the supported flat-address target. Caller provides valid
 * aligned objects of sufficient size. Returns 0 or -1 with EINVAL/EDOM.
 * On error ctx is unchanged; on success errno=0. */
int crypto_montgomery_ctx_init(crypto_montgomery_ctx *ctx, const limb_t *modulus, size_t n);

/* Constant-flow CANDIDATE for secret reduced a,b with PUBLIC modulus/length.
 * out <- a*b*R^-1 mod m, R=2^(64*n). Output has n limbs.
 * Capacity parameters, addresses, and initialized ctx are public. Every array
 * has at least its stated capacity; out must not overlap a,b,ctx. a and b may
 * alias. Requires 0<=a,b<m; this SECRET precondition is deliberately NOT checked.
 * For out-of-domain operands the result is unspecified, without a secret error.
 * Returns -1/EINVAL on public pointer/capacity/count errors before writing out.
 * No claim of formal verification or portable constant-time machine execution.
 */
int crypto_montgomery_mul_ct(limb_t *out, size_t out_capacity,
                             const limb_t *a, size_t a_capacity,
                             const limb_t *b, size_t b_capacity,
                             const crypto_montgomery_ctx *ctx);
#ifdef __cplusplus
}
#endif
#endif
