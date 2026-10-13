#ifndef CRYPTO_MONTGOMERY_CT_H
#define CRYPTO_MONTGOMERY_CT_H
#include "crypto_multiplier.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Public, immutable after initialization. Not for secret RSA factors/moduli.
 * Do not construct or alter by hand; initialize with the function below.
 */
typedef struct {
    size_t limbs;
    limb_t inverse;
    limb_t modulus[CRYPTO_MAX_LIMBS];
} crypto_montgomery_ctx;

/* Public setup: modulus is n limbs, odd and >1; 1<=n<=1024.
 * ctx and modulus must not overlap. Returns 0 or -1 with EINVAL/EDOM.
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
