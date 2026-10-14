#include "crypto_montgomery_ct.h"
#include <errno.h>
#include <string.h>

int crypto_montgomery_ctx_init(crypto_montgomery_ctx *ctx, const limb_t *modulus, size_t n) {
    if (!ctx || !modulus || n == 0 || n > CRYPTO_MAX_LIMBS) { errno = EINVAL; return -1; }
    /* Supported flat-address target: compare distances, never wrapping endpoints.
     * Caller still supplies valid, aligned, sufficiently large objects. */
    uintptr_t ms = (uintptr_t)modulus;
    uintptr_t cs = (uintptr_t)ctx;
    size_t mb = n * sizeof(*modulus);
    size_t cb = sizeof(*ctx);
    if (ms <= cs ? cs - ms < mb : ms - cs < cb) { errno = EINVAL; return -1; }
    /* Only public modulus bits are inspected by this setup routine. */
    limb_t upper = 0;
    for (size_t i = 1; i < n; ++i) upper |= modulus[i];
    if (!(modulus[0] & 1) || (upper == 0 && modulus[0] <= 1)) { errno = EDOM; return -1; }
    limb_t inv = 1;
    for (unsigned i = 0; i < 6; ++i) inv *= 2 - modulus[0] * inv;
    memset(ctx, 0, sizeof(*ctx));
    memcpy(ctx->modulus, modulus, n * sizeof(limb_t));
    ctx->inverse = (limb_t)0 - inv;
    ctx->limbs = n;
    errno = 0; return 0;
}

int crypto_montgomery_mul_ct(limb_t *out, size_t out_capacity,
                             const limb_t *a, size_t a_capacity,
                             const limb_t *b, size_t b_capacity,
                             const crypto_montgomery_ctx *ctx) {
    if (!ctx || !out || !a || !b) { errno = EINVAL; return -1; }
    size_t n = ctx->limbs;
    if (n == 0 || n > CRYPTO_MAX_LIMBS || out_capacity < n || a_capacity < n || b_capacity < n) {
        errno = EINVAL; return -1;
    }
    /* All subsequent loop bounds and addresses depend only on public n.
     * Do not insert early-exit operand validation or reuse variable-time APIs. */
    limb_t t[2*n], difference[n];
    memset(t, 0, 2*n * sizeof(limb_t));
    for (size_t i = 0; i < n; ++i) {
        dlimb_t carry = 0;
        for (size_t j = 0; j < n; ++j) {
            dlimb_t sum = (dlimb_t)a[i] * b[j] + t[i+j] + carry;
            t[i+j] = (limb_t)sum;
            carry = sum >> 64;
        }
        t[i+n] = (limb_t)carry;
    }
    limb_t high = 0;
    for (size_t i = 0; i < n; ++i) {
        limb_t u = t[i] * ctx->inverse;
        dlimb_t carry = 0;
        for (size_t j = 0; j < n; ++j) {
            dlimb_t sum = (dlimb_t)u * ctx->modulus[j] + t[i+j] + carry;
            t[i+j] = (limb_t)sum;
            carry = sum >> 64;
        }
        for (size_t k = i+n; k < 2*n; ++k) {
            dlimb_t sum = (dlimb_t)t[k] + carry;
            t[k] = (limb_t)sum;
            carry = sum >> 64;
        }
        high += (limb_t)carry;
    }
    /* a,b<m -> T<mR -> quotient<2m; high is 0 or 1. */
    limb_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t sub = (dlimb_t)ctx->modulus[i] + borrow;
        limb_t next = (dlimb_t)t[n+i] < sub;
        difference[i] = (limb_t)((dlimb_t)t[n+i] - sub);
        borrow = next;
    }
    limb_t mask = (limb_t)0 - (high | (borrow ^ 1));
    for (size_t i = 0; i < n; ++i) out[i] = (difference[i] & mask) | (t[n+i] & ~mask);
    /* Wipe this routine's secret stack arrays using public, fixed bounds. */
    volatile limb_t *p = t;
    for (size_t i = 0; i < 2*n; ++i) p[i] = 0;
    p = difference;
    for (size_t i = 0; i < n; ++i) p[i] = 0;
    errno = 0; return 0;
}
