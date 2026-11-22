/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:1fb00f05-adce-42ed-9fe7-f420cf332db5
 */

/* ECPP certificate checker. See crypto_ecpp.h for the contract and
 * docs/engineering/CRYPTO-ECPP.md for the soundness argument. Variable-time,
 * public values only. */

#include "crypto_ecpp.h"
#include "crypto_bigint.h"
#include "crypto_cert.h"
#include <errno.h>

static int fail(int code) { errno = code; return -1; }
static int valid_count(size_t n) { return n > 0 && n <= CRYPTO_MAX_LIMBS; }

/* Curve context in Montgomery form: a, 3b, the modulus and scratch. */
typedef struct {
    const limb_t *m;
    size_t n;
    const limb_t *a;
    const limb_t *b3;
    limb_t *wide;     /* 2n limbs */
} curve_ctx;

/* (X3:Y3:Z3) = (X1:Y1:Z1) + (X2:Y2:Z2): Renes-Costello-Batina 2016,
 * Algorithm 1 (complete projective addition for y^2 = x^3 + a x + b, b3 = 3b),
 * step for step. Outputs may alias inputs. */
static int point_add(limb_t *X3o, limb_t *Y3o, limb_t *Z3o,
                     const limb_t *X1, const limb_t *Y1, const limb_t *Z1,
                     const limb_t *X2, const limb_t *Y2, const limb_t *Z2, const curve_ctx *c) {
    const size_t n = c->n;
    const limb_t *m = c->m;
    limb_t t0[n], t1[n], t2[n], t3[n], t4[n], t5[n], X3[n], Y3[n], Z3[n];
#define MUL(r, x, y) do { if (crypto_bn_mont_mul((r), (x), (y), m, n, c->wide) != 0) return -1; } while (0)
#define ADD(r, x, y) crypto_bn_add_mod((r), (x), (y), m, n)
#define SUB(r, x, y) crypto_bn_sub_mod((r), (x), (y), m, n)
    MUL(t0, X1, X2);  MUL(t1, Y1, Y2);  MUL(t2, Z1, Z2);
    ADD(t3, X1, Y1);  ADD(t4, X2, Y2);  MUL(t3, t3, t4);
    ADD(t4, t0, t1);  SUB(t3, t3, t4);  ADD(t4, X1, Z1);
    ADD(t5, X2, Z2);  MUL(t4, t4, t5);  ADD(t5, t0, t2);
    SUB(t4, t4, t5);  ADD(t5, Y1, Z1);  ADD(X3, Y2, Z2);
    MUL(t5, t5, X3);  ADD(X3, t1, t2);  SUB(t5, t5, X3);
    MUL(Z3, c->a, t4); MUL(X3, c->b3, t2); ADD(Z3, X3, Z3);
    SUB(X3, t1, Z3);  ADD(Z3, t1, Z3);  MUL(Y3, X3, Z3);
    ADD(t1, t0, t0);  ADD(t1, t1, t0);  MUL(t2, c->a, t2);
    MUL(t4, c->b3, t4); ADD(t1, t1, t2); SUB(t2, t0, t2);
    MUL(t2, c->a, t2); ADD(t4, t4, t2); MUL(t2, t1, t4);
    ADD(Y3, Y3, t2);  MUL(t2, t5, t4);  MUL(X3, t3, X3);
    SUB(X3, X3, t2);  MUL(t2, t3, t1);  MUL(Z3, t5, Z3);
    ADD(Z3, Z3, t2);
#undef MUL
#undef ADD
#undef SUB
    crypto_bn_copy(X3o, X3, n);
    crypto_bn_copy(Y3o, Y3, n);
    crypto_bn_copy(Z3o, Z3, n);
    return 0;
}

/* R = k * P by left-to-right double-and-add with point_add only, from the
 * identity (0:one:0). k has n limbs. R must not alias P. */
static int scalar_mul(limb_t *RX, limb_t *RY, limb_t *RZ,
                      const limb_t *PX, const limb_t *PY, const limb_t *PZ,
                      const limb_t *k, const limb_t *one, const curve_ctx *c) {
    const size_t n = c->n;
    crypto_bn_set_small(RX, 0, n);
    crypto_bn_copy(RY, one, n);
    crypto_bn_set_small(RZ, 0, n);
    for (size_t bit = crypto_bn_bit_length(k, n); bit-- > 0;) {
        if (point_add(RX, RY, RZ, RX, RY, RZ, RX, RY, RZ, c) != 0) return -1;
        if ((k[bit / 64] >> (bit % 64)) & 1) {
            if (point_add(RX, RY, RZ, RX, RY, RZ, PX, PY, PZ, c) != 0) return -1;
        }
    }
    return 0;
}

/* value < m, or value < m when padded from `width` limbs to n. */
static int fits_below(const limb_t *v, size_t width, const limb_t *m, size_t n) {
    return crypto_bn_significant(v, width) <= n && crypto_bn_compare(v, m, n) < 0;
}

/* Checks one step's arithmetic. Chain links are checked by the caller. */
static int check_step(const crypto_ecpp_step *s, uint64_t *used, uint64_t max_ops) {
    if (!s->n || !s->a || !s->x || !s->y || !s->k || !s->q || !valid_count(s->limbs)) {
        return fail(EINVAL);
    }
    const size_t width = s->limbs;
    const limb_t *big_n = s->n;
    const size_t n = crypto_bn_significant(big_n, width);
    if (!(big_n[0] & 1) || crypto_bn_mod3(big_n, n) == 0 || (n == 1 && big_n[0] <= 3)) {
        return fail(EDOM);
    }
    if (!fits_below(s->a, width, big_n, n) || !fits_below(s->x, width, big_n, n) ||
        !fits_below(s->y, width, big_n, n) || !fits_below(s->k, width, big_n, n) ||
        !fits_below(s->q, width, big_n, n) || crypto_bn_is_zero(s->k, n)) {
        return fail(EDOM);
    }

    /* q >= (floor(N^(1/4)) + 2)^2 > (N^(1/4) + 1)^2. */
    limb_t root[n], root4[n], two[n], wide[2 * n];
    crypto_bn_isqrt(root, big_n, n);
    crypto_bn_isqrt(root4, root, n);
    crypto_bn_set_small(two, 2, n);
    (void)crypto_bn_add(root4, root4, two, n);         /* <= 2^(16n) + 2: no carry */
    crypto_bn_mul_full(wide, root4, root4, n);
    if (!crypto_bn_is_zero(wide + n, n) || crypto_bn_compare(s->q, wide, n) < 0) return fail(EDOM);

    const uint64_t ops = 2 * ((uint64_t)crypto_bn_bit_length(s->k, n) +
                              crypto_bn_bit_length(s->q, n));
    *used += ops;
    if (max_ops != 0 && *used > max_ops) return fail(ERANGE);

    /* Montgomery form of a, x, y; b = y^2 - x^3 - a x; 3b. */
    limb_t one[n], r2[n], am[n], xm[n], ym[n], bm[n], b3m[n], t[n], u[n];
    crypto_bn_montgomery_constants(one, r2, big_n, n);
#define MUL(r, x, y) do { if (crypto_bn_mont_mul((r), (x), (y), big_n, n, wide) != 0) return -1; } while (0)
    MUL(am, s->a, r2);
    MUL(xm, s->x, r2);
    MUL(ym, s->y, r2);
    MUL(bm, ym, ym);                                   /* y^2 */
    MUL(t, xm, xm);
    MUL(t, t, xm);                                     /* x^3 */
    crypto_bn_sub_mod(bm, bm, t, big_n, n);
    MUL(t, am, xm);                                    /* a x */
    crypto_bn_sub_mod(bm, bm, t, big_n, n);
    crypto_bn_add_mod(b3m, bm, bm, big_n, n);
    crypto_bn_add_mod(b3m, b3m, bm, big_n, n);

    /* Non-singular modulo every prime factor: gcd(4a^3 + 27b^2, N) = 1.
     * Montgomery form multiplies by R, a unit, so the gcd is unchanged. */
    MUL(t, am, am);
    MUL(t, t, am);                                     /* a^3 */
    crypto_bn_add_mod(t, t, t, big_n, n);
    crypto_bn_add_mod(t, t, t, big_n, n);              /* 4a^3 */
    MUL(u, bm, bm);                                    /* b^2 */
    limb_t acc[n];
    crypto_bn_set_small(acc, 0, n);
    for (int i = 0; i < 27; ++i) crypto_bn_add_mod(acc, acc, u, big_n, n);
    crypto_bn_add_mod(t, t, acc, big_n, n);
    if (!crypto_bn_coprime_odd(t, big_n, n)) return fail(EDOM);
#undef MUL

    curve_ctx c = {big_n, n, am, b3m, wide};
    limb_t QX[n], QY[n], QZ[n], RX[n], RY[n], RZ[n];
    if (scalar_mul(QX, QY, QZ, xm, ym, one, s->k, one, &c) != 0) return -1;
    /* Q is a proper point, not the identity, modulo every prime factor. */
    if (!crypto_bn_coprime_odd(QZ, big_n, n)) return fail(EDOM);
    if (scalar_mul(RX, RY, RZ, QX, QY, QZ, s->q, one, &c) != 0) return -1;
    /* qQ = (0 : unit : 0): the identity modulo every prime factor, with no
     * collapse to (0:0:0) anywhere. */
    if (!crypto_bn_is_zero(RZ, n) || !crypto_bn_coprime_odd(RY, big_n, n)) return fail(EDOM);
    return 0;
}

int crypto_ecpp_verify(const crypto_ecpp_step *steps, size_t count, uint64_t max_point_ops,
                       size_t *failed_step) {
    if (!steps || count == 0 || count > CRYPTO_ECPP_MAX_STEPS) return fail(EINVAL);
    uint64_t used = 0;
    for (size_t i = 0; i < count; ++i) {
        const crypto_ecpp_step *s = &steps[i];
        if (check_step(s, &used, max_point_ops) != 0) {
            if (failed_step) *failed_step = i;
            return -1;
        }
        int bad;
        if (i + 1 < count) {
            const crypto_ecpp_step *next = &steps[i + 1];
            bad = !next->n || !valid_count(next->limbs) ||
                  !crypto_bn_same_value(next->n, next->limbs, s->q, s->limbs);
            if (bad) errno = EINVAL;
        } else if (crypto_bn_significant(s->q, s->limbs) != 1) {
            bad = 1;
            errno = EINVAL;                            /* chain ends above 2^64 */
        } else {
            bad = !crypto_is_prime_u64(s->q[0]);
            if (bad) errno = EDOM;
        }
        if (bad) {
            if (failed_step) *failed_step = i;
            return -1;
        }
    }
    errno = 0;
    return 0;
}

int crypto_ecpp_scalar_mul(limb_t *X, limb_t *Y, limb_t *Z, const limb_t *a, const limb_t *b,
                           const limb_t *x, const limb_t *y, const limb_t *k, const limb_t *m,
                           size_t n) {
    if (!X || !Y || !Z || !a || !b || !x || !y || !k || !m || !valid_count(n) || !(m[0] & 1) ||
        (crypto_bn_significant(m, n) == 1 && m[0] < 3)) {
        return fail(EINVAL);
    }
    if (crypto_bn_compare(a, m, n) >= 0 || crypto_bn_compare(b, m, n) >= 0 ||
        crypto_bn_compare(x, m, n) >= 0 || crypto_bn_compare(y, m, n) >= 0) {
        return fail(EINVAL);
    }
    limb_t one[n], r2[n], am[n], bm[n], b3m[n], xm[n], ym[n], wide[2 * n], plain[n];
    limb_t RX[n], RY[n], RZ[n];
    crypto_bn_montgomery_constants(one, r2, m, n);
#define MUL(r, p, q) do { if (crypto_bn_mont_mul((r), (p), (q), m, n, wide) != 0) return -1; } while (0)
    MUL(am, a, r2);
    MUL(bm, b, r2);
    MUL(xm, x, r2);
    MUL(ym, y, r2);
    crypto_bn_add_mod(b3m, bm, bm, m, n);
    crypto_bn_add_mod(b3m, b3m, bm, m, n);
    curve_ctx c = {m, n, am, b3m, wide};
    if (scalar_mul(RX, RY, RZ, xm, ym, one, k, one, &c) != 0) return -1;
    crypto_bn_set_small(plain, 1, n);
    MUL(X, RX, plain);
    MUL(Y, RY, plain);
    MUL(Z, RZ, plain);
#undef MUL
    errno = 0;
    return 0;
}
