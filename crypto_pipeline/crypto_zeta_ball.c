/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:d668712c-dcb7-4b87-aa98-27ae57b5c611
 */

/* Ball arithmetic over MPFR. Each operation bounds two errors separately:
 *  - propagation: how far f(x) can move as x ranges over the input ball,
 *    from an explicit derivative bound, computed in radius arithmetic
 *    (ZZ_RAD_PREC bits, every step rounded upward, all terms nonnegative;
 *    a quantity that is subtracted is first bounded in the right direction);
 *  - rounding: MPFR rounds each result correctly (to nearest), so when it
 *    reports an inexact result the error is below half an ulp of the exact
 *    value's binade; one ulp of the rounded midpoint, 2^(EXP(y) - PREC(y)),
 *    is added, which covers that even when rounding crosses a power of 2.
 *    An inexact zero (underflow) adds the smallest positive number instead.
 * Radius temporaries are fixed-precision stack variables (MPFR_DECL_INIT, see
 * the MPFR manual), so most operations make no heap allocation. */

#include "crypto_zeta_ball.h"

/* r += the rounding error of y, given MPFR's ternary value for y. */
static void add_rounding(mpfr_t r, const mpfr_t y, int ternary) {
    if (ternary == 0) {
        return;
    }
    MPFR_DECL_INIT(u, ZZ_RAD_PREC);
    if (mpfr_zero_p(y)) {
        mpfr_set_ui_2exp(u, 1, mpfr_get_emin() - 1, MPFR_RNDU);
    } else {
        mpfr_set_ui_2exp(u, 1, mpfr_get_exp(y) - (mpfr_exp_t)mpfr_get_prec(y), MPFR_RNDU);
    }
    mpfr_add(r, r, u, MPFR_RNDU);
}

void zz_init(zz_ball *x, mpfr_prec_t prec) {
    mpfr_init2(x->m, prec);
    mpfr_init2(x->r, ZZ_RAD_PREC);
    mpfr_set_zero(x->m, 1);
    mpfr_set_zero(x->r, 1);
}

void zz_clear(zz_ball *x) {
    mpfr_clear(x->m);
    mpfr_clear(x->r);
}

void zz_cinit(zz_cball *z, mpfr_prec_t prec) {
    zz_init(&z->re, prec);
    zz_init(&z->im, prec);
}

void zz_cclear(zz_cball *z) {
    zz_clear(&z->re);
    zz_clear(&z->im);
}

void zz_set(zz_ball *y, const zz_ball *x) {
    if (y == x) {
        return;
    }
    int t = mpfr_set(y->m, x->m, MPFR_RNDN);
    mpfr_set(y->r, x->r, MPFR_RNDU);
    add_rounding(y->r, y->m, t);
}

void zz_set_fr(zz_ball *y, const mpfr_t v) {
    int t = mpfr_set(y->m, v, MPFR_RNDN);
    mpfr_set_zero(y->r, 1);
    add_rounding(y->r, y->m, t);
}

void zz_set_d(zz_ball *y, double v) {
    int t = mpfr_set_d(y->m, v, MPFR_RNDN);
    mpfr_set_zero(y->r, 1);
    add_rounding(y->r, y->m, t);
}

void zz_set_si_frac(zz_ball *y, long p, long q) {
    MPFR_DECL_INIT(num, 64);
    mpfr_set_si(num, p, MPFR_RNDN); /* exact: 64 bits hold any long */
    int t = mpfr_div_si(y->m, num, q, MPFR_RNDN);
    mpfr_set_zero(y->r, 1);
    add_rounding(y->r, y->m, t);
}

void zz_set_pi(zz_ball *y) {
    int t = mpfr_const_pi(y->m, MPFR_RNDN);
    mpfr_set_zero(y->r, 1);
    add_rounding(y->r, y->m, t);
}

void zz_add_error(zz_ball *y, const mpfr_t rad) {
    mpfr_add(y->r, y->r, rad, MPFR_RNDU);
}

void zz_upper(mpfr_t u, const zz_ball *x) {
    mpfr_add(u, x->m, x->r, MPFR_RNDU);
}

void zz_lower(mpfr_t l, const zz_ball *x) {
    mpfr_sub(l, x->m, x->r, MPFR_RNDD);
}

void zz_abs_upper(mpfr_t u, const zz_ball *x) {
    MPFR_DECL_INIT(a, ZZ_RAD_PREC);
    mpfr_abs(a, x->m, MPFR_RNDU);
    mpfr_add(u, a, x->r, MPFR_RNDU);
}

void zz_abs_lower(mpfr_t l, const zz_ball *x) {
    MPFR_DECL_INIT(a, ZZ_RAD_PREC);
    mpfr_abs(a, x->m, MPFR_RNDD);
    mpfr_sub(l, a, x->r, MPFR_RNDD);
    if (mpfr_sgn(l) < 0) {
        mpfr_set_zero(l, 1);
    }
}

int zz_sign(const zz_ball *x) {
    MPFR_DECL_INIT(a, ZZ_RAD_PREC);
    zz_abs_lower(a, x);
    int s = mpfr_sgn(a) > 0 ? mpfr_sgn(x->m) : 0;
    return s;
}

void zz_add(zz_ball *z, const zz_ball *x, const zz_ball *y) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_add(r, x->r, y->r, MPFR_RNDU);
    int t = mpfr_add(z->m, x->m, y->m, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

void zz_sub(zz_ball *z, const zz_ball *x, const zz_ball *y) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_add(r, x->r, y->r, MPFR_RNDU);
    int t = mpfr_sub(z->m, x->m, y->m, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

/* |xy - mx my| <= |mx| ry + |my| rx + rx ry. */
void zz_mul(zz_ball *z, const zz_ball *x, const zz_ball *y) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    MPFR_DECL_INIT(a, ZZ_RAD_PREC);
    MPFR_DECL_INIT(b, ZZ_RAD_PREC);
    mpfr_abs(a, x->m, MPFR_RNDU);
    mpfr_mul(a, a, y->r, MPFR_RNDU);
    mpfr_abs(b, y->m, MPFR_RNDU);
    mpfr_mul(b, b, x->r, MPFR_RNDU);
    mpfr_mul(r, x->r, y->r, MPFR_RNDU);
    mpfr_add(r, r, a, MPFR_RNDU);
    mpfr_add(r, r, b, MPFR_RNDU);
    int t = mpfr_mul(z->m, x->m, y->m, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

void zz_mul_si(zz_ball *z, const zz_ball *x, long k) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_mul_ui(r, x->r, k < 0 ? (unsigned long)(-(k + 1)) + 1u : (unsigned long)k, MPFR_RNDU);
    int t = mpfr_mul_si(z->m, x->m, k, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

void zz_div_si(zz_ball *z, const zz_ball *x, long k) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_div_ui(r, x->r, k < 0 ? (unsigned long)(-(k + 1)) + 1u : (unsigned long)k, MPFR_RNDU);
    int t = mpfr_div_si(z->m, x->m, k, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

void zz_mul_2si(zz_ball *z, const zz_ball *x, long e) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_mul_2si(r, x->r, e, MPFR_RNDU);
    int t = mpfr_mul_2si(z->m, x->m, e, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

void zz_neg(zz_ball *z, const zz_ball *x) {
    int t = mpfr_neg(z->m, x->m, MPFR_RNDN);
    if (z != x) {
        mpfr_set(z->r, x->r, MPFR_RNDU);
    }
    add_rounding(z->r, z->m, t);
}

/* |1/x - 1/m| = |x - m| / (|x| |m|) <= r / ((|m| - r) |m|). */
int zz_inv(zz_ball *z, const zz_ball *x) {
    MPFR_DECL_INIT(lo, ZZ_RAD_PREC);
    MPFR_DECL_INIT(am, ZZ_RAD_PREC);
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    int rc = -1;
    mpfr_abs(am, x->m, MPFR_RNDD);
    mpfr_sub(lo, am, x->r, MPFR_RNDD);
    if (mpfr_sgn(lo) > 0) {
        mpfr_mul(lo, lo, am, MPFR_RNDD);
        mpfr_div(r, x->r, lo, MPFR_RNDU);
        int t = mpfr_ui_div(z->m, 1, x->m, MPFR_RNDN);
        mpfr_set(z->r, r, MPFR_RNDU);
        add_rounding(z->r, z->m, t);
        rc = 0;
    }
    return rc;
}

int zz_div(zz_ball *z, const zz_ball *x, const zz_ball *y) {
    zz_ball w;
    zz_init(&w, mpfr_get_prec(z->m) + 16);
    int rc = zz_inv(&w, y);
    if (rc == 0) {
        zz_mul(z, x, &w);
    }
    zz_clear(&w);
    return rc;
}

/* |sqrt(x) - sqrt(m)| = |x - m| / (sqrt(x) + sqrt(m)) <= r / sqrt(m - r). */
int zz_sqrt(zz_ball *z, const zz_ball *x) {
    MPFR_DECL_INIT(lo, ZZ_RAD_PREC);
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    int rc = -1;
    mpfr_sub(lo, x->m, x->r, MPFR_RNDD);
    if (mpfr_sgn(lo) > 0) {
        mpfr_sqrt(lo, lo, MPFR_RNDD);
        mpfr_div(r, x->r, lo, MPFR_RNDU);
        int t = mpfr_sqrt(z->m, x->m, MPFR_RNDN);
        mpfr_set(z->r, r, MPFR_RNDU);
        add_rounding(z->r, z->m, t);
        rc = 0;
    }
    return rc;
}

/* |e^x - e^m| <= e^m (e^r - 1); e^m <= |y| + ulp(y) for the rounded y. */
void zz_exp(zz_ball *z, const zz_ball *x) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    MPFR_DECL_INIT(e, ZZ_RAD_PREC);
    mpfr_expm1(r, x->r, MPFR_RNDU);
    int t = mpfr_exp(z->m, x->m, MPFR_RNDN);
    mpfr_set_zero(e, 1);
    add_rounding(e, z->m, t != 0 ? 1 : 0);
    mpfr_add(e, e, z->m, MPFR_RNDU); /* e^m > 0, so |y| = y */
    mpfr_mul(r, r, e, MPFR_RNDU);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

/* |log x - log m| <= r / (m - r). */
int zz_log(zz_ball *z, const zz_ball *x) {
    MPFR_DECL_INIT(lo, ZZ_RAD_PREC);
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    int rc = -1;
    mpfr_sub(lo, x->m, x->r, MPFR_RNDD);
    if (mpfr_sgn(lo) > 0) {
        mpfr_div(r, x->r, lo, MPFR_RNDU);
        int t = mpfr_log(z->m, x->m, MPFR_RNDN);
        mpfr_set(z->r, r, MPFR_RNDU);
        add_rounding(z->r, z->m, t);
        rc = 0;
    }
    return rc;
}

/* |sin'|, |cos'| <= 1. mpfr_sin_cos returns s + 4c, where s (resp. c) is 0
 * when the sine (resp. cosine) is exact. */
void zz_sin_cos(zz_ball *s, zz_ball *c, const zz_ball *x) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_set(r, x->r, MPFR_RNDU);
    int t = mpfr_sin_cos(s->m, c->m, x->m, MPFR_RNDN);
    int ts = t & 3;
    int tc = (t >> 2) & 3;
    mpfr_set(s->r, r, MPFR_RNDU);
    mpfr_set(c->r, r, MPFR_RNDU);
    add_rounding(s->r, s->m, ts);
    add_rounding(c->r, c->m, tc);
}

void zz_atan(zz_ball *z, const zz_ball *x) {
    MPFR_DECL_INIT(r, ZZ_RAD_PREC);
    mpfr_set(r, x->r, MPFR_RNDU);
    int t = mpfr_atan(z->m, x->m, MPFR_RNDN);
    mpfr_set(z->r, r, MPFR_RNDU);
    add_rounding(z->r, z->m, t);
}

void zz_log_ui(zz_ball *z, unsigned long n) {
    int t = mpfr_log_ui(z->m, n, MPFR_RNDN);
    mpfr_set_zero(z->r, 1);
    add_rounding(z->r, z->m, t);
}

void zz_rec_sqrt_ui(zz_ball *z, unsigned long n) {
    MPFR_DECL_INIT(x, 64);
    mpfr_set_ui(x, n, MPFR_RNDN); /* exact: 64 bits hold any unsigned long */
    int t = mpfr_rec_sqrt(z->m, x, MPFR_RNDN);
    mpfr_set_zero(z->r, 1);
    add_rounding(z->r, z->m, t);
}

void zz_cadd(zz_cball *z, const zz_cball *x, const zz_cball *y) {
    zz_add(&z->re, &x->re, &y->re);
    zz_add(&z->im, &x->im, &y->im);
}

void zz_csub(zz_cball *z, const zz_cball *x, const zz_cball *y) {
    zz_sub(&z->re, &x->re, &y->re);
    zz_sub(&z->im, &x->im, &y->im);
}

void zz_cmul(zz_cball *z, const zz_cball *x, const zz_cball *y) {
    mpfr_prec_t p = mpfr_get_prec(z->re.m);
    zz_ball ac, bd, ad, bc;
    zz_init(&ac, p);
    zz_init(&bd, p);
    zz_init(&ad, p);
    zz_init(&bc, p);
    zz_mul(&ac, &x->re, &y->re);
    zz_mul(&bd, &x->im, &y->im);
    zz_mul(&ad, &x->re, &y->im);
    zz_mul(&bc, &x->im, &y->re);
    zz_sub(&z->re, &ac, &bd);
    zz_add(&z->im, &ad, &bc);
    zz_clear(&ac);
    zz_clear(&bd);
    zz_clear(&ad);
    zz_clear(&bc);
}

void zz_cmul_real(zz_cball *z, const zz_cball *x, const zz_ball *y) {
    zz_mul(&z->re, &x->re, y);
    zz_mul(&z->im, &x->im, y);
}

int zz_cinv(zz_cball *z, const zz_cball *x) {
    mpfr_prec_t p = mpfr_get_prec(z->re.m) + 16;
    zz_ball n, a2, b2;
    zz_init(&n, p);
    zz_init(&a2, p);
    zz_init(&b2, p);
    zz_mul(&a2, &x->re, &x->re);
    zz_mul(&b2, &x->im, &x->im);
    zz_add(&n, &a2, &b2);
    int rc = zz_inv(&n, &n);
    if (rc == 0) {
        zz_neg(&b2, &x->im);
        zz_mul(&z->re, &x->re, &n);
        zz_mul(&z->im, &b2, &n);
    }
    zz_clear(&n);
    zz_clear(&a2);
    zz_clear(&b2);
    return rc;
}

int zz_cdiv(zz_cball *z, const zz_cball *x, const zz_cball *y) {
    zz_cball w;
    zz_cinit(&w, mpfr_get_prec(z->re.m) + 16);
    int rc = zz_cinv(&w, y);
    if (rc == 0) {
        zz_cmul(z, x, &w);
    }
    zz_cclear(&w);
    return rc;
}

/* For Re x > 0 on the whole ball: log x = (1/2) log(a^2 + b^2) + i atan(b/a),
 * and atan(b/a) is the principal argument, in (-pi/2, pi/2). */
int zz_clog(zz_cball *z, const zz_cball *x) {
    if (zz_sign(&x->re) <= 0) {
        return -1;
    }
    mpfr_prec_t p = mpfr_get_prec(z->re.m) + 16;
    zz_ball n, a2, q;
    zz_init(&n, p);
    zz_init(&a2, p);
    zz_init(&q, p);
    zz_mul(&a2, &x->re, &x->re);
    zz_mul(&n, &x->im, &x->im);
    zz_add(&n, &n, &a2);
    int rc = zz_div(&q, &x->im, &x->re);
    if (rc == 0) {
        rc = zz_log(&n, &n);
    }
    if (rc == 0) {
        zz_atan(&z->im, &q);
        zz_mul_2si(&z->re, &n, -1);
    }
    zz_clear(&n);
    zz_clear(&a2);
    zz_clear(&q);
    return rc;
}

void zz_cexp(zz_cball *z, const zz_cball *x) {
    mpfr_prec_t p = mpfr_get_prec(z->re.m) + 16;
    zz_ball e, s, c;
    zz_init(&e, p);
    zz_init(&s, p);
    zz_init(&c, p);
    zz_exp(&e, &x->re);
    zz_sin_cos(&s, &c, &x->im);
    zz_mul(&z->re, &e, &c);
    zz_mul(&z->im, &e, &s);
    zz_clear(&e);
    zz_clear(&s);
    zz_clear(&c);
}

void zz_cabs_upper(mpfr_t u, const zz_cball *x) {
    MPFR_DECL_INIT(a, ZZ_RAD_PREC);
    MPFR_DECL_INIT(b, ZZ_RAD_PREC);
    zz_abs_upper(a, &x->re);
    zz_abs_upper(b, &x->im);
    mpfr_sqr(a, a, MPFR_RNDU);
    mpfr_sqr(b, b, MPFR_RNDU);
    mpfr_add(a, a, b, MPFR_RNDU);
    mpfr_sqrt(u, a, MPFR_RNDU);
}

void zz_cabs_lower(mpfr_t l, const zz_cball *x) {
    MPFR_DECL_INIT(a, ZZ_RAD_PREC);
    MPFR_DECL_INIT(b, ZZ_RAD_PREC);
    zz_abs_lower(a, &x->re);
    zz_abs_lower(b, &x->im);
    mpfr_sqr(a, a, MPFR_RNDD);
    mpfr_sqr(b, b, MPFR_RNDD);
    mpfr_add(a, a, b, MPFR_RNDD);
    mpfr_sqrt(l, a, MPFR_RNDD);
}

void zz_cadd_error(zz_cball *z, const mpfr_t rad) {
    zz_add_error(&z->re, rad);
    zz_add_error(&z->im, rad);
}
