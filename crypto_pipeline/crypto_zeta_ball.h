/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:1546e4bb-bbfc-41b1-9105-423c00399941
 */

#ifndef CRYPTO_ZETA_BALL_H
#define CRYPTO_ZETA_BALL_H

/**
 * @file crypto_zeta_ball.h
 * @brief Midpoint-radius ("ball") arithmetic over MPFR with proven enclosures.
 *
 * A ball [m +/- r] stands for every real x with |x - m| <= r. Every operation
 * returns a ball that contains f(x) for every x in the input balls: the
 * radius adds a bound on how far f moves over the inputs (from a derivative
 * bound) and, when MPFR reports an inexact result (nonzero ternary value), one
 * unit in the last place of the rounded midpoint, which bounds MPFR's
 * correctly rounded error. Radii are 64-bit MPFR numbers rounded upward.
 *
 * Functions return 0, or -1 when the result is not defined on the whole input
 * ball (division by a ball containing 0, log or sqrt of a ball reaching 0 or
 * below, a complex log off the right half-plane). On -1 the output is left in
 * an unspecified but valid (initialized) state. Outputs may alias inputs.
 * See docs/engineering/CRYPTO-ZETA-ZEROS.md.
 */

#include <mpfr.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Precision of radii, in bits. */
#define ZZ_RAD_PREC 64

/** A real ball: every x with |x - m| <= r. Invariant: r >= 0, finite. */
typedef struct zz_ball {
    mpfr_t m; /**< midpoint, at the ball's working precision */
    mpfr_t r; /**< radius, ZZ_RAD_PREC bits, an upper bound */
} zz_ball;

/** A complex ball: a rectangle given by two real balls. */
typedef struct zz_cball {
    zz_ball re; /**< real part */
    zz_ball im; /**< imaginary part */
} zz_cball;

/** Initializes x = [0 +/- 0] with a midpoint of prec bits. */
void zz_init(zz_ball *x, mpfr_prec_t prec);
/** Releases x. */
void zz_clear(zz_ball *x);
/** Initializes z = 0 with midpoints of prec bits. */
void zz_cinit(zz_cball *z, mpfr_prec_t prec);
/** Releases z. */
void zz_cclear(zz_cball *z);

/** y = x (the radius absorbs any rounding to y's precision). */
void zz_set(zz_ball *y, const zz_ball *x);
/** y = the exact value v (rounded midpoint plus rounding error). */
void zz_set_fr(zz_ball *y, const mpfr_t v);
/** y = the exact double v. */
void zz_set_d(zz_ball *y, double v);
/** y = p / q exactly enclosed; q != 0. */
void zz_set_si_frac(zz_ball *y, long p, long q);
/** y = pi. */
void zz_set_pi(zz_ball *y);
/** y = [m +/- (r + rad)]: widens x by rad >= 0. */
void zz_add_error(zz_ball *y, const mpfr_t rad);

/** z = x + y. */
void zz_add(zz_ball *z, const zz_ball *x, const zz_ball *y);
/** z = x - y. */
void zz_sub(zz_ball *z, const zz_ball *x, const zz_ball *y);
/** z = x * y. */
void zz_mul(zz_ball *z, const zz_ball *x, const zz_ball *y);
/** z = x * k for an integer k. */
void zz_mul_si(zz_ball *z, const zz_ball *x, long k);
/** z = x / k for an integer k != 0. */
void zz_div_si(zz_ball *z, const zz_ball *x, long k);
/** z = x * 2^e (exact on the midpoint unless it underflows). */
void zz_mul_2si(zz_ball *z, const zz_ball *x, long e);
/** z = -x. */
void zz_neg(zz_ball *z, const zz_ball *x);
/** z = 1 / x; -1 if x contains 0. */
int zz_inv(zz_ball *z, const zz_ball *x);
/** z = x / y; -1 if y contains 0. */
int zz_div(zz_ball *z, const zz_ball *x, const zz_ball *y);
/** z = sqrt(x); -1 unless x > 0 on the whole ball. */
int zz_sqrt(zz_ball *z, const zz_ball *x);
/** z = exp(x). */
void zz_exp(zz_ball *z, const zz_ball *x);
/** z = log(x); -1 unless x > 0 on the whole ball. */
int zz_log(zz_ball *z, const zz_ball *x);
/** s = sin(x), c = cos(x); s and c must be distinct. */
void zz_sin_cos(zz_ball *s, zz_ball *c, const zz_ball *x);
/** z = atan(x). */
void zz_atan(zz_ball *z, const zz_ball *x);
/** z = log(n) for an integer n >= 1 (correctly rounded, exact input). */
void zz_log_ui(zz_ball *z, unsigned long n);
/** z = n^(-1/2) for an integer n >= 1 (correctly rounded, exact input). */
void zz_rec_sqrt_ui(zz_ball *z, unsigned long n);

/** Sign of every point of x: +1, -1, or 0 if x contains 0. */
int zz_sign(const zz_ball *x);
/** u = an upper bound of every point of x, rounded up to u's precision. */
void zz_upper(mpfr_t u, const zz_ball *x);
/** l = a lower bound of every point of x, rounded down to l's precision. */
void zz_lower(mpfr_t l, const zz_ball *x);
/** u = an upper bound of |x| over the ball, rounded up. */
void zz_abs_upper(mpfr_t u, const zz_ball *x);
/** l = a lower bound of |x| over the ball (0 if it contains 0), rounded down. */
void zz_abs_lower(mpfr_t l, const zz_ball *x);

/** z = x + y (complex). */
void zz_cadd(zz_cball *z, const zz_cball *x, const zz_cball *y);
/** z = x - y (complex). */
void zz_csub(zz_cball *z, const zz_cball *x, const zz_cball *y);
/** z = x * y (complex). */
void zz_cmul(zz_cball *z, const zz_cball *x, const zz_cball *y);
/** z = x * y for a real ball y. */
void zz_cmul_real(zz_cball *z, const zz_cball *x, const zz_ball *y);
/** z = 1 / x; -1 if |x|^2 may be 0. */
int zz_cinv(zz_cball *z, const zz_cball *x);
/** z = x / y; -1 if |y|^2 may be 0. */
int zz_cdiv(zz_cball *z, const zz_cball *x, const zz_cball *y);
/** z = principal log(x); -1 unless Re x > 0 on the whole ball. */
int zz_clog(zz_cball *z, const zz_cball *x);
/** z = exp(x) (complex). */
void zz_cexp(zz_cball *z, const zz_cball *x);
/** u = an upper bound of |x| over the ball, rounded up. */
void zz_cabs_upper(mpfr_t u, const zz_cball *x);
/** l = a lower bound of |x| over the ball, rounded down (may be 0). */
void zz_cabs_lower(mpfr_t l, const zz_cball *x);
/** Widens both parts of z by rad: encloses z + e for every complex |e| <= rad. */
void zz_cadd_error(zz_cball *z, const mpfr_t rad);

#ifdef __cplusplus
}
#endif
#endif
