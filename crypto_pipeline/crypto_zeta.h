/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:3c3416c0-0ccc-466d-acab-5b2e2489f373
 */

#ifndef CRYPTO_ZETA_H
#define CRYPTO_ZETA_H

/**
 * @file crypto_zeta.h
 * @brief Rigorous verification that the zeros of zeta up to height T lie on
 * the critical line.
 *
 * Hardy's function Z(t) = e^{i theta(t)} zeta(1/2 + i t) is real for real t,
 * and a sign change of Z between two points proves a zero of zeta on the
 * critical line between them. All values are MPFR balls (crypto_zeta_ball.h)
 * that provably contain the exact value, so every sign used is proven.
 *
 * - theta(t) = Im log Gamma(1/4 + i t/2) - (t/2) log pi, from Stirling's
 *   series with its complex-argument remainder bound (DLMF 5.11.1, 5.11(ii)).
 * - zeta(1/2 + i t) by Euler-Maclaurin summation with the remainder bound
 *   derived from DLMF 2.10.1 and 24.9.1 (ZZ_METHOD_EM uses it for every t).
 * - Z(t) by the Riemann-Siegel formula with the C_0 correction for t >= 200
 *   and Gabcke's remainder bound |R_0(t)| < 0.127 t^(-3/4) (W. Gabcke,
 *   Dissertation, Goettingen 1979, Satz 3.2.2 (b); ZZ_METHOD_RS_GABCKE).
 * - Turing's method with Trudgian's constants (Math. Comp. 80 (2011),
 *   Theorem 2.2: for t2 > t1 > 168 pi, the integral of S(t) from t1 to t2 is
 *   at most 2.067 + 0.059 log t2) bounds N(t0) from above; equality with the
 *   number of sign changes found proves every zero up to t0 is on the line
 *   and simple.
 *
 * See docs/engineering/CRYPTO-ZETA-ZEROS.md. Research code: proofs depend on
 * MPFR's documented correct rounding and on the cited theorems.
 */

#include <stddef.h>

#include "crypto_zeta_ball.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The verification proved its claim. */
#define ZZ_OK 0
/** Invalid arguments or allocation failure. */
#define ZZ_ERR_ARGS (-1)
/** A value could not be enclosed tightly enough to decide its sign. */
#define ZZ_ERR_EVAL (-2)
/** An internal consistency check failed: Im(e^{i theta} zeta) is provably
 *  nonzero, or more sign changes were found than Turing's bound allows. */
#define ZZ_ERR_INCONSISTENT (-3)
/** A Gram block did not yield its expected number of sign changes within
 *  the refinement budget. */
#define ZZ_ERR_ROSSER (-4)
/** Turing's upper bound exceeds the number of sign changes found. */
#define ZZ_ERR_TURING (-5)

/** Euler-Maclaurin everywhere: slower, independent of Gabcke's bound. */
#define ZZ_METHOD_EM 0
/** Riemann-Siegel with Gabcke's remainder bound for t >= 200,
 *  Euler-Maclaurin below 200 and wherever Riemann-Siegel cannot decide the
 *  sign. */
#define ZZ_METHOD_RS_GABCKE 1

/** Gabcke's bound |R_0(t)| < (127/1000) t^(-3/4) for t >= 200, for the
 *  Riemann-Siegel formula with the C_0 term only: W. Gabcke, Neue Herleitung
 *  und explizite Restabschaetzung der Riemann-Siegel-Formel, Dissertation,
 *  Goettingen 1979, Satz 3.2.2 (b), p. 55, and Einleitung (6), (8), p. 3. */
#define ZZ_GABCKE_D0_NUM 127
/** Denominator of Gabcke's constant 0.127. */
#define ZZ_GABCKE_D0_DEN 1000

/** Trudgian (2011), Theorem 2.2: a = 2.067 = 2067/1000. */
#define ZZ_TRUDGIAN_A_NUM 2067
/** Trudgian (2011), Theorem 2.2: b = 0.059 = 59/1000. */
#define ZZ_TRUDGIAN_B_NUM 59
/** Denominator of both Trudgian constants. */
#define ZZ_TRUDGIAN_DEN 1000

/** Deliberate defects for negative tests; all zero in normal use. */
typedef struct zz_faults {
    double theta_offset; /**< added to theta(t) everywhere */
    int drop_term;       /**< 1: Euler-Maclaurin drops its k = 1 Bernoulli
                              term; 2: Riemann-Siegel drops the n = N term
                              of its main sum */
    long skip_interval;  /**< do not count the sign change(s) found in the
                              Gram interval [g_n, g_{n+1}) with this n
                              (ZZ_NO_SKIP for none) */
    double min_t;        /**< theta_offset and drop_term act only where the
                              evaluation point is >= min_t (0: everywhere) */
} zz_faults;

/** Value of zz_faults.skip_interval meaning no interval is skipped. */
#define ZZ_NO_SKIP (-1000000L)

/** Bernoulli numbers B_0..B_{2 ZZ_BERN_K} are held by the context. */
#define ZZ_BERN_K 80

/** Evaluation context: working precision, Bernoulli numbers, counters.
 *  Owned by the caller (no heap allocation of its own); set up with
 *  zz_ctx_init and released with zz_ctx_clear. Fields are internal. */
typedef struct zz_ctx {
    mpfr_prec_t prec;                /**< midpoint precision, bits */
    zz_faults faults;                /**< fault injection (tests only) */
    zz_ball bern_fact[ZZ_BERN_K + 1]; /**< B_2k / (2k)! */
    zz_ball stirling[ZZ_BERN_K + 1];  /**< B_2k / (2k (2k - 1)), k >= 1 */
    zz_ball pi;                      /**< pi */
    zz_ball log_pi;                  /**< log pi */
    long evaluations;                /**< Z evaluations */
    long em_evaluations;             /**< of which Euler-Maclaurin */
    long rs_fallbacks;               /**< RS evaluations that fell back */
    int used_riemann_siegel;         /**< a sign came from Riemann-Siegel */
} zz_ctx;

/** Initializes ctx with prec-bit midpoints (64..4096). Returns ZZ_OK or
 *  ZZ_ERR_ARGS (ctx then needs no zz_ctx_clear). */
int zz_ctx_init(zz_ctx *ctx, mpfr_prec_t prec);
/** Releases a context set up by zz_ctx_init. */
void zz_ctx_clear(zz_ctx *ctx);
/** Installs fault injection (copied); NULL clears it. */
void zz_ctx_set_faults(zz_ctx *ctx, const zz_faults *faults);

/** out = theta(t) for a ball t (t may be wide; t > 0 on the whole ball is
 *  not required, but |t| must stay below 2^40). Returns ZZ_OK or an error. */
int zz_theta(zz_ctx *ctx, zz_ball *out, const zz_ball *t);

/** out = zeta(1/2 + i t) by Euler-Maclaurin, 0 < t < 2^40 (ball). */
int zz_zeta_em(zz_ctx *ctx, zz_cball *out, const zz_ball *t);

/** out = Z(t) by Euler-Maclaurin. Fails with ZZ_ERR_INCONSISTENT if the
 *  enclosure of Im(e^{i theta} zeta) excludes 0 (which happens if theta or
 *  zeta is wrong by more than the claimed error). */
int zz_z_em(zz_ctx *ctx, zz_ball *out, const zz_ball *t);

/** out = Z(t) by Riemann-Siegel with C_0 and the radius widened by
 *  Gabcke's bound 0.127 t^(-3/4) (see above). Requires t >= 200.
 *  Fails with ZZ_ERR_EVAL when floor(sqrt(tau)) or C_0 cannot be resolved
 *  (t near a square of sqrt(2 pi) n, or p near 1/4 or 3/4). */
int zz_z_rs(zz_ctx *ctx, zz_ball *out, const zz_ball *t);

/** Proven sign of Z at the exact double t > 0: +1 or -1, or 0 if undecided.
 *  method is ZZ_METHOD_EM or ZZ_METHOD_RS_GABCKE. *err receives ZZ_OK or
 *  the error that stopped evaluation (may be NULL). */
int zz_z_sign(zz_ctx *ctx, double t, int method, int *err);

/** Encloses the n-th Gram point (theta(g_n) = n pi, n >= -1, the root above
 *  t = 7) in [lo, hi] with theta(lo) < n pi < theta(hi) proven, and width
 *  below 2^-(prec/2) relative. Returns ZZ_OK or an error. */
int zz_gram_point(zz_ctx *ctx, long n, mpfr_t lo, mpfr_t hi);

/** Proven sign of Z on the whole interval [lo, hi] (evaluated as one ball):
 *  +1, -1, or 0 if undecided. Uses Euler-Maclaurin. */
int zz_z_sign_interval(zz_ctx *ctx, const mpfr_t lo, const mpfr_t hi);

/** A sample point of Z with its proven sign (workspace element). */
typedef struct zz_sample {
    double t;  /**< exact sample point */
    int sign;  /**< +1 or -1 */
    long gram; /**< Gram index n if t approximates g_n, else ZZ_NOT_GRAM */
} zz_sample;

/** zz_sample.gram for a point that is not a Gram point. */
#define ZZ_NOT_GRAM (-2000000000L)

/** A proven sign change: Z(lo) and Z(hi) have opposite proven signs. */
typedef struct zz_bracket {
    double lo; /**< left sample point (exact double) */
    double hi; /**< right sample point (exact double) */
} zz_bracket;

/** Parameters of zz_verify. All storage is provided by the caller. */
typedef struct zz_verify_params {
    double height;             /**< T: prove all zeros with 0 < t <= T are on
                                    the critical line; 0 < T <= 1e6 */
    const double *checkpoints; /**< heights <= T at which to report N(h) */
    size_t n_checkpoints;      /**< number of checkpoints (0..64) */
    int method;                /**< ZZ_METHOD_EM or ZZ_METHOD_RS_GABCKE */
    long turing_intervals;     /**< Gram intervals in Turing's range (>= 8) */
    mpfr_prec_t prec;          /**< working precision, bits (64..1024) */
    zz_faults faults;          /**< fault injection for negative tests */
    zz_sample *work;           /**< workspace (see zz_verify_work_size) */
    size_t work_cap;           /**< elements in work */
    zz_bracket *brackets;      /**< optional output: sign changes found */
    size_t brackets_cap;       /**< elements in brackets (0 if NULL) */
    long *bad_gram;            /**< optional output: bad Gram indices */
    size_t bad_gram_cap;       /**< elements in bad_gram (0 if NULL) */
} zz_verify_params;

/** Result of zz_verify. */
typedef struct zz_verify_result {
    int status;               /**< ZZ_OK or an error code */
    const char *reason;       /**< static description when status != 0 */
    double fail_t;            /**< where the failure was detected */
    long zeros_to_height;     /**< N(T), valid when status == ZZ_OK */
    long checkpoint_counts[64]; /**< N(checkpoints[i]) when status == ZZ_OK */
    double t0;                /**< Turing point (a sample, a Gram point >= T) */
    double t1;                /**< end of Turing's range */
    long gram_t0;             /**< index n of the Gram point t0 */
    long sign_changes_to_t0;  /**< sign changes found in (0, t0] */
    long turing_bound;        /**< floor of the proven upper bound of N(t0) */
    double turing_upper;      /**< that upper bound, rounded up to double */
    size_t n_brackets;        /**< sign changes found up to t1 (the first
                                   brackets_cap are stored) */
    size_t n_bad_gram;        /**< n <= gram_t0 with (-1)^n Z(~g_n) < 0 at
                                   the sampled approximate Gram points (the
                                   first bad_gram_cap are stored) */
    long evaluations;         /**< Z evaluations */
    long em_evaluations;      /**< of which by Euler-Maclaurin */
    long rs_fallbacks;        /**< RS evaluations that fell back to EM */
    int used_riemann_siegel;  /**< 1 if any sign came from Riemann-Siegel */
} zz_verify_result;

/** Runs the verification. Returns result->status. On ZZ_OK every zero of
 *  zeta with 0 < t <= result->t0 (>= T) is on the critical line and simple,
 *  and zeros_to_height is N(T) exactly. Fails with
 *  ZZ_ERR_ARGS ("workspace too small") if work_cap is exhausted. */
int zz_verify(const zz_verify_params *params, zz_verify_result *result);

/** A workspace size that suffices for zz_verify at this height unless a
 *  Gram block needs unusually deep refinement: twice the number of Gram
 *  points up to the end of Turing's range, plus 2^16 for refinement. */
size_t zz_verify_work_size(double height, long turing_intervals);

/** Fills params with defaults: method EM, 40 Turing intervals, 128 bits,
 *  no checkpoints, no workspace or outputs (the caller must set work). */
void zz_verify_params_default(zz_verify_params *params, double height);

#ifdef __cplusplus
}
#endif
#endif
