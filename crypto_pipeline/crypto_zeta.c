/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:3d6b677d-a027-4fea-a06e-2eec28a2fdbe
 */

/* Verification that the zeros of zeta up to a height T lie on the critical
 * line. Every quantity that decides the result is a ball from
 * crypto_zeta_ball.h; doubles appear only as exact sample points and in
 * heuristics that choose where to sample (which cannot affect soundness).
 * See crypto_zeta.h and docs/engineering/CRYPTO-ZETA-ZEROS.md. */

#include "crypto_zeta.h"

#include <math.h>

/* No heap allocation here: the context and all arrays belong to the caller
 * (MPFR and GMP manage the storage of their own numbers). */

#define BERN_K ZZ_BERN_K
#define BERN_COUNT (2 * ZZ_BERN_K + 1)
/* Stirling series: terms k = 1..STIRLING_N-1, remainder from k = STIRLING_N. */
#define STIRLING_N 20
/* Shift log Gamma's argument to |w| >= 40 before using Stirling's series. */
#define STIRLING_SHIFT 40
/* Euler-Maclaurin: stop adding terms when the remainder bound is below
 * 2^-EM_TARGET_BITS; fail if it never gets below 2^-EM_MAX_BITS. */
#define EM_TARGET_BITS 80
#define EM_MAX_BITS 40
/* Rosser-block refinement: bisection rounds per block (each round splits
 * every subinterval, since one sign change may hide three zeros). */
#define ROSSER_ROUNDS 12
/* Heights accepted by zz_verify. */
#define VERIFY_MAX_HEIGHT 1e6
/* RS mode: every RS_SPOT_CHECK-th Gram sample at t >= 200 is also evaluated
 * by Euler-Maclaurin, as a consistency check (see rs_spot_check). */
#define RS_SPOT_CHECK 64
/* Pieces per unit length in Turing's integral of theta. */
#define TURING_PIECES_PER_UNIT 64

/* ---- Bernoulli numbers (exact, Akiyama-Tanigawa) ---------------------- */

static void bernoulli_exact(mpq_t *b, int count) {
    mpq_t a[BERN_COUNT];
    mpq_t d;
    mpq_init(d);
    for (int m = 0; m < count; m++) {
        mpq_init(a[m]);
        mpq_set_ui(a[m], 1, (unsigned long)m + 1u);
        for (int j = m; j >= 1; j--) {
            mpq_sub(d, a[j - 1], a[j]);
            mpq_set_ui(a[j - 1], (unsigned long)j, 1);
            mpq_mul(a[j - 1], a[j - 1], d);
        }
        mpq_set(b[m], a[0]); /* B_m with B_1 = +1/2; only even m are used */
    }
    for (int m = 0; m < count; m++) {
        mpq_clear(a[m]);
    }
    mpq_clear(d);
}

static void ball_from_q(zz_ball *y, const mpq_t q) {
    int t = mpfr_set_q(y->m, q, MPFR_RNDN);
    mpfr_set_zero(y->r, 1);
    if (t != 0) {
        mpfr_t u;
        mpfr_init2(u, ZZ_RAD_PREC);
        mpfr_set_ui_2exp(u, 1, mpfr_get_exp(y->m) - (mpfr_exp_t)mpfr_get_prec(y->m), MPFR_RNDU);
        mpfr_add(y->r, y->r, u, MPFR_RNDU);
        mpfr_clear(u);
    }
}

int zz_ctx_init(zz_ctx *ctx, mpfr_prec_t prec) {
    if (ctx == NULL || prec < 64 || prec > 4096) {
        return ZZ_ERR_ARGS;
    }
    ctx->prec = prec;
    zz_ctx_set_faults(ctx, NULL);
    ctx->evaluations = 0;
    ctx->em_evaluations = 0;
    ctx->rs_fallbacks = 0;
    ctx->used_riemann_siegel = 0;
    mpq_t b[BERN_COUNT];
    for (int i = 0; i < BERN_COUNT; i++) {
        mpq_init(b[i]);
    }
    bernoulli_exact(b, BERN_COUNT);
    mpq_t q;
    mpz_t f;
    mpq_init(q);
    mpz_init(f);
    for (int k = 0; k <= BERN_K; k++) {
        zz_init(&ctx->bern_fact[k], prec + 32);
        zz_init(&ctx->stirling[k], prec + 32);
        mpz_fac_ui(f, (unsigned long)(2 * k));
        mpq_set_z(q, f);
        mpq_div(q, b[2 * k], q);
        ball_from_q(&ctx->bern_fact[k], q);
        if (k >= 1) {
            mpq_set_ui(q, (unsigned long)(2 * k * (2 * k - 1)), 1);
            mpq_div(q, b[2 * k], q);
            ball_from_q(&ctx->stirling[k], q);
        }
    }
    mpq_clear(q);
    mpz_clear(f);
    for (int i = 0; i < BERN_COUNT; i++) {
        mpq_clear(b[i]);
    }
    zz_init(&ctx->pi, prec + 32);
    zz_init(&ctx->log_pi, prec + 32);
    zz_set_pi(&ctx->pi);
    (void)zz_log(&ctx->log_pi, &ctx->pi); /* pi > 0: cannot fail */
    return ZZ_OK;
}

void zz_ctx_clear(zz_ctx *ctx) {
    for (int k = 0; k <= BERN_K; k++) {
        zz_clear(&ctx->bern_fact[k]);
        zz_clear(&ctx->stirling[k]);
    }
    zz_clear(&ctx->pi);
    zz_clear(&ctx->log_pi);
}

void zz_ctx_set_faults(zz_ctx *ctx, const zz_faults *faults) {
    if (faults == NULL) {
        ctx->faults.theta_offset = 0.0;
        ctx->faults.drop_term = 0;
        ctx->faults.skip_interval = ZZ_NO_SKIP;
        ctx->faults.min_t = 0.0;
    } else {
        ctx->faults = *faults;
    }
}

/* ---- theta ------------------------------------------------------------- */

int zz_theta(zz_ctx *ctx, zz_ball *out, const zz_ball *t) {
    mpfr_prec_t p = ctx->prec + 32;
    int rc = ZZ_OK;
    mpfr_t tabs, wl, wu, rel, rem;
    mpfr_inits2(ZZ_RAD_PREC, tabs, wl, wu, rel, rem, (mpfr_ptr)0);
    zz_abs_upper(tabs, t);
    if (mpfr_cmp_d(tabs, 1099511627776.0) >= 0) { /* 2^40 */
        mpfr_clears(tabs, wl, wu, rel, rem, (mpfr_ptr)0);
        return ZZ_ERR_ARGS;
    }
    long shift = mpfr_cmp_ui(tabs, 2 * STIRLING_SHIFT) < 0 ? STIRLING_SHIFT : 0;

    zz_cball w, lw, winv, winv2, pw, tmp;
    zz_ball a, b, half_t;
    zz_cinit(&w, p);
    zz_cinit(&lw, p);
    zz_cinit(&winv, p);
    zz_cinit(&winv2, p);
    zz_cinit(&pw, p);
    zz_cinit(&tmp, p);
    zz_init(&a, p);
    zz_init(&b, p);
    zz_init(&half_t, p);

    /* z = 1/4 + i t/2, w = z + shift. */
    zz_mul_2si(&half_t, t, -1);
    zz_set_si_frac(&w.re, 4 * shift + 1, 4);
    zz_set(&w.im, &half_t);
    if (zz_clog(&lw, &w) != 0 || zz_cinv(&winv, &w) != 0) {
        rc = ZZ_ERR_EVAL;
        goto done;
    }
    /* Im[(w - 1/2) log w - w] = Im w Re log w + (Re w - 1/2) Im log w - Im w. */
    zz_mul(&a, &w.im, &lw.re);
    zz_set_si_frac(&b, 4 * shift - 1, 4); /* Re w - 1/2 */
    zz_mul(&b, &b, &lw.im);
    zz_add(&a, &a, &b);
    zz_sub(&a, &a, &w.im);
    /* + sum_{k=1}^{N-1} B_2k / (2k (2k-1)) Im w^(1-2k). */
    zz_cmul(&winv2, &winv, &winv);
    zz_set(&pw.re, &winv.re);
    zz_set(&pw.im, &winv.im);
    for (int k = 1; k < STIRLING_N; k++) {
        zz_mul(&b, &ctx->stirling[k], &pw.im);
        zz_add(&a, &a, &b);
        zz_cmul(&tmp, &pw, &winv2);
        zz_set(&pw.re, &tmp.re);
        zz_set(&pw.im, &tmp.im);
    }
    /* Remainder (DLMF 5.11(ii)): at most sec^(2N)(ph(w)/2) times the first
     * neglected term, |B_2N| / (2N (2N-1) |w|^(2N-1)). With Re w > 0,
     * sec^2(ph(w)/2) = 2|w| / (|w| + Re w). */
    zz_cabs_lower(wl, &w);
    zz_cabs_upper(wu, &w);
    zz_abs_lower(rel, &w.re);
    mpfr_add(rel, rel, wl, MPFR_RNDD);
    mpfr_mul_2ui(rem, wu, 1, MPFR_RNDU);
    mpfr_div(rem, rem, rel, MPFR_RNDU);          /* sec^2 bound */
    mpfr_pow_ui(rem, rem, STIRLING_N, MPFR_RNDU); /* sec^(2N) */
    zz_abs_upper(rel, &ctx->stirling[STIRLING_N]);
    mpfr_mul(rem, rem, rel, MPFR_RNDU);
    mpfr_pow_ui(rel, wl, 2 * STIRLING_N - 1, MPFR_RNDD);
    mpfr_div(rem, rem, rel, MPFR_RNDU);
    zz_add_error(&a, rem);
    /* log Gamma(z) = log Gamma(z + K) - sum_{k<K} log(z + k), and for
     * Re > 0, Im log(z + k) = atan((t/2) / (1/4 + k)). */
    for (long k = 0; k < shift; k++) {
        zz_div_si(&b, &half_t, 1);
        zz_mul_si(&b, &b, 4);
        zz_div_si(&b, &b, 4 * k + 1);
        zz_atan(&b, &b);
        zz_sub(&a, &a, &b);
    }
    /* theta = Im log Gamma(1/4 + i t/2) - (t/2) log pi. */
    zz_mul(&b, &half_t, &ctx->log_pi);
    zz_sub(&a, &a, &b);
    if (ctx->faults.theta_offset != 0.0 && mpfr_cmp_d(t->m, ctx->faults.min_t) >= 0) {
        zz_set_d(&b, ctx->faults.theta_offset);
        zz_add(&a, &a, &b);
    }
    zz_set(out, &a);
done:
    zz_cclear(&w);
    zz_cclear(&lw);
    zz_cclear(&winv);
    zz_cclear(&winv2);
    zz_cclear(&pw);
    zz_cclear(&tmp);
    zz_clear(&a);
    zz_clear(&b);
    zz_clear(&half_t);
    mpfr_clears(tabs, wl, wu, rel, rem, (mpfr_ptr)0);
    return rc;
}

/* ---- zeta by Euler-Maclaurin ------------------------------------------- */

/* With f(x) = x^(-s) in DLMF 2.10.1 (a = N, n -> infinity, sigma > 1, then
 * analytic continuation to sigma > 1 - 2M):
 *   zeta(s) = sum_{n<N} n^-s + N^(1-s)/(s-1) + N^-s / 2
 *           + sum_{k=1}^{M} B_2k/(2k)! s(s+1)...(s+2k-2) N^(-s-2k+1) + E,
 *   E = -int_N^inf B~_2M(x)/(2M)! s(s+1)...(s+2M-1) x^(-s-2M) dx,
 * and |B~_2M(x)| <= |B_2M| (DLMF 24.9.1, 24.2.11) gives
 *   |E| <= |B_2M|/(2M)! |s(s+1)...(s+2M-1)| N^(1-sigma-2M) / (sigma+2M-1). */
int zz_zeta_em(zz_ctx *ctx, zz_cball *out, const zz_ball *t) {
    mpfr_prec_t p = ctx->prec + 32;
    double td = mpfr_get_d(t->m, MPFR_RNDN);
    if (!(td > 0.0) || td >= 1099511627776.0) {
        return ZZ_ERR_ARGS;
    }
    long N = (long)(td / 3.141592653589793) + 10;
    int rc = ZZ_OK;
    zz_cball sum, s, ns, pk, term, tmp;
    zz_ball a, sn, cs, x, invn2, ln, is;
    mpfr_t e, u, v;
    zz_cinit(&sum, p);
    zz_cinit(&s, p);
    zz_cinit(&ns, p);
    zz_cinit(&pk, p);
    zz_cinit(&term, p);
    zz_cinit(&tmp, p);
    zz_init(&a, p);
    zz_init(&sn, p);
    zz_init(&cs, p);
    zz_init(&x, p);
    zz_init(&invn2, p);
    zz_init(&ln, p);
    zz_init(&is, p);
    mpfr_inits2(ZZ_RAD_PREC, e, u, v, (mpfr_ptr)0);

    /* sum_{n=1}^{N-1} n^(-1/2) (cos(t log n) - i sin(t log n)). */
    for (long n = 1; n < N; n++) {
        zz_log_ui(&ln, (unsigned long)n);
        zz_rec_sqrt_ui(&is, (unsigned long)n);
        zz_mul(&a, t, &ln);
        zz_sin_cos(&sn, &cs, &a);
        zz_mul(&x, &cs, &is);
        zz_add(&sum.re, &sum.re, &x);
        zz_mul(&x, &sn, &is);
        zz_sub(&sum.im, &sum.im, &x);
    }
    /* N^-s. */
    zz_log_ui(&ln, (unsigned long)N);
    zz_rec_sqrt_ui(&is, (unsigned long)N);
    zz_mul(&a, t, &ln);
    zz_sin_cos(&sn, &cs, &a);
    zz_mul(&ns.re, &cs, &is);
    zz_mul(&ns.im, &sn, &is);
    zz_neg(&ns.im, &ns.im);
    /* s = 1/2 + i t. */
    zz_set_si_frac(&s.re, 1, 2);
    zz_set(&s.im, t);
    /* + N^(1-s)/(s-1) = N N^-s / (s - 1). */
    zz_set_si_frac(&tmp.re, -1, 2);
    zz_set(&tmp.im, t);
    if (zz_cdiv(&term, &ns, &tmp) != 0) {
        rc = ZZ_ERR_EVAL;
        goto done;
    }
    zz_mul_si(&term.re, &term.re, N);
    zz_mul_si(&term.im, &term.im, N);
    zz_cadd(&sum, &sum, &term);
    /* + N^-s / 2. */
    zz_mul_2si(&term.re, &ns.re, -1);
    zz_mul_2si(&term.im, &ns.im, -1);
    zz_cadd(&sum, &sum, &term);
    /* Bernoulli terms: pk = s(s+1)...(s+2k-2) (2k-1 factors), and the power
     * N^(-s-2k+1) = N^-s * N^(1-2k). */
    zz_set(&pk.re, &s.re);
    zz_set(&pk.im, &s.im);
    zz_set_si_frac(&invn2, 1, N);
    zz_mul(&invn2, &invn2, &invn2);
    zz_set_si_frac(&x, 1, N); /* N^(1-2k) for k = 1 */
    int converged = 0;
    for (int k = 1; k < BERN_K; k++) {
        if (!(ctx->faults.drop_term == 1 && k == 1 && td >= ctx->faults.min_t)) {
            zz_cmul(&term, &pk, &ns);
            zz_cmul_real(&term, &term, &x);
            zz_cmul_real(&term, &term, &ctx->bern_fact[k]);
            zz_cadd(&sum, &sum, &term);
        }
        /* pk <- pk (s + 2k - 1): now s...(s+2k-1), for the remainder. */
        zz_set(&tmp.re, &s.re);
        zz_set_si_frac(&a, 2 * k - 1, 1);
        zz_add(&tmp.re, &tmp.re, &a);
        zz_set(&tmp.im, &s.im);
        zz_cmul(&term, &pk, &tmp);
        zz_set(&pk.re, &term.re);
        zz_set(&pk.im, &term.im);
        /* Remainder bound with M = k. */
        zz_abs_upper(e, &ctx->bern_fact[k]);
        zz_cabs_upper(u, &pk);
        mpfr_mul(e, e, u, MPFR_RNDU);
        mpfr_set_si(u, N, MPFR_RNDD);
        mpfr_set_d(v, 0.5 - 2.0 * k, MPFR_RNDN); /* exact */
        mpfr_pow(u, u, v, MPFR_RNDU);             /* N^(1 - 1/2 - 2k) */
        mpfr_mul(e, e, u, MPFR_RNDU);
        mpfr_set_d(v, 2.0 * k - 0.5, MPFR_RNDN); /* exact */
        mpfr_div(e, e, v, MPFR_RNDU);
        if (mpfr_cmp_ui_2exp(e, 1, -EM_TARGET_BITS) < 0) {
            converged = 1;
            break;
        }
        /* pk <- pk (s + 2k): s...(s+2k), the next term's product. */
        zz_set(&tmp.re, &s.re);
        zz_set_si_frac(&a, 2 * k, 1);
        zz_add(&tmp.re, &tmp.re, &a);
        zz_cmul(&term, &pk, &tmp);
        zz_set(&pk.re, &term.re);
        zz_set(&pk.im, &term.im);
        zz_mul(&x, &x, &invn2);
    }
    if (!converged && mpfr_cmp_ui_2exp(e, 1, -EM_MAX_BITS) >= 0) {
        rc = ZZ_ERR_EVAL;
        goto done;
    }
    zz_cadd_error(&sum, e);
    zz_set(&out->re, &sum.re);
    zz_set(&out->im, &sum.im);
done:
    zz_cclear(&sum);
    zz_cclear(&s);
    zz_cclear(&ns);
    zz_cclear(&pk);
    zz_cclear(&term);
    zz_cclear(&tmp);
    zz_clear(&a);
    zz_clear(&sn);
    zz_clear(&cs);
    zz_clear(&x);
    zz_clear(&invn2);
    zz_clear(&ln);
    zz_clear(&is);
    mpfr_clears(e, u, v, (mpfr_ptr)0);
    return rc;
}

int zz_z_em(zz_ctx *ctx, zz_ball *out, const zz_ball *t) {
    mpfr_prec_t p = ctx->prec + 32;
    zz_cball zeta;
    zz_ball th, sn, cs, a, b;
    zz_cinit(&zeta, p);
    zz_init(&th, p);
    zz_init(&sn, p);
    zz_init(&cs, p);
    zz_init(&a, p);
    zz_init(&b, p);
    int rc = zz_zeta_em(ctx, &zeta, t);
    if (rc == ZZ_OK) {
        rc = zz_theta(ctx, &th, t);
    }
    if (rc == ZZ_OK) {
        /* e^{i theta} zeta = (cos th Re - sin th Im) + i (sin th Re + cos th Im). */
        zz_sin_cos(&sn, &cs, &th);
        zz_mul(&a, &cs, &zeta.re);
        zz_mul(&b, &sn, &zeta.im);
        zz_sub(out, &a, &b);
        zz_mul(&a, &sn, &zeta.re);
        zz_mul(&b, &cs, &zeta.im);
        zz_add(&a, &a, &b);
        if (zz_sign(&a) != 0) {
            rc = ZZ_ERR_INCONSISTENT;
        }
    }
    ctx->evaluations++;
    ctx->em_evaluations++;
    zz_cclear(&zeta);
    zz_clear(&th);
    zz_clear(&sn);
    zz_clear(&cs);
    zz_clear(&a);
    zz_clear(&b);
    return rc;
}

/* ---- Riemann-Siegel with Gabcke's remainder bound ---------------------- */

/* Z(t) = 2 sum_{n<=N} n^(-1/2) cos(theta - t log n)
 *        + (-1)^(N-1) tau^(-1/4) C_0(p) + R_0(t),
 * tau = t/(2 pi), N = floor(sqrt(tau)), p = sqrt(tau) - N,
 * C_0(p) = cos(2 pi (p^2 - p - 1/16)) / cos(2 pi p),
 * |R_0(t)| < 0.127 t^(-3/4) for t >= 200 (Gabcke 1979, Satz 3.2.2 (b)).
 * Gabcke writes the correction as C_0(z) = cos(pi/2 (z^2 + 3/4)) / cos(pi z)
 * with z = 1 - 2p; substituting z turns numerator and denominator each into
 * -1 times the ones above, so the two are the same function. */
int zz_z_rs(zz_ctx *ctx, zz_ball *out, const zz_ball *t) {
    mpfr_prec_t p = ctx->prec + 32;
    int rc = ZZ_OK;
    mpfr_t lo, hi, e;
    mpfr_inits2(p, lo, hi, e, (mpfr_ptr)0);
    zz_lower(lo, t);
    if (mpfr_cmp_ui(lo, 200) < 0) {
        mpfr_clears(lo, hi, e, (mpfr_ptr)0);
        return ZZ_ERR_ARGS;
    }
    zz_ball tau, sq, th, sum, a, sn, cs, frac, num, den, f;
    zz_init(&tau, p);
    zz_init(&sq, p);
    zz_init(&th, p);
    zz_init(&sum, p);
    zz_init(&a, p);
    zz_init(&sn, p);
    zz_init(&cs, p);
    zz_init(&frac, p);
    zz_init(&num, p);
    zz_init(&den, p);
    zz_init(&f, p);
    long N = 0;

    zz_mul_2si(&a, &ctx->pi, 1);
    if (zz_div(&tau, t, &a) != 0 || zz_sqrt(&sq, &tau) != 0) {
        rc = ZZ_ERR_EVAL;
        goto done;
    }
    zz_lower(lo, &sq);
    zz_upper(hi, &sq);
    mpfr_floor(lo, lo);
    mpfr_floor(hi, hi);
    if (!mpfr_equal_p(lo, hi)) {
        rc = ZZ_ERR_EVAL; /* sqrt(tau) may straddle an integer */
        goto done;
    }
    N = mpfr_get_si(lo, MPFR_RNDN);
    rc = zz_theta(ctx, &th, t);
    if (rc != ZZ_OK) {
        goto done;
    }
    for (long n = 1; n <= N; n++) {
        if (ctx->faults.drop_term == 2 && n == N && mpfr_cmp_d(t->m, ctx->faults.min_t) >= 0) {
            continue;
        }
        zz_log_ui(&f, (unsigned long)n);
        zz_mul(&a, t, &f);
        zz_sub(&a, &th, &a);
        zz_sin_cos(&sn, &cs, &a);
        zz_rec_sqrt_ui(&f, (unsigned long)n);
        zz_mul(&a, &cs, &f);
        zz_add(&sum, &sum, &a);
    }
    zz_mul_2si(&sum, &sum, 1);
    /* C_0(p). */
    zz_set_si_frac(&a, N, 1);
    zz_sub(&frac, &sq, &a);
    zz_mul(&num, &frac, &frac);
    zz_sub(&num, &num, &frac);
    zz_set_si_frac(&a, 1, 16);
    zz_sub(&num, &num, &a);
    zz_mul_2si(&a, &ctx->pi, 1);
    zz_mul(&num, &num, &a);
    zz_sin_cos(&sn, &cs, &num);
    zz_set(&num, &cs);
    zz_mul(&den, &frac, &a);
    zz_sin_cos(&sn, &cs, &den);
    if (zz_div(&num, &num, &cs) != 0) {
        rc = ZZ_ERR_EVAL; /* p too close to 1/4 or 3/4 */
        goto done;
    }
    /* tau^(-1/4). */
    if (zz_log(&f, &tau) != 0) {
        rc = ZZ_ERR_EVAL;
        goto done;
    }
    zz_mul_2si(&f, &f, -2);
    zz_neg(&f, &f);
    zz_exp(&f, &f);
    zz_mul(&a, &f, &num);
    if (N % 2 == 1) {
        zz_add(&sum, &sum, &a);
    } else {
        zz_sub(&sum, &sum, &a);
    }
    /* Remainder: (127/1000) t^(-3/4), at the smallest t of the ball (where
     * t^(-3/4) is largest), rounded up. */
    zz_lower(e, t);
    mpfr_set_d(hi, -0.75, MPFR_RNDN); /* exact */
    mpfr_pow(e, e, hi, MPFR_RNDU);
    mpfr_mul_ui(e, e, ZZ_GABCKE_D0_NUM, MPFR_RNDU);
    mpfr_div_ui(e, e, ZZ_GABCKE_D0_DEN, MPFR_RNDU);
    zz_add_error(&sum, e);
    zz_set(out, &sum);
done:
    zz_clear(&tau);
    zz_clear(&sq);
    zz_clear(&th);
    zz_clear(&sum);
    zz_clear(&a);
    zz_clear(&sn);
    zz_clear(&cs);
    zz_clear(&frac);
    zz_clear(&num);
    zz_clear(&den);
    zz_clear(&f);
    mpfr_clears(lo, hi, e, (mpfr_ptr)0);
    ctx->evaluations++;
    return rc;
}

int zz_z_sign(zz_ctx *ctx, double t, int method, int *err) {
    zz_ball tb, z;
    zz_init(&tb, ctx->prec + 32);
    zz_init(&z, ctx->prec + 32);
    zz_set_d(&tb, t);
    int sign = 0;
    int rc = ZZ_OK;
    if (method == ZZ_METHOD_RS_GABCKE && t >= 200.0) {
        rc = zz_z_rs(ctx, &z, &tb);
        if (rc == ZZ_OK) {
            sign = zz_sign(&z);
        }
        if (sign != 0) {
            ctx->used_riemann_siegel = 1;
        } else {
            ctx->rs_fallbacks++;
        }
    }
    if (sign == 0) {
        rc = zz_z_em(ctx, &z, &tb);
        sign = rc == ZZ_OK ? zz_sign(&z) : 0;
        if (rc == ZZ_OK && sign == 0) {
            rc = ZZ_ERR_EVAL;
        }
    }
    if (err != NULL) {
        *err = rc;
    }
    zz_clear(&tb);
    zz_clear(&z);
    return rc == ZZ_OK ? sign : 0;
}

int zz_z_sign_interval(zz_ctx *ctx, const mpfr_t lo, const mpfr_t hi) {
    zz_ball tb, z;
    zz_init(&tb, ctx->prec + 32);
    zz_init(&z, ctx->prec + 32);
    mpfr_add(tb.m, lo, hi, MPFR_RNDN);
    mpfr_mul_2si(tb.m, tb.m, -1, MPFR_RNDN);
    mpfr_sub(tb.r, hi, tb.m, MPFR_RNDU);
    mpfr_t w;
    mpfr_init2(w, ZZ_RAD_PREC);
    mpfr_sub(w, tb.m, lo, MPFR_RNDU);
    if (mpfr_cmp(w, tb.r) > 0) {
        mpfr_set(tb.r, w, MPFR_RNDU);
    }
    mpfr_clear(w);
    int sign = zz_z_em(ctx, &z, &tb) == ZZ_OK ? zz_sign(&z) : 0;
    zz_clear(&tb);
    zz_clear(&z);
    return sign;
}

/* ---- Gram points ------------------------------------------------------- */

/* Heuristic only (chooses sample points): the asymptotic expansion
 * theta(t) ~ t/2 log(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760t^3). */
static double theta_approx(double t) {
    const double pi = 3.141592653589793;
    return t / 2.0 * log(t / (2.0 * pi)) - t / 2.0 - pi / 8.0 + 1.0 / (48.0 * t) +
           7.0 / (5760.0 * t * t * t);
}

/* Approximate n-th Gram point (n >= -1), from a guess above 7. */
static double gram_approx(long n, double guess) {
    const double pi = 3.141592653589793;
    double t = guess > 7.5 ? guess : 9.6;
    for (int i = 0; i < 60; i++) {
        double d = (theta_approx(t) - (double)n * pi) / (0.5 * log(t / (2.0 * pi)));
        t -= d;
        if (t < 7.5) {
            t = 7.5;
        }
        if (fabs(d) < 1e-14 * t) {
            break;
        }
    }
    return t;
}

int zz_gram_point(zz_ctx *ctx, long n, mpfr_t lo, mpfr_t hi) {
    if (n < -1 || n > 100000000L) {
        return ZZ_ERR_ARGS;
    }
    mpfr_prec_t p = ctx->prec + 32;
    int rc = ZZ_OK;
    zz_ball tb, th, target;
    mpfr_t t, d;
    zz_init(&tb, p);
    zz_init(&th, p);
    zz_init(&target, p);
    mpfr_inits2(p, t, d, (mpfr_ptr)0);
    mpfr_set_d(t, gram_approx(n, 9.6 + 2.0 * (double)(n + 1)), MPFR_RNDN);
    zz_mul_si(&target, &ctx->pi, n);
    /* Newton with the approximate derivative (1/2) log(t / 2pi). */
    for (int i = 0; i < 200; i++) {
        zz_set_fr(&tb, t);
        rc = zz_theta(ctx, &th, &tb);
        if (rc != ZZ_OK) {
            goto done;
        }
        mpfr_sub(d, th.m, target.m, MPFR_RNDN);
        double td = mpfr_get_d(t, MPFR_RNDN);
        mpfr_div_d(d, d, 0.5 * log(td / (2.0 * 3.141592653589793)), MPFR_RNDN);
        mpfr_sub(t, t, d, MPFR_RNDN);
        if (mpfr_zero_p(d) || mpfr_get_exp(d) < mpfr_get_exp(t) - (mpfr_exp_t)ctx->prec + 8) {
            break;
        }
    }
    /* Certify: theta(lo) < n pi < theta(hi). */
    mpfr_set_ui_2exp(d, 1, mpfr_get_exp(t) - (mpfr_exp_t)(ctx->prec / 2), MPFR_RNDN);
    mpfr_sub(lo, t, d, MPFR_RNDD);
    mpfr_add(hi, t, d, MPFR_RNDU);
    {
        mpfr_t a, b;
        mpfr_inits2(p, a, b, (mpfr_ptr)0);
        zz_set_fr(&tb, lo);
        rc = zz_theta(ctx, &th, &tb);
        if (rc == ZZ_OK) {
            zz_upper(a, &th);
            zz_lower(b, &target);
            if (mpfr_cmp(a, b) >= 0) {
                rc = ZZ_ERR_EVAL;
            }
        }
        if (rc == ZZ_OK) {
            zz_set_fr(&tb, hi);
            rc = zz_theta(ctx, &th, &tb);
        }
        if (rc == ZZ_OK) {
            zz_lower(a, &th);
            zz_upper(b, &target);
            if (mpfr_cmp(a, b) <= 0) {
                rc = ZZ_ERR_EVAL;
            }
        }
        mpfr_clears(a, b, (mpfr_ptr)0);
    }
done:
    zz_clear(&tb);
    zz_clear(&th);
    zz_clear(&target);
    mpfr_clears(t, d, (mpfr_ptr)0);
    return rc;
}

/* ---- Verification ------------------------------------------------------ */

typedef struct verifier {
    zz_ctx *ctx;
    int method;
    zz_verify_result *res;
    zz_sample *w; /* caller's workspace */
    size_t cap;
    size_t n;     /* samples in use, sorted by t */
} verifier;

/* Records the first failure only. */
static void fail(verifier *vf, int status, const char *msg, double t) {
    if (vf->res->reason == NULL) {
        vf->res->status = status;
        vf->res->reason = msg;
        vf->res->fail_t = t;
    }
}

static int push(verifier *vf, double t, int sign, long gram) {
    if (vf->n >= vf->cap) {
        fail(vf, ZZ_ERR_ARGS, "workspace too small", t);
        return -1;
    }
    vf->w[vf->n].t = t;
    vf->w[vf->n].sign = sign;
    vf->w[vf->n].gram = gram;
    vf->n++;
    return 0;
}

/* Proven sign at t; if undecided and nudge != 0, tries t + k * nudge for
 * k = 1, -1, 2, -2, ... and stores the point used in *used. */
static int sign_at(verifier *vf, double t, double nudge, double *used) {
    for (int k = 0; k < 9; k++) {
        double off = (double)((k + 1) / 2) * ((k % 2) ? 1.0 : -1.0);
        double x = t + off * nudge;
        int err = ZZ_OK;
        int s = zz_z_sign(vf->ctx, x, vf->method, &err);
        if (err == ZZ_ERR_INCONSISTENT) {
            fail(vf, err, "Im(e^{i theta} zeta) provably nonzero", x);
            return 0;
        }
        if (err == ZZ_ERR_ARGS) {
            fail(vf, err, "evaluation out of range", x);
            return 0;
        }
        if (s != 0) {
            *used = x;
            return s;
        }
        if (nudge == 0.0) {
            break;
        }
    }
    fail(vf, ZZ_ERR_EVAL, "sign of Z undecided", t);
    return 0;
}

/* RS mode cross-check at t: Euler-Maclaurin must pass its Im check (which
 * catches a wrong theta) and its enclosure of Z must overlap the
 * Riemann-Siegel one (which catches a wrong RS value or bound). This guards
 * against defects in the implementation, which Gabcke's bound cannot. */
static int rs_spot_check(verifier *vf, double t) {
    zz_ctx *ctx = vf->ctx;
    zz_ball tb, zr, ze;
    mpfr_t d, r;
    zz_init(&tb, ctx->prec + 32);
    zz_init(&zr, ctx->prec + 32);
    zz_init(&ze, ctx->prec + 32);
    mpfr_inits2(ctx->prec + 32, d, r, (mpfr_ptr)0);
    zz_set_d(&tb, t);
    int rc = zz_z_em(ctx, &ze, &tb);
    if (rc == ZZ_ERR_INCONSISTENT) {
        fail(vf, rc, "Im(e^{i theta} zeta) provably nonzero", t);
    } else if (rc == ZZ_OK && zz_z_rs(ctx, &zr, &tb) == ZZ_OK) {
        mpfr_sub(d, zr.m, ze.m, MPFR_RNDN);
        mpfr_abs(d, d, MPFR_RNDD);
        mpfr_add(r, zr.r, ze.r, MPFR_RNDU);
        if (mpfr_cmp(d, r) > 0) {
            rc = ZZ_ERR_INCONSISTENT;
            fail(vf, rc, "Riemann-Siegel and Euler-Maclaurin enclosures are disjoint", t);
        }
    } else {
        rc = ZZ_OK; /* an undecidable spot check is skipped, not a failure */
    }
    zz_clear(&tb);
    zz_clear(&zr);
    zz_clear(&ze);
    mpfr_clears(d, r, (mpfr_ptr)0);
    return rc;
}

static size_t count_changes(const zz_sample *v, size_t a, size_t b) {
    size_t c = 0;
    for (size_t i = a; i < b; i++) {
        if (v[i].sign != v[i + 1].sign) {
            c++;
        }
    }
    return c;
}

/* The Gram block w[ip..n-1] ends the workspace. Bisects every pair of
 * neighbours, round by round (one sign change may hide three zeros), until
 * the block has at least want sign changes. Each round is built in scratch
 * space at the end of the workspace and copied back. Returns 0, or -1. */
static int refine_tail(verifier *vf, size_t ip, size_t want) {
    zz_sample *w = vf->w;
    for (int round = 0; round < ROSSER_ROUNDS; round++) {
        if (count_changes(w, ip, vf->n - 1) >= want) {
            return 0;
        }
        size_t len = vf->n - ip;
        size_t need = 2 * len - 1;
        if (need > vf->cap || vf->n > vf->cap - need) {
            fail(vf, ZZ_ERR_ARGS, "workspace too small", w[ip].t);
            return -1;
        }
        size_t base = vf->cap - need; /* >= n: scratch never overlaps */
        size_t m = 0;
        for (size_t i = 0; i + 1 < len; i++) {
            w[base + m++] = w[ip + i];
            double u = w[ip + i].t;
            double v = w[ip + i + 1].t;
            double mid = 0.5 * (u + v);
            if (!(mid > u && mid < v)) {
                continue; /* cannot split further in double */
            }
            double used = mid;
            int s = sign_at(vf, mid, (v - u) * 1e-3, &used);
            if (s == 0) {
                return -1;
            }
            if (used > u && used < v) {
                w[base + m].t = used;
                w[base + m].sign = s;
                w[base + m].gram = ZZ_NOT_GRAM;
                m++;
            }
        }
        w[base + m++] = w[ip + len - 1];
        for (size_t k = 0; k < m; k++) {
            w[ip + k] = w[base + k]; /* ip < base: forward copy is safe */
        }
        vf->n = ip + m;
    }
    if (count_changes(w, ip, vf->n - 1) >= want) {
        return 0;
    }
    fail(vf, ZZ_ERR_ROSSER, "Gram block lacks expected sign changes", w[ip].t);
    return -1;
}

static int gram_good(long n, int sign) {
    int expect = (n % 2 == 0) ? 1 : -1;
    return sign == expect;
}

void zz_verify_params_default(zz_verify_params *params, double height) {
    *params = (zz_verify_params){0};
    params->height = height;
    params->method = ZZ_METHOD_EM;
    params->turing_intervals = 40;
    params->prec = 128;
    params->faults.skip_interval = ZZ_NO_SKIP;
}

size_t zz_verify_work_size(double height, long turing_intervals) {
    const double two_pi = 6.283185307179586;
    double t = height > 528.0 ? height : 528.0;
    if (!(t <= VERIFY_MAX_HEIGHT) || turing_intervals < 0 || turing_intervals > 100000) {
        return 0;
    }
    t += (double)(turing_intervals + 4) * two_pi / log(t / two_pi);
    double grams = theta_approx(t) / 3.141592653589793 + 4.0;
    return 2 * (size_t)grams + 65536u + 130u;
}

/* Upper bound of N(t0) by Turing's method (see the header and the note):
 *   N(t0) <= 1 + (a + b log t1 - int L + (1/pi) int_{t0}^{t1} theta) / (t1 - t0),
 * where il <= int L (the caller sums t1 - hi over sign changes in [t0, t1]),
 * and int theta is bounded above piecewise by theta evaluated on balls. */
static int turing_bound(zz_ctx *ctx, const mpfr_t il, double t0, double t1, mpfr_t upper) {
    mpfr_prec_t p = ctx->prec + 64;
    int rc = ZZ_OK;
    mpfr_t ith, u, v, x, y;
    mpfr_inits2(p, ith, u, v, x, y, (mpfr_ptr)0);
    zz_ball tb, th;
    zz_init(&tb, ctx->prec + 32);
    zz_init(&th, ctx->prec + 32);
    /* int theta over [t0, t1] <= sum over pieces of (v - u) sup theta([u, v]). */
    mpfr_set_zero(ith, 1);
    mpfr_set_d(u, t0, MPFR_RNDN);
    mpfr_set_d(y, t1, MPFR_RNDN);
    while (mpfr_cmp(u, y) < 0) {
        mpfr_add_d(v, u, 1.0 / TURING_PIECES_PER_UNIT, MPFR_RNDN); /* exact */
        if (mpfr_cmp(v, y) > 0) {
            mpfr_set(v, y, MPFR_RNDN);
        }
        mpfr_add(tb.m, u, v, MPFR_RNDN);
        mpfr_mul_2si(tb.m, tb.m, -1, MPFR_RNDN); /* exact at this precision */
        mpfr_sub(tb.r, v, tb.m, MPFR_RNDU);
        rc = zz_theta(ctx, &th, &tb);
        if (rc != ZZ_OK) {
            goto done;
        }
        zz_upper(x, &th);
        if (mpfr_sgn(x) < 0) {
            rc = ZZ_ERR_INCONSISTENT; /* theta > 0 for t > 168 pi */
            goto done;
        }
        mpfr_sub(v, v, u, MPFR_RNDN); /* exact */
        mpfr_mul(x, x, v, MPFR_RNDU);
        mpfr_add(ith, ith, x, MPFR_RNDU);
        mpfr_add(u, u, v, MPFR_RNDN); /* exact */
    }
    /* ith >= 0, so dividing by a lower bound of pi rounds up. */
    zz_lower(x, &ctx->pi);
    mpfr_div(ith, ith, x, MPFR_RNDU);
    /* a + b log t1 - il + ith/pi, rounded up. */
    mpfr_set_d(x, t1, MPFR_RNDN);
    mpfr_log(x, x, MPFR_RNDU);
    mpfr_mul_ui(x, x, ZZ_TRUDGIAN_B_NUM, MPFR_RNDU);
    mpfr_add_ui(x, x, ZZ_TRUDGIAN_A_NUM, MPFR_RNDU);
    mpfr_div_ui(x, x, ZZ_TRUDGIAN_DEN, MPFR_RNDU);
    mpfr_sub(x, x, il, MPFR_RNDU);
    mpfr_add(x, x, ith, MPFR_RNDU);
    if (mpfr_sgn(x) < 0) {
        rc = ZZ_ERR_INCONSISTENT;
        goto done;
    }
    mpfr_set_d(v, t1, MPFR_RNDN);
    mpfr_sub_d(v, v, t0, MPFR_RNDN); /* exact */
    mpfr_div(x, x, v, MPFR_RNDU);
    mpfr_add_ui(upper, x, 1, MPFR_RNDU);
done:
    zz_clear(&tb);
    zz_clear(&th);
    mpfr_clears(ith, u, v, x, y, (mpfr_ptr)0);
    return rc;
}

static int params_valid(const zz_verify_params *p) {
    if (p == NULL || !(p->height > 0.0) || p->height > VERIFY_MAX_HEIGHT ||
        p->n_checkpoints > 64 || (p->n_checkpoints > 0 && p->checkpoints == NULL) ||
        p->turing_intervals < 8 || p->turing_intervals > 100000 || p->prec < 64 ||
        p->prec > 1024 || (p->method != ZZ_METHOD_EM && p->method != ZZ_METHOD_RS_GABCKE) ||
        p->work == NULL || p->work_cap < 16 || (p->brackets == NULL && p->brackets_cap != 0) ||
        (p->bad_gram == NULL && p->bad_gram_cap != 0)) {
        return 0;
    }
    for (size_t i = 0; i < p->n_checkpoints; i++) {
        if (!(p->checkpoints[i] > 0.0) || p->checkpoints[i] > p->height) {
            return 0;
        }
    }
    return 1;
}

/* Samples approximate Gram points from g_-1 with proven signs, merging in
 * the checkpoints, refining each Gram block as soon as it closes, up to the
 * first good Gram point at or above `need` (t0) and turing_intervals beyond
 * it, ending at a good Gram point (t1). */
static int sample_all(verifier *vf, const zz_verify_params *params, const double *cps, size_t ncp,
                      double need, long *gram_t0_out) {
    size_t ci = 0;
    long gram_t0 = ZZ_NOT_GRAM;
    size_t ip = 0;
    long ip_gram = ZZ_NOT_GRAM;
    double g = 9.6;
    for (long n = -1;; n++) {
        g = gram_approx(n, g);
        double spacing = 6.283185307179586 / log(g / 6.283185307179586);
        double used = g;
        int s = sign_at(vf, g, spacing * 1e-4, &used);
        if (s == 0) {
            return -1;
        }
        for (; ci < ncp && cps[ci] < used; ci++) {
            double cu = cps[ci];
            if (vf->n > 0 && cps[ci] > vf->w[vf->n - 1].t) {
                int cs = sign_at(vf, cps[ci], 0.0, &cu);
                if (cs == 0 || push(vf, cps[ci], cs, ZZ_NOT_GRAM) != 0) {
                    return -1;
                }
            }
        }
        if (vf->method == ZZ_METHOD_RS_GABCKE && used >= 200.0 && n % RS_SPOT_CHECK == 0 &&
            rs_spot_check(vf, used) != ZZ_OK) {
            return -1;
        }
        if (push(vf, used, s, n) != 0) {
            return -1;
        }
        if (gram_good(n, s)) {
            if (ip_gram != ZZ_NOT_GRAM) {
                size_t want = (size_t)(n - ip_gram);
                if (count_changes(vf->w, ip, vf->n - 1) < want && refine_tail(vf, ip, want) != 0) {
                    return -1;
                }
            }
            ip = vf->n - 1;
            ip_gram = n;
            if (gram_t0 == ZZ_NOT_GRAM && used >= need) {
                gram_t0 = n;
            }
            if (gram_t0 != ZZ_NOT_GRAM && n >= gram_t0 + params->turing_intervals) {
                break;
            }
        }
        if (gram_t0 != ZZ_NOT_GRAM && n > gram_t0 + params->turing_intervals + 1000) {
            fail(vf, ZZ_ERR_TURING, "no good Gram point ends Turing's range", used);
            return -1;
        }
        g += spacing;
    }
    *gram_t0_out = gram_t0;
    return 0;
}

int zz_verify(const zz_verify_params *params, zz_verify_result *res) {
    if (res == NULL) {
        return ZZ_ERR_ARGS;
    }
    *res = (zz_verify_result){0};
    if (!params_valid(params)) {
        res->status = ZZ_ERR_ARGS;
        res->reason = "invalid parameters";
        return res->status;
    }
    zz_ctx ctx;
    if (zz_ctx_init(&ctx, params->prec) != ZZ_OK) {
        res->status = ZZ_ERR_ARGS;
        res->reason = "invalid precision";
        return res->status;
    }
    zz_ctx_set_faults(&ctx, &params->faults);
    verifier vf = {&ctx, params->method, res, params->work, params->work_cap, 0};
    const long skip = params->faults.skip_interval;

    /* Checkpoints and T, sorted. */
    double cps[65];
    size_t ncp = params->n_checkpoints;
    for (size_t i = 0; i < ncp; i++) {
        cps[i] = params->checkpoints[i];
    }
    cps[ncp++] = params->height;
    for (size_t i = 1; i < ncp; i++) { /* insertion sort */
        double key = cps[i];
        size_t j = i;
        while (j > 0 && cps[j - 1] > key) {
            cps[j] = cps[j - 1];
            j--;
        }
        cps[j] = key;
    }
    double turing_floor = 168.0 * 3.141592653589793 + 1.0; /* t0 > 168 pi */
    double need = params->height > turing_floor ? params->height : turing_floor;
    long gram_t0 = ZZ_NOT_GRAM;
    if (sample_all(&vf, params, cps, ncp, need, &gram_t0) != 0) {
        goto out;
    }

    const zz_sample *w = vf.w;
    double t0 = 0.0;
    for (size_t k = 0; k < vf.n; k++) {
        if (w[k].gram == gram_t0) {
            t0 = w[k].t;
        }
    }
    double t1 = w[vf.n - 1].t;
    res->t0 = t0;
    res->t1 = t1;
    res->gram_t0 = gram_t0;
    if (!(t1 > t0) || !(t0 > 168.0 * 3.141592653589793)) {
        fail(&vf, ZZ_ERR_TURING, "Turing range is empty", t0);
        goto out;
    }

    /* Sign changes: count to t0, store, and sum t1 - hi over [t0, t1]. */
    {
        mpfr_t il, x, upper, fl;
        mpfr_inits2(ctx.prec + 64, il, x, upper, fl, (mpfr_ptr)0);
        mpfr_set_zero(il, 1);
        long cur_gram = ZZ_NOT_GRAM;
        long count_t0 = 0;
        for (size_t k = 0; k + 1 < vf.n; k++) {
            if (w[k].gram != ZZ_NOT_GRAM) {
                cur_gram = w[k].gram;
            }
            if (w[k].sign == w[k + 1].sign || cur_gram == skip) {
                continue; /* no sign change, or (fault injection) not counted */
            }
            zz_bracket b = {w[k].t, w[k + 1].t};
            if (res->n_brackets < params->brackets_cap) {
                params->brackets[res->n_brackets] = b;
            }
            res->n_brackets++;
            if (b.hi <= t0) {
                count_t0++;
            } else {
                mpfr_set_d(x, t1, MPFR_RNDN);
                mpfr_sub_d(x, x, b.hi, MPFR_RNDD);
                mpfr_add(il, il, x, MPFR_RNDD);
            }
        }
        res->sign_changes_to_t0 = count_t0;
        for (size_t k = 0; k < vf.n; k++) {
            if (w[k].gram != ZZ_NOT_GRAM && w[k].gram <= gram_t0 && !gram_good(w[k].gram, w[k].sign)) {
                if (res->n_bad_gram < params->bad_gram_cap) {
                    params->bad_gram[res->n_bad_gram] = w[k].gram;
                }
                res->n_bad_gram++;
            }
        }
        int rc = turing_bound(&ctx, il, t0, t1, upper);
        if (rc == ZZ_OK) {
            mpfr_floor(fl, upper);
            res->turing_bound = mpfr_get_si(fl, MPFR_RNDN);
            res->turing_upper = mpfr_get_d(upper, MPFR_RNDU);
        }
        mpfr_clears(il, x, upper, fl, (mpfr_ptr)0);
        if (rc != ZZ_OK) {
            fail(&vf, rc, "Turing bound could not be computed", t0);
            goto out;
        }
        if (count_t0 > res->turing_bound) {
            fail(&vf, ZZ_ERR_INCONSISTENT, "more sign changes than Turing's bound allows", t0);
            goto out;
        }
        if (count_t0 < res->turing_bound) {
            fail(&vf, ZZ_ERR_TURING, "Turing's bound exceeds the sign changes found", t0);
            goto out;
        }
    }

    /* Proven: every zero up to t0 is a counted sign change. T and every
     * checkpoint (at least g_-1) are sample points, so N(h) is the number of
     * sign changes ending at or before h. */
    {
        long cur_gram = ZZ_NOT_GRAM;
        for (size_t k = 0; k + 1 < vf.n; k++) {
            if (w[k].gram != ZZ_NOT_GRAM) {
                cur_gram = w[k].gram;
            }
            if (w[k].sign == w[k + 1].sign || cur_gram == skip) {
                continue;
            }
            if (w[k + 1].t <= params->height) {
                res->zeros_to_height++;
            }
            for (size_t c = 0; c < params->n_checkpoints; c++) {
                if (w[k + 1].t <= params->checkpoints[c]) {
                    res->checkpoint_counts[c]++;
                }
            }
        }
    }
    res->status = ZZ_OK;
out:
    res->evaluations = ctx.evaluations;
    res->em_evaluations = ctx.em_evaluations;
    res->rs_fallbacks = ctx.rs_fallbacks;
    res->used_riemann_siegel = ctx.used_riemann_siegel;
    zz_ctx_clear(&ctx);
    mpfr_free_cache();
    return res->status;
}
