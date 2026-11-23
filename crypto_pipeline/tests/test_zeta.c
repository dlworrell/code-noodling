/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:0ac0b0d5-a1e4-4c99-be74-cf0bb9da9cc1
 */

/* Tests for crypto_zeta_ball.h and crypto_zeta.h. Kept small enough to run
 * under Valgrind and ASan/UBSan; tests/test_zeta.py runs the full T = 10^4
 * verification through zeta_verify.
 *
 * Published values used here (OEIS, retrieved 2026-11-23):
 *   A058303, A065434, A065452, A065453, A192492, A305741..A305744, A306004:
 *     imaginary parts of the first ten zeros;
 *   A072080: N(10) = 0, N(100) = 29, N(1000) = 649, N(10^4) = 10142;
 *   A114856: bad Gram points (-1)^n Z(g_n) < 0 begin 126, 134, 195, 211,
 *     232, 254, 288;
 *   A002505: g_126 rounds to 282;
 *   A013629: zeros 6709 and 6710 both have floor 7005 (Lehmer's pair). */

#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_zeta.h"

#define PREC 128

/* Static workspace (no heap) for the T = 600 verifications. */
#define WORK_CAP (1u << 17)
static zz_sample work[WORK_CAP];

static const char *const first_zeros[10] = {
    "14.13472514173469379045725198356247", "21.02203963877155499262847959389690",
    "25.01085758014568876321379099256282", "30.42487612585951321031189753058409",
    "32.93506158773918969066236896407490", "37.58617815882567125721776348070533",
    "40.91871901214749518739812691463325", "43.32707328091499951949612216540680",
    "48.00515088116715972794247274942751", "49.77383247767230218191678467856372",
};

/* Encloses exact values at higher precision and checks containment. */
static int contains(const zz_ball *x, const mpfr_t v) {
    mpfr_t d;
    mpfr_init2(d, 512);
    mpfr_sub(d, v, x->m, MPFR_RNDN);
    mpfr_abs(d, d, MPFR_RNDU);
    int ok = mpfr_cmp(d, x->r) <= 0;
    mpfr_clear(d);
    return ok;
}

static void test_balls(void) {
    zz_ball a, b, c;
    zz_init(&a, 64);
    zz_init(&b, 64);
    zz_init(&c, 64);
    mpfr_t ref, one;
    mpfr_inits2(512, ref, one, (mpfr_ptr)0);

    /* e^1: the 64-bit ball must contain the 512-bit value. */
    zz_set_si_frac(&a, 1, 1);
    zz_exp(&b, &a);
    mpfr_set_ui(one, 1, MPFR_RNDN);
    mpfr_exp(ref, one, MPFR_RNDN);
    assert(contains(&b, ref));
    assert(mpfr_cmp_d(b.r, 1e-17) < 0);

    /* 1/3 is inexact: radius > 0 and the ball contains it. */
    zz_set_si_frac(&a, 1, 3);
    assert(mpfr_sgn(a.r) > 0);
    mpfr_set_ui(ref, 1, MPFR_RNDN);
    mpfr_div_ui(ref, ref, 3, MPFR_RNDN);
    assert(contains(&a, ref));

    /* Propagation: sin over [1e6 +/- 0.5] has radius at least 0.5. */
    zz_set_d(&a, 1e6);
    mpfr_set_d(a.r, 0.5, MPFR_RNDU);
    zz_sin_cos(&b, &c, &a);
    assert(mpfr_cmp_d(b.r, 0.5) >= 0 && mpfr_cmp_d(c.r, 0.5) >= 0);

    /* log, sqrt, 1/x refuse balls that reach 0. */
    zz_set_d(&a, 0.25);
    mpfr_set_d(a.r, 0.25, MPFR_RNDU);
    assert(zz_log(&b, &a) == -1);
    assert(zz_sqrt(&b, &a) == -1);
    assert(zz_inv(&b, &a) == -1);
    assert(zz_sign(&a) == 0);
    mpfr_set_d(a.r, 0.125, MPFR_RNDU);
    assert(zz_log(&b, &a) == 0 && zz_sign(&b) == -1);
    assert(zz_sqrt(&b, &a) == 0 && zz_sign(&b) == 1);
    assert(zz_inv(&b, &a) == 0);
    /* 1/[0.125, 0.375] = [2.67, 8]: the enclosure must cover both ends. */
    mpfr_t lo, hi;
    mpfr_inits2(64, lo, hi, (mpfr_ptr)0);
    zz_lower(lo, &b);
    zz_upper(hi, &b);
    assert(mpfr_cmp_d(lo, 8.0 / 3.0) <= 0 && mpfr_cmp_d(hi, 8.0) >= 0);
    mpfr_clears(lo, hi, (mpfr_ptr)0);

    /* Complex log: refuses Re <= 0; log(i + 1) = log(2)/2 + i pi/4. */
    zz_cball z, w;
    zz_cinit(&z, 64);
    zz_cinit(&w, 64);
    zz_set_si_frac(&z.re, -1, 1);
    zz_set_si_frac(&z.im, 1, 1);
    assert(zz_clog(&w, &z) == -1);
    zz_set_si_frac(&z.re, 1, 1);
    assert(zz_clog(&w, &z) == 0);
    mpfr_const_pi(ref, MPFR_RNDN);
    mpfr_div_ui(ref, ref, 4, MPFR_RNDN);
    assert(contains(&w.im, ref));
    mpfr_set_ui(ref, 2, MPFR_RNDN);
    mpfr_log(ref, ref, MPFR_RNDN);
    mpfr_div_ui(ref, ref, 2, MPFR_RNDN);
    assert(contains(&w.re, ref));
    zz_cclear(&z);
    zz_cclear(&w);

    zz_clear(&a);
    zz_clear(&b);
    zz_clear(&c);
    mpfr_clears(ref, one, (mpfr_ptr)0);
    printf("[PASS] ball arithmetic encloses exact values\n");
}

/* theta against Gabcke's expansion (Dissertation 1979, Einleitung (4), (5)):
 *   theta(t) = t/2 log(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760t^3)
 *              + 31/(80640t^5) + R(t),  |R(t)| < 1/(3322 t^7) for t >= 10.
 * Our theta comes from Stirling's series instead (DLMF 5.11); the two
 * independent routes must agree within Gabcke's bound, plus our radius.
 * Without its last term the expansion must miss (the check has teeth). */
static void expansion(mpfr_t out, double tt, int terms, const mpfr_t pi) {
    mpfr_t x, a;
    mpfr_inits2(256, x, a, (mpfr_ptr)0);
    mpfr_set_d(x, tt, MPFR_RNDN);
    mpfr_div(out, x, pi, MPFR_RNDN);
    mpfr_div_ui(out, out, 2, MPFR_RNDN);
    mpfr_log(out, out, MPFR_RNDN);
    mpfr_sub_ui(out, out, 1, MPFR_RNDN);
    mpfr_mul(out, out, x, MPFR_RNDN);
    mpfr_div_ui(out, out, 2, MPFR_RNDN); /* t/2 (log(t/2pi) - 1) */
    mpfr_div_ui(a, pi, 8, MPFR_RNDN);
    mpfr_sub(out, out, a, MPFR_RNDN);
    const unsigned long num[3] = {1, 7, 31};
    const unsigned long den[3] = {48, 5760, 80640};
    for (int k = 0; k < terms; k++) {
        mpfr_pow_si(a, x, -(2 * k + 1), MPFR_RNDN);
        mpfr_mul_ui(a, a, num[k], MPFR_RNDN);
        mpfr_div_ui(a, a, den[k], MPFR_RNDN);
        mpfr_add(out, out, a, MPFR_RNDN);
    }
    mpfr_clears(x, a, (mpfr_ptr)0);
}

static void test_theta(zz_ctx *ctx) {
    zz_ball t, th;
    zz_init(&t, PREC);
    zz_init(&th, 192); /* 128 bits would round theta(10^4) by ~1e-34 */
    mpfr_t x, a, pi;
    mpfr_inits2(256, x, a, pi, (mpfr_ptr)0);
    mpfr_const_pi(pi, MPFR_RNDN);
    const double ts[] = {10.0, 20.0, 50.0, 200.0, 1000.0, 7005.0, 10000.0};
    for (int i = 0; i < 7; i++) {
        double tt = ts[i];
        zz_set_d(&t, tt);
        assert(zz_theta(ctx, &th, &t) == ZZ_OK);
        assert(mpfr_cmp_d(th.r, 1e-35) < 0); /* far below 1/(3322 t^7) */
        mpfr_set_d(x, tt, MPFR_RNDN); /* Gabcke's bound 1/(3322 t^7) */
        mpfr_pow_si(x, x, -7, MPFR_RNDN);
        mpfr_div_ui(x, x, 3322, MPFR_RNDN);
        expansion(a, tt, 3, pi);
        mpfr_sub(a, a, th.m, MPFR_RNDN);
        mpfr_abs(a, a, MPFR_RNDN);
        mpfr_sub(a, a, th.r, MPFR_RNDN);
        assert(mpfr_cmp(a, x) < 0);
        expansion(a, tt, 2, pi); /* last term dropped */
        mpfr_sub(a, a, th.m, MPFR_RNDN);
        mpfr_abs(a, a, MPFR_RNDN);
        assert(mpfr_cmp(a, x) > 0);
    }
    /* A wrong offset is visible far above the radius. */
    zz_faults f = {.theta_offset = 0.01, .skip_interval = ZZ_NO_SKIP};
    zz_set_d(&t, 1000.0);
    assert(zz_theta(ctx, &th, &t) == ZZ_OK);
    mpfr_set(a, th.m, MPFR_RNDN);
    zz_ctx_set_faults(ctx, &f);
    assert(zz_theta(ctx, &th, &t) == ZZ_OK);
    zz_ctx_set_faults(ctx, NULL);
    mpfr_sub(a, th.m, a, MPFR_RNDN);
    assert(fabs(mpfr_get_d(a, MPFR_RNDN) - 0.01) < 1e-12);
    zz_clear(&t);
    zz_clear(&th);
    mpfr_clears(x, a, pi, (mpfr_ptr)0);
    printf("[PASS] theta (Stirling) matches Gabcke's expansion within 1/(3322 t^7)\n");
}

/* Each published zero gamma_k: Z(gamma_k - 1e-20) and Z(gamma_k + 1e-20)
 * have opposite proven signs, so the first 20 decimals are confirmed. */
static void test_first_zeros(zz_ctx *ctx) {
    zz_ball t, z;
    zz_init(&t, PREC);
    zz_init(&z, PREC);
    mpfr_t g, eps;
    mpfr_inits2(PREC, g, eps, (mpfr_ptr)0);
    mpfr_set_str(eps, "1e-20", 10, MPFR_RNDN);
    int prev = -1; /* Z < 0 before the first zero */
    for (int k = 0; k < 10; k++) {
        mpfr_set_str(g, first_zeros[k], 10, MPFR_RNDN);
        mpfr_sub(t.m, g, eps, MPFR_RNDN);
        mpfr_set_zero(t.r, 1);
        assert(zz_z_em(ctx, &z, &t) == ZZ_OK);
        int below = zz_sign(&z);
        mpfr_add(t.m, g, eps, MPFR_RNDN);
        assert(zz_z_em(ctx, &z, &t) == ZZ_OK);
        int above = zz_sign(&z);
        assert(below == prev && above == -below);
        prev = above;
    }
    zz_clear(&t);
    zz_clear(&z);
    mpfr_clears(g, eps, (mpfr_ptr)0);
    printf("[PASS] first ten published zeros, to 20 decimals\n");
}

/* Gram's law (-1)^n Z(g_n) > 0 holds for n = -1..125 and first fails at 126,
 * with the sign proven on a certified enclosure of each g_n. */
static void test_gram_law(zz_ctx *ctx) {
    mpfr_t lo, hi;
    mpfr_inits2(PREC + 32, lo, hi, (mpfr_ptr)0);
    for (long n = -1; n <= 135; n++) {
        assert(zz_gram_point(ctx, n, lo, hi) == ZZ_OK);
        int s = zz_z_sign_interval(ctx, lo, hi);
        assert(s != 0);
        int good = s == ((n % 2 == 0) ? 1 : -1);
        if (n == 126) {
            assert(!good);
            assert(mpfr_cmp_d(lo, 282.0) > 0 && mpfr_cmp_d(hi, 283.0) < 0);
            assert(fabs(mpfr_get_d(lo, MPFR_RNDN) - 282.0) < 0.5); /* A002505 */
        } else if (n == 134) {
            assert(!good);
        } else {
            assert(good);
        }
    }
    mpfr_clears(lo, hi, (mpfr_ptr)0);
    printf("[PASS] Gram's law first fails at g_126 (and next at g_134)\n");
}

/* Lehmer's pair near 7005.06 and 7005.10: Z changes sign twice. */
static void test_lehmer_pair(zz_ctx *ctx) {
    int err = 0;
    int a = zz_z_sign(ctx, 7005.04, ZZ_METHOD_EM, &err);
    assert(err == ZZ_OK);
    int b = zz_z_sign(ctx, 7005.08, ZZ_METHOD_EM, &err);
    assert(err == ZZ_OK);
    int c = zz_z_sign(ctx, 7005.12, ZZ_METHOD_EM, &err);
    assert(err == ZZ_OK);
    assert(a != 0 && b == -a && c == a);
    /* Riemann-Siegel decides the same signs here. */
    zz_ball t, z;
    zz_init(&t, PREC);
    zz_init(&z, PREC);
    zz_set_d(&t, 7005.08);
    assert(zz_z_rs(ctx, &z, &t) == ZZ_OK && zz_sign(&z) == b);
    zz_clear(&t);
    zz_clear(&z);
    printf("[PASS] Lehmer's pair: sign changes in (7005.04, 7005.08) and (7005.08, 7005.12)\n");
}

/* RS (with Gabcke's claimed bound) and EM enclosures must overlap. */
static void test_rs_against_em(zz_ctx *ctx) {
    zz_ball t, zr, ze;
    zz_init(&t, PREC);
    zz_init(&zr, PREC);
    zz_init(&ze, PREC);
    mpfr_t d, r;
    mpfr_inits2(PREC, d, r, (mpfr_ptr)0);
    int checked = 0;
    for (double x = 200.5; x < 2000.0; x += 37.3) {
        zz_set_d(&t, x);
        if (zz_z_rs(ctx, &zr, &t) != ZZ_OK) {
            continue;
        }
        assert(zz_z_em(ctx, &ze, &t) == ZZ_OK);
        mpfr_sub(d, zr.m, ze.m, MPFR_RNDN);
        mpfr_abs(d, d, MPFR_RNDN);
        mpfr_add(r, zr.r, ze.r, MPFR_RNDU);
        assert(mpfr_cmp(d, r) <= 0);
        checked++;
    }
    assert(checked > 40);
    /* With the n = N term dropped, the RS enclosure misses the EM one. */
    zz_faults f = {.drop_term = 2, .skip_interval = ZZ_NO_SKIP};
    zz_ctx_set_faults(ctx, &f);
    zz_set_d(&t, 285.75172222589299);
    assert(zz_z_rs(ctx, &zr, &t) == ZZ_OK);
    assert(zz_z_em(ctx, &ze, &t) == ZZ_OK); /* EM is unaffected by drop_term 2 */
    zz_ctx_set_faults(ctx, NULL);
    mpfr_sub(d, zr.m, ze.m, MPFR_RNDN);
    mpfr_abs(d, d, MPFR_RNDN);
    mpfr_add(r, zr.r, ze.r, MPFR_RNDU);
    assert(mpfr_cmp(d, r) > 0);
    /* Below 200 RS refuses. */
    zz_set_d(&t, 199.0);
    assert(zz_z_rs(ctx, &zr, &t) == ZZ_ERR_ARGS);
    zz_clear(&t);
    zz_clear(&zr);
    zz_clear(&ze);
    mpfr_clears(d, r, (mpfr_ptr)0);
    printf("[PASS] Riemann-Siegel agrees with Euler-Maclaurin within the claimed bound\n");
}

static int run_verify(double height, int method, const zz_faults *f, long *n100) {
    double cps[1] = {100.0};
    zz_verify_params p;
    zz_verify_params_default(&p, height);
    p.method = method;
    p.checkpoints = cps;
    p.n_checkpoints = 1;
    if (f != NULL) {
        p.faults = *f;
    }
    p.work = work;
    p.work_cap = WORK_CAP;
    assert(zz_verify_work_size(height, p.turing_intervals) <= WORK_CAP);
    zz_verify_result r;
    int st = zz_verify(&p, &r);
    if (st == ZZ_OK) {
        assert(r.reason == NULL);
        assert(r.sign_changes_to_t0 == r.turing_bound);
        assert(r.t0 >= height && r.t1 > r.t0);
        *n100 = r.checkpoint_counts[0];
    } else {
        assert(r.reason != NULL);
    }
    return st;
}

static void test_verify(void) {
    long n100 = 0;
    /* Euler-Maclaurin, with outputs whose capacity is smaller than the
     * number found (the totals are still reported). */
    zz_verify_params p;
    zz_verify_result r;
    zz_bracket br[8];
    long bad[2];
    double cp100 = 100.0;
    zz_verify_params_default(&p, 600.0);
    p.work = work;
    p.work_cap = WORK_CAP;
    p.checkpoints = &cp100;
    p.n_checkpoints = 1;
    p.brackets = br;
    p.brackets_cap = 8;
    p.bad_gram = bad;
    p.bad_gram_cap = 2;
    assert(zz_verify(&p, &r) == ZZ_OK && r.reason == NULL);
    assert(r.zeros_to_height == 341 && r.checkpoint_counts[0] == 29); /* A013629 */
    assert(r.sign_changes_to_t0 == r.turing_bound && r.t0 >= 600.0 && r.t1 > r.t0);
    assert(r.n_brackets > 341 && r.n_bad_gram > 2);
    assert(bad[0] == 126 && bad[1] == 134);
    assert(br[0].lo < 14.134725141734694 && 14.134725141734693 < br[0].hi);
    assert(run_verify(600.0, ZZ_METHOD_RS_GABCKE, NULL, &n100) == ZZ_OK && n100 == 29);
    printf("[PASS] verification to T = 600 (both methods), N(100) = 29, N(600) = 341\n");

    /* Negative cases: each defect must make verification fail. */
    zz_faults skip = {.skip_interval = -1}; /* 14.13... in [g_-1, g_0) */
    assert(run_verify(600.0, ZZ_METHOD_EM, &skip, &n100) == ZZ_ERR_TURING);
    /* Past t0 a skipped interval only weakens Turing's integral; the count
     * up to t0 is still complete, so the proof may still go through. With
     * T = 600, t0 = g_340 and g_420 lies inside Turing's range. */
    skip.skip_interval = 420;
    assert(run_verify(600.0, ZZ_METHOD_EM, &skip, &n100) == ZZ_OK && n100 == 29);
    printf("[PASS] a skipped Gram interval fails Turing's check\n");

    /* A wrong theta. Euler-Maclaurin computes zeta without theta, so
     * Im(e^{i theta} zeta) stops containing 0 for any offset above the radius
     * (1e-6 here). In RS mode the faults act only at t >= 200, where Z comes
     * from Riemann-Siegel; the Euler-Maclaurin spot checks catch them. Without
     * the spot checks an offset of 0.05 kept all signs (and the count of 341)
     * unchanged, which is why they exist. */
    zz_faults theta = {.theta_offset = 1e-6, .skip_interval = ZZ_NO_SKIP};
    assert(run_verify(600.0, ZZ_METHOD_EM, &theta, &n100) == ZZ_ERR_INCONSISTENT);
    theta.min_t = 200.0;
    const double offsets[3] = {1e-6, 0.05, 3.141592653589793};
    for (int i = 0; i < 3; i++) {
        theta.theta_offset = offsets[i];
        assert(run_verify(600.0, ZZ_METHOD_RS_GABCKE, &theta, &n100) == ZZ_ERR_INCONSISTENT);
    }
    printf("[PASS] a wrong theta breaks the verification\n");

    /* A dropped term: Euler-Maclaurin's k = 1 term (Im check), or the n = N
     * term of the Riemann-Siegel main sum, which breaks the count: a Gram
     * block near t = 230 loses its sign changes before the first spot check
     * (t = 285.75) is reached. test_rs_against_em checks that the spot check
     * would also catch it. */
    zz_faults drop = {.drop_term = 1, .skip_interval = ZZ_NO_SKIP};
    assert(run_verify(600.0, ZZ_METHOD_EM, &drop, &n100) == ZZ_ERR_INCONSISTENT);
    drop.drop_term = 2;
    drop.min_t = 200.0;
    assert(run_verify(600.0, ZZ_METHOD_RS_GABCKE, &drop, &n100) == ZZ_ERR_ROSSER);
    printf("[PASS] a dropped term breaks the verification\n");

    zz_verify_params_default(&p, 0.0);
    p.work = work;
    p.work_cap = WORK_CAP;
    assert(zz_verify(&p, &r) == ZZ_ERR_ARGS);
    zz_verify_params_default(&p, 1000.0);
    assert(zz_verify(&p, &r) == ZZ_ERR_ARGS); /* no workspace */
    p.work = work;
    p.work_cap = WORK_CAP;
    p.turing_intervals = 2;
    assert(zz_verify(&p, &r) == ZZ_ERR_ARGS);
    double bad_cp = 2000.0;
    zz_verify_params_default(&p, 1000.0);
    p.work = work;
    p.work_cap = WORK_CAP;
    p.checkpoints = &bad_cp;
    p.n_checkpoints = 1;
    assert(zz_verify(&p, &r) == ZZ_ERR_ARGS);
    /* A workspace that runs out fails cleanly. */
    zz_verify_params_default(&p, 600.0);
    p.work = work;
    p.work_cap = 100;
    assert(zz_verify(&p, &r) == ZZ_ERR_ARGS && r.reason != NULL);

    printf("[PASS] argument checks\n");
}

int main(void) {
    printf("=== crypto_zeta tests ===\n");
    test_balls();
    static zz_ctx context;
    zz_ctx *ctx = &context;
    assert(zz_ctx_init(ctx, 16) == ZZ_ERR_ARGS);
    assert(zz_ctx_init(ctx, PREC) == ZZ_OK);
    test_theta(ctx);
    test_first_zeros(ctx);
    test_gram_law(ctx);
    test_lehmer_pair(ctx);
    test_rs_against_em(ctx);
    zz_ctx_clear(ctx);
    test_verify();
    mpfr_free_cache();
    printf("All crypto_zeta tests passed.\n");
    return 0;
}
