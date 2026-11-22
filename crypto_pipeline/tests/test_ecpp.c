/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:655ca414-6698-4373-b629-98a1c4c38d95
 */

/* Smoke and contract tests for crypto_ecpp.h. Kept small enough to run under
 * Valgrind; tests/test_ecpp.py does the large chains and the forgeries. */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_ecpp.h"

/* A 100-bit prime and its two-step chain, from ecpp_cert.provable_prime(100,
 * random.Random(5)). The second q, 0xef56230c3e5, is a 64-bit leaf. */
static limb_t n0[2] = {0x71e0c07e9e115e4bULL, 0xa206f5c66ULL};
static limb_t a0[2] = {0x88b6719788b46a67ULL, 0x7a29e49cbULL};
static limb_t x0[2] = {0x78f845f57b3120dfULL, 0x2b4653252ULL};
static limb_t y0[2] = {0x29b7caf68e7bb682ULL, 0x96d2bbca4ULL};
static limb_t k0[2] = {0xff530ULL, 0};
static limb_t q0[2] = {0xbeab40d87ebd7525ULL, 0xa274ULL};
static limb_t a1[2] = {0xc70a68c8859c23eeULL, 0x5cbeULL};
static limb_t x1[2] = {0xe4cc4132f7108e96ULL, 0x621aULL};
static limb_t y1[2] = {0xf4906f6fce327e46ULL, 0x77bdULL};
static limb_t k1[2] = {0xadc450429ULL, 0};
static limb_t q1[2] = {0xef56230c3e5ULL, 0};

static int run(const crypto_ecpp_step *steps, size_t count, uint64_t cap, size_t *failed) {
    *failed = 77;
    errno = 0;
    int rc = crypto_ecpp_verify(steps, count, cap, failed);
    if (rc == 0) {
        assert(errno == 0 && *failed == 77);
        return 0;
    }
    assert(*failed < count);
    return errno;
}

int main(void) {
    printf("=== crypto_ecpp tests ===\n");
    crypto_ecpp_step chain[2] = {{n0, a0, x0, y0, k0, q0, 2}, {q0, a1, x1, y1, k1, q1, 2}};
    size_t failed;

    /* 1. The valid chain, and its exact work need. */
    assert(run(chain, 2, 0, &failed) == 0);
    /* 2 * (bits(k) + bits(q)) per step: k0 20, q0 80, k1 36, q1 44 bits. */
    assert(run(chain, 2, 2 * (20 + 80) + 2 * (36 + 44), &failed) == 0);
    assert(run(chain, 2, 2 * (20 + 80) + 2 * (36 + 44) - 1, &failed) == ERANGE && failed == 1);
    printf("[PASS] valid chain\n");

    /* 2. One defect at a time. */
    limb_t bad[2];
    crypto_ecpp_step s;

    s = chain[0]; bad[0] = k0[0] + 1; bad[1] = 0; s.k = bad;          /* wrong cofactor */
    crypto_ecpp_step c1[2] = {s, chain[1]};
    assert(run(c1, 2, 0, &failed) == EDOM && failed == 0);

    s = chain[0]; bad[0] = x0[0] ^ 1; bad[1] = x0[1]; s.x = bad;      /* different curve */
    c1[0] = s;
    assert(run(c1, 2, 0, &failed) == EDOM);

    s = chain[0]; s.x = n0;                                          /* x = N: out of range */
    c1[0] = s;
    assert(run(c1, 2, 0, &failed) == EDOM);

    limb_t zero[2] = {0, 0}, one[2] = {1, 0};
    s = chain[0]; s.a = zero; s.x = one; s.y = one;                  /* y^2 = x^3: singular */
    c1[0] = s;
    assert(run(c1, 2, 0, &failed) == EDOM);

    s = chain[0]; s.k = zero;
    c1[0] = s;
    assert(run(c1, 2, 0, &failed) == EDOM);

    limb_t n3[2] = {0x71e0c07e9e115e4bULL - 1, 0xa206f5c66ULL};      /* even N */
    s = chain[0]; s.n = n3;
    c1[0] = s;
    assert(run(c1, 2, 0, &failed) == EDOM);

    limb_t small_q[2] = {1000003, 0};                                /* far below the bound */
    s = chain[0]; s.q = small_q;
    c1[0] = s;
    assert(run(c1, 2, 0, &failed) == EDOM);

    c1[0] = chain[0]; c1[1] = chain[1]; c1[1].n = n0;                /* broken link */
    assert(run(c1, 2, 0, &failed) == EINVAL && failed == 0);

    assert(run(chain, 1, 0, &failed) == EINVAL && failed == 0);       /* ends above 2^64 */

    limb_t q_composite[2] = {0xef56230c3e5ULL + 2, 0};
    c1[0] = chain[0]; c1[1] = chain[1]; c1[1].q = q_composite;
    assert(run(c1, 2, 0, &failed) == EDOM && failed == 1);
    printf("[PASS] rejections\n");

    /* 3. Arguments. */
    failed = 5;
    errno = 0;
    assert(crypto_ecpp_verify(NULL, 1, 0, &failed) == -1 && errno == EINVAL && failed == 5);
    assert(crypto_ecpp_verify(chain, 0, 0, &failed) == -1 && errno == EINVAL && failed == 5);
    c1[0] = chain[0]; c1[0].y = NULL;
    assert(crypto_ecpp_verify(c1, 1, 0, &failed) == -1 && errno == EINVAL && failed == 0);
    c1[0] = chain[0]; c1[0].limbs = 0;
    assert(crypto_ecpp_verify(c1, 1, 0, NULL) == -1 && errno == EINVAL);
    assert(crypto_ecpp_verify(chain, 2, 0, NULL) == 0);
    printf("[PASS] arguments\n");

    /* 4. Scalar hook: 0 * P is the identity (0:1:0); bad modulus refused. */
    limb_t X[2], Y[2], Z[2], b0[2] = {5, 0};
    assert(crypto_ecpp_scalar_mul(X, Y, Z, a0, b0, x0, y0, zero, n0, 2) == 0);
    assert(X[0] == 0 && X[1] == 0 && Y[0] == 1 && Y[1] == 0 && Z[0] == 0 && Z[1] == 0);
    assert(crypto_ecpp_scalar_mul(X, Y, Z, a0, b0, x0, y0, one, n3, 2) == -1 && errno == EINVAL);
    printf("[PASS] scalar hook\n");

    printf("=== all crypto_ecpp tests passed ===\n");
    return 0;
}
