/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:3541ecdf-17e1-4ae9-9f93-0e5e772ccafa
 */

/* Smoke and contract tests for crypto_cert.h. Kept small enough to run under
 * Valgrind; tests/test_pocklington.py does the large nested certificates. */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_cert.h"

static int trial_division_is_prime(uint64_t v) {
    if (v < 2) return 0;
    for (uint64_t f = 2; f * f <= v; ++f) {
        if (v % f == 0) return 0;
    }
    return 1;
}

static int verify_one(const crypto_pocklington_node *node, uint64_t cap) {
    size_t failed = 77;
    errno = 0;
    int rc = crypto_pocklington_verify(node, 1, 0, cap, &failed);
    if (rc == 0) assert(failed == 77 && errno == 0);
    else assert(failed == 0);
    return rc == 0 ? 0 : errno;
}

int main(void) {
    printf("=== crypto_cert tests ===\n");

    /* 1. 64-bit leaves agree with trial division below 3000, and reject the
     *    strong pseudoprimes to small base sets. */
    for (uint64_t v = 0; v < 3000; ++v) {
        assert(crypto_is_prime_u64(v) == trial_division_is_prime(v));
    }
    assert(!crypto_is_prime_u64(3215031751ULL));            /* spsp(2,3,5,7) */
    assert(!crypto_is_prime_u64(3825123056546413051ULL));   /* spsp(2..23) */
    assert(!crypto_is_prime_u64(UINT64_MAX));
    assert(crypto_is_prime_u64(18446744073709551557ULL));   /* 2^64 - 59 */
    printf("[PASS] 64-bit leaves\n");

    /* 2. 65537 = 2^16 + 1: F = 2^16, base 3. */
    limb_t n65537[1] = {65537}, two[1] = {2}, three[1] = {3};
    crypto_pocklington_factor f65537 = {two, three, 16, -1};
    crypto_pocklington_node node = {n65537, 1, &f65537, 1};
    assert(verify_one(&node, 0) == 0);
    printf("[PASS] 65537\n");

    /* 3. Each defect is rejected with its documented code. */
    limb_t four[1] = {4}, n65535[1] = {65535}, big_q[2] = {1, 1};
    crypto_pocklington_factor bad = f65537;
    node.factors = &bad;
    bad.exponent = 9;                                         /* F^2 = 2^18 > N: still valid */
    assert(verify_one(&node, 0) == 0);
    bad.exponent = 8;                                         /* F^2 = 2^16 < N */
    assert(verify_one(&node, 0) == EDOM);
    bad = f65537; bad.exponent = 17;                          /* 2^17 does not divide N - 1 */
    assert(verify_one(&node, 0) == EDOM);
    bad = f65537; bad.base = four;                            /* 4 is a square: x = 1 */
    assert(verify_one(&node, 0) == EDOM);
    bad = f65537; bad.child = 0;                              /* child must prove q = 2 */
    assert(verify_one(&node, 0) == EINVAL);
    bad = f65537; bad.exponent = 0;
    assert(verify_one(&node, 0) == EINVAL);
    bad = f65537; bad.prime = NULL;
    assert(verify_one(&node, 0) == EINVAL);
    node.factors = &f65537;
    node.n = n65535;                                          /* composite, F^2 > N */
    assert(verify_one(&node, 0) == EDOM);
    node.n = n65537;
    assert(verify_one(&node, 1) == ERANGE);                   /* needs 2 modexps */
    assert(verify_one(&node, 2) == 0);

    /* q of two limbs with child = -1 is malformed (q >= 2^64). */
    limb_t n2[2] = {65537, 0x100}, a2[2] = {3, 0}, two2[2] = {2, 0};
    crypto_pocklington_factor wide[2] = {{two2, a2, 1, -1}, {big_q, a2, 1, -1}};
    crypto_pocklington_node wide_node = {n2, 2, wide, 2};
    assert(verify_one(&wide_node, 0) == EINVAL);
    printf("[PASS] rejections\n");

    /* 4. Structural arguments. */
    size_t failed = 5;
    errno = 0;
    assert(crypto_pocklington_verify(NULL, 1, 0, 0, &failed) == -1 && errno == EINVAL);
    assert(crypto_pocklington_verify(&node, 0, 0, 0, &failed) == -1 && errno == EINVAL);
    assert(crypto_pocklington_verify(&node, 1, 1, 0, &failed) == -1 && errno == EINVAL);
    assert(failed == 5);
    assert(crypto_pocklington_verify(&node, 1, 0, 0, NULL) == 0);
    printf("[PASS] arguments\n");

    printf("=== all crypto_cert tests passed ===\n");
    return 0;
}
