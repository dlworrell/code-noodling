/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:3541ecdf-17e1-4ae9-9f93-0e5e772ccafa
 */

/* Smoke and contract tests for crypto_cert.h, including the multi-node
 * structural cases (shared child, bad child index, cycle, depth limit), so they
 * also run under Valgrind and ASan/UBSan. Kept small enough for Valgrind;
 * tests/test_pocklington.py does the large nested certificates. */

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

/* Verifies a multi-node certificate; returns 0 or the errno, and the failing
 * node index through *failed (left at 77 on success). */
static int verify_tree(const crypto_pocklington_node *nodes, size_t count, size_t root,
                       uint64_t cap, size_t *failed) {
    *failed = 77;
    errno = 0;
    int rc = crypto_pocklington_verify(nodes, count, root, cap, failed);
    if (rc == 0) assert(*failed == 77 && errno == 0);
    return rc == 0 ? 0 : errno;
}

/* A chain for the depth limit: N_0 = 65537 and N_(i+1) = 2 k_i N_i + 1, each
 * prime, proven by F = N_i with base 2 (N_0 by F = 2^16, base 3). The k_i are
 * the smallest that give a prime; they were found in Python, and the
 * certificate check below proves every N_i prime independently. */
#define CHAIN_NODES 66u
#define CHAIN_LIMBS 7u
static const uint16_t chain_k[CHAIN_NODES - 1] = {
    7, 4, 1, 10, 3, 29, 7, 1, 4, 18, 32, 9, 3, 23, 4, 3, 21, 5, 16, 60, 9, 47,
    27, 62, 33, 86, 78, 158, 30, 35, 4, 70, 43, 69, 59, 190, 45, 80, 10, 238,
    97, 84, 92, 135, 261, 17, 28, 249, 30, 36, 158, 30, 309, 66, 131, 430, 100,
    76, 115, 196, 81, 104, 334, 399, 95};

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

    /* 5. Multi-node trees (all values fit one limb unless noted). */
    limb_t nC[1] = {65537}, nA[1] = {917519}, nB[1] = {1310741},
           nR[1] = {UINT64_C(76968305381057)};          /* A = 14 C + 1, B = 20 C + 1, R = 64 A B + 1 */
    crypto_pocklington_factor fR[2] = {{nA, two, 1, 1}, {nB, two, 1, 2}};
    crypto_pocklington_factor fA = {nC, two, 1, 3}, fB = {nC, two, 1, 3};
    crypto_pocklington_node dag[4] = {{nR, 1, fR, 2}, {nA, 1, &fA, 1}, {nB, 1, &fB, 1},
                                      {nC, 1, &f65537, 1}};
    /* Shared child: C is reachable through A and B but checked once, so the
     * tree needs 2 modexps per factor of R, A, B and C: 10, not 12. */
    assert(verify_tree(dag, 4, 0, 0, &failed) == 0);
    assert(verify_tree(dag, 4, 0, 10, &failed) == 0);
    assert(verify_tree(dag, 4, 0, 9, &failed) == ERANGE);
    /* Child index out of range: rejected at the node that names it. */
    fR[1].child = 4;
    assert(verify_tree(dag, 4, 0, 0, &failed) == EINVAL && failed == 0);
    fR[1].child = 2;
    /* Two-node cycle A -> C -> A. Every edge goes from N to a q < N, so the
     * back edge is caught by the child-value check (C's q = 2 is not A),
     * before the in-progress guard; EINVAL either way. */
    crypto_pocklington_factor f_back = {two, three, 16, 0};
    crypto_pocklington_node cycle[2] = {{nA, 1, &fA, 1}, {nC, 1, &f_back, 1}};
    fA.child = 1;
    assert(verify_tree(cycle, 2, 0, 0, &failed) == EINVAL && failed == 1);
    fA.child = 3;
    printf("[PASS] multi-node trees\n");

    /* 6. Depth: a chain of 65 nodes (depths 0..64) is accepted, 66 is not. */
    static limb_t chain_n[CHAIN_NODES][CHAIN_LIMBS];
    static crypto_pocklington_factor chain_f[CHAIN_NODES];
    static crypto_pocklington_node chain[CHAIN_NODES];
    static const limb_t two_w[CHAIN_LIMBS] = {2}, three_w[CHAIN_LIMBS] = {3};
    chain_n[0][0] = 65537;
    for (size_t i = 1; i < CHAIN_NODES; ++i) {         /* N_i = 2 k N_(i-1) + 1 */
        dlimb_t carry = 1;
        for (size_t j = 0; j < CHAIN_LIMBS; ++j) {
            dlimb_t t = (dlimb_t)chain_n[i - 1][j] * (2u * chain_k[i - 1]) + carry;
            chain_n[i][j] = (limb_t)t;
            carry = t >> 64;
        }
        assert(carry == 0);
    }
    chain_f[0] = (crypto_pocklington_factor){two_w, three_w, 16, -1};
    for (size_t i = 1; i < CHAIN_NODES; ++i) {
        chain_f[i] = (crypto_pocklington_factor){chain_n[i - 1], two_w, 1, (int32_t)(i - 1)};
    }
    for (size_t i = 0; i < CHAIN_NODES; ++i) {
        chain[i] = (crypto_pocklington_node){chain_n[i], CHAIN_LIMBS, &chain_f[i], 1};
    }
    assert(verify_tree(chain, CHAIN_NODES - 1, CHAIN_NODES - 2, 0, &failed) == 0);
    assert(verify_tree(chain, CHAIN_NODES, CHAIN_NODES - 1, 0, &failed) == EINVAL && failed == 0);
    printf("[PASS] depth limit\n");

    printf("=== all crypto_cert tests passed ===\n");
    return 0;
}
