/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:0c70c031-6e3f-4aae-bc87-a2660fae8ec5
 */

/* Smoke and contract tests for crypto_mixed.h, small enough for Valgrind and
 * ASan/UBSan. All values are single-limb; tests/test_mixed.py covers large
 * generated certificates. Values were generated with mixed_cert.py helpers and
 * are proven here by the checker itself. */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_mixed.h"

static int verify(const crypto_mixed_node *nodes, size_t count, size_t root, uint64_t modexps,
                  uint64_t point_ops, size_t *failed) {
    *failed = 77;
    errno = 0;
    int rc = crypto_mixed_verify(nodes, count, root, modexps, point_ops, failed);
    if (rc == 0) assert(*failed == 77 && errno == 0);
    return rc == 0 ? 0 : errno;
}

int main(void) {
    printf("=== crypto_mixed tests ===\n");
    size_t failed = 0;

    /* A. Pocklington root N = 42 q + 1 with F = q; q is proven by an ECPP step
     *    whose q' = 62773 is a 64-bit leaf. */
    limb_t a_n[1] = {UINT64_C(25164164803)}, a_q[1] = {UINT64_C(599146781)}, two[1] = {2};
    crypto_pocklington_factor a_fac = {a_q, two, 1, 1};
    crypto_pocklington_node a_root = {a_n, 1, &a_fac, 1};
    limb_t e_a[1] = {UINT64_C(544854974)}, e_x[1] = {UINT64_C(40260662)},
           e_y[1] = {UINT64_C(197163010)}, e_k[1] = {9544}, e_q[1] = {62773};
    crypto_ecpp_step a_step = {a_q, e_a, e_x, e_y, e_k, e_q, 1};
    crypto_mixed_node graph_a[2] = {{CRYPTO_MIXED_POCKLINGTON, -1, &a_root, NULL},
                                    {CRYPTO_MIXED_ECPP, -1, NULL, &a_step}};
    assert(verify(graph_a, 2, 0, 0, 0, &failed) == 0);
    /* Costs: 2 modexps for the root's one factor; 2 * (14 + 16) = 60 point additions. */
    assert(verify(graph_a, 2, 0, 2, 60, &failed) == 0);
    assert(verify(graph_a, 2, 0, 1, 0, &failed) == ERANGE && failed == 0);
    assert(verify(graph_a, 2, 0, 0, 59, &failed) == ERANGE && failed == 1);
    printf("[PASS] Pocklington node proven through an ECPP step\n");

    /* B. ECPP root whose q = 4304389 is proven by a Pocklington node with
     *    q - 1 = 2^2 * 3 * 11 * 32609 fully factored (all 64-bit leaves). */
    limb_t b_n[1] = {UINT64_C(10149273415921)}, b_a[1] = {UINT64_C(6977541769974)},
           b_x[1] = {UINT64_C(816993325456)}, b_y[1] = {UINT64_C(9708642625126)},
           b_k[1] = {2357888}, b_q[1] = {4304389};
    crypto_ecpp_step b_step = {b_n, b_a, b_x, b_y, b_k, b_q, 1};
    limb_t p3[1] = {3}, p11[1] = {11}, p32609[1] = {32609};
    crypto_pocklington_factor b_facs[4] = {{two, two, 2, -1}, {p3, two, 1, -1},
                                           {p11, two, 1, -1}, {p32609, two, 1, -1}};
    crypto_pocklington_node b_pock = {b_q, 1, b_facs, 4};
    crypto_mixed_node graph_b[2] = {{CRYPTO_MIXED_ECPP, 1, NULL, &b_step},
                                    {CRYPTO_MIXED_POCKLINGTON, -1, &b_pock, NULL}};
    assert(verify(graph_b, 2, 0, 0, 0, &failed) == 0);
    assert(verify(graph_b, 2, 0, 8, 90, &failed) == 0);          /* 4 factors; 2 * (22 + 23) */
    assert(verify(graph_b, 2, 0, 7, 0, &failed) == ERANGE && failed == 1);
    assert(verify(graph_b, 2, 0, 0, 89, &failed) == ERANGE && failed == 0);
    printf("[PASS] ECPP step proven through a Pocklington node\n");

    /* C. Links. */
    a_fac.child = 0;                                             /* N is not q */
    assert(verify(graph_a, 2, 0, 0, 0, &failed) == EINVAL && failed == 0);
    a_fac.child = 2;                                             /* out of range */
    assert(verify(graph_a, 2, 0, 0, 0, &failed) == EINVAL && failed == 0);
    a_fac.child = -1;                                            /* q < 2^64 may also be a leaf */
    assert(verify(graph_a, 2, 0, 0, 0, &failed) == 0);
    /* A link to a different node that is itself valid: only the value check can
     * reject it (node 2 proves 4304389, not q). */
    crypto_mixed_node graph_ab[3] = {graph_a[0], graph_a[1], graph_b[1]};
    assert(verify(graph_ab, 3, 2, 0, 0, &failed) == 0);
    a_fac.child = 2;
    assert(verify(graph_ab, 3, 0, 0, 0, &failed) == EINVAL && failed == 0);
    a_fac.child = 1;
    graph_b[0].q_child = 0;                                      /* self-link: q is not N */
    assert(verify(graph_b, 2, 0, 0, 0, &failed) == EINVAL && failed == 0);
    graph_b[0].q_child = -2;
    assert(verify(graph_b, 2, 0, 0, 0, &failed) == EINVAL && failed == 0);
    graph_b[0].q_child = 1;
    /* A composite 64-bit leaf: q = 3 * 62773. Q still has order 62773, which
     * divides q, so the step check passes and only the leaf test can reject it. */
    limb_t composite_q[1] = {UINT64_C(188319)};
    a_step.q = composite_q;
    assert(verify(graph_a, 2, 1, 0, 0, &failed) == EDOM && failed == 1);  /* node 1 on its own */
    a_step.q = e_q;
    assert(verify(graph_a, 2, 1, 0, 0, &failed) == 0);
    printf("[PASS] link errors\n");

    /* D. Each node is checked by its own method's rules. */
    limb_t bad_k[1] = {9545};
    a_step.k = bad_k;                                            /* wrong cofactor */
    assert(verify(graph_a, 2, 0, 0, 0, &failed) == EDOM && failed == 1);
    a_step.k = e_k;
    limb_t cube_root_of_one[1] = {UINT64_C(10880230342)};       /* order 3: a^((N-1)/q) = a^42 = 1 */
    a_fac.base = cube_root_of_one;
    assert(verify(graph_a, 2, 0, 0, 0, &failed) == EDOM && failed == 0);
    a_fac.base = two;
    b_facs[3].prime = b_q;                                       /* leaf q >= N is out of range */
    assert(verify(graph_b, 2, 0, 0, 0, &failed) == EDOM && failed == 1);
    b_facs[3].prime = p32609;
    printf("[PASS] node checks\n");

    /* E. Structure. */
    crypto_mixed_node bad_kind[1] = {{7u, -1, &a_root, NULL}};
    assert(verify(bad_kind, 1, 0, 0, 0, &failed) == EINVAL && failed == 0);
    crypto_mixed_node null_ecpp[1] = {{CRYPTO_MIXED_ECPP, -1, NULL, NULL}};
    assert(verify(null_ecpp, 1, 0, 0, 0, &failed) == EINVAL && failed == 0);
    failed = 5;
    errno = 0;
    assert(crypto_mixed_verify(NULL, 1, 0, 0, 0, &failed) == -1 && errno == EINVAL);
    assert(crypto_mixed_verify(graph_a, 0, 0, 0, 0, &failed) == -1 && errno == EINVAL);
    assert(crypto_mixed_verify(graph_a, 2, 2, 0, 0, &failed) == -1 && errno == EINVAL);
    assert(crypto_mixed_verify(graph_a, CRYPTO_MIXED_MAX_NODES + 1u, 0, 0, 0, &failed) == -1 &&
           errno == EINVAL);
    assert(failed == 5);
    assert(crypto_mixed_verify(graph_a, 2, 0, 0, 0, NULL) == 0);
    /* The second graph's root on its own is fine; node 1 alone proves q. */
    assert(verify(graph_b, 2, 1, 0, 0, &failed) == 0);
    printf("[PASS] structural arguments\n");

    printf("=== all crypto_mixed tests passed ===\n");
    return 0;
}
