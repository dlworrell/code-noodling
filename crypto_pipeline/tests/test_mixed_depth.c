/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:3dbef44f-9ff6-4511-b5b9-e3826c454622
 */

/* The depth rule of crypto_mixed_verify on a graph with a shared node. Built
 * with -DCRYPTO_MIXED_MAX_DEPTH=2 (see the makefile), together with its own
 * copy of crypto_mixed.c, so the rule can be tested at a small limit instead
 * of with 4097 nested primes.
 *
 * Nodes (Pocklington, n - 1 fully factored, bases from pocklington_cert.py):
 *   0  U = 36 s r + 1 = 3123675690065215185460789 (two limbs): 2^2, 3^2, s, r
 *   1  R = 32 s t + 1 = 20158208091433697:                    2^5, s, t
 *   2  S = s = 4304389 = 2^2 * 3 * 11 * 32609 + 1:             all leaves
 *   3  T = t = 2 * 17 * s + 1 = 146349227:                     2, 17, s
 * S is shared: U -> S (1 link), U -> R -> S (2), U -> R -> T -> S (3). */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_mixed.h"

static int verify(const crypto_mixed_node *nodes, size_t root, size_t *failed) {
    *failed = 77;
    errno = 0;
    return crypto_mixed_verify(nodes, 4, root, 0, 0, failed) == 0 ? 0 : errno;
}

int main(void) {
    printf("=== crypto_mixed depth tests (limit %u) ===\n", (unsigned)CRYPTO_MIXED_MAX_DEPTH);
    assert(CRYPTO_MIXED_MAX_DEPTH == 2u);

    /* Node 2: S, with its factors as leaves. */
    limb_t s1[1] = {4304389}, two1[1] = {2}, three1[1] = {3}, p11[1] = {11}, p32609[1] = {32609};
    crypto_pocklington_factor s_facs[4] = {{two1, two1, 2, -1}, {three1, two1, 1, -1},
                                           {p11, two1, 1, -1}, {p32609, two1, 1, -1}};
    crypto_pocklington_node s_node = {s1, 1, s_facs, 4};

    /* Node 3: T, whose factor s is proven by node 2. */
    limb_t t1[1] = {UINT64_C(146349227)}, p17[1] = {17};
    crypto_pocklington_factor t_facs[3] = {{two1, two1, 1, -1}, {p17, two1, 1, -1},
                                           {s1, two1, 1, 2}};
    crypto_pocklington_node t_node = {t1, 1, t_facs, 3};

    /* Node 1: R, which reaches S directly and through T. */
    limb_t r1[1] = {UINT64_C(20158208091433697)}, base3[1] = {3};
    crypto_pocklington_factor r_facs[3] = {{two1, base3, 5, -1}, {s1, two1, 1, 2}, {t1, two1, 1, 3}};
    crypto_pocklington_node r_node = {r1, 1, r_facs, 3};

    /* Node 0: U, two limbs; s is listed before r. */
    limb_t u2[2] = {UINT64_C(14729087681972117045), UINT64_C(169334)};
    limb_t two2[2] = {2, 0}, three2[2] = {3, 0}, s2[2] = {4304389, 0},
           r2[2] = {UINT64_C(20158208091433697), 0};
    crypto_pocklington_factor u_facs[4] = {{two2, two2, 2, -1}, {three2, two2, 2, -1},
                                           {s2, two2, 1, 2}, {r2, two2, 1, 1}};
    crypto_pocklington_node u_node = {u2, 2, u_facs, 4};

    crypto_mixed_node graph[4] = {{CRYPTO_MIXED_POCKLINGTON, -1, &u_node, NULL},
                                  {CRYPTO_MIXED_POCKLINGTON, -1, &r_node, NULL},
                                  {CRYPTO_MIXED_POCKLINGTON, -1, &s_node, NULL},
                                  {CRYPTO_MIXED_POCKLINGTON, -1, &t_node, NULL}};
    size_t failed = 0;

    /* Longest path from R is R -> T -> S: exactly the limit, in either order. */
    assert(verify(graph, 1, &failed) == 0);
    crypto_pocklington_factor swap = r_facs[1];
    r_facs[1] = r_facs[2];
    r_facs[2] = swap;                                            /* R: 2^5, t, s */
    assert(verify(graph, 1, &failed) == 0);
    r_facs[2] = r_facs[1];
    r_facs[1] = swap;                                            /* back to 2^5, s, t */
    printf("[PASS] a path of exactly the limit is accepted in either order\n");

    /* From U the longest path, U -> R -> T -> S, has 3 links. S is checked
     * first at depth 1 (U lists s before r). Its later visits at depths 2 and
     * 3 must still count: a checker that skips verified nodes before checking
     * depth accepts this graph. */
    assert(verify(graph, 0, &failed) == EINVAL && failed == 2);
    /* Listing r first reaches S at depth 3 on its first visit; same result. */
    swap = u_facs[2];
    u_facs[2] = u_facs[3];
    u_facs[3] = swap;
    assert(verify(graph, 0, &failed) == EINVAL && failed == 2);
    printf("[PASS] a longer path through a shared node is rejected in either order\n");

    /* Each node on its own: S has no links, T and R stay within the limit. */
    assert(verify(graph, 2, &failed) == 0);
    assert(verify(graph, 3, &failed) == 0);
    printf("[PASS] each node within the limit is accepted\n");

    printf("=== all crypto_mixed depth tests passed ===\n");
    return 0;
}
