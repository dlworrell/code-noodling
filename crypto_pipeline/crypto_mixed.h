/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:36d533b1-4076-41bd-9802-d71a3522a470
 */

#ifndef CRYPTO_MIXED_H
#define CRYPTO_MIXED_H
#include "crypto_cert.h"
#include "crypto_ecpp.h"

/* Mixed primality certificates: Pocklington nodes (crypto_cert.h) and ECPP
 * steps (crypto_ecpp.h) in one graph, so that each prime is proven by
 * whichever method suits it.
 *
 * A node proves its N prime with one method. Every prime it relies on is
 * proven either by another node or, when below 2^64, by deterministic
 * Miller-Rabin (crypto_is_prime_u64):
 *   - a Pocklington node's factors use their `child` field, which here indexes
 *     this node array (-1 for a 64-bit leaf);
 *   - an ECPP node is one step; its q is proven by node `q_child`, or by a
 *     64-bit leaf when q_child is -1.
 * Each node is checked exactly as by crypto_pocklington_verify or
 * crypto_ecpp_verify, so the soundness argument is unchanged: a linked node
 * proves the q its parent assumed prime. No hypothesis beyond those checkers'
 * is used. See docs/engineering/CRYPTO-ECPP.md, "Mixed certificates".
 *
 * Research prototype, variable-time, public values only. */
#ifdef __cplusplus
extern "C" {
#endif

#define CRYPTO_MIXED_MAX_NODES 8192u
#ifndef CRYPTO_MIXED_MAX_DEPTH            /* overridable only for tests of the walk */
#define CRYPTO_MIXED_MAX_DEPTH 4096u
#endif

enum { CRYPTO_MIXED_POCKLINGTON = 1u, CRYPTO_MIXED_ECPP = 2u };

typedef struct crypto_mixed_node {
    uint32_t kind;                                 /* CRYPTO_MIXED_POCKLINGTON or CRYPTO_MIXED_ECPP */
    int32_t q_child;                               /* ECPP: node proving q, or -1 for q < 2^64 */
    const crypto_pocklington_node *pocklington;    /* kind POCKLINGTON (child fields index this array) */
    const crypto_ecpp_step *ecpp;                  /* kind ECPP: one step */
} crypto_mixed_node;

/* Verifies that nodes[root] proves its N prime. Each node reachable from root
 * is checked once, however many times it is referenced.
 *
 * Depth: every path from root, counted in links (root at 0), must have at most
 * CRYPTO_MIXED_MAX_DEPTH links, including paths that reach an already-checked
 * shared node a second time. Whether a certificate is accepted therefore does
 * not depend on the order of factors or links; for a certificate with several
 * defects, which error and failed_node are reported may.
 *
 * Returns 0 when the certificate proves N prime. Otherwise returns -1 and sets:
 *   EINVAL  malformed: null pointer, unknown kind, count or index out of range,
 *           a child whose N differs from the prime it proves, a cycle, a
 *           path longer than CRYPTO_MIXED_MAX_DEPTH links, or a leaf (child
 *           or q_child -1) of 2^64 or more;
 *   EDOM    a mathematical check failed (as documented by the two checkers),
 *           including a 64-bit leaf that is not prime;
 *   ERANGE  more than max_modexps modular exponentiations (Pocklington) or
 *           max_point_ops point additions (ECPP) would be needed; 0 means no cap.
 * A rejected certificate says nothing about whether N is prime.
 * If failed_node is not NULL, it receives the index of the node where checking
 * stopped (unchanged on success). Variable-time. */
int crypto_mixed_verify(const crypto_mixed_node *nodes, size_t count, size_t root,
                        uint64_t max_modexps, uint64_t max_point_ops, size_t *failed_node);

#ifdef __cplusplus
}
#endif
#endif
