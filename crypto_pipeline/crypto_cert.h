/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:8554f182-05f9-4822-80d1-1962670459b2
 */

#ifndef CRYPTO_CERT_H
#define CRYPTO_CERT_H
#include "crypto_multiplier.h"

/* Unconditional primality certificates (Pocklington's theorem).
 *
 * Theorem (Pocklington). Let N > 2 and N - 1 = F * R with F > sqrt(N). If for
 * every prime q dividing F there is a base a with
 *     a^(N-1) = 1 (mod N)  and  gcd(a^((N-1)/q) - 1, N) = 1,
 * then N is prime.
 *
 * A certificate is a tree of nodes. Each node gives N and the factorization of
 * F as a list of (q, exponent, base) entries. Every q must itself be proven
 * prime: either by another node (child >= 0), or, when q < 2^64, by
 * deterministic Miller-Rabin with the twelve prime bases 2..37, which is proven
 * correct for every n < 3.18 * 10^23 (Jiang and Deng 2014; Sorenson and
 * Webster 2017). No unproven hypothesis (such as GRH) is used.
 *
 * Within a node, N, every q and every base use the node's `limbs` count
 * (little-endian base-2^64, zero-padded). Research prototype, variable-time,
 * public values only. See docs/engineering/CRYPTO-POCKLINGTON.md. */
#ifdef __cplusplus
extern "C" {
#endif

#define CRYPTO_CERT_MAX_NODES 4096u
#define CRYPTO_CERT_MAX_FACTORS 512u
#ifndef CRYPTO_CERT_MAX_DEPTH              /* overridable only for tests of the walk */
#define CRYPTO_CERT_MAX_DEPTH 64u
#endif

typedef struct crypto_pocklington_factor {
    const limb_t *prime;   /* q: node.limbs limbs, 2 <= q < N */
    const limb_t *base;    /* a: node.limbs limbs, 1 < a < N - 1 */
    uint32_t exponent;     /* q^exponent divides F; at least 1 */
    int32_t child;         /* index of the node proving q prime, or -1 when q < 2^64 */
} crypto_pocklington_factor;

typedef struct crypto_pocklington_node {
    const limb_t *n;       /* N: limbs limbs, odd, N > 2 */
    size_t limbs;          /* 1..CRYPTO_MAX_LIMBS */
    const crypto_pocklington_factor *factors;
    size_t factor_count;   /* 1..CRYPTO_CERT_MAX_FACTORS */
} crypto_pocklington_node;

/* Verifies that nodes[root] proves its N prime. Each node reachable from root
 * is checked once, however many times it is referenced.
 *
 * Depth: every path from root, counted in links (root at 0), must have at most
 * CRYPTO_CERT_MAX_DEPTH links, including paths that reach an already-checked
 * shared node a second time. Whether a certificate is accepted therefore does
 * not depend on the order of factors; for a certificate with several defects,
 * which error and failed_node are reported may.
 *
 * Returns 0 when the certificate proves N prime. Otherwise returns -1 and sets:
 *   EINVAL  malformed certificate: null pointer, count or index out of range,
 *           a child whose N differs from its q, a cycle, a path longer
 *           than CRYPTO_CERT_MAX_DEPTH links, or child = -1 for q >= 2^64;
 *   EDOM    a mathematical check failed: N even or <= 2, q or a out of range,
 *           F does not divide N - 1, F^2 <= N, a base condition fails, or a
 *           64-bit q is not prime;
 *   ERANGE  more than max_modexps modular exponentiations would be needed
 *           (0 means no cap).
 * A rejected certificate says nothing about whether N is prime.
 * If failed_node is not NULL, it receives the index of the node where checking
 * stopped (unchanged on success). Variable-time. */
int crypto_pocklington_verify(const crypto_pocklington_node *nodes, size_t node_count,
                              size_t root, uint64_t max_modexps, size_t *failed_node);

/* Deterministic primality of a 64-bit value: Miller-Rabin with bases 2..37,
 * proven correct for all n < 2^64. Returns 1 for prime, 0 otherwise. */
int crypto_is_prime_u64(uint64_t n);

#ifdef __cplusplus
}
#endif
#endif
