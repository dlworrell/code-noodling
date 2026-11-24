/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:70632e95-51e2-4aa0-83b6-abfe30ded83d
 */

/* Pocklington certificate checker. See crypto_cert.h for the contract and
 * docs/engineering/CRYPTO-POCKLINGTON.md for the mathematics. Everything here
 * is variable-time and works on public values. */

#include "crypto_cert.h"
#include "crypto_bigint.h"
#include "crypto_prime.h"
#include "crypto_checks_internal.h"
#include <errno.h>

/* ---- 64-bit leaves ---------------------------------------------------- */

static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((dlimb_t)a * b) % m);
}

static uint64_t powmod64(uint64_t base, uint64_t exponent, uint64_t m) {
    uint64_t result = 1 % m;
    base %= m;
    while (exponent) {
        if (exponent & 1) result = mulmod64(result, base, m);
        base = mulmod64(base, base, m);
        exponent >>= 1;
    }
    return result;
}

int crypto_is_prime_u64(uint64_t n) {
    static const uint64_t bases[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2) return 0;
    for (size_t i = 0; i < 12; ++i) {
        if (n == bases[i]) return 1;
        if (n % bases[i] == 0) return 0;
    }
    /* n > 37 and odd here, so every base is in 2..n-2. */
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (size_t i = 0; i < 12; ++i) {
        uint64_t x = powmod64(bases[i], d, n);
        if (x == 1 || x == n - 1) continue;
        int passed = 0;
        for (unsigned r = 1; r < s; ++r) {
            x = mulmod64(x, x, n);
            if (x == n - 1) { passed = 1; break; }
        }
        if (!passed) return 0;
    }
    return 1;
}

static int valid_count(size_t n) { return n > 0 && n <= CRYPTO_MAX_LIMBS; }

/* ---- one node ------------------------------------------------------------ */

static int fail(int code) { errno = code; return -1; }

/* Checks one node's arithmetic: Pocklington's conditions, and 64-bit leaves.
 * Children are checked by the caller. Holds no buffers across recursion. */
static int check_node(const crypto_pocklington_node *node, uint64_t *used, uint64_t max_modexps) {
    if (!node->n || !valid_count(node->limbs) || !node->factors ||
        node->factor_count == 0 || node->factor_count > CRYPTO_CERT_MAX_FACTORS) return fail(EINVAL);
    const size_t width = node->limbs;
    const limb_t *big_n = node->n;
    const size_t n = crypto_bn_significant(big_n, width);
    if (!(big_n[0] & 1) || (n == 1 && big_n[0] <= 2)) return fail(EDOM);

    limb_t n_minus_1[n], f[n], next[n], one[n], wide[2 * n];
    limb_t quot[n], rem[n], x[n], y[n], x_minus_1[n], g[n];
    crypto_bn_copy(n_minus_1, big_n, n);
    n_minus_1[0] -= 1;                                 /* N is odd: no borrow. */
    crypto_bn_set_small(one, 1, n);
    crypto_bn_set_small(f, 1, n);

    /* Range checks, leaves, and F = product of q^exponent (must fit: F <= N - 1). */
    for (size_t k = 0; k < node->factor_count; ++k) {
        const crypto_pocklington_factor *fac = &node->factors[k];
        if (!fac->prime || !fac->base || fac->exponent == 0 || fac->child < -1) return fail(EINVAL);
        if (crypto_bn_significant(fac->prime, width) > n ||
            crypto_bn_significant(fac->base, width) > n) return fail(EDOM);
        const limb_t *q = fac->prime, *a = fac->base;
        if (crypto_bn_compare(q, big_n, n) >= 0 || (crypto_bn_significant(q, n) == 1 && q[0] < 2)) return fail(EDOM);
        if (crypto_bn_compare(a, n_minus_1, n) >= 0 || (crypto_bn_significant(a, n) == 1 && a[0] < 2)) return fail(EDOM);
        if (fac->child == -1) {
            if (crypto_bn_significant(q, n) != 1) return fail(EINVAL);
            if (!crypto_is_prime_u64(q[0])) return fail(EDOM);
        }
        if (fac->exponent > 64 * n) return fail(EDOM);   /* q >= 2, so F would exceed N. */
        for (uint32_t e = 0; e < fac->exponent; ++e) {
            crypto_bn_mul_full(wide, f, q, n);
            if (!crypto_bn_is_zero(wide + n, n)) return fail(EDOM); /* F > N - 1: cannot divide it. */
            crypto_bn_copy(next, wide, n);
            crypto_bn_copy(f, next, n);
        }
    }

    /* F divides N - 1, and F^2 > N. */
    crypto_bn_divide(quot, rem, n_minus_1, f, n);
    if (!crypto_bn_is_zero(rem, n)) return fail(EDOM);
    crypto_bn_mul_full(wide, f, f, n);
    if (crypto_bn_is_zero(wide + n, n) && crypto_bn_compare(wide, big_n, n) <= 0) return fail(EDOM);

    /* For each q: a^(N-1) = 1 and gcd(a^((N-1)/q) - 1, N) = 1. */
    for (size_t k = 0; k < node->factor_count; ++k) {
        const crypto_pocklington_factor *fac = &node->factors[k];
        *used += 2;
        if (max_modexps != 0 && *used > max_modexps) return fail(ERANGE);
        crypto_bn_divide(quot, rem, n_minus_1, fac->prime, n);    /* exact: q | F | N - 1 */
        if (crypto_mod_exp(x, fac->base, quot, n, big_n, n) != 0) return -1;
        if (crypto_mod_exp(y, x, fac->prime, n, big_n, n) != 0) return -1;
        if (crypto_bn_compare(y, one, n) != 0) return fail(EDOM);
        (void)crypto_bn_sub(x_minus_1, x, one, n);          /* x >= 1 because x^q = 1. */
        crypto_bn_gcd_odd(g, x_minus_1, big_n, n);
        if (crypto_bn_compare(g, one, n) != 0) return fail(EDOM);
    }
    return 0;
}

/* ---- the tree ------------------------------------------------------------ */

enum { UNSEEN = 0, IN_PROGRESS = 1, VERIFIED = 2 };

static int verify_node(const crypto_pocklington_node *nodes, size_t node_count, size_t index,
                       unsigned depth, unsigned char *state, uint64_t *used,
                       uint64_t max_modexps, size_t *failed) {
    if (state[index] == VERIFIED) return 0;
    if (state[index] == IN_PROGRESS || depth > CRYPTO_CERT_MAX_DEPTH) {
        *failed = index; return fail(EINVAL);
    }
    state[index] = IN_PROGRESS;
    const crypto_pocklington_node *node = &nodes[index];
    if (check_node(node, used, max_modexps) != 0) { *failed = index; return -1; }
    for (size_t k = 0; k < node->factor_count; ++k) {
        const crypto_pocklington_factor *fac = &node->factors[k];
        if (fac->child < 0) continue;
        size_t c = (size_t)fac->child;
        if (c >= node_count || !nodes[c].n || !valid_count(nodes[c].limbs) ||
            !crypto_bn_same_value(nodes[c].n, nodes[c].limbs, fac->prime, node->limbs)) {
            *failed = index; return fail(EINVAL);
        }
        if (verify_node(nodes, node_count, c, depth + 1, state, used, max_modexps, failed) != 0) {
            return -1;
        }
    }
    state[index] = VERIFIED;
    return 0;
}

int crypto_pocklington_verify(const crypto_pocklington_node *nodes, size_t node_count,
                              size_t root, uint64_t max_modexps, size_t *failed_node) {
    if (!nodes || node_count == 0 || node_count > CRYPTO_CERT_MAX_NODES || root >= node_count) {
        return fail(EINVAL);
    }
    unsigned char state[node_count];
    for (size_t i = 0; i < node_count; ++i) state[i] = UNSEEN;
    uint64_t used = 0;
    size_t failed = root;
    if (verify_node(nodes, node_count, root, 0, state, &used, max_modexps, &failed) != 0) {
        if (failed_node) *failed_node = failed;
        return -1;
    }
    errno = 0;
    return 0;
}

/* Single-node check for the mixed-certificate checker (crypto_checks_internal.h).
 * Factor children are not followed; the caller links them. */
int crypto_pocklington_check_node(const crypto_pocklington_node *node, uint64_t *used,
                                  uint64_t max_modexps) {
    return check_node(node, used, max_modexps);
}
