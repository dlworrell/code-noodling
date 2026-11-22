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
#include "crypto_prime.h"
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

/* ---- multi-limb helpers ------------------------------------------------ */

static int valid_count(size_t n) { return n > 0 && n <= CRYPTO_MAX_LIMBS; }

static size_t significant(const limb_t *a, size_t n) {
    while (n > 1 && a[n - 1] == 0) --n;
    return n;
}

static void copy_limbs(limb_t *dst, const limb_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

static void set_small(limb_t *dst, limb_t value, size_t n) {
    dst[0] = value;
    for (size_t i = 1; i < n; ++i) dst[i] = 0;
}

static int is_zero(const limb_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) if (a[i]) return 0;
    return 1;
}

static int compare(const limb_t *a, const limb_t *b, size_t n) {
    for (size_t i = n; i-- > 0;) {
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}

/* Compares a (an limbs) with b (bn limbs) as integers. */
static int same_value(const limb_t *a, size_t an, const limb_t *b, size_t bn) {
    an = significant(a, an);
    bn = significant(b, bn);
    return an == bn && compare(a, b, an) == 0;
}

/* r = a - b mod 2^(64n); returns the borrow. r may alias a. */
static limb_t subtract(limb_t *r, const limb_t *a, const limb_t *b, size_t n) {
    limb_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t sub = (dlimb_t)b[i] + borrow;
        limb_t next = (dlimb_t)a[i] < sub;
        r[i] = (limb_t)((dlimb_t)a[i] - sub);
        borrow = next;
    }
    return borrow;
}

static void shift_right_one(limb_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = (a[i] >> 1) | (i + 1 < n ? a[i + 1] << 63 : 0);
}

/* wide (2n limbs) = a * b (n limbs each), schoolbook. */
static void multiply_full(limb_t *wide, const limb_t *a, const limb_t *b, size_t n) {
    for (size_t i = 0; i < 2 * n; ++i) wide[i] = 0;
    for (size_t i = 0; i < n; ++i) {
        limb_t carry = 0;
        for (size_t j = 0; j < n; ++j) {
            dlimb_t t = (dlimb_t)a[i] * b[j] + wide[i + j] + carry;
            wide[i + j] = (limb_t)t;
            carry = (limb_t)(t >> 64);
        }
        wide[i + n] = carry;
    }
}

/* quot = a / d and rem = a mod d (all n limbs, d nonzero), by binary long
 * division. Invariant rem < d; after the shift rem < 2d, so one subtraction
 * restores it, using the bit shifted out when rem overflowed n limbs. */
static void divide(limb_t *quot, limb_t *rem, const limb_t *a, const limb_t *d, size_t n) {
    set_small(quot, 0, n);
    set_small(rem, 0, n);
    for (size_t bit = 64 * n; bit-- > 0;) {
        limb_t carry = rem[n - 1] >> 63;
        for (size_t i = n; i-- > 1;) rem[i] = (rem[i] << 1) | (rem[i - 1] >> 63);
        rem[0] = (rem[0] << 1) | ((a[bit / 64] >> (bit % 64)) & 1);
        if (carry || compare(rem, d, n) >= 0) {
            (void)subtract(rem, rem, d, n);
            quot[bit / 64] |= (limb_t)1 << (bit % 64);
        }
    }
}

/* g = gcd(a, b) for odd b (all n limbs), binary GCD. b stays odd throughout. */
static void gcd_odd(limb_t *g, const limb_t *a_in, const limb_t *b_in, size_t n) {
    limb_t a[n], b[n];
    copy_limbs(a, a_in, n);
    copy_limbs(b, b_in, n);
    while (!is_zero(a, n)) {
        while ((a[0] & 1) == 0) shift_right_one(a, n);
        if (compare(a, b, n) < 0) {
            for (size_t i = 0; i < n; ++i) { limb_t t = a[i]; a[i] = b[i]; b[i] = t; }
        }
        (void)subtract(a, a, b, n);
    }
    copy_limbs(g, b, n);
}

/* ---- one node ------------------------------------------------------------ */

static int fail(int code) { errno = code; return -1; }

/* Checks one node's arithmetic: Pocklington's conditions, and 64-bit leaves.
 * Children are checked by the caller. Holds no buffers across recursion. */
static int check_node(const crypto_pocklington_node *node, uint64_t *used, uint64_t max_modexps) {
    if (!node->n || !valid_count(node->limbs) || !node->factors ||
        node->factor_count == 0 || node->factor_count > CRYPTO_CERT_MAX_FACTORS) return fail(EINVAL);
    const size_t width = node->limbs;
    const limb_t *big_n = node->n;
    const size_t n = significant(big_n, width);
    if (!(big_n[0] & 1) || (n == 1 && big_n[0] <= 2)) return fail(EDOM);

    limb_t n_minus_1[n], f[n], next[n], one[n], wide[2 * n];
    limb_t quot[n], rem[n], x[n], y[n], x_minus_1[n], g[n];
    copy_limbs(n_minus_1, big_n, n);
    n_minus_1[0] -= 1;                                 /* N is odd: no borrow. */
    set_small(one, 1, n);
    set_small(f, 1, n);

    /* Range checks, leaves, and F = product of q^exponent (must fit: F <= N - 1). */
    for (size_t k = 0; k < node->factor_count; ++k) {
        const crypto_pocklington_factor *fac = &node->factors[k];
        if (!fac->prime || !fac->base || fac->exponent == 0 || fac->child < -1) return fail(EINVAL);
        if (significant(fac->prime, width) > n || significant(fac->base, width) > n) return fail(EDOM);
        const limb_t *q = fac->prime, *a = fac->base;
        if (compare(q, big_n, n) >= 0 || (significant(q, n) == 1 && q[0] < 2)) return fail(EDOM);
        if (compare(a, n_minus_1, n) >= 0 || (significant(a, n) == 1 && a[0] < 2)) return fail(EDOM);
        if (fac->child == -1) {
            if (significant(q, n) != 1) return fail(EINVAL);
            if (!crypto_is_prime_u64(q[0])) return fail(EDOM);
        }
        if (fac->exponent > 64 * n) return fail(EDOM);   /* q >= 2, so F would exceed N. */
        for (uint32_t e = 0; e < fac->exponent; ++e) {
            multiply_full(wide, f, q, n);
            if (!is_zero(wide + n, n)) return fail(EDOM); /* F > N - 1: cannot divide it. */
            copy_limbs(next, wide, n);
            copy_limbs(f, next, n);
        }
    }

    /* F divides N - 1, and F^2 > N. */
    divide(quot, rem, n_minus_1, f, n);
    if (!is_zero(rem, n)) return fail(EDOM);
    multiply_full(wide, f, f, n);
    if (is_zero(wide + n, n) && compare(wide, big_n, n) <= 0) return fail(EDOM);

    /* For each q: a^(N-1) = 1 and gcd(a^((N-1)/q) - 1, N) = 1. */
    for (size_t k = 0; k < node->factor_count; ++k) {
        const crypto_pocklington_factor *fac = &node->factors[k];
        *used += 2;
        if (max_modexps != 0 && *used > max_modexps) return fail(ERANGE);
        divide(quot, rem, n_minus_1, fac->prime, n);    /* exact: q | F | N - 1 */
        if (crypto_mod_exp(x, fac->base, quot, n, big_n, n) != 0) return -1;
        if (crypto_mod_exp(y, x, fac->prime, n, big_n, n) != 0) return -1;
        if (compare(y, one, n) != 0) return fail(EDOM);
        (void)subtract(x_minus_1, x, one, n);          /* x >= 1 because x^q = 1. */
        gcd_odd(g, x_minus_1, big_n, n);
        if (compare(g, one, n) != 0) return fail(EDOM);
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
            !same_value(nodes[c].n, nodes[c].limbs, fac->prime, node->limbs)) {
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
