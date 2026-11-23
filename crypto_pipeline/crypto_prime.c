/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:f7b84c41-d295-40e2-a119-5ea1eb1ba42c
 */

/* Variable-time modular exponentiation and the GRH-conditional Miller test.
 * Everything here works on public values. See crypto_prime.h for contracts
 * and docs/engineering/CRYPTO-PRIMALITY-GRH.md for the mathematics. */

#include "crypto_prime.h"
#include <errno.h>

static int valid_count(size_t n) { return n > 0 && n <= CRYPTO_MAX_LIMBS; }

static void copy_limbs(limb_t *dst, const limb_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

static void set_small(limb_t *dst, limb_t value, size_t n) {
    dst[0] = value;
    for (size_t i = 1; i < n; ++i) dst[i] = 0;
}

static int compare_limbs(const limb_t *a, const limb_t *b, size_t n) {
    for (size_t i = n; i-- > 0;) {
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}

static int equal_limbs(const limb_t *a, const limb_t *b, size_t n) {
    return compare_limbs(a, b, n) == 0;
}

/* r = a - b mod 2^(64n); returns the borrow. r may alias a. */
static limb_t subtract_limbs(limb_t *r, const limb_t *a, const limb_t *b, size_t n) {
    limb_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t sub = (dlimb_t)b[i] + borrow;
        limb_t next = (dlimb_t)a[i] < sub;
        r[i] = (limb_t)((dlimb_t)a[i] - sub);
        borrow = next;
    }
    return borrow;
}

/* Number of limbs up to and including the highest nonzero one (at least 1). */
static size_t significant_limbs(const limb_t *a, size_t n) {
    while (n > 1 && a[n - 1] == 0) --n;
    return n;
}

static size_t bit_length(const limb_t *a, size_t n) {
    n = significant_limbs(a, n);
    limb_t top = a[n - 1];
    size_t bits = 0;
    while (top) { ++bits; top >>= 1; }
    return (n - 1) * 64 + bits;
}

/* r = 2r mod m. Requires r < m. 2r < 2m, so one subtraction suffices; when
 * the doubling carries out of n limbs the true value is 2^(64n) + r and the
 * wrapped subtraction still yields 2r - m. */
static void double_mod(limb_t *r, const limb_t *m, size_t n) {
    limb_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        limb_t next = r[i] >> 63;
        r[i] = (r[i] << 1) | carry;
        carry = next;
    }
    if (carry || compare_limbs(r, m, n) >= 0) (void)subtract_limbs(r, r, m, n);
}

/* Montgomery context for an odd modulus m > 1 with R = 2^(64n).
 * one = R mod m (Montgomery form of 1), r2 = R^2 mod m. */
static void montgomery_constants(limb_t *one, limb_t *r2, const limb_t *m, size_t n) {
    set_small(one, 1, n);
    for (size_t i = 0; i < 64 * n; ++i) double_mod(one, m, n);
    copy_limbs(r2, one, n);
    for (size_t i = 0; i < 64 * n; ++i) double_mod(r2, m, n);
}

/* out = a*b*R^-1 mod m through crypto_multiply_reduce. a, b < m.
 * wide is 2n limbs of scratch; out may alias a or b. */
static int mont_mul(limb_t *out, const limb_t *a, const limb_t *b, const limb_t *m,
                    size_t n, limb_t *wide) {
    crypto_multiply_reduce(wide, a, b, m, n, 0);
    if (errno != 0) return -1;
    copy_limbs(out, wide, n);
    return 0;
}

/* acc = base_m^exponent in Montgomery form, left to right. base_m and one are
 * in Montgomery form; acc must not alias base_m. */
static int mont_pow(limb_t *acc, const limb_t *base_m, const limb_t *exponent, size_t exp_n,
                    const limb_t *one, const limb_t *m, size_t n, limb_t *wide) {
    copy_limbs(acc, one, n);
    for (size_t bit = bit_length(exponent, exp_n); bit-- > 0;) {
        if (mont_mul(acc, acc, acc, m, n, wide) != 0) return -1;
        if ((exponent[bit / 64] >> (bit % 64)) & 1) {
            if (mont_mul(acc, acc, base_m, m, n, wide) != 0) return -1;
        }
    }
    return 0;
}

static int odd_modulus_above_one(const limb_t *m, size_t n) {
    if (!(m[0] & 1)) return 0;
    return significant_limbs(m, n) > 1 || m[0] > 1;
}

int crypto_mod_exp(limb_t *result, const limb_t *base, const limb_t *exponent,
                   size_t exp_n, const limb_t *mod, size_t n) {
    if (!result || !base || !exponent || !mod || !valid_count(n) || !valid_count(exp_n)) {
        errno = EINVAL; return -1;
    }
    if (!odd_modulus_above_one(mod, n)) { errno = EDOM; return -1; }

    limb_t one[n], r2[n], unit[n], reduced[n], base_m[n], acc[n], wide[2 * n];
    set_small(unit, 1, n);
    if (crypto_mod_multiply(reduced, base, unit, mod, n) != 0) return -1;
    montgomery_constants(one, r2, mod, n);
    if (mont_mul(base_m, reduced, r2, mod, n, wide) != 0) return -1;
    if (mont_pow(acc, base_m, exponent, exp_n, one, mod, n, wide) != 0) return -1;
    if (mont_mul(acc, acc, unit, mod, n, wide) != 0) return -1;
    copy_limbs(result, acc, n);
    errno = 0;
    return 0;
}

/* 0.69315 = 13863/20000 exceeds ln 2 = 0.693147...; the bound is
 * ceil(2 * L^2 * 13863^2 / 20000^2). L <= 65536, so the numerator is at most
 * 2 * 2^32 * 192182769 < 2^61 and cannot overflow. */
uint64_t crypto_grh_witness_bound(const limb_t *candidate, size_t n) {
    if (!candidate || !valid_count(n)) { errno = EINVAL; return 0; }
    uint64_t bits = bit_length(candidate, n);
    const uint64_t num = 13863u, den = 20000u;
    uint64_t numerator = 2u * bits * bits * num * num;
    uint64_t denominator = den * den;
    errno = 0;
    return (numerator + denominator - 1) / denominator;
}

int crypto_is_prime_grh(int *is_prime, const limb_t *candidate, size_t n, uint64_t max_bases) {
    if (!is_prime || !candidate || !valid_count(n)) { errno = EINVAL; return -1; }
    n = significant_limbs(candidate, n);
    const limb_t *m = candidate;

    if (n == 1 && m[0] < 4) { *is_prime = m[0] >= 2; errno = 0; return 0; }
    if (!(m[0] & 1)) { *is_prime = 0; errno = 0; return 0; }

    /* Bases 2..last, last = min(bound, N - 2). N - 1 is never a witness. */
    uint64_t last = crypto_grh_witness_bound(m, n);
    if (n == 1 && last > m[0] - 2) last = m[0] - 2;
    uint64_t count = last >= 2 ? last - 1 : 0;
    if (max_bases != 0 && count > max_bases) { errno = ERANGE; return -1; }

    /* N - 1 = d * 2^s with d odd. */
    limb_t d[n], one[n], r2[n], minus_one[n], base[n], base_m[n], x[n], wide[2 * n];
    copy_limbs(d, m, n);
    d[0] -= 1; /* N is odd, so no borrow. */
    size_t s = 0;
    while (((d[s / 64] >> (s % 64)) & 1) == 0) ++s;
    for (size_t k = 0; k < s; ++k) {
        for (size_t i = 0; i < n; ++i) {
            d[i] = (d[i] >> 1) | (i + 1 < n ? d[i + 1] << 63 : 0);
        }
    }

    montgomery_constants(one, r2, m, n);
    (void)subtract_limbs(minus_one, m, one, n); /* Montgomery form of N - 1. */

    for (uint64_t a = 2; a <= last; ++a) {
        set_small(base, (limb_t)a, n);
        if (mont_mul(base_m, base, r2, m, n, wide) != 0) return -1;
        if (mont_pow(x, base_m, d, n, one, m, n, wide) != 0) return -1;
        if (equal_limbs(x, one, n) || equal_limbs(x, minus_one, n)) continue;
        int passed = 0;
        for (size_t j = 1; j < s; ++j) {
            if (mont_mul(x, x, x, m, n, wide) != 0) return -1;
            if (equal_limbs(x, minus_one, n)) { passed = 1; break; }
            if (equal_limbs(x, one, n)) break;
        }
        if (!passed) { *is_prime = 0; errno = 0; return 0; } /* a is a witness. */
    }
    *is_prime = 1;
    errno = 0;
    return 0;
}

/* a >>= 1 over n limbs. */
static void shift_right_one(limb_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = (a[i] >> 1) | (i + 1 < n ? a[i + 1] << 63 : 0);
}

/* a += 1 over n limbs; the caller guarantees no carry out. */
static void add_one(limb_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (++a[i] != 0) break;
    }
}

int crypto_mod_sqrt_grh(limb_t *root, const limb_t *a, const limb_t *p, size_t n,
                        uint64_t max_trials) {
    if (!root || !a || !p || !valid_count(n)) { errno = EINVAL; return -1; }
    size_t pn = significant_limbs(p, n);

    if (pn == 1 && p[0] == 2) { /* The only even prime: every value is its own root. */
        limb_t low = a[0] & 1;
        set_small(root, low, n);
        errno = 0; return 0;
    }
    if (!odd_modulus_above_one(p, pn)) { errno = EDOM; return -1; }

    limb_t unit[pn], value[pn], one[pn], r2[pn], minus_one[pn], wide[2 * pn];
    limb_t e[pn], q[pn], q1[pn], value_m[pn], z[pn], z_m[pn];
    limb_t c[pn], t[pn], r[pn], b[pn], x[pn], check[pn];
    set_small(unit, 1, pn);
    /* value = a mod p; a has n limbs, p has pn <= n. */
    {
        limb_t a_wide[n], p_wide[n], reduced[n];
        copy_limbs(p_wide, p, n);
        for (size_t i = pn; i < n; ++i) p_wide[i] = 0;
        set_small(a_wide, 1, n);
        if (crypto_mod_multiply(reduced, a, a_wide, p_wide, n) != 0) return -1;
        copy_limbs(value, reduced, pn);
    }
    int value_is_zero = 1;
    for (size_t i = 0; i < pn; ++i) value_is_zero &= value[i] == 0;
    if (value_is_zero) { set_small(root, 0, n); errno = 0; return 0; }

    montgomery_constants(one, r2, p, pn);
    (void)subtract_limbs(minus_one, p, one, pn);     /* Montgomery form of p - 1. */
    if (mont_mul(value_m, value, r2, p, pn, wide) != 0) return -1;

    /* Euler's criterion: value^((p-1)/2) must be 1 for a quadratic residue. */
    copy_limbs(e, p, pn);
    e[0] -= 1;                                       /* p is odd: no borrow. */
    shift_right_one(e, pn);
    if (mont_pow(x, value_m, e, pn, one, p, pn, wide) != 0) return -1;
    if (!equal_limbs(x, one, pn)) { errno = EDOM; return -1; } /* not a residue, or p composite */

    /* p - 1 = q * 2^s with q odd. */
    copy_limbs(q, p, pn);
    q[0] -= 1;
    size_t s = 0;
    while ((q[0] & 1) == 0) { shift_right_one(q, pn); ++s; }

    /* Least quadratic non-residue z: below 2 (ln p)^2 under ERH (Bach 1990). */
    uint64_t last = crypto_grh_witness_bound(p, pn);
    if (pn == 1 && last > p[0] - 1) last = p[0] - 1;
    uint64_t trials = 0;
    int found = 0;
    for (uint64_t candidate = 2; candidate <= last; ++candidate) {
        if (max_trials != 0 && ++trials > max_trials) { errno = ERANGE; return -1; }
        set_small(z, (limb_t)candidate, pn);
        if (mont_mul(z_m, z, r2, p, pn, wide) != 0) return -1;
        if (mont_pow(x, z_m, e, pn, one, p, pn, wide) != 0) return -1;
        if (equal_limbs(x, minus_one, pn)) { found = 1; break; }
        if (!equal_limbs(x, one, pn)) { errno = EDOM; return -1; } /* p is composite */
    }
    if (!found) { errno = EDOM; return -1; } /* p composite, or a counterexample to ERH */

    /* Tonelli-Shanks in Montgomery form. */
    copy_limbs(q1, q, pn);
    add_one(q1, pn);                                 /* q < p, so q + 1 fits. */
    shift_right_one(q1, pn);                         /* (q + 1) / 2 */
    if (mont_pow(c, z_m, q, pn, one, p, pn, wide) != 0) return -1;
    if (mont_pow(t, value_m, q, pn, one, p, pn, wide) != 0) return -1;
    if (mont_pow(r, value_m, q1, pn, one, p, pn, wide) != 0) return -1;
    size_t m = s;
    while (!equal_limbs(t, one, pn)) {
        /* Least i, 0 < i < m, with t^(2^i) = 1. */
        size_t i = 0;
        copy_limbs(x, t, pn);
        while (!equal_limbs(x, one, pn)) {
            if (++i >= m) { errno = EDOM; return -1; } /* p is composite */
            if (mont_mul(x, x, x, p, pn, wide) != 0) return -1;
        }
        copy_limbs(b, c, pn);                        /* b = c^(2^(m-i-1)) */
        for (size_t k = 0; k + i + 1 < m; ++k) {
            if (mont_mul(b, b, b, p, pn, wide) != 0) return -1;
        }
        m = i;
        if (mont_mul(c, b, b, p, pn, wide) != 0) return -1;
        if (mont_mul(t, t, c, p, pn, wide) != 0) return -1;
        if (mont_mul(r, r, b, p, pn, wide) != 0) return -1;
    }

    /* Leave Montgomery form, confirm r^2 = value, and return min(r, p - r). */
    if (mont_mul(r, r, unit, p, pn, wide) != 0) return -1;
    {
        limb_t r_m[pn], sq[pn];
        if (mont_mul(r_m, r, r2, p, pn, wide) != 0) return -1;
        if (mont_mul(sq, r_m, r_m, p, pn, wide) != 0) return -1;
        if (mont_mul(check, sq, unit, p, pn, wide) != 0) return -1;
    }
    if (!equal_limbs(check, value, pn)) { errno = EDOM; return -1; }
    limb_t other[pn];
    (void)subtract_limbs(other, p, r, pn);
    const limb_t *smaller = compare_limbs(other, r, pn) < 0 ? other : r;
    for (size_t i = 0; i < n; ++i) root[i] = i < pn ? smaller[i] : 0;
    errno = 0;
    return 0;
}
