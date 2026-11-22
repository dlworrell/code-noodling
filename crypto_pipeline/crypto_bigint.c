/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:fd15a5fc-a4e8-4877-b71f-0079d4e3a634
 */

/* Internal multi-limb helpers; see crypto_bigint.h. */

#include "crypto_bigint.h"
#include <errno.h>

size_t crypto_bn_significant(const limb_t *a, size_t n) {
    while (n > 1 && a[n - 1] == 0) --n;
    return n;
}

size_t crypto_bn_bit_length(const limb_t *a, size_t n) {
    n = crypto_bn_significant(a, n);
    size_t bits = 0;
    for (limb_t top = a[n - 1]; top; top >>= 1) ++bits;
    return (n - 1) * 64 + bits;
}

void crypto_bn_copy(limb_t *dst, const limb_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

void crypto_bn_set_small(limb_t *dst, limb_t value, size_t n) {
    dst[0] = value;
    for (size_t i = 1; i < n; ++i) dst[i] = 0;
}

int crypto_bn_is_zero(const limb_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) if (a[i]) return 0;
    return 1;
}

int crypto_bn_compare(const limb_t *a, const limb_t *b, size_t n) {
    for (size_t i = n; i-- > 0;) {
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}

int crypto_bn_same_value(const limb_t *a, size_t an, const limb_t *b, size_t bn) {
    an = crypto_bn_significant(a, an);
    bn = crypto_bn_significant(b, bn);
    return an == bn && crypto_bn_compare(a, b, an) == 0;
}

limb_t crypto_bn_add(limb_t *r, const limb_t *a, const limb_t *b, size_t n) {
    limb_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t t = (dlimb_t)a[i] + b[i] + carry;
        r[i] = (limb_t)t;
        carry = (limb_t)(t >> 64);
    }
    return carry;
}

limb_t crypto_bn_sub(limb_t *r, const limb_t *a, const limb_t *b, size_t n) {
    limb_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
        dlimb_t sub = (dlimb_t)b[i] + borrow;
        limb_t next = (dlimb_t)a[i] < sub;
        r[i] = (limb_t)((dlimb_t)a[i] - sub);
        borrow = next;
    }
    return borrow;
}

void crypto_bn_shift_right_one(limb_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = (a[i] >> 1) | (i + 1 < n ? a[i + 1] << 63 : 0);
}

void crypto_bn_mul_full(limb_t *wide, const limb_t *a, const limb_t *b, size_t n) {
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

/* Binary long division. Invariant rem < d; after the shift rem < 2d, so one
 * subtraction restores it, using the bit shifted out when rem overflowed. */
void crypto_bn_divide(limb_t *quot, limb_t *rem, const limb_t *a, const limb_t *d, size_t n) {
    crypto_bn_set_small(quot, 0, n);
    crypto_bn_set_small(rem, 0, n);
    for (size_t bit = 64 * n; bit-- > 0;) {
        limb_t carry = rem[n - 1] >> 63;
        for (size_t i = n; i-- > 1;) rem[i] = (rem[i] << 1) | (rem[i - 1] >> 63);
        rem[0] = (rem[0] << 1) | ((a[bit / 64] >> (bit % 64)) & 1);
        if (carry || crypto_bn_compare(rem, d, n) >= 0) {
            (void)crypto_bn_sub(rem, rem, d, n);
            quot[bit / 64] |= (limb_t)1 << (bit % 64);
        }
    }
}

/* Binary GCD; b stays odd throughout. */
void crypto_bn_gcd_odd(limb_t *g, const limb_t *a_in, const limb_t *b_in, size_t n) {
    limb_t a[n], b[n];
    crypto_bn_copy(a, a_in, n);
    crypto_bn_copy(b, b_in, n);
    while (!crypto_bn_is_zero(a, n)) {
        while ((a[0] & 1) == 0) crypto_bn_shift_right_one(a, n);
        if (crypto_bn_compare(a, b, n) < 0) {
            for (size_t i = 0; i < n; ++i) { limb_t t = a[i]; a[i] = b[i]; b[i] = t; }
        }
        (void)crypto_bn_sub(a, a, b, n);
    }
    crypto_bn_copy(g, b, n);
}

int crypto_bn_coprime_odd(const limb_t *a, const limb_t *m, size_t n) {
    limb_t g[n], one[n];
    crypto_bn_gcd_odd(g, a, m, n);
    crypto_bn_set_small(one, 1, n);
    return crypto_bn_compare(g, one, n) == 0;
}

/* Bit by bit from the top: set a bit of r when (r | bit)^2 <= a still holds. */
void crypto_bn_isqrt(limb_t *r, const limb_t *a, size_t n) {
    limb_t trial[n], wide[2 * n], padded[2 * n];
    crypto_bn_set_small(r, 0, n);
    crypto_bn_copy(padded, a, n);
    for (size_t i = n; i < 2 * n; ++i) padded[i] = 0;
    for (size_t bit = (crypto_bn_bit_length(a, n) + 1) / 2 + 1; bit-- > 0;) {
        crypto_bn_copy(trial, r, n);
        trial[bit / 64] |= (limb_t)1 << (bit % 64);
        crypto_bn_mul_full(wide, trial, trial, n);
        if (crypto_bn_compare(wide, padded, 2 * n) <= 0) crypto_bn_copy(r, trial, n);
    }
}

/* 2^64 = 1 (mod 3), so a = sum of its limbs (mod 3). */
unsigned crypto_bn_mod3(const limb_t *a, size_t n) {
    unsigned sum = 0;
    for (size_t i = 0; i < n; ++i) sum = (sum + (unsigned)(a[i] % 3)) % 3;
    return sum;
}

void crypto_bn_add_mod(limb_t *r, const limb_t *a, const limb_t *b, const limb_t *m, size_t n) {
    limb_t carry = crypto_bn_add(r, a, b, n);
    if (carry || crypto_bn_compare(r, m, n) >= 0) (void)crypto_bn_sub(r, r, m, n);
}

void crypto_bn_sub_mod(limb_t *r, const limb_t *a, const limb_t *b, const limb_t *m, size_t n) {
    if (crypto_bn_sub(r, a, b, n)) (void)crypto_bn_add(r, r, m, n);
}

void crypto_bn_montgomery_constants(limb_t *one, limb_t *r2, const limb_t *m, size_t n) {
    crypto_bn_set_small(one, 1, n);
    if (crypto_bn_compare(one, m, n) >= 0) (void)crypto_bn_sub(one, one, m, n);
    for (size_t i = 0; i < 64 * n; ++i) crypto_bn_add_mod(one, one, one, m, n);
    crypto_bn_copy(r2, one, n);
    for (size_t i = 0; i < 64 * n; ++i) crypto_bn_add_mod(r2, r2, r2, m, n);
}

int crypto_bn_mont_mul(limb_t *out, const limb_t *a, const limb_t *b, const limb_t *m,
                       size_t n, limb_t *wide) {
    errno = 0;
    crypto_multiply_reduce(wide, a, b, m, n, 0);
    if (errno != 0) return -1;
    crypto_bn_copy(out, wide, n);
    return 0;
}
