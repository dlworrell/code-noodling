/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:7b373942-a7d2-4080-9c56-44b6b169b947
 */

#include "crypto_multiplier.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>

/* Static storage leaves stack space for the implementation's bounded VLAs. */
static limb_t a[CRYPTO_MAX_LIMBS], b[CRYPTO_MAX_LIMBS], mod[CRYPTO_MAX_LIMBS];
static limb_t output[2 * CRYPTO_MAX_LIMBS + 2];
static const limb_t sentinel = UINT64_C(0xdeadbeefcafef00d);

static void reset_output(size_t n) {
    for (size_t i = 0; i < 2 * CRYPTO_MAX_LIMBS + 2; ++i) output[i] = sentinel;
    output[2 * n + 1] = sentinel;
}

static void check_guards(size_t n) {
    assert(output[0] == sentinel);
    assert(output[2 * n + 1] == sentinel);
}

#define REJECT(call) do { \
    errno = 0; call; assert(errno == EINVAL); \
    for (size_t j = 0; j < 2 * CRYPTO_MAX_LIMBS + 2; ++j) \
        assert(output[j] == sentinel); \
} while (0)

int main(void) {
    const size_t invalid[] = {0, CRYPTO_MAX_LIMBS + 1, SIZE_MAX};
    reset_output(1);
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        size_t n = invalid[i];
        REJECT(secure_comba_multiply(output + 1, a, b, n));
        REJECT(secure_montgomery_reduction(output + 1, mod, n, 1));
        REJECT(ct_modular_reduce_final(output + 1, mod, n));
        REJECT(ct_mod_inverse_multiprecision(output + 1, a, mod, n));
        REJECT(execute_unblinding(output + 1, a, b, mod, n));
        REJECT(assert(execute_blinded_multiplication(output + 1, a, n) == -1));
        REJECT(assert(crypto_mod_multiply(output + 1, a, b, mod, n) == -1));
        REJECT(assert(crypto_mod_inverse(output + 1, a, mod, n) == -1));
        REJECT(assert(crypto_blind(output + 1, output + 2, a, mod, n) == -1));
        REJECT(karatsuba_multiply(output + 1, a, b, n));
        REJECT(ntt_polynomial_multiply(output + 1, a, b, n));
        for (int mode = 0; mode <= 1; ++mode)
            REJECT(crypto_multiply_reduce(output + 1, a, b, mod, n, mode));
    }
    REJECT(secure_comba_multiply(NULL, a, b, 1));
    REJECT(secure_comba_multiply(output + 1, NULL, b, 1));
    REJECT(secure_comba_multiply(output + 1, a, NULL, 1));
    REJECT(secure_montgomery_reduction(output + 1, NULL, 1, 1));
    REJECT(ct_modular_reduce_final(output + 1, NULL, 1));
    REJECT(ct_mod_inverse_multiprecision(output + 1, NULL, mod, 1));
    REJECT(execute_unblinding(output + 1, a, NULL, mod, 1));
    REJECT(assert(execute_blinded_multiplication(output + 1, NULL, 1) == -1));
    REJECT(karatsuba_multiply(output + 1, NULL, b, 1));
    REJECT(ntt_polynomial_multiply(output + 1, a, NULL, 1));
    REJECT(crypto_multiply_reduce(output + 1, a, b, NULL, 1, 0));

    const size_t sizes[] = {1, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024};
    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); ++t) {
        size_t n = sizes[t];
        for (size_t i = 0; i < n; ++i) {
            a[i] = UINT64_MAX - (i == 0);
            b[i] = UINT64_MAX - i - (i == 0);
            mod[i] = UINT64_MAX;
        }
        reset_output(n);
        secure_comba_multiply(output + 1, a, b, n);
        secure_montgomery_reduction(output + 1, mod, n, 1);
        ct_modular_reduce_final(output + 1, mod, n);
        check_guards(n);
        reset_output(n);
        ct_mod_inverse_multiprecision(output + 1, a, mod, n);
        execute_unblinding(output + 1, a, b, mod, n);
        assert(execute_blinded_multiplication(output + 1, a, n) == -1);
        assert(errno == ENOTSUP);
        check_guards(n);
        /* These operations write only n limbs, not 2*n. */
        assert(output[n + 1] == sentinel);
        reset_output(n);
        karatsuba_multiply(output + 1, a, b, n);
        check_guards(n);
        ntt_polynomial_multiply(output + 1, a, b, n);
        check_guards(n);
        for (int mode = 0; mode <= 1; ++mode) {
            reset_output(n);
            crypto_multiply_reduce(output + 1, a, b, mod, n, mode);
            check_guards(n);
        }
    }
    /* Deterministic mathematical assertions also run under sanitizers. */
    limb_t aa[33], bb[33], rr[66];
    for (size_t i = 0; i < 33; ++i) aa[i] = bb[i] = UINT64_MAX;
    karatsuba_multiply(rr, aa, bb, 33);
    assert(rr[0] == 1);
    for (size_t i = 1; i < 33; ++i) assert(rr[i] == 0);
    assert(rr[33] == UINT64_MAX - 1);
    for (size_t i = 34; i < 66; ++i) assert(rr[i] == UINT64_MAX);
    ntt_polynomial_multiply(rr, aa, bb, 33);
    assert(rr[0] == 1);
    for (size_t i = 1; i < 33; ++i) assert(rr[i] == 0);
    assert(rr[33] == UINT64_MAX - 1);
    for (size_t i = 34; i < 66; ++i) assert(rr[i] == UINT64_MAX);
    limb_t x = UINT64_MAX-1, m = UINT64_MAX, reduced[2];
    crypto_multiply_reduce(reduced, &x, &x, &m, 1, 1);
    assert(errno == 0 && reduced[0] == 1 && reduced[1] == 0);
    limb_t three=3, eleven=11, inv=0, factor=0, blinded=0, plain=0;
    assert(crypto_mod_inverse(&inv, &three, &eleven, 1) == 0 && inv == 4);
    assert(crypto_blind(&blinded, &factor, &three, &eleven, 1) == 0);
    execute_unblinding(&plain, &blinded, &factor, &eleven, 1);
    assert(errno == 0 && plain == 3);
    puts("[PASS] Invalid inputs rejected; boundary-size memory checks passed.");
    return 0;
}
