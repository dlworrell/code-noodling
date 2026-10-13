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
            a[i] = UINT64_MAX;
            b[i] = UINT64_MAX - i;
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
        assert(execute_blinded_multiplication(output + 1, a, n) == 0);
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
    puts("[PASS] Invalid inputs rejected; boundary-size memory checks passed.");
    return 0;
}
