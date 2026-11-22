/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:97459378-1449-4825-83cc-c6d0a5fe48e4
 */

/* Smoke and contract tests for crypto_prime.h. Kept small enough to run under
 * Valgrind; tests/test_prime.py does the larger differential checks. */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_prime.h"

static int trial_division_is_prime(uint64_t v) {
    if (v < 2) return 0;
    for (uint64_t f = 2; f * f <= v; ++f) {
        if (v % f == 0) return 0;
    }
    return 1;
}

static int grh_is_prime(const limb_t *v, size_t n) {
    int result = -1;
    int rc = crypto_is_prime_grh(&result, v, n, 0);
    assert(rc == 0 && errno == 0);
    assert(result == 0 || result == 1);
    return result;
}

int main(void) {
    printf("=== crypto_prime tests ===\n");

    /* 1. Every value below 3000 agrees with trial division. */
    for (uint64_t v = 0; v < 3000; ++v) {
        limb_t x[1] = {v};
        assert(grh_is_prime(x, 1) == trial_division_is_prime(v));
    }
    printf("[PASS] 0..2999 match trial division\n");

    /* 2. Carmichael numbers and strong pseudoprimes are composite. */
    const uint64_t composites[] = {
        561u, 1105u, 1729u, 2465u, 2821u, 6601u, 8911u, 41041u, 825265u,
        2047u,                        /* strong pseudoprime to base 2 */
        1373653u,                     /* to bases 2 and 3 */
        25326001u,                    /* to bases 2, 3 and 5 */
        3215031751u,                  /* to bases 2, 3, 5 and 7 */
        UINT64_C(3825123056546413051) /* to bases 2 through 23 */
    };
    for (size_t i = 0; i < sizeof composites / sizeof composites[0]; ++i) {
        limb_t x[1] = {composites[i]};
        assert(grh_is_prime(x, 1) == 0);
    }
    printf("[PASS] Carmichael numbers and strong pseudoprimes are composite\n");

    /* 3. Known primes, including padding with a zero high limb. */
    limb_t m61[2] = {UINT64_C(0x1FFFFFFFFFFFFFFF), 0};        /* 2^61 - 1 */
    assert(grh_is_prime(m61, 1) == 1);
    assert(grh_is_prime(m61, 2) == 1);
    limb_t largest64[1] = {UINT64_C(18446744073709551557)};  /* 2^64 - 59 */
    assert(grh_is_prime(largest64, 1) == 1);
    limb_t m89[2] = {UINT64_MAX, UINT64_C(0x1FFFFFF)};       /* 2^89 - 1 */
    assert(grh_is_prime(m89, 2) == 1);
    limb_t m89_times_3[2];                                    /* 3 * (2^89 - 1) */
    m89_times_3[0] = UINT64_MAX - 2;                          /* low limb of 3*2^89 - 3 */
    m89_times_3[1] = UINT64_C(0x5FFFFFF);
    assert(grh_is_prime(m89_times_3, 2) == 0);
    printf("[PASS] 2^61-1, 2^64-59 and 2^89-1 are prime; 3(2^89-1) is not\n");

    /* 4. Witness bound and the work cap. */
    assert(crypto_grh_witness_bound(m61, 1) == 3576 && errno == 0);
    int untouched = 7;
    assert(crypto_is_prime_grh(&untouched, m61, 1, 3574) == -1 && errno == ERANGE);
    assert(untouched == 7);
    assert(crypto_is_prime_grh(&untouched, m61, 1, 3575) == 0 && untouched == 1); /* bases 2..3576 */
    printf("[PASS] witness bound 3576 for 2^61-1; cap refuses with ERANGE\n");

    /* 5. Modular exponentiation. */
    limb_t base[1] = {3}, exponent[1] = {UINT64_C(1000000007)}, mod[1] = {UINT64_C(1000000009)};
    limb_t out[1] = {0};
    assert(crypto_mod_exp(out, base, exponent, 1, mod, 1) == 0 && errno == 0);
    assert(out[0] == UINT64_C(666666673)); /* pow(3, 1000000007, 1000000009) */
    limb_t zero[1] = {0};
    assert(crypto_mod_exp(out, base, zero, 1, mod, 1) == 0 && out[0] == 1);
    limb_t big_base[1] = {UINT64_MAX}; /* unreduced base is reduced first */
    assert(crypto_mod_exp(out, big_base, exponent, 1, mod, 1) == 0);
    assert(out[0] == UINT64_C(304865480)); /* pow(2^64 - 1, 1000000007, 1000000009) */
    printf("[PASS] modular exponentiation\n");

    /* 6. Domain and argument errors leave outputs unchanged. */
    limb_t even[1] = {1000000008}, unit[1] = {1};
    out[0] = 42;
    assert(crypto_mod_exp(out, base, exponent, 1, even, 1) == -1 && errno == EDOM && out[0] == 42);
    assert(crypto_mod_exp(out, base, exponent, 1, unit, 1) == -1 && errno == EDOM && out[0] == 42);
    assert(crypto_mod_exp(out, base, exponent, 0, mod, 1) == -1 && errno == EINVAL);
    assert(crypto_mod_exp(NULL, base, exponent, 1, mod, 1) == -1 && errno == EINVAL);
    assert(crypto_is_prime_grh(&untouched, m61, 0, 0) == -1 && errno == EINVAL);
    assert(crypto_is_prime_grh(NULL, m61, 1, 0) == -1 && errno == EINVAL);
    assert(crypto_grh_witness_bound(NULL, 1) == 0 && errno == EINVAL);
    printf("[PASS] error contracts\n");

    printf("=== crypto_prime tests passed ===\n");
    return 0;
}
