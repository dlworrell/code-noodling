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

    /* 7. Square roots modulo primes, checked by squaring. */
    const uint64_t primes[] = {2u, 3u, 5u, 7u, 13u, 17u, 97u, 65537u, 998244353u,
                               UINT64_C(0xFFFFFFFF00000001) /* 2^64 - 2^32 + 1, s = 32 */};
    for (size_t i = 0; i < sizeof primes / sizeof primes[0]; ++i) {
        uint64_t pr = primes[i];
        limb_t pm[1] = {pr};
        for (uint64_t v = 0; v < 40; ++v) {
            uint64_t sq = (uint64_t)(((dlimb_t)v * v) % pr);
            limb_t av[1] = {sq}, rt[1] = {UINT64_MAX};
            assert(crypto_mod_sqrt_grh(rt, av, pm, 1, 0) == 0 && errno == 0);
            assert((uint64_t)(((dlimb_t)rt[0] * rt[0]) % pr) == sq);
            assert(rt[0] <= pr - rt[0] || rt[0] == 0);
        }
    }
    /* A non-residue fails with EDOM and leaves the output unchanged. */
    limb_t p7[1] = {7}, three[1] = {3}, kept[1] = {99};
    assert(crypto_mod_sqrt_grh(kept, three, p7, 1, 0) == -1 && errno == EDOM && kept[0] == 99);
    /* 2^89 - 1 is 3 mod 4; 2^64 + 13 is a two-limb prime. */
    limb_t nine[2] = {9, 0}, rt2[2];
    assert(crypto_mod_sqrt_grh(rt2, nine, m89, 2, 0) == 0 && rt2[0] == 3 && rt2[1] == 0);
    /* 2^64 + 13 is 5 mod 8, so this takes the general (non 3 mod 4) path. The
     * smaller root of (2^64 + 13 - 0xFEDCBA9876543210)^2 is that difference;
     * expected values computed independently in Python. 3 is a non-residue. */
    limb_t p64_13[2] = {13, 1};
    limb_t sq64[2] = {UINT64_C(0xFC70A5227119A54D), 0}, rt64[2] = {0, 0};
    assert(crypto_mod_sqrt_grh(rt64, sq64, p64_13, 2, 0) == 0 && errno == 0);
    assert(rt64[0] == UINT64_C(0x0123456789ABCDFD) && rt64[1] == 0);
    limb_t three2[2] = {3, 0}, kept2[2] = {99, 99};
    assert(crypto_mod_sqrt_grh(kept2, three2, p64_13, 2, 0) == -1 && errno == EDOM);
    assert(kept2[0] == 99 && kept2[1] == 99);
    limb_t even_mod[1] = {10}, one_mod[1] = {1};
    assert(crypto_mod_sqrt_grh(kept, three, even_mod, 1, 0) == -1 && errno == EDOM);
    assert(crypto_mod_sqrt_grh(kept, three, one_mod, 1, 0) == -1 && errno == EDOM);
    assert(crypto_mod_sqrt_grh(kept, three, p7, 0, 0) == -1 && errno == EINVAL);
    printf("[PASS] square roots modulo primes\n");

    printf("=== crypto_prime tests passed ===\n");
    return 0;
}
