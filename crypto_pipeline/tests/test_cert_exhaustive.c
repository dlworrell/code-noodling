/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:b23f8ee9-4e6b-4997-b2a1-bcc340d66222
 */

/* Exhaustive soundness check of the Pocklington node rules on small numbers.
 *
 * For every odd N in [5, LIMIT), every certificate made of one or two prime
 * powers q^e dividing N - 1 (each q a 64-bit leaf) and every base a in
 * [2, N - 2] used for all factors, crypto_pocklington_verify must accept only
 * if N is prime (decided here by trial division). Random fuzzing rarely
 * reaches the few composites a weakened rule would accept (for example
 * N = 49, q = 3, a = 18 without F^2 > N, or N = 341 without the gcd
 * condition); enumerating the small space reaches all of them.
 *
 * It also counts, for each prime N, whether some certificate was accepted, as
 * a check that the enumeration is not vacuous. */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_cert.h"

#ifndef LIMIT
#define LIMIT 512u
#endif

static int is_prime(unsigned n) {
    if (n < 2) return 0;
    for (unsigned d = 2; d * d <= n; ++d) {
        if (n % d == 0) return 0;
    }
    return 1;
}

/* Prime-power divisors q^e of m, as (q, e) pairs with e at most the
 * multiplicity of q. Returns how many were written. */
static size_t prime_powers(unsigned m, unsigned q_out[], unsigned e_out[], size_t cap) {
    size_t count = 0;
    for (unsigned q = 2; q <= m; ++q) {
        if (!is_prime(q) || m % q != 0) continue;
        unsigned mult = 0;
        for (unsigned r = m; r % q == 0; r /= q) ++mult;
        for (unsigned e = 1; e <= mult && count < cap; ++e) {
            q_out[count] = q;
            e_out[count] = e;
            ++count;
        }
    }
    return count;
}

static unsigned long long checks, accepted_primes;

static int accepts(unsigned n, const unsigned *q, const unsigned *e, size_t factors, unsigned a) {
    limb_t big_n[1] = {n}, qs[2][1], base[1] = {a};
    crypto_pocklington_factor facs[2];
    for (size_t i = 0; i < factors; ++i) {
        qs[i][0] = q[i];
        facs[i].prime = qs[i];
        facs[i].base = base;
        facs[i].exponent = e[i];
        facs[i].child = -1;
    }
    crypto_pocklington_node node = {big_n, 1, facs, factors};
    size_t failed = 0;
    errno = 0;
    ++checks;
    int rc = crypto_pocklington_verify(&node, 1, 0, 0, &failed);
    if (rc != 0) assert(errno == EINVAL || errno == EDOM);
    return rc == 0;
}

int main(void) {
    printf("=== exhaustive Pocklington soundness, N < %u ===\n", (unsigned)LIMIT);
    unsigned primes_with_certificate = 0, primes = 0;
    for (unsigned n = 5; n < LIMIT; n += 2) {
        unsigned q[32], e[32];
        size_t pp = prime_powers(n - 1, q, e, 32);
        int prime = is_prime(n), proven = 0;
        if (prime) ++primes;
        for (unsigned a = 2; a + 2 <= n; ++a) {
            for (size_t i = 0; i < pp; ++i) {
                unsigned one_q[1] = {q[i]}, one_e[1] = {e[i]};
                if (accepts(n, one_q, one_e, 1, a)) {
                    if (!prime) {
                        printf("accepted composite N=%u with %u^%u, a=%u\n", n, q[i], e[i], a);
                        return 1;
                    }
                    proven = 1;
                }
                for (size_t j = i + 1; j < pp; ++j) {
                    if (q[j] == q[i]) continue;                  /* distinct primes only */
                    unsigned two_q[2] = {q[i], q[j]}, two_e[2] = {e[i], e[j]};
                    if (accepts(n, two_q, two_e, 2, a)) {
                        if (!prime) {
                            printf("accepted composite N=%u with %u^%u * %u^%u, a=%u\n", n,
                                    q[i], e[i], q[j], e[j], a);
                            return 1;
                        }
                        proven = 1;
                    }
                }
            }
        }
        if (proven) {
            ++primes_with_certificate;
            ++accepted_primes;
        }
    }
    printf("%llu certificates checked; no composite accepted\n", checks);
    printf("%u of %u odd primes in [5, %u) proven by some enumerated certificate\n",
           primes_with_certificate, primes, (unsigned)LIMIT);
    /* Every odd prime p >= 5 has a certificate here: p - 1 = 2^a 3^b ... has a
     * prime-power divisor set with F^2 > p and a base meeting both conditions
     * for all factors (a primitive root works). */
    assert(primes_with_certificate == primes);
    printf("=== exhaustive Pocklington soundness passed ===\n");
    return 0;
}
