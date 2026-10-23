/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:f3c94ad3-116e-4b63-88ce-b43591ad9a48
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#ifdef NDEBUG
#error "Test assertions must remain enabled"
#endif
#include "crypto_multiplier.h"

// Simple printing helper
void print_limbs(const char* label, const limb_t *src, size_t size) {
    printf("%s: 0x", label);
    for (size_t i = size; i > 0; i--) {
        printf("%016llX", (unsigned long long)src[i-1]);
    }
    printf("\n");
}

int main() {
    printf("=== Starting Cryptographic Wrapper Test Suite ===\n");

    // Setting up a standard 256-bit modulus (4 x 64-bit limbs)
    // Note: Montgomery requires an ODD modulus
    size_t size = 4; 
    limb_t mod[4] = { 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 
                      0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
                      
    // Test Case 1: Dense, complex inputs
    limb_t a1[4] = { 0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x4444444444444444ULL };
    limb_t b1[4] = { 0x5555555555555555ULL, 0x6666666666666666ULL, 0x7777777777777777ULL, 0x8888888888888888ULL };
    limb_t res1[8] = {0};

    // Test Case 2: Sparse inputs (mostly zeroes to check execution uniformity)
    limb_t a2[4] = { 0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL };
    limb_t b2[4] = { 0x0000000000000002ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL };
    limb_t res2[8] = {0};

    // Verify inverse pre-computation
    limb_t inv = compute_montgomery_inverse(mod[0]);
    assert(errno == 0 && mod[0] * inv == UINT64_MAX);
    printf("[PASS] Inverse computed successfully: 0x%016llX\n", (unsigned long long)inv);

    // Run execution routes
    printf("\nExecuting Secure Path for Case 1 (Dense)...\n");
    crypto_multiply_reduce(res1, a1, b1, mod, size, 1);
    print_limbs("Case 1 Modular Result", res1, size);
    /* Independent Python integer oracle: a*b*pow(2**256,-1,m) % m. */
    const limb_t expected1[4] = {UINT64_C(0xedcba987654320fe), UINT64_C(0x13579be02468acf1), UINT64_C(0x6d3a06d3a06d3a06), UINT64_C(0xe6f8091a2b3c4d5e)};
    assert(errno == 0);
    for (size_t i = 0; i < size; ++i) assert(res1[i] == expected1[i]);
    for (size_t i = size; i < 2 * size; ++i) assert(res1[i] == 0);

    printf("\nExecuting Secure Path for Case 2 (Sparse)...\n");
    crypto_multiply_reduce(res2, a2, b2, mod, size, 1);
    print_limbs("Case 2 Modular Result", res2, size);
    /* Independent Python integer oracle: a*b*pow(2**256,-1,m) % m. */
    const limb_t expected2[4] = {UINT64_C(0xffffffffffffffff), UINT64_C(0x0000000000000004), UINT64_C(0xffffffffffffffff), UINT64_C(0xfffffffffffffffd)};
    assert(errno == 0);
    for (size_t i = 0; i < size; ++i) assert(res2[i] == expected2[i]);
    for (size_t i = size; i < 2 * size; ++i) assert(res2[i] == 0);

    printf("\n[SUCCESS] Functional pipeline validation complete.\n");
    return 0;
}
