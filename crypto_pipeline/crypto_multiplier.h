#ifndef CRYPTO_MULTIPLIER_H
#define CRYPTO_MULTIPLIER_H

#include <stdint.h>
#include <stddef.h>

/* Architectural multi-precision types */
typedef uint64_t limb_t;
typedef unsigned __int128 dlimb_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculates the negated modular inverse (-mod^-1 mod 2^64)
 * dynamically in constant-time for Montgomery operations.
 */
limb_t compute_montgomery_inverse(limb_t mod0);

/**
 * Multiplies two large integers using a highly secure, 
 * register-efficient Comba loop. Result buffer must be 2*size in length.
 */
void secure_comba_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t size);

/**
 * Performs a branchless Montgomery modular reduction in-place.
 */
void secure_montgomery_reduction(limb_t *product, const limb_t *mod, size_t size, limb_t inv);

/**
 * Adaptive top-level wrapper interface. Automatically routes to the 
 * most efficient algorithm unless is_constant_time is activated.
 */
void crypto_multiply_reduce(
    limb_t *result, 
    const limb_t *a, 
    const limb_t *b, 
    const limb_t *mod, 
    size_t size, 
    int is_constant_time
);

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_MULTIPLIER_H */