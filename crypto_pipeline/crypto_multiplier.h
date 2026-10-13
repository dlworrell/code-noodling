#ifndef CRYPTO_MULTIPLIER_H
#define CRYPTO_MULTIPLIER_H
#include <stdint.h>
#include <stddef.h>

/* Experimental arithmetic: not approved for production cryptography.
 * All array APIs require 1 <= size <= CRYPTO_MAX_LIMBS, valid aligned
 * pointers, and caller-owned arrays of the documented length. A non-NULL
 * pointer does not prove allocation capacity: callers must supply it.
 * Distinct input/output arrays must not overlap unless explicitly stated.
 * Invalid count or NULL array: void APIs set errno=EINVAL and return without
 * writing output; execute_blinded_multiplication returns -1 with EINVAL.
 * Other precondition violations (undersized/overlapping buffers) are errors
 * by the caller and cannot be detected by these legacy pointer-only APIs.
 * Count cap bounds VLA stack use; callers must provide adequate stack space.
 */
#define CRYPTO_MAX_LIMBS 1024u

/* Architectural multi-precision types */
typedef uint64_t limb_t;
typedef unsigned __int128 dlimb_t;
#ifdef __cplusplus
extern "C" {
#endif

/* r and mod: size limbs; r is updated in place. */
void ct_modular_reduce_final(limb_t *r, const limb_t *mod, size_t size);
/* Experimental placeholder: result_inv, input, mod each have size limbs.
 * This does NOT yet compute a modular inverse. */
void ct_mod_inverse_multiprecision(limb_t *result_inv, const limb_t *input,
                                  const limb_t *mod, size_t size);
/* Each array has size limbs. Depends on the unfinished inverse above. */
void execute_unblinding(limb_t *final_output, const limb_t *blinded_result,
                        const limb_t *original_k, const limb_t *mod, size_t size);
/* Both arrays have size limbs. */
int execute_blinded_multiplication(limb_t *blinded_a, const limb_t *original_a, size_t size);
/* result: 2*size limbs; a and b: size limbs. Experimental algorithms. */
void karatsuba_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t size);
void ntt_polynomial_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t size);

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
 * product: 2*size limbs; mod: size limbs, disjoint from product.
 * Performs Montgomery reduction in-place; constant-time behavior is unproven.
 */
void secure_montgomery_reduction(limb_t *product, const limb_t *mod, size_t size, limb_t inv);

/**
 * result: 2*size limbs; a, b, mod: size limbs, disjoint from result.
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
#endif
