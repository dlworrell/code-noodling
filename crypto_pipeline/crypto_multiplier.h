#ifndef CRYPTO_MULTIPLIER_H
#define CRYPTO_MULTIPLIER_H
#include <stdint.h>
#include <stddef.h>

/* Experimental arithmetic, not production cryptography. No API is certified
 * constant-time, including legacy names containing ct_/secure_.
 * Arrays are little-endian base-2^64 limbs; 1 <= n <= 1024.
 * Callers supply valid aligned buffers of the stated length and adequate stack.
 * Outputs must not overlap any other argument; read-only inputs may alias.
 * Reduction routines explicitly operate in place on their r/product argument.
 * Pointer-only APIs cannot detect insufficient capacity or forbidden overlap.
 * Invalid pointer/count: EINVAL; invalid mathematical domain: EDOM.
 * On failure, output is unchanged. All successful calls clear errno.
 * int APIs return 0 on success, -1 on failure. void APIs report through errno.
 */
#define CRYPTO_MAX_LIMBS 1024u
typedef uint64_t limb_t;
typedef unsigned __int128 dlimb_t;
#ifdef __cplusplus
extern "C" {
#endif

/* One conditional subtraction, NOT general reduction. r,mod: n limbs;
 * requires mod>1 and 0<=r<2*mod. */
void ct_modular_reduce_final(limb_t *r, const limb_t *mod, size_t n);
/* Returns -mod0^-1 mod 2^64 for odd mod0; even mod0: 0 with EDOM. */
limb_t compute_montgomery_inverse(limb_t mod0);
/* Exact full product: result has 2*n limbs; a,b have n. */
void secure_comba_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t n);
void karatsuba_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t n);
/* Exact integer product via two-prime radix-2 NTT and CRT; result: 2*n limbs. */
void ntt_polynomial_multiply(limb_t *result, const limb_t *a, const limb_t *b, size_t n);
/* R=2^(64*n); product has 2*n limbs, mod has n. Requires odd mod>1,
 * product<mod*R and mod[0]*inv == -1 mod 2^64. Returns product*R^-1 mod mod
 * in the low n limbs and zeros the upper n limbs. */
void secure_montgomery_reduction(limb_t *product, const limb_t *mod, size_t n, limb_t inv);
/* Montgomery multiplication, NOT ordinary a*b mod mod. a,b,mod: n limbs;
 * result: 2*n limbs. Requires odd mod>1 and a,b<mod. Legacy flag selects
 * Comba when nonzero; otherwise Comba below 64, Karatsuba at 64 and above.
 * NTT is explicitly callable; dispatch thresholds are provisional. */
void crypto_multiply_reduce(limb_t *result, const limb_t *a, const limb_t *b,
                            const limb_t *mod, size_t n, int is_constant_time);
/* Ordinary a*b mod mod. All arrays n limbs, mod>1 (odd OR even).
 * Inputs need not be reduced. Variable-time. */
int crypto_mod_multiply(limb_t *result, const limb_t *a, const limb_t *b,
                        const limb_t *mod, size_t n);
/* Ordinary inverse. All arrays n limbs, odd mod>1; input need not be reduced.
 * Returns EDOM if gcd(input,mod)!=1. Variable-time binary extended GCD. */
int crypto_mod_inverse(limb_t *result, const limb_t *input, const limb_t *mod, size_t n);
/* Deprecated compatibility name; calls the VARIABLE-TIME inverse above. */
void ct_mod_inverse_multiprecision(limb_t *result, const limb_t *input, const limb_t *mod, size_t n);
/* Uniform rejection sampling of invertible k in 1..mod-1. All arrays n limbs.
 * Returns blinded=a*k mod mod AND factor=k; odd mod>1. Variable-time.
 * May fail with EAGAIN after 128 samples or an OS entropy error. */
int crypto_blind(limb_t *blinded, limb_t *factor, const limb_t *a, const limb_t *mod, size_t n);
/* Ordinary blinded*k^-1 mod mod. All arrays n limbs; odd mod>1 and gcd(k,mod)=1. */
void execute_unblinding(limb_t *out, const limb_t *blinded, const limb_t *k, const limb_t *mod, size_t n);
/* Retired ABI cannot specify modulus or return k. Valid calls fail ENOTSUP;
 * migrate to crypto_blind. Invalid pointer/count still fails EINVAL. */
int execute_blinded_multiplication(limb_t *out, const limb_t *a, size_t n);
/* Up to CRYPTO_MAX_LIMBS*sizeof(limb_t) random bytes. Buffer must be non-NULL;
 * on OS failure it may contain a partial read. */
int secure_get_random_bytes(uint8_t *buffer, size_t length);
#ifdef __cplusplus
}
#endif
#endif
