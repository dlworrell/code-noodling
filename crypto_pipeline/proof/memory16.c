/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:9102f67e-a9fe-48c4-b4e5-0a854345ad98
 */

/* Verify the actual C, with arbitrary operand/modulus bits and public n=16.
 * Arithmetic reducedness is not needed for memory safety. */
#include "../crypto_montgomery_ct.c"
_Static_assert(sizeof(limb_t) == 8, "64-bit limbs");
_Static_assert(sizeof(dlimb_t) == 16, "128-bit intermediates");
_Static_assert((limb_t)-1 > 0, "unsigned limbs");
_Static_assert((dlimb_t)-1 > 0, "unsigned intermediates");
extern limb_t nondet_limb(void);
extern _Bool nondet_bool(void);
int main(void) {
    limb_t a[16], b[16], out[16];
    crypto_montgomery_ctx ctx;
    ctx.limbs = 16;
    ctx.inverse = nondet_limb();
    for (size_t i = 0; i < 16; ++i) {
        a[i] = nondet_limb(); b[i] = nondet_limb();
        ctx.modulus[i] = nondet_limb();
    }
    const limb_t *bp = nondet_bool() ? a : b;
    int rc = crypto_montgomery_mul_ct(out, 16, a, 16, bp, 16, &ctx);
    __CPROVER_assert(rc == 0, "valid public arguments succeed");
    __CPROVER_assert(errno == 0, "success clears errno");
    return 0;
}
