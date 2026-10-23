/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:3f17c387-b909-45d9-b692-6d508207bb14
 */

/* Build against pinned upstream dudect; run n=4,16,32 and both fixed classes. */
#include "crypto_montgomery_ct.h"
#include <stdio.h>
#define DUDECT_IMPLEMENTATION
#include "dudect.h"
#ifndef CT_LIMBS
#define CT_LIMBS 4
#endif
#ifndef CT_FIXED_CLASS
#define CT_FIXED_CLASS 0
#endif
#ifndef CT_BATCHES
#define CT_BATCHES 21
#endif
static crypto_montgomery_ctx mont;
static volatile uint8_t sink;

uint8_t do_one_computation(uint8_t *data) {
    /* dudect allocates aligned storage; chunk stride is a multiple of 8. */
    limb_t *a = (limb_t *)(void *)data;
#ifdef CT_TIMING_INPUT_ONLY
    /* Fixed-address volatile reads retain all input traffic, without arithmetic. */
    const volatile limb_t *input = a;
    limb_t observed = 0;
    for (size_t i = 0; i < 2*CT_LIMBS; ++i) observed ^= input[i];
    sink ^= (uint8_t)observed;
    return sink;
#else
    limb_t out[CT_LIMBS];
#ifdef CT_TIMING_NEGATIVE_CONTROL
    /* Deliberate timing leak to validate the measurement setup. */
    for (limb_t i = 0; i < (a[0] & 1023); ++i) sink ^= 1;
#endif
#ifdef CT_TIMING_BASELINE
    /* Measurement control: operation is independent even of secret values. */
    limb_t public_a[CT_LIMBS] = {1};
    a = public_a;
    const limb_t *b = public_a;
#else
    const limb_t *b = a + CT_LIMBS;
#endif
    int rc = crypto_montgomery_mul_ct(out, CT_LIMBS, a, CT_LIMBS,
                                     b, CT_LIMBS, &mont);
    sink ^= (uint8_t)(out[0] ^ (limb_t)rc);
    return sink;
#endif
}
void prepare_inputs(dudect_config_t *config, uint8_t *data, uint8_t *classes) {
    randombytes(data, config->number_measurements * config->chunk_size);
    for (size_t i = 0; i < config->number_measurements; ++i) {
        classes[i] = randombit();
        limb_t *p = (limb_t *)(void *)(data + i*config->chunk_size);
        if (classes[i] == 0) {
            for (size_t j = 0; j < 2*CT_LIMBS; ++j)
                p[j] = CT_FIXED_CLASS ? UINT64_MAX : 0;
            if (CT_FIXED_CLASS) { p[0]--; p[CT_LIMBS]--; }
        } else {
            /* Public modulus R-1: clearing the low bit guarantees a,b<m. */
            p[0] &= ~UINT64_C(1); p[CT_LIMBS] &= ~UINT64_C(1);
        }
#ifdef CT_TIMING_RANDOM_LABELS
        /* Preserve the mixture but break the link between values and labels. */
        classes[i] = randombit();
#endif
    }
}
int main(void) {
    limb_t modulus[CT_LIMBS];
    for (size_t i = 0; i < CT_LIMBS; ++i) modulus[i] = UINT64_MAX;
    if (crypto_montgomery_ctx_init(&mont, modulus, CT_LIMBS)) return 2;
    dudect_config_t config = {.chunk_size = 2*CT_LIMBS*sizeof(limb_t), .number_measurements = 10000};
    dudect_ctx_t ctx;
    dudect_init(&ctx, &config);
    dudect_state_t state = DUDECT_NO_LEAKAGE_EVIDENCE_YET;
    unsigned batches = 0;
    while (batches < CT_BATCHES && state == DUDECT_NO_LEAKAGE_EVIDENCE_YET) {
        state = dudect_main(&ctx); ++batches;
    }
    printf("limbs=%d fixed_class=%d batches=%u: %s\n", CT_LIMBS, CT_FIXED_CLASS, batches,
           state == DUDECT_LEAKAGE_FOUND ? "LEAKAGE EVIDENCE DETECTED" : "NO LEAKAGE DETECTED IN THIS BOUNDED RUN; NOT A PROOF");
    dudect_free(&ctx);
    return state == DUDECT_LEAKAGE_FOUND ? 1 : 0;
}
