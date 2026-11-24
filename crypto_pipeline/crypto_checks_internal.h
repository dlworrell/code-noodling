/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:9d63caf5-236d-4946-a025-15430ca585d7
 */

#ifndef CRYPTO_CHECKS_INTERNAL_H
#define CRYPTO_CHECKS_INTERNAL_H
#include "crypto_cert.h"
#include "crypto_ecpp.h"

/* Internal: single-node checks shared with the mixed-certificate checker
 * (crypto_mixed.c). Not part of the public API.
 *
 * Each checks one node exactly as crypto_pocklington_verify or
 * crypto_ecpp_verify does, except that it does not follow links: a Pocklington
 * factor with child >= 0, and an ECPP step's q, are left to the caller. Both
 * return 0, or -1 with errno set as the public verifiers document, and add the
 * node's cost to *used (refusing with ERANGE once it exceeds a non-zero cap). */
#ifdef __cplusplus
extern "C" {
#endif

int crypto_pocklington_check_node(const crypto_pocklington_node *node, uint64_t *used,
                                  uint64_t max_modexps);
int crypto_ecpp_check_step(const crypto_ecpp_step *step, uint64_t *used, uint64_t max_point_ops);

#ifdef __cplusplus
}
#endif
#endif
