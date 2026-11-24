/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:71b8915c-3a0d-4434-b0b7-1d8ed9894eb2
 */

/* Mixed Pocklington/ECPP certificate checker. See crypto_mixed.h for the
 * contract. Each node is checked by the existing single-node checks
 * (crypto_checks_internal.h); this file only links nodes. Everything here is
 * variable-time and works on public values. */

#include "crypto_mixed.h"
#include "crypto_bigint.h"
#include "crypto_checks_internal.h"
#include <errno.h>

_Static_assert(CRYPTO_MIXED_MAX_DEPTH <= 0xFFFFu, "node heights are stored in 16 bits");

enum { UNSEEN = 0, IN_PROGRESS = 1, VERIFIED = 2 };

typedef struct walk {
    const crypto_mixed_node *nodes;
    size_t count;
    unsigned char *state;
    uint16_t *height;               /* VERIFIED nodes: longest path below, in links */
    uint64_t modexps, max_modexps;
    uint64_t point_ops, max_point_ops;
    size_t failed;
} walk;

static int fail(int code) { errno = code; return -1; }
static int valid_count(size_t n) { return n > 0 && n <= CRYPTO_MAX_LIMBS; }

/* The N a node proves and its limb count, or NULL for a malformed node. */
static const limb_t *node_n(const crypto_mixed_node *node, size_t *limbs) {
    if (node->kind == CRYPTO_MIXED_POCKLINGTON && node->pocklington) {
        *limbs = node->pocklington->limbs;
        return valid_count(*limbs) ? node->pocklington->n : NULL;
    }
    if (node->kind == CRYPTO_MIXED_ECPP && node->ecpp) {
        *limbs = node->ecpp->limbs;
        return valid_count(*limbs) ? node->ecpp->n : NULL;
    }
    return NULL;
}

static int verify_node(walk *w, size_t index, unsigned depth, unsigned *height);

/* Proves the prime p (p_limbs limbs) through node `child`, or as a 64-bit leaf
 * when child is -1. On success *reach is the number of links below the parent
 * on the longest path through this prime (0 for a leaf). On failure w->failed
 * names the node that found it. */
static int prove(walk *w, size_t parent, const limb_t *p, size_t p_limbs, int32_t child,
                 unsigned depth, unsigned *reach) {
    *reach = 0;
    if (child == -1) {
        if (crypto_bn_significant(p, p_limbs) != 1) { w->failed = parent; return fail(EINVAL); }
        if (!crypto_is_prime_u64(p[0])) { w->failed = parent; return fail(EDOM); }
        return 0;
    }
    size_t c_limbs = 0;
    if (child < 0 || (size_t)child >= w->count) { w->failed = parent; return fail(EINVAL); }
    const limb_t *c_n = node_n(&w->nodes[child], &c_limbs);
    if (!c_n || !crypto_bn_same_value(c_n, c_limbs, p, p_limbs)) {
        w->failed = parent;
        return fail(EINVAL);
    }
    unsigned child_height = 0;
    if (verify_node(w, (size_t)child, depth + 1, &child_height) != 0) return -1;
    *reach = child_height + 1;
    return 0;
}

/* Verifies node `index`, reached at `depth` links from the root, and sets
 * *height to the longest path below it. The limit applies to every path: a
 * node verified earlier is not re-walked, but its stored height is checked
 * against the new depth, so the result does not depend on factor order. */
static int verify_node(walk *w, size_t index, unsigned depth, unsigned *height) {
    if (w->state[index] == VERIFIED) {
        *height = w->height[index];
        if (depth + *height > CRYPTO_MIXED_MAX_DEPTH) {
            w->failed = index;
            return fail(EINVAL);
        }
        return 0;
    }
    if (w->state[index] == IN_PROGRESS || depth > CRYPTO_MIXED_MAX_DEPTH) {
        w->failed = index;
        return fail(EINVAL);
    }
    w->state[index] = IN_PROGRESS;
    unsigned longest = 0, reach = 0;
    const crypto_mixed_node *node = &w->nodes[index];
    if (node->kind == CRYPTO_MIXED_POCKLINGTON && node->pocklington) {
        const crypto_pocklington_node *pn = node->pocklington;
        if (crypto_pocklington_check_node(pn, &w->modexps, w->max_modexps) != 0) {
            w->failed = index;
            return -1;
        }
        for (size_t k = 0; k < pn->factor_count; ++k) {
            const crypto_pocklington_factor *fac = &pn->factors[k];
            if (fac->child < 0) continue;              /* leaves were checked with the node */
            if (prove(w, index, fac->prime, pn->limbs, fac->child, depth, &reach) != 0) return -1;
            if (reach > longest) longest = reach;
        }
    } else if (node->kind == CRYPTO_MIXED_ECPP && node->ecpp) {
        const crypto_ecpp_step *s = node->ecpp;
        if (crypto_ecpp_check_step(s, &w->point_ops, w->max_point_ops) != 0) {
            w->failed = index;
            return -1;
        }
        if (node->q_child < -1) { w->failed = index; return fail(EINVAL); }
        if (prove(w, index, s->q, s->limbs, node->q_child, depth, &reach) != 0) return -1;
        longest = reach;
    } else {
        w->failed = index;
        return fail(EINVAL);
    }
    w->height[index] = (uint16_t)longest;    /* depth + longest <= MAX_DEPTH */
    w->state[index] = VERIFIED;
    *height = longest;
    return 0;
}

int crypto_mixed_verify(const crypto_mixed_node *nodes, size_t count, size_t root,
                        uint64_t max_modexps, uint64_t max_point_ops, size_t *failed_node) {
    if (!nodes || count == 0 || count > CRYPTO_MIXED_MAX_NODES || root >= count) {
        return fail(EINVAL);
    }
    unsigned char state[count];
    uint16_t height[count];
    for (size_t i = 0; i < count; ++i) {
        state[i] = UNSEEN;
        height[i] = 0;
    }
    walk w = {nodes, count, state, height, 0, max_modexps, 0, max_point_ops, root};
    unsigned root_height = 0;
    if (verify_node(&w, root, 0, &root_height) != 0) {
        if (failed_node) *failed_node = w.failed;
        return -1;
    }
    errno = 0;
    return 0;
}
