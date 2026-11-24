/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:163f6ef3-6e06-45f7-8fc2-c70e5ada079f
 */

/* Fuzz harness for the three primality-certificate checkers:
 * crypto_pocklington_verify, crypto_ecpp_verify and crypto_mixed_verify.
 *
 * Each input decodes into one small certificate (byte format below; see
 * tests/fuzz_corpus.py, which writes valid seeds in the same format). The
 * harness checks, for every input:
 *   - memory and undefined-behaviour safety (run under ASan/UBSan);
 *   - the error contract: -1 with errno EINVAL, EDOM or ERANGE, and a
 *     failed-node index inside the certificate; on success errno is 0 and the
 *     failed-node index is left unchanged;
 *   - soundness where it is cheap to decide: an accepted certificate whose
 *     root N is below 2^64 must prove a prime (crypto_is_prime_u64 is exact
 *     there). An accepted composite aborts.
 *
 * Built two ways (ci/fuzz.sh): with libFuzzer (clang -fsanitize=fuzzer), or
 * with -DFUZZ_STANDALONE, where main() replays the files named on the command
 * line and then mutates them deterministically, so CI can run it with any
 * compiler.
 *
 * Byte format (a missing byte reads as 0):
 *   selector   1 byte, % 3: 0 Pocklington, 1 ECPP, 2 mixed
 *   limbs      1 byte, 1 + b % 3 (every value in the certificate)
 *   value      1 tag byte, then by tag % 4: 0 one byte, 1 two bytes,
 *              2 four bytes, 3 limbs * 8 bytes (little-endian)
 *   child      1 byte: 255 -> -1 (a 64-bit leaf), 254 -> -2 (invalid),
 *              otherwise b % (count + 1) (so count itself is out of range)
 * Pocklington: count (1 + b % 6), root (b % count), cap (1 byte, 0 = none),
 *   then per node: n value, factor count (1 + b % 6), per factor: prime
 *   value, base value, exponent (1 + b % 16), child.
 * ECPP: count (1 + b % 4), cap (2 bytes, 0 = none), per step: n, a, x, y, k,
 *   q values.
 * Mixed: count (1 + b % 6), root, modexp cap (1 byte), point-op cap (2
 *   bytes), per node: kind (b % 3: 0 Pocklington, 1 ECPP, 2 an unknown kind),
 *   q_child (child byte), then the node as above (an unknown kind carries an
 *   ECPP step). */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "crypto_mixed.h"

#define FZ_MAX_LIMBS 3u
#define FZ_MAX_NODES 6u
#define FZ_MAX_FACTORS 6u
#define FZ_VALUES_PER_NODE (1u + 2u * FZ_MAX_FACTORS)
#define FZ_ECPP_VALUES 6u

typedef struct reader {
    const uint8_t *data;
    size_t size, at;
} reader;

static uint8_t next(reader *r) { return r->at < r->size ? r->data[r->at++] : 0; }

static void read_value(reader *r, limb_t *out, size_t limbs) {
    for (size_t i = 0; i < limbs; ++i) out[i] = 0;
    unsigned tag = next(r) % 4u;
    size_t bytes = tag == 0 ? 1 : tag == 1 ? 2 : tag == 2 ? 4 : limbs * 8u;
    for (size_t i = 0; i < bytes; ++i) out[i / 8u] |= (limb_t)next(r) << (8u * (i % 8u));
}

static int32_t read_child(reader *r, size_t count) {
    uint8_t b = next(r);
    if (b == 255) return -1;
    if (b == 254) return -2;
    return (int32_t)(b % (count + 1u));
}

typedef struct pock_storage {
    limb_t values[FZ_VALUES_PER_NODE][FZ_MAX_LIMBS];
    crypto_pocklington_factor factors[FZ_MAX_FACTORS];
} pock_storage;

typedef struct ecpp_storage {
    limb_t values[FZ_ECPP_VALUES][FZ_MAX_LIMBS];
} ecpp_storage;

static void read_pock(reader *r, size_t limbs, size_t count, pock_storage *s,
                      crypto_pocklington_node *node) {
    read_value(r, s->values[0], limbs);
    size_t fc = 1u + next(r) % FZ_MAX_FACTORS;
    for (size_t k = 0; k < fc; ++k) {
        read_value(r, s->values[1 + 2 * k], limbs);
        read_value(r, s->values[2 + 2 * k], limbs);
        s->factors[k].prime = s->values[1 + 2 * k];
        s->factors[k].base = s->values[2 + 2 * k];
        s->factors[k].exponent = 1u + next(r) % 16u;
        s->factors[k].child = read_child(r, count);
    }
    node->n = s->values[0];
    node->limbs = limbs;
    node->factors = s->factors;
    node->factor_count = fc;
}

static void read_ecpp(reader *r, size_t limbs, ecpp_storage *s, crypto_ecpp_step *step) {
    for (size_t v = 0; v < FZ_ECPP_VALUES; ++v) read_value(r, s->values[v], limbs);
    step->n = s->values[0];
    step->a = s->values[1];
    step->x = s->values[2];
    step->y = s->values[3];
    step->k = s->values[4];
    step->q = s->values[5];
    step->limbs = limbs;
}

/* N fits in 64 bits: return 1 and its value. */
static int as_u64(const limb_t *n, size_t limbs, uint64_t *out) {
    for (size_t i = 1; i < limbs; ++i) {
        if (n[i] != 0) return 0;
    }
    *out = n[0];
    return 1;
}

static long accepted;   /* inputs every checker accepted, for the standalone report */

static void check_outcome(int rc, int err, size_t failed, size_t count, const limb_t *root_n,
                          size_t limbs) {
    if (rc == 0) {
        ++accepted;
        if (err != 0 || failed != (size_t)-7) abort();     /* contract on success */
        uint64_t n = 0;
        if (as_u64(root_n, limbs, &n) && !crypto_is_prime_u64(n)) {
            printf("SOUNDNESS: accepted composite N = %llu\n", (unsigned long long)n);
            abort();
        }
        return;
    }
    if (rc != -1 || (err != EINVAL && err != EDOM && err != ERANGE)) abort();
    if (failed != (size_t)-7 && failed >= count) abort();  /* index inside the certificate */
}

static void fuzz_pocklington(reader *r, size_t limbs) {
    static pock_storage store[FZ_MAX_NODES];
    crypto_pocklington_node nodes[FZ_MAX_NODES];
    size_t count = 1u + next(r) % FZ_MAX_NODES;
    size_t root = next(r) % count;
    uint64_t cap = next(r);
    for (size_t i = 0; i < count; ++i) read_pock(r, limbs, count, &store[i], &nodes[i]);
    size_t failed = (size_t)-7;
    errno = 0;
    int rc = crypto_pocklington_verify(nodes, count, root, cap, &failed);
    check_outcome(rc, errno, failed, count, nodes[root].n, limbs);
}

static void fuzz_ecpp(reader *r, size_t limbs) {
    static ecpp_storage store[FZ_MAX_NODES];
    crypto_ecpp_step steps[FZ_MAX_NODES];
    size_t count = 1u + next(r) % 4u;
    uint64_t cap = next(r);
    cap |= (uint64_t)next(r) << 8;
    for (size_t i = 0; i < count; ++i) read_ecpp(r, limbs, &store[i], &steps[i]);
    size_t failed = (size_t)-7;
    errno = 0;
    int rc = crypto_ecpp_verify(steps, count, cap, &failed);
    check_outcome(rc, errno, failed, count, steps[0].n, limbs);
}

static void fuzz_mixed(reader *r, size_t limbs) {
    static pock_storage pstore[FZ_MAX_NODES];
    static ecpp_storage estore[FZ_MAX_NODES];
    crypto_pocklington_node pnodes[FZ_MAX_NODES];
    crypto_ecpp_step steps[FZ_MAX_NODES];
    crypto_mixed_node nodes[FZ_MAX_NODES];
    size_t count = 1u + next(r) % FZ_MAX_NODES;
    size_t root = next(r) % count;
    uint64_t modexps = next(r);
    uint64_t point_ops = next(r);
    point_ops |= (uint64_t)next(r) << 8;
    for (size_t i = 0; i < count; ++i) {
        unsigned kind = next(r) % 3u;
        nodes[i].q_child = read_child(r, count);
        nodes[i].pocklington = NULL;
        nodes[i].ecpp = NULL;
        if (kind == 0) {
            nodes[i].kind = CRYPTO_MIXED_POCKLINGTON;
            read_pock(r, limbs, count, &pstore[i], &pnodes[i]);
            nodes[i].pocklington = &pnodes[i];
        } else {
            nodes[i].kind = kind == 1 ? CRYPTO_MIXED_ECPP : 9u;
            read_ecpp(r, limbs, &estore[i], &steps[i]);
            nodes[i].ecpp = &steps[i];
        }
    }
    size_t failed = (size_t)-7;
    errno = 0;
    int rc = crypto_mixed_verify(nodes, count, root, modexps, point_ops, &failed);
    const limb_t *root_n = nodes[root].kind == CRYPTO_MIXED_POCKLINGTON ? pnodes[root].n
                                                                         : steps[root].n;
    check_outcome(rc, errno, failed, count, root_n, limbs);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    reader r = {data, size, 0};
    unsigned selector = next(&r) % 3u;
    size_t limbs = 1u + next(&r) % FZ_MAX_LIMBS;
    if (selector == 0) fuzz_pocklington(&r, limbs);
    else if (selector == 1) fuzz_ecpp(&r, limbs);
    else fuzz_mixed(&r, limbs);
    return 0;
}

#ifdef FUZZ_STANDALONE
/* Replays each file named on the command line, then runs `iterations`
 * deterministic mutations of them (FUZZ_ITERATIONS, default 20000). Every
 * replayed file must be accepted: the seeds are valid certificates, and a
 * rejected seed means the corpus and the harness format disagree. */
#define FZ_MAX_INPUT 8192u

static uint64_t rng_state = 0x9E3779B97F4A7C15u;
static uint64_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static size_t load(const char *path, uint8_t *buf) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("cannot open %s\n", path);
        exit(2);
    }
    size_t n = fread(buf, 1, FZ_MAX_INPUT, f);
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    static uint8_t seeds[64][FZ_MAX_INPUT];
    static size_t sizes[64];
    static uint8_t work[FZ_MAX_INPUT];
    int nseeds = argc - 1 < 64 ? argc - 1 : 64;
    for (int i = 0; i < nseeds; ++i) {
        sizes[i] = load(argv[i + 1], seeds[i]);
        long before = accepted;
        LLVMFuzzerTestOneInput(seeds[i], sizes[i]);
        if (accepted == before) {
            printf("seed rejected: %s\n", argv[i + 1]);
            return 1;
        }
    }
    long seed_accepted = accepted;
    const char *env = getenv("FUZZ_ITERATIONS");
    long iterations = env ? strtol(env, NULL, 10) : 20000;
    for (long it = 0; it < iterations; ++it) {
        size_t n;
        if (nseeds > 0 && (rng() % 8u) != 0) {
            int s = (int)(rng() % (uint64_t)nseeds);
            n = sizes[s];
            for (size_t j = 0; j < n; ++j) work[j] = seeds[s][j];
            unsigned edits = 1u + (unsigned)(rng() % 4u);
            for (unsigned e = 0; e < edits && n > 0; ++e) {
                size_t at = (size_t)(rng() % n);
                switch (rng() % 3u) {
                case 0: work[at] ^= (uint8_t)(1u << (rng() % 8u)); break;   /* flip a bit */
                case 1: work[at] = (uint8_t)rng(); break;                   /* new byte */
                default: work[at] = (uint8_t)(work[at] + 1u); break;        /* off by one */
                }
            }
        } else {
            n = (size_t)(rng() % 512u);                                     /* random input */
            for (size_t j = 0; j < n; ++j) work[j] = (uint8_t)rng();
        }
        LLVMFuzzerTestOneInput(work, n);
    }
    printf("fuzz_certificates: %d seed(s) accepted, %ld mutation(s) (%ld accepted), no failure\n",
           nseeds, iterations, accepted - seed_accepted);
    return 0;
}
#endif
