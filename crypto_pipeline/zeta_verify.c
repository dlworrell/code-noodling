/*
 * SPDX-FileCopyrightText: 2026 Donovan Worrell
 *
 * SPDX-License-Identifier: Apache-2.0
 * Catalyst-File-ID: urn:uuid:1e7d1067-efac-4451-a54a-0f9ff02f1ada
 */

/* Command-line driver for zz_verify (crypto_zeta.h).
 *
 * Usage: zeta_verify [--rs] [--json] [--brackets] T [checkpoint...]
 *   --rs        Riemann-Siegel with Gabcke's constant (conditional result)
 *   --json      one JSON object on stdout
 *   --brackets  include every sign change in the JSON output
 * Exit status: 0 when the claim is proven, 1 when verification failed,
 * 2 for usage errors. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "crypto_zeta.h"

/* Static storage (no heap): enough for heights up to about 3 * 10^5. */
#define WORK_CAP (1u << 20)
#define BRACKETS_CAP (1u << 19)
#define BAD_GRAM_CAP (1u << 17)
static zz_sample work[WORK_CAP];
static zz_bracket brackets_buf[BRACKETS_CAP];
static long bad_gram_buf[BAD_GRAM_CAP];

/* Writes "<what> <arg>\n" to stderr. */
static void complain(const char *what, const char *arg) {
    fputs(what, stderr);
    fputs(" ", stderr);
    fputs(arg, stderr);
    fputs("\n", stderr);
}

static int parse_double(const char *s, double *out) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || *end != '\0') {
        return -1;
    }
    *out = v;
    return 0;
}

int main(int argc, char **argv) {
    int rs = 0;
    int json = 0;
    int brackets = 0;
    int i = 1;
    for (; i < argc && strncmp(argv[i], "--", 2) == 0; i++) {
        if (strcmp(argv[i], "--rs") == 0) {
            rs = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            json = 1;
        } else if (strcmp(argv[i], "--brackets") == 0) {
            brackets = 1;
        } else {
            complain("unknown option", argv[i]);
            return 2;
        }
    }
    if (i >= argc || argc - i - 1 > 64) {
        fputs("usage: zeta_verify [--rs] [--json] [--brackets] T [checkpoint...]\n", stderr);
        return 2;
    }
    double height = 0.0;
    double cps[64];
    size_t ncp = 0;
    if (parse_double(argv[i], &height) != 0) {
        complain("bad height", argv[i]);
        return 2;
    }
    for (int k = i + 1; k < argc; k++) {
        if (parse_double(argv[k], &cps[ncp]) != 0) {
            complain("bad checkpoint", argv[k]);
            return 2;
        }
        ncp++;
    }
    zz_verify_params params;
    zz_verify_params_default(&params, height);
    params.method = rs ? ZZ_METHOD_RS_GABCKE : ZZ_METHOD_EM;
    params.checkpoints = cps;
    params.n_checkpoints = ncp;
    size_t need = zz_verify_work_size(height, params.turing_intervals);
    if (need == 0 || need > WORK_CAP) {
        complain("height outside this driver's range:", argv[i]);
        return 2;
    }
    params.work = work;
    params.work_cap = WORK_CAP;
    params.brackets = brackets_buf;
    params.brackets_cap = BRACKETS_CAP;
    params.bad_gram = bad_gram_buf;
    params.bad_gram_cap = BAD_GRAM_CAP;
    zz_verify_result res;
    int status = zz_verify(&params, &res);
    const char *reason = res.reason != NULL ? res.reason : "";
    size_t nb = res.n_brackets < BRACKETS_CAP ? res.n_brackets : BRACKETS_CAP;
    size_t ng = res.n_bad_gram < BAD_GRAM_CAP ? res.n_bad_gram : BAD_GRAM_CAP;
    if (json) {
        printf("{\"status\": %d, \"reason\": \"%s\", \"fail_t\": %.17g, \"method\": \"%s\","
               " \"conditional\": %d,\n",
               status, reason, res.fail_t, rs ? "rs-gabcke" : "em", res.conditional);
        printf(" \"height\": %.17g, \"zeros_to_height\": %ld, \"t0\": %.17g, \"t1\": %.17g,\n",
               height, res.zeros_to_height, res.t0, res.t1);
        printf(" \"gram_t0\": %ld, \"sign_changes_to_t0\": %ld, \"turing_bound\": %ld,"
               " \"turing_upper\": %.17g,\n",
               res.gram_t0, res.sign_changes_to_t0, res.turing_bound, res.turing_upper);
        printf(" \"evaluations\": %ld, \"em_evaluations\": %ld, \"rs_fallbacks\": %ld,\n",
               res.evaluations, res.em_evaluations, res.rs_fallbacks);
        printf(" \"checkpoints\": [");
        for (size_t k = 0; k < ncp; k++) {
            printf("%s[%.17g, %ld]", k ? ", " : "", cps[k], res.checkpoint_counts[k]);
        }
        printf("],\n \"bad_gram\": [");
        for (size_t k = 0; k < ng; k++) {
            printf("%s%ld", k ? ", " : "", bad_gram_buf[k]);
        }
        printf("],\n \"n_brackets\": %zu", res.n_brackets);
        if (brackets) {
            printf(",\n \"brackets\": [");
            for (size_t k = 0; k < nb; k++) {
                printf("%s[%.17g, %.17g]", k ? ", " : "", brackets_buf[k].lo, brackets_buf[k].hi);
            }
            printf("]");
        }
        printf("}\n");
    } else if (status == ZZ_OK) {
        printf("PROVEN%s: all %ld zeros of zeta with 0 < t <= %.17g lie on the critical line"
               " and are simple; N(%.17g) = %ld\n",
               res.conditional ? " (conditional on Gabcke's constant)" : "", res.sign_changes_to_t0,
               res.t0, height, res.zeros_to_height);
        for (size_t k = 0; k < ncp; k++) {
            printf("N(%.17g) = %ld\n", cps[k], res.checkpoint_counts[k]);
        }
        printf("Turing: t0 = g_%ld = %.17g, t1 = %.17g, N(t0) <= %.6f\n", res.gram_t0, res.t0,
               res.t1, res.turing_upper);
        printf("Evaluations: %ld (Euler-Maclaurin %ld, RS fallbacks %ld)\n", res.evaluations,
               res.em_evaluations, res.rs_fallbacks);
    } else {
        printf("FAILED (%d): %s (t = %.17g)\n", status, reason, res.fail_t);
    }
    return status == ZZ_OK ? 0 : 1;
}
