#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:51596e6e-d492-4bf0-a04f-82040f106c19

# Certificate-checker fuzzing and exhaustive soundness checks, run by
# .github/workflows/crypto_fuzz.yml. See docs/engineering/CRYPTO-FUZZING.md.
# Usage: ci/fuzz.sh <step>
#   install-tools  clang and the libFuzzer runtime (CI only)
#   exhaustive     every small Pocklington certificate (N < 512): only primes accepted
#   smoke          ASan/UBSan build with any C compiler: replay the seeds (all must be
#                  accepted), then FUZZ_ITERATIONS deterministic mutations (default 20000)
#   libfuzzer      coverage-guided run with clang for FUZZ_SECONDS (default 60)
# Builds go to a temporary directory outside the repository; with
# FUZZ_OUTPUT=<dir>, the corpus, any crash input and logs are kept there.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

SRCS=(crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c crypto_ecpp.c crypto_mixed.c)
SAN=(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
work=$(mktemp -d)
out=${FUZZ_OUTPUT:-$work/out}
mkdir -p "$out"

corpus() {
  python3 tests/fuzz_corpus.py "$out/seeds" >/dev/null
}

case "${1:-}" in
  install-tools)
    sudo apt-get update
    sudo apt-get install -y clang libclang-rt-18-dev
    ;;
  exhaustive)
    "${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror -I. tests/test_cert_exhaustive.c \
      crypto_multiplier.c crypto_prime.c crypto_bigint.c crypto_cert.c -o "$work/exhaustive"
    "$work/exhaustive" | tee "$out/exhaustive.log"
    ;;
  smoke)
    corpus
    "${CC:-cc}" -std=c11 -O1 -g -Wall -Wextra -Werror "${SAN[@]}" -DFUZZ_STANDALONE -I. \
      tests/fuzz_certificates.c "${SRCS[@]}" -o "$work/fuzz_standalone"
    "$work/fuzz_standalone" "$out"/seeds/* | tee "$out/smoke.log"
    ;;
  libfuzzer)
    corpus
    clang -std=c11 -O1 -g -fsanitize=fuzzer "${SAN[@]}" -I. \
      tests/fuzz_certificates.c "${SRCS[@]}" -o "$work/fuzz_libfuzzer"
    mkdir -p "$out/corpus"
    cp "$out"/seeds/* "$out/corpus/"
    "$work/fuzz_libfuzzer" -max_total_time="${FUZZ_SECONDS:-60}" -print_final_stats=1 \
      -artifact_prefix="$out/" "$out/corpus" 2>&1 | tee "$out/libfuzzer.log"
    ;;
  *)
    echo "usage: $0 {install-tools|exhaustive|smoke|libfuzzer}" >&2
    exit 2
    ;;
esac
