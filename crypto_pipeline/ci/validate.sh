#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:9d322d83-f3f4-4602-bf28-0265680f5f33

# Cryptographic wrapper validation steps, run by .github/workflows/crypto_ci.yml.
# Usage: ci/validate.sh <step>
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  install-tools)
    sudo apt-get update
    sudo apt-get install -y build-essential clang clang-tidy valgrind libssl-dev
    ;;
  static-analysis)
    clang-tidy crypto_multiplier.c crypto_montgomery_ct.c crypto_prime.c crypto_bigint.c crypto_cert.c crypto_ecpp.c \
      tests/test_montgomery_ct.c tests/test_suite.c tests/test_bounds.c tests/test_prime.c \
      tests/test_cert.c tests/test_ecpp.c -- -std=c11 -I.
    ;;
  build)
    make clean
    make shared
    make all
    ;;
  memcheck)
    # Run the test executable inside Valgrind's Memcheck engine.
    valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_suite
    valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_prime
    valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_cert
    valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_ecpp
    ;;
  smoke)
    # This smoke test does not establish constant-time or side-channel safety.
    ./test_suite
    ./test_prime
    ./test_cert
    ./test_ecpp
    echo "[CI Pipeline Pass] Build and functional smoke tests completed."
    ;;
  bounds)
    make bounds-sanitize
    ;;
  differential)
    python3 tests/test_arithmetic.py
    python3 tests/test_montgomery_ct.py
    python3 tests/test_prime.py
    python3 tests/test_pocklington.py
    python3 tests/test_ecpp.py
    python3 tests/test_repository_files.py
    ;;
  *)
    echo "usage: $0 {install-tools|static-analysis|build|memcheck|smoke|bounds|differential}" >&2
    exit 2
    ;;
esac
