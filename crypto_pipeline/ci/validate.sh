#!/usr/bin/env bash
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
    clang-tidy crypto_multiplier.c crypto_montgomery_ct.c tests/test_montgomery_ct.c \
      tests/test_suite.c tests/test_bounds.c -- -std=c11 -I.
    ;;
  build)
    make clean
    make shared
    make all
    ;;
  memcheck)
    # Run the test executable inside Valgrind's Memcheck engine.
    valgrind --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./test_suite
    ;;
  smoke)
    # This smoke test does not establish constant-time or side-channel safety.
    ./test_suite
    echo "[CI Pipeline Pass] Build and functional smoke tests completed."
    ;;
  bounds)
    make bounds-sanitize
    ;;
  differential)
    python3 tests/test_arithmetic.py
    python3 tests/test_montgomery_ct.py
    ;;
  *)
    echo "usage: $0 {install-tools|static-analysis|build|memcheck|smoke|bounds|differential}" >&2
    exit 2
    ;;
esac
