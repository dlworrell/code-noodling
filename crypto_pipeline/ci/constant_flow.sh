#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:0af20b28-7fa2-4907-8652-73462b581684

# Montgomery constant-flow steps, run by .github/workflows/crypto_constant_flow.yml.
# Usage: ci/constant_flow.sh <step>
# The constant-flow step reads CT_CC and CT_OPT; the timing step takes
# --optimization, --limbs and --output, passed through to run_timing_diagnosis.py.
set -euo pipefail
scope="$(cd "$(dirname "$0")/.." && pwd)"
step="${1:-}"
shift || true

case "$step" in
  install-tools)
    sudo apt-get update
    sudo apt-get install -y gcc clang valgrind binutils python3
    ;;
  timing-runner-tests)
    python3 -m unittest discover -s "$scope/tests" -p test_timing_runner.py
    ;;
  constant-flow)
    "$scope/run_constant_flow.sh"
    ;;
  timing)
    # Timing measurements on shared runners are evidence, not a portable proof.
    python3 "$scope/run_timing_diagnosis.py" "$@"
    ;;
  *)
    echo "usage: $0 {install-tools|timing-runner-tests|constant-flow|timing ...}" >&2
    exit 2
    ;;
esac
