#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:d966a88a-7a27-4c56-bf6a-591b1201acfc

# Report-only provenance checks, run by .github/workflows/crypto_provenance.yml
# (PRM-0012 step 3; logic in the scope per Catylist ADR-0004).
set -euo pipefail
scope=$(cd "$(dirname "$0")/.." && pwd)

case "${1:-}" in
  install-tools)
    python3 -m pip install -r "$scope/ci/requirements-provenance.txt"
    sudo apt-get update
    sudo apt-get install -y doxygen
    ;;
  tests)
    python3 -m unittest discover -s "$scope/tests" -p test_provenance_checks.py
    ;;
  report)
    # Report-only: record the checks' status (0 pass, 1 violation,
    # 2 indeterminate) without failing the job.
    output=${2:-build/provenance}
    status=0
    python3 "$scope/ci/provenance.py" all --output "$output" || status=$?
    echo "$status" > "$output/status"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "status=$status" >> "$GITHUB_OUTPUT"; fi
    ;;
  *)
    echo "usage: $0 {install-tools|tests|report [OUTPUT]}" >&2
    exit 2
    ;;
esac
