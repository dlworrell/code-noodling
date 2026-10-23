#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:1df71ae8-7f71-4a10-9e3a-ad960e9e9a1f

# Report-only provenance checks, run by .github/workflows/election_provenance.yml
# (election slice, step 3; logic in the scope per Catylist ADR-0004).
set -euo pipefail
scope=$(cd "$(dirname "$0")/.." && pwd)

case "${1:-}" in
  install-tools)
    python3 -m pip install -r "$scope/ci/requirements-provenance.txt"
    ;;
  tests)
    PYTHONPATH="$scope" python3 -m unittest discover -s "$scope/tests" -p test_provenance_checks.py
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
