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
  test-results)
    # Test-results file and predicate for the step 4 test attestation.
    output=${2:-build/provenance}
    make -s -C "$scope" shared
    status=0
    python3 "$scope/ci/test_results.py" --output "$output" --manifest "$output/scope-manifest.json" || status=$?
    echo "$status" > "$output/test-status"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "test_status=$status" >> "$GITHUB_OUTPUT"; fi
    ;;
  verify)
    # Step 4/5 evidence, run by the attest job: each attestation verifies for
    # the exact artifact, workflow, commit and ref, and verification fails for
    # a copy with one byte changed and for any other commit.
    output=${2:?usage: provenance.sh verify OUTPUT}
    repo=${GITHUB_REPOSITORY:?}; sha=${GITHUB_SHA:?}; ref=${GITHUB_REF:?}
    other=${OTHER_SHA:-}
    if [[ -z "$other" || "$other" =~ ^0+$ || "$other" == "$sha" ]]; then other=$(printf '%040d' 1); fi
    workflow="$repo/.github/workflows/crypto_provenance.yml"
    base=https://catalyst.dlworrell.dev/attestation
    check() {
      gh attestation verify "$1" --repo "$repo" --predicate-type "$2" \
        --signer-workflow "$workflow" --source-digest "$3" --source-ref "$ref" \
        --deny-self-hosted-runners >/dev/null 2>&1
    }
    expect() {  # expect pass|fail DESCRIPTION -- command...
      local want=$1 what=$2; shift 3
      if "$@"; then got=pass; else got=fail; fi
      echo "| $what | expected $want | $got |"
      [[ $got == "$want" ]]
    }
    tampered=$(mktemp -d)/scope-manifest.json
    python3 - "$output/scope-manifest.json" "$tampered" <<'PY'
import sys
data = bytearray(open(sys.argv[1], "rb").read())
data[len(data) // 2] ^= 0x01  # one byte changed
open(sys.argv[2], "wb").write(bytes(data))
PY
    failed=0
    {
      echo "## Attestation verification"
      echo ""
      echo "| Case | Expected | Result |"
      echo "|---|---|---|"
      expect pass "scope-manifest.json, check/v1, commit $sha" -- check "$output/scope-manifest.json" "$base/check/v1" "$sha" || true
      expect pass "test-results.json, test/v1, commit $sha" -- check "$output/test-results.json" "$base/test/v1" "$sha" || true
      expect fail "scope-manifest.json with one byte changed" -- check "$tampered" "$base/check/v1" "$sha" || true
      expect fail "scope-manifest.json claimed for commit $other" -- check "$output/scope-manifest.json" "$base/check/v1" "$other" || true
    } | tee "$output/verification.md"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then cat "$output/verification.md" >> "$GITHUB_STEP_SUMMARY"; fi
    grep -q "expected pass | fail\|expected fail | pass" "$output/verification.md" && failed=1
    exit $failed
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
    echo "usage: $0 {install-tools|tests|report [OUTPUT]|test-results [OUTPUT]|verify OUTPUT}" >&2
    exit 2
    ;;
esac
