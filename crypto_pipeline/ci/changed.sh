#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:4dd5e980-d214-47f0-87e2-a86fd389cd07

# Decide whether crypto_ci.yml's required audit-and-test job must run: writes
# crypto=true|false to $GITHUB_OUTPUT. Runs when crypto_pipeline/ or the
# workflow changed; also on manual runs and whenever the diff base is unknown
# (new branch, shallow history), so a doubtful case always runs the tests.
set -euo pipefail

decide() {
  case "${EVENT:-}" in
    pull_request) base="${PR_BASE:-}" ;;
    push) base="${BEFORE:-}" ;;
    *) echo true; return ;;
  esac
  if [[ -z "$base" || "$base" =~ ^0+$ ]] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
    echo true; return
  fi
  if git diff --name-only "$base" HEAD -- crypto_pipeline .github/workflows/crypto_ci.yml | grep -q .; then
    echo true
  else
    echo false
  fi
}

result=$(decide)
echo "crypto=$result" >> "${GITHUB_OUTPUT:-/dev/stdout}"
echo "crypto_pipeline changed or undetermined: $result"
