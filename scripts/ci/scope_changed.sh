#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:4dd5e980-d214-47f0-87e2-a86fd389cd07

# Decide whether a project's required test job must run: writes
# changed=true|false to $GITHUB_OUTPUT. Usage: scope_changed.sh <path>...
# True when anything under the given project paths changed; also on manual
# runs and whenever the diff base is unknown (new branch, shallow history),
# so a doubtful case always runs the tests. Repository-level CI files are not
# a project's paths: changing them does not run another project's tests
# (owner decision 2026-11-06).
set -euo pipefail
(( $# > 0 )) || { echo "usage: $0 <path>..." >&2; exit 2; }

decide() {
  case "${EVENT:-}" in
    pull_request) base="${PR_BASE:-}" ;;
    push) base="${BEFORE:-}" ;;
    *) echo true; return ;;
  esac
  if [[ -z "$base" || "$base" =~ ^0+$ ]] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
    echo true; return
  fi
  if git diff --name-only "$base" HEAD -- "$@" | grep -q .; then
    echo true
  else
    echo false
  fi
}

result=$(decide "$@")
echo "changed=$result" >> "${GITHUB_OUTPUT:-/dev/stdout}"
echo "$* changed or undetermined: $result"
