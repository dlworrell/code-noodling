#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:2a061c24-9853-4499-8fc4-bf3ba775b949
set -euo pipefail
runs=$(gh api --paginate "repos/$1/commits/$2/check-runs?per_page=100" \
  --jq '.check_runs[] | select(.name != "Verify main" and .name != "Package and attest release source" and .name != "Publish (parks for owner approval)") | [.name, .status, (.conclusion // "")] | @tsv')
if [[ -z $runs ]]; then
  echo "::error::No check has run on selected main commit $2."
  exit 1
fi
bad=$(awk -F'\t' '$2 != "completed" || ($3 != "success" && $3 != "neutral" && $3 != "skipped")' <<<"$runs")
if [[ -n $bad ]]; then
  printf '::error::Some checks on %s have not passed:\n%s\n' "$2" "$bad"
  exit 1
fi
