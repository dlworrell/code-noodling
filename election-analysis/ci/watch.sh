#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:88acb08e-2587-4c56-83a3-82f7410521f9

# Election Source Watch logic, run by .github/workflows/election-source-watch.yml.
# The root workflow only picks triggers, permissions and checkouts; the logic
# lives in the scope (Catylist ADR-0004). Paths are relative to the repository
# root.
set -euo pipefail
scope=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$scope/.." && pwd)
rel=${scope#"$root"/}
export PYTHONPATH="$scope"
cd "$root"

case "${1:-}" in
  test)
    python3 -m unittest discover -s "$rel/tests" -p 'test_election_watch*.py'
    ;;
  pending)
    # Writes exists=true|false to $GITHUB_OUTPUT: whether the automation
    # branch with unmerged snapshots exists.
    output=${GITHUB_OUTPUT:-/dev/null}
    if gh api "repos/${GITHUB_REPOSITORY:?}/branches/${SNAPSHOT_BRANCH:?}" --silent 2>/dev/null; then
      echo "exists=true" >> "$output"
    else
      echo "exists=false" >> "$output"
    fi
    ;;
  carry)
    # carry CHECKOUT: snapshots recorded on the unmerged automation branch
    # checked out at CHECKOUT, which is removed afterwards. A branch written
    # before the move to election-analysis/ keeps the old layout.
    checkout=${2:?usage: watch.sh carry CHECKOUT}
    pending=$checkout/$rel/data/elections
    [[ -d $pending ]] || pending=$checkout/election-data/elections
    python3 -m election_reporting.watch carry --from "$pending"
    rm -rf "$checkout"
    ;;
  fetch)
    # FORCE=true fetches even outside the checking window.
    args=()
    if [[ ${FORCE:-false} == true ]]; then args+=(--force); fi
    python3 -m election_reporting.watch fetch "${args[@]}"
    ;;
  publish)
    python3 -m election_reporting.ci.publish_reports \
      --base "${BASE_BRANCH:?}" \
      --base-sha "${GITHUB_SHA:?}" \
      --branch "${SNAPSHOT_BRANCH:?}" \
      --path "$rel/data/elections" \
      --title "data: new official election result snapshots" \
      --summary "New official result snapshots from the **Election Source Watch** workflow. See each election's SOURCES.json for source URLs, retrieval times and SHA-256 checksums. Merging regenerates the reports." \
      --run-url "${RUN_URL:?}"
    ;;
  *)
    echo "usage: watch.sh {test|pending|carry CHECKOUT|fetch|publish}" >&2
    exit 2
    ;;
esac
