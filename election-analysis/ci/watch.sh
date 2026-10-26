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

# The existing workflow calls these commands as separate steps. Gate every
# live/publication entry point here as well as per-state requests in Python.
# This works for scheduled and manual runs without broadening token permissions.
case "${1:-}" in
  pending|carry|fetch|publish)
    gate_args=()
    if [[ ${FORCE:-false} == true ]]; then gate_args+=(--force); fi
    if python3 -m election_reporting.watch status --require-eligible "${gate_args[@]}"; then
      :
    else
      gate_status=$?
      if [[ $gate_status != 3 ]]; then exit "$gate_status"; fi
      if [[ ${1:-} == pending ]]; then
        echo "exists=false" >> "${GITHUB_OUTPUT:-/dev/null}"
      fi
      exit 0
    fi
    ;;
esac

case "${1:-}" in
  gate)
    # Offline, real-clock status for the workflow's publication/fetch gate.
    args=()
    if [[ ${FORCE:-false} == true ]]; then args+=(--force); fi
    python3 -m election_reporting.watch status "${args[@]}"
    ;;
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
    # FORCE=true ignores cadence/end, never the state's not_before gate.
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
    echo "usage: watch.sh {gate|test|pending|carry CHECKOUT|fetch|publish}" >&2
    exit 2
    ;;
esac
