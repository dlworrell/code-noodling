#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:5ae94741-2232-4bd8-bebf-bfc6f1d35d3e

# Election Result Reports logic, run by
# .github/workflows/election-result-reports.yml. The root workflow only picks
# triggers, permissions and artifacts; the logic lives in the scope
# (Catylist ADR-0004). Paths are relative to the repository root.
set -euo pipefail
scope=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$scope/.." && pwd)
rel=${scope#"$root"/}
edt=${EDT_DIR:-$root/.tools/engineering-docs-toolkit}
export PYTHONPATH="$scope:$edt"
cd "$root"

case "${1:-}" in
  test)
    # test PATTERN: one group of the scope's offline tests.
    python3 -m unittest discover -s "$rel/tests" -p "${2:?usage: reports.sh test PATTERN}" -v
    ;;
  generate)
    # One report directory per state and election:
    # data/elections/<state>/<date>/ builds into reports/<state>/<date>/. An
    # election without any snapshots yet (the watcher created only its
    # election.toml) is skipped.
    shopt -s nullglob
    built=0
    for config in "$rel"/data/elections/*/*/election.toml; do
      directory=$(dirname "$config")
      election="$(basename "$(dirname "$directory")")/$(basename "$directory")"
      if ! compgen -G "$directory/input/*/*" > /dev/null; then
        echo "Skipping $election: no snapshots yet."
        continue
      fi
      python3 -m election_reporting --config "$config" --output "$rel/reports/$election"
      built=$((built + 1))
    done
    if [[ $built -eq 0 ]]; then
      echo "::error::No election has snapshots to report on."
      exit 1
    fi
    ;;
  detect)
    # Writes changed=true|false to $GITHUB_OUTPUT.
    output=${GITHUB_OUTPUT:-/dev/null}
    git add -A -- "$rel/reports"
    if git diff --cached --quiet -- "$rel/reports"; then
      echo "changed=false" >> "$output"
      echo "Reports are already current."
    else
      echo "changed=true" >> "$output"
      git diff --cached --stat -- "$rel/reports"
      if [[ ${GITHUB_EVENT_NAME:-} == pull_request ]]; then
        echo "::warning::Committed reports differ from regenerated reports. Nothing is committed from pull requests; the regenerated reports are in the election-result-reports artifact and will arrive through an automation pull request after merge."
      fi
    fi
    ;;
  clear)
    # Before the regenerated reports artifact is downloaded in their place.
    rm -rf "${rel:?}/reports"
    ;;
  publish)
    python3 -m election_reporting.ci.publish_reports \
      --base "${BASE_BRANCH:?}" \
      --base-sha "${GITHUB_SHA:?}" \
      --branch automation/election-reports \
      --path "$rel/reports" \
      --run-url "${RUN_URL:?}"
    ;;
  *)
    echo "usage: reports.sh {test PATTERN|generate|detect|clear|publish}" >&2
    exit 2
    ;;
esac
