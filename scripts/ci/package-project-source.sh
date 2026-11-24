#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:3f254a2c-e08c-4b93-a967-c1cf4826945a
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 PROJECT SHA VERSION OUTPUT [EXCLUDED_SCOPE_PREFIX...]" >&2
  exit 2
fi
project=$1 sha=$2 version=$3 output=$4
shift 4
[[ $sha =~ ^[a-f0-9]{40}$ && $(git cat-file -t "$sha") == commit ]]
selection=$(python3 scripts/ci/project_release.py select "$project" "$version" refs/heads/main)
scope=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["root"])' <<<"$selection")
git cat-file -e "$sha:LICENSES/Apache-2.0.txt"
git cat-file -e "$sha:LICENSES/MIT.txt"
tmp=$(mktemp "${output}.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
paths=("$scope" LICENSES)
for prefix in "$@"; do
  [[ $prefix =~ ^[a-zA-Z0-9_/-]+$ && $prefix != /* ]] || exit 2
  paths+=(":(exclude,top)$scope/$prefix")
done
git archive --format=tar --prefix="$project-$version/" "$sha" -- "${paths[@]}" | gzip -n >"$tmp"
mv -- "$tmp" "$output"
