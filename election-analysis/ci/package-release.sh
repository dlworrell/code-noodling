#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:487313d7-744c-4d3a-b511-2d117a17d0af
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
exec bash scripts/ci/package-project-source.sh election-analysis "$@" data/crosswalks data/elections reports tests/fixtures
