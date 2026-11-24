#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:02bdb165-d31a-422c-96d8-ae34a62c40cc
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
exec bash scripts/ci/package-project-source.sh crypto-pipeline "$@"
