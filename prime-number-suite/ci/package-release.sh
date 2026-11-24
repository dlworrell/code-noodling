#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:c8e1587d-66c3-452c-b43e-c43bec848991
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
exec bash scripts/ci/package-project-source.sh prime-number-suite "$@"
