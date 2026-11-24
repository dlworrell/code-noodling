#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Donovan Worrell
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:70e75475-acb2-432b-bbf0-1146d5173db3
set -euo pipefail
# Repository ruleset: protect-release-tags. Never infer empty bypasses from
# a missing (permission-redacted) field or accept a failed API read.
gh api "repos/$1/rulesets/25747644" | python3 scripts/ci/project_release.py rules "$2" -
