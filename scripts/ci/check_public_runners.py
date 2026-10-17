#!/usr/bin/env python3
"""Reject public workflows that select anything except approved hosted runners.

This is a source regression check, not a runner authorization boundary. Keep
repository runner inventories empty; a fork can change or bypass this check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

HOSTED = re.compile(
    r"(?:ubuntu-(?:latest|22\.04|24\.04)(?:-arm)?|windows-(?:latest|2022|2025)|"
    r"macos-(?:latest|14|15|15-intel|26))"
)
# Existing governance delegation; changes require review of the called workflow.
REUSABLE = {"dlworrell/AEMS/.github/workflows/aes-sec-001-distributed.yml@main"}


def hosted(value: object) -> bool:
    return isinstance(value, str) and HOSTED.fullmatch(value) is not None


def validate(path: Path) -> list[str]:
    errors = []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs = document["jobs"]
        if not isinstance(jobs, dict) or not jobs:
            raise ValueError("jobs must be a nonempty mapping")
        for name, job in jobs.items():
            if not isinstance(job, dict):
                raise ValueError("job must be a mapping")
            if "uses" in job:
                safe = job["uses"] in REUSABLE and "runs-on" not in job
            else:
                runner = job.get("runs-on")
                safe = hosted(runner)
                if runner == "${{ matrix.os }}":
                    matrix = job.get("strategy", {}).get("matrix", {})
                    safe = isinstance(matrix, dict) and bool(matrix)
                    if safe:
                        values = list(matrix.get("os", []))
                        includes = matrix.get("include", [])
                        values += [entry["os"] for entry in includes if "os" in entry]
                        safe = bool(values) and all(hosted(value) for value in values)
                        safe = safe and (
                            bool(matrix.get("os")) or all("os" in entry for entry in includes)
                        )
            if not safe:
                errors.append(
                    f"{path}:{name}: runner must be an approved GitHub-hosted label "
                    "or literal hosted matrix.os"
                )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as error:
        errors.append(f"{path}: cannot establish hosted-only execution: {error}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / ".github/workflows").glob("*.y*ml"))
    errors = [error for path in paths for error in validate(path)]
    if not paths:
        errors.append("No workflows found")
    for error in errors:
        print(error)
    print(f"public-runner-policy: {len(paths)} workflows, {len(errors)} error(s)")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
