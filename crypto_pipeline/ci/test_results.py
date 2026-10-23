# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:83b9fe38-9c84-4816-98de-459e87e3489d

"""Run crypto_pipeline's Python tests and write a test-results file (PRM-0012 step 4).

Writes, to --output:

- test-results.json: the commit and every test's id and outcome; the subject
  of the test attestation;
- test-predicate.json: the commit, totals and the scope manifest's SHA-256;
  the predicate of that attestation.

Exits 0 when every test passed and 1 otherwise. Requires the shared library
(`make shared`) for the arithmetic and Montgomery tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

SCOPE = Path(__file__).resolve().parents[1]


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.outcomes: dict[str, str] = {}

    def addSuccess(self, test) -> None:
        super().addSuccess(test)
        self.outcomes[test.id()] = "passed"

    def addFailure(self, test, err) -> None:
        super().addFailure(test, err)
        self.outcomes[test.id()] = "failed"

    def addError(self, test, err) -> None:
        super().addError(test, err)
        self.outcomes[test.id()] = "error"

    def addSkip(self, test, reason) -> None:
        super().addSkip(test, reason)
        self.outcomes[test.id()] = "skipped"

    def addExpectedFailure(self, test, err) -> None:
        super().addExpectedFailure(test, err)
        self.outcomes[test.id()] = "expected-failure"

    def addUnexpectedSuccess(self, test) -> None:
        super().addUnexpectedSuccess(test)
        self.outcomes[test.id()] = "unexpected-success"


def commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SCOPE, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build/provenance"))
    parser.add_argument("--manifest", type=Path, help="scope-manifest.json to bind the predicate to")
    args = parser.parse_args(argv)
    output = args.output.resolve()  # before chdir below
    manifest = args.manifest.resolve() if args.manifest else None
    output.mkdir(parents=True, exist_ok=True)

    # Tests load the shared library and fixtures relative to the scope.
    os.chdir(SCOPE)
    sys.path.insert(0, str(SCOPE / "tests"))
    suite = unittest.defaultTestLoader.discover(str(SCOPE / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(resultclass=RecordingResult, verbosity=1)
    result = runner.run(suite)

    outcomes = dict(sorted(result.outcomes.items()))
    totals = {"run": result.testsRun}
    for outcome in outcomes.values():
        totals[outcome] = totals.get(outcome, 0) + 1
    passed = result.wasSuccessful() and result.testsRun > 0
    sha = commit()
    results = {
        "schema": "https://catalyst.dlworrell.dev/test-results/v1-draft",
        "scope": "crypto_pipeline/",
        "commit": sha,
        "python": sys.version.split()[0],
        "totals": totals,
        "passed": passed,
        "tests": [{"id": test_id, "outcome": outcome} for test_id, outcome in outcomes.items()],
    }
    results_path = output / "test-results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    predicate = {
        "scope": "crypto_pipeline/",
        "commit": sha,
        "passed": passed,
        "totals": totals,
        "test_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }
    if manifest:
        predicate["scope_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (output / "test-predicate.json").write_text(json.dumps(predicate, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
