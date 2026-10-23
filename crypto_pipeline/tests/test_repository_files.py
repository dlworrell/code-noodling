# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:51e197ca-5f13-4fd3-be15-222e874ebd61

"""Checks that crypto_pipeline's evidence and provenance records reach git."""

import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class RepositoryFileTests(unittest.TestCase):
    def test_evidence_and_provenance_records_are_not_git_ignored(self) -> None:
        if not (REPO / ".git").exists():
            self.skipTest("not a git checkout")
        for path in (
            "crypto_pipeline/docs/engineering/provenance-baseline.json",
            "crypto_pipeline/docs/engineering/evidence/new-evidence.json",
            "crypto_pipeline/ci/scope-manifest.json",
        ):
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", "--no-index", path], cwd=REPO
            ).returncode == 0
            self.assertFalse(ignored, f"{path} is ignored by .gitignore")

    def test_manifest_evidence_paths_exist(self) -> None:
        manifest = (REPO / "aes-manifest.yaml").read_text(encoding="utf-8")
        for key in ("profile:", "waiver_log:", "review_dispositions:"):
            for line in manifest.splitlines():
                if line.strip().startswith(key):
                    path = line.split(":", 1)[1].strip()
                    self.assertTrue((REPO / path).is_file(), f"aes-manifest.yaml {key} {path} does not exist")


if __name__ == "__main__":
    unittest.main()
