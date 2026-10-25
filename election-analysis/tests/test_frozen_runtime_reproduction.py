# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:869985c1-0150-41db-b134-fd5a75901e78

import hashlib
import importlib.util
import io
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("frozen_reproduction", Path(__file__).parents[1]/"ci/reproduce_official_experiment.py")
reproduction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduction)


class FrozenRuntimeTests(unittest.TestCase):
    def mini_repo(self, root):
        root = Path(root)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        for name, content in {"election_reporting/core.py": b"VALUE=1\n", "ci/requirements-provenance.txt": b"pypdf==6.10.0\n"}.items():
            path = root/"election-analysis"/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(content)
        return self.commit(root)

    def commit(self, root):
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=Disposable regression", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=root, check=True)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    def hashes(self, root):
        return {name: hashlib.sha256((Path(root)/"election-analysis"/name).read_bytes()).hexdigest()
                for name in ("election_reporting/core.py", "ci/requirements-provenance.txt")}

    def test_dependency_upgrade_does_not_rewrite_frozen_source(self):
        with tempfile.TemporaryDirectory() as td:
            commit = self.mini_repo(td); hashes = self.hashes(td)
            pin = Path(td)/"election-analysis/ci/requirements-provenance.txt"
            pin.write_text("pypdf==6.21.2\n")
            self.assertEqual(reproduction.verify_frozen_tree(td, commit, hashes), hashes)
            changed = self.commit(td)
            with self.assertRaisesRegex(reproduction.ReproductionError, "requirements-provenance"):
                reproduction.verify_frozen_tree(td, changed, hashes)

    def test_unlisted_python_source_is_not_an_equivalent_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            self.mini_repo(td);hashes = self.hashes(td)
            (Path(td)/"election-analysis/election_reporting/extra.py").write_text("VALUE=2\n")
            changed = self.commit(td)
            with self.assertRaisesRegex(reproduction.ReproductionError, "inventory"):
                reproduction.verify_frozen_tree(td, changed, hashes)

    def test_runtime_and_unpinned_requirement_mismatch_fail_closed(self):
        pins = reproduction.pinned_versions(b"# frozen\npypdf==6.10.0\nreuse==6.2.0\n")
        reproduction.require_runtime(pins, pins)
        for observed in ({"pypdf": "6.21.2", "reuse": "6.2.0"}, {"pypdf": "6.10.0", "reuse": None}):
            with self.assertRaises(reproduction.ReproductionError):reproduction.require_runtime(pins, observed)
        for raw in (b"pypdf>=6\n", b"pypdf==6\npypdf==7\n", b"--index-url https://example.invalid\n"):
            with self.assertRaises(reproduction.ReproductionError):reproduction.pinned_versions(raw)

    def test_invalid_paths_and_source_refs_are_rejected_before_git(self):
        for name in ("../x", "/tmp/x", "x\\y", ""):
            with self.assertRaises(reproduction.ReproductionError):reproduction.safe_path(name)
        with self.assertRaises(reproduction.ReproductionError):
            reproduction.verify_frozen_tree("/no/repo", "--exec=bad", {"a": "0"*64})
        with tempfile.TemporaryDirectory() as td, self.assertRaises(reproduction.ReproductionError):
            reproduction.reproduce("/no/repo", Path(td)/"out", source_commit="--output=unexpected")
        with tempfile.TemporaryDirectory() as td:
            occupied = Path(td)/"seal.json";occupied.write_text("first seal")
            with self.assertRaises(reproduction.ReproductionError):reproduction.reproduce(td, occupied)
            self.assertEqual(occupied.read_text(), "first seal")

    def test_git_archive_rejects_traversal_links_and_duplicate_members(self):
        for kind in ("traversal", "symlink", "duplicate"):
            raw = io.BytesIO()
            with tarfile.open(fileobj=raw, mode="w:") as archive:
                info = tarfile.TarInfo("../escape" if kind == "traversal" else "election-analysis/file")
                if kind == "symlink":info.type=tarfile.SYMTYPE;info.linkname="/tmp/escape"
                else:info.size=1
                archive.addfile(info, None if kind == "symlink" else io.BytesIO(b"x"))
                if kind == "duplicate":archive.addfile(info, io.BytesIO(b"x"))
            with tempfile.TemporaryDirectory() as td, self.assertRaises(reproduction.ReproductionError):
                reproduction.extract_snapshot(raw.getvalue(), Path(td)/"snapshot")

    def test_normalization_retains_source_clocks_probabilities_and_versions(self):
        a = {"generated_at": "new", "rows": [{"prediction_generated_at": "new", "information_time": "old", "probability": .2, "result_version": "v1"}], "sealed_predictions_sha256": "a"}
        b = {**a, "generated_at": "another", "rows": [{**a["rows"][0], "prediction_generated_at": "another"}]}
        self.assertEqual(reproduction.semantic(a), reproduction.semantic(b))
        for key, value in (("information_time", "later"), ("probability", .3), ("result_version", "v2")):
            changed = {**b, "rows": [{**b["rows"][0], key: value}]}
            self.assertNotEqual(reproduction.semantic(a), reproduction.semantic(changed))
        changed = {**b, "sealed_predictions_sha256": "b"}
        self.assertNotEqual(reproduction.semantic(a), reproduction.semantic(changed))
        self.assertEqual(reproduction.semantic(a, scoring=True), reproduction.semantic(changed, scoring=True))


if __name__ == "__main__":unittest.main()
