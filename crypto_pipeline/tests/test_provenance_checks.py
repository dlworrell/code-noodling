# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:95b33455-e799-4c0e-92f2-664076effde2

"""Tests for ci/provenance.py: File-ID rules, the scope manifest and exit codes."""

# Fixtures below contain SPDX and File-ID text that is test data, not this
# file's own licensing; REUSE must not read it.
# REUSE-IgnoreStart

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "provenance", Path(__file__).resolve().parents[1] / "ci" / "provenance.py"
)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)

ID_A = "urn:uuid:0f1e2d3c-4b5a-4968-8776-655443322110"
ID_B = "urn:uuid:11223344-5566-4778-9899-aabbccddeeff"


def header(file_id: str) -> str:
    return (
        "/*\n * SPDX-FileCopyrightText: 2026 Donovan Worrell\n *\n"
        f" * SPDX-License-Identifier: Apache-2.0\n * Catalyst-File-ID: {file_id}\n */\n"
    )


class ScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.scope = Path(self.tmp.name)
        # Not a git checkout: governed_files() walks the directory.
        (self.scope / "LICENSES").mkdir()
        (self.scope / "LICENSES" / "Apache-2.0.txt").write_text("license text\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> None:
        path = self.scope / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


class FileIdTests(ScopeTest):
    def test_inline_and_sidecar_ids_pass(self) -> None:
        self.write("a.c", header(ID_A) + "int a;\n")
        self.write("data.json", "{}\n")
        self.write("data.json.license", f"SPDX-License-Identifier: Apache-2.0\nCatalyst-File-ID: {ID_B}\n")
        status, details = provenance.file_ids(self.scope)
        self.assertEqual(status, provenance.PASS, details["violations"])
        self.assertEqual(details["ids"], {"a.c": ID_A, "data.json": ID_B})

    def test_every_rule_is_a_violation(self) -> None:
        self.write("missing.c", "int m;\n")
        self.write("bad.c", header("urn:uuid:not-a-uuid") + "int b;\n")
        self.write("v1.c", header("urn:uuid:0f1e2d3c-4b5a-1968-8776-655443322110") + "int v;\n")
        self.write("twice.c", header(ID_A) + "int t;\n")
        self.write("twice.c.license", f"Catalyst-File-ID: {ID_B}\n")
        self.write("dup1.c", header(ID_B.replace("ff", "fe")) + "int d;\n")
        self.write("dup2.c", header(ID_B.replace("ff", "fe")) + "int e;\n")
        self.write("orphan.txt.license", f"Catalyst-File-ID: {ID_A}\n")
        status, details = provenance.file_ids(self.scope)
        self.assertEqual(status, provenance.VIOLATION)
        text = "\n".join(details["violations"])
        self.assertIn("missing.c: no Catalyst-File-ID", text)
        self.assertIn("bad.c: 'urn:uuid:not-a-uuid' is not a version 4 urn:uuid", text)
        self.assertIn("v1.c:", text)  # version 1 UUID rejected
        self.assertIn("twice.c: 2 Catalyst-File-IDs", text)
        self.assertIn("dup2.c: Catalyst-File-ID", text)
        self.assertIn("orphan.txt.license: sidecar for a file that is not in the scope", text)

    def test_ids_below_the_header_are_ignored(self) -> None:
        body = "\n" * 30 + f"/* Catalyst-File-ID: {ID_B} quoted in a comment */\n"
        self.write("a.c", header(ID_A) + body)
        status, details = provenance.file_ids(self.scope)
        self.assertEqual(status, provenance.PASS, details["violations"])


class ManifestAndRunTests(ScopeTest):
    def test_manifest_lists_every_file_with_id_and_hash(self) -> None:
        self.write("a.c", header(ID_A) + "int a;\n")
        _, ids = provenance.file_ids(self.scope)
        with mock.patch.dict("os.environ", {"GITHUB_SHA": "abc123"}):
            status, document = provenance.manifest(self.scope, ids["ids"])
        self.assertEqual(status, provenance.PASS)
        self.assertEqual(document["commit"], "abc123")
        by_path = {entry["path"]: entry for entry in document["files"]}
        self.assertEqual(by_path["crypto_pipeline/a.c"]["file_id"], ID_A)
        self.assertEqual(
            by_path["crypto_pipeline/a.c"]["sha256"],
            hashlib.sha256((self.scope / "a.c").read_bytes()).hexdigest(),
        )
        self.assertEqual(by_path["crypto_pipeline/LICENSES/Apache-2.0.txt"]["kind"], "license-text")

    def test_missing_tools_are_indeterminate_and_violations_win(self) -> None:
        self.write("a.c", "int a;\n")  # no File-ID: a violation
        output = self.scope / "out"
        with mock.patch.object(provenance.shutil, "which", return_value=None), mock.patch.dict(
            "os.environ", {"GITHUB_SHA": "abc123", "GITHUB_STEP_SUMMARY": ""}
        ):
            overall = provenance.main(["all", "--scope", str(self.scope), "--output", str(output)])
        results = json.loads((output / "results.json").read_text())
        self.assertEqual(results["reuse"]["status"], provenance.INDETERMINATE)
        self.assertEqual(results["doxygen"]["status"], provenance.INDETERMINATE)
        self.assertEqual(results["file-ids"]["status"], provenance.VIOLATION)
        self.assertEqual(overall, provenance.VIOLATION)
        self.assertTrue((output / "scope-manifest.json").exists())
        self.assertIn("Report-only", (output / "summary.md").read_text())

    def test_only_indeterminate_gives_two(self) -> None:
        self.write("a.c", header(ID_A) + "int a;\n")
        output = self.scope / "out"
        with mock.patch.object(provenance.shutil, "which", return_value=None), mock.patch.dict(
            "os.environ", {"GITHUB_SHA": "abc123", "GITHUB_STEP_SUMMARY": ""}
        ):
            overall = provenance.main(["all", "--scope", str(self.scope), "--output", str(output)])
        self.assertEqual(overall, provenance.INDETERMINATE)


TEST_RESULTS = Path(__file__).resolve().parents[1] / "ci" / "test_results.py"
PROVENANCE_SH = Path(__file__).resolve().parents[1] / "ci" / "provenance.sh"


class TestResultsTests(ScopeTest):
    def test_outcomes_totals_predicate_and_exit_code(self) -> None:
        spec = importlib.util.spec_from_file_location("test_results", TEST_RESULTS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.write(
            "tests/test_sample.py",
            "import unittest\n"
            "class Sample(unittest.TestCase):\n"
            "    def test_ok(self):\n        pass\n"
            "    def test_bad(self):\n        self.fail('x')\n",
        )
        self.write("manifest.json", "{}\n")
        output = self.scope / "out"
        cwd = Path.cwd()
        try:
            with mock.patch.object(module, "SCOPE", self.scope), mock.patch.dict("os.environ", {"GITHUB_SHA": "abc123"}):
                status = module.main(["--output", str(output), "--manifest", str(self.scope / "manifest.json")])
        finally:
            import os
            os.chdir(cwd)
        results = json.loads((output / "test-results.json").read_text())
        predicate = json.loads((output / "test-predicate.json").read_text())
        self.assertEqual(status, 1)
        self.assertEqual(results["totals"], {"run": 2, "passed": 1, "failed": 1})
        self.assertFalse(predicate["passed"])
        self.assertEqual(predicate["commit"], "abc123")
        self.assertEqual(predicate["test_results_sha256"], hashlib.sha256((output / "test-results.json").read_bytes()).hexdigest())
        self.assertEqual(predicate["scope_manifest_sha256"], hashlib.sha256(b"{}\n").hexdigest())


class VerifyScriptTests(ScopeTest):
    """provenance.sh verify, with a stub `gh` standing in for GitHub's verifier."""

    def run_verify(self, honest: bool):
        import subprocess

        out = self.scope / "out"
        out.mkdir()
        (out / "scope-manifest.json").write_text('{"files": []}\n')
        (out / "test-results.json").write_text('{"tests": []}\n')
        stub = self.scope / "bin" / "gh"
        stub.parent.mkdir()
        rule = (
            'case "$3" in */out/scope-manifest.json|*/out/test-results.json) '
            '[[ "$*" == *"--source-digest aaaa "* ]] && exit 0;; esac; exit 1'
            if honest else "exit 0"
        )
        stub.write_text("#!/usr/bin/env bash\n" + rule + "\n")
        stub.chmod(0o755)
        env = {
            "PATH": f"{stub.parent}:/usr/bin:/bin",
            "GITHUB_REPOSITORY": "dlworrell/code-noodling",
            "GITHUB_SHA": "aaaa",
            "GITHUB_REF": "refs/heads/main",
            "OTHER_SHA": "bbbb",
        }
        return subprocess.run(["bash", str(PROVENANCE_SH), "verify", str(out)], env=env, capture_output=True, text=True)

    def test_genuine_attestations_pass_and_forgeries_fail(self) -> None:
        run = self.run_verify(honest=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("one byte changed | expected fail | fail", run.stdout)
        self.assertIn("commit bbbb | expected fail | fail", run.stdout)

    def test_a_verifier_that_accepts_everything_is_caught(self) -> None:
        run = self.run_verify(honest=False)
        self.assertEqual(run.returncode, 1)
        self.assertIn("one byte changed | expected fail | pass", run.stdout)


if __name__ == "__main__":
    unittest.main()

# REUSE-IgnoreEnd
