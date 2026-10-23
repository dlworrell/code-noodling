# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:3f1b6d28-b0fb-49ad-9c44-8095a952fced

"""Tests for ci/provenance.py, ci/test_results.py and provenance.sh verify (election slice, steps 3-4)."""

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
    "election_provenance", Path(__file__).resolve().parents[1] / "ci" / "provenance.py"
)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)

ID_A = "urn:uuid:0f1e2d3c-4b5a-4968-8776-655443322110"
ID_B = "urn:uuid:11223344-5566-4778-9899-aabbccddeeff"


def header(file_id: str) -> str:
    return (
        "# SPDX-FileCopyrightText: 2026 Donovan Worrell\n#\n"
        f"# SPDX-License-Identifier: Apache-2.0\n# Catalyst-File-ID: {file_id}\n\n"
    )


class ScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.scope = Path(self.tmp.name)  # not a git checkout: governed_files() walks it

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> None:
        path = self.scope / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


class KindTests(unittest.TestCase):
    def test_automation_written_paths_are_exempt_kinds(self) -> None:
        cases = {
            "reports/washington/2026-08-04/analysis.json": "generated-report",
            "data/elections/washington/2026-11-03/input/pierce-county/2026-11-03-2015.csv": "source-data",
            "data/elections/alaska/2026-11-03/raw/alaska/2026-11-03-2015.csv": "source-data",
            "data/elections/washington/2026-08-04/areas/washington/county/x.csv": "derived-data",
            "data/elections/maine/2026-11-03/SOURCES.json": "provenance-record",
            "data/crosswalks/alaska/precinct-borough.csv": "derived-data",
            "docs/engineering/provenance-baseline.json": "engineering-record",
            "LICENSES/Apache-2.0.txt": "license-text",
            "REUSE.toml": "reuse-config",
            "data/elections/washington/2026-11-03/election.toml": "authored",
            "election_reporting/cli.py": "authored",
            "data/README.md": "authored",
        }
        for name, expected in cases.items():
            self.assertEqual(provenance.kind(name), expected, name)


class FileIdTests(ScopeTest):
    def test_authored_files_need_ids_and_automation_files_do_not(self) -> None:
        self.write("a.py", header(ID_A) + "x = 1\n")
        self.write("reports/r.json", "{}\n")
        self.write("data/elections/washington/2026-11-03/input/washington/s.csv", "a,b\n")
        status, details = provenance.file_ids(self.scope)
        self.assertEqual(status, provenance.PASS, details["violations"])
        self.assertEqual(details["ids"], {"a.py": ID_A})
        self.assertEqual(details["exempt"], 2)

    def test_every_rule_is_a_violation(self) -> None:
        self.write("missing.py", "x = 1\n")
        self.write("bad.sh", header("urn:uuid:not-a-uuid"))
        self.write("v1.toml", header("urn:uuid:0f1e2d3c-4b5a-1968-8776-655443322110"))
        self.write("twice.md", header(ID_A) + header(ID_B))
        self.write("dup1.py", header(ID_B))
        self.write("dup2.py", header(ID_B))
        status, details = provenance.file_ids(self.scope)
        self.assertEqual(status, provenance.VIOLATION)
        text = "\n".join(details["violations"])
        self.assertIn("missing.py: no Catalyst-File-ID", text)
        self.assertIn("bad.sh: 'urn:uuid:not-a-uuid' is not a version 4 urn:uuid", text)
        self.assertIn("v1.toml:", text)
        self.assertIn("twice.md: 2 Catalyst-File-IDs", text)
        self.assertIn("dup2.py: Catalyst-File-ID", text)

    def test_ids_below_the_header_are_ignored(self) -> None:
        self.write("a.py", header(ID_A) + "\n" * 30 + f"# Catalyst-File-ID: {ID_B} quoted later\n")
        status, details = provenance.file_ids(self.scope)
        self.assertEqual(status, provenance.PASS, details["violations"])


def reuse_json(unused=(), missing=()) -> str:
    return json.dumps({
        "reuse_tool_version": "6.2.0", "reuse_spec_version": "3.3",
        "summary": {"compliant": not unused and not missing, "files_total": 3,
                    "files_with_copyright_info": 3, "files_with_licensing_info": 3},
        "non_compliant": {"unused_licenses": list(unused), "missing_licenses": list(missing), "bad_licenses": []},
    })


class ReuseTests(ScopeTest):
    def run_reuse(self, stdout: str):
        completed = mock.Mock(stdout=stdout, stderr="")
        with mock.patch.object(provenance.shutil, "which", return_value="/usr/bin/reuse"), \
                mock.patch.object(provenance.subprocess, "run", return_value=completed):
            return provenance.reuse_lint(self.scope)

    def test_pending_publishers_unused_licenses_pass(self) -> None:
        status, details = self.run_reuse(reuse_json(unused=["LicenseRef-PierceCounty-PublicData"]))
        self.assertEqual(status, provenance.PASS)
        self.assertEqual(details["pending_licenses"], ["LicenseRef-PierceCounty-PublicData"])
        self.assertEqual(details["non_compliant"], {})

    def test_other_unused_or_missing_licenses_are_violations(self) -> None:
        status, details = self.run_reuse(reuse_json(unused=["LicenseRef-Stray"]))
        self.assertEqual(status, provenance.VIOLATION)
        self.assertEqual(details["non_compliant"], {"unused_licenses": ["LicenseRef-Stray"]})
        status, details = self.run_reuse(reuse_json(missing=["LicenseRef-New"]))
        self.assertEqual(status, provenance.VIOLATION)

    def test_unreadable_output_is_indeterminate(self) -> None:
        status, _ = self.run_reuse("not json")
        self.assertEqual(status, provenance.INDETERMINATE)


class ManifestAndRunTests(ScopeTest):
    def test_manifest_lists_every_file_with_kind_id_and_hash(self) -> None:
        self.write("a.py", header(ID_A) + "x = 1\n")
        self.write("reports/r.json", "{}\n")
        _, ids = provenance.file_ids(self.scope)
        with mock.patch.dict("os.environ", {"GITHUB_SHA": "abc123"}):
            status, document = provenance.manifest(self.scope, ids["ids"])
        self.assertEqual(status, provenance.PASS)
        by_path = {entry["path"]: entry for entry in document["files"]}
        self.assertEqual(by_path["election-analysis/a.py"]["file_id"], ID_A)
        self.assertEqual(by_path["election-analysis/reports/r.json"]["kind"], "generated-report")
        self.assertIsNone(by_path["election-analysis/reports/r.json"]["file_id"])
        self.assertEqual(by_path["election-analysis/a.py"]["sha256"],
                         hashlib.sha256((self.scope / "a.py").read_bytes()).hexdigest())

    def test_missing_reuse_is_indeterminate_and_violations_win(self) -> None:
        self.write("a.py", "x = 1\n")  # no File-ID: a violation
        output = self.scope / "out"
        with mock.patch.object(provenance.shutil, "which", return_value=None), \
                mock.patch.dict("os.environ", {"GITHUB_SHA": "abc123", "GITHUB_STEP_SUMMARY": ""}):
            overall = provenance.main(["all", "--scope", str(self.scope), "--output", str(output)])
        results = json.loads((output / "results.json").read_text())
        self.assertEqual(results["reuse"]["status"], provenance.INDETERMINATE)
        self.assertEqual(results["file-ids"]["status"], provenance.VIOLATION)
        self.assertEqual(overall, provenance.VIOLATION)
        self.assertTrue((output / "scope-manifest.json").exists())
        self.assertIn("Report-only", (output / "summary.md").read_text())

    def test_only_indeterminate_gives_two(self) -> None:
        self.write("a.py", header(ID_A) + "x = 1\n")
        output = self.scope / "out"
        with mock.patch.object(provenance.shutil, "which", return_value=None), \
                mock.patch.dict("os.environ", {"GITHUB_SHA": "abc123", "GITHUB_STEP_SUMMARY": ""}):
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
