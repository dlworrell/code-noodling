"""Exercise bypass attempts against the public runner regression check."""
import tempfile
import unittest
from pathlib import Path

from scripts.ci.check_public_runners import validate


class PublicRunnerPolicyTests(unittest.TestCase):
    def check(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new-workflow.yaml"
            path.write_text(body, encoding="utf-8")
            return validate(path)

    def test_runner_selectors(self):
        for selector in ("self-hosted", "[self-hosted, k80]", "[linux, x64, kepler]",
                         "{group: lab}", "${{ inputs.runner }}", "k80", "kepler",
                         "${{ fromJSON(inputs.runner) }}"):
            with self.subTest(selector=selector):
                self.assertTrue(self.check("jobs:\n  added:\n    runs-on: " + selector + "\n"))
        self.assertFalse(self.check("jobs:\n  test:\n    runs-on: ubuntu-latest\n"))

    def test_matrix_values(self):
        for values, blocked in (("[ubuntu-latest, macos-15]", False),
                                ("[ubuntu-latest, self-hosted]", True),
                                ("${{ inputs.labels }}", True)):
            with self.subTest(values=values):
                body = ("jobs:\n  test:\n    runs-on: ${{ matrix.os }}\n"
                        "    strategy:\n      matrix:\n        os: " + values + "\n")
                self.assertEqual(bool(self.check(body)), blocked)

    def test_include_and_groups(self):
        prefix = ("jobs:\n  test:\n    runs-on: ${{ matrix.os }}\n"
                  "    strategy:\n      matrix:\n        include:\n          - os: ")
        self.assertFalse(self.check(prefix + "ubuntu-latest\n"))
        self.assertTrue(self.check(prefix + "{group: lab}\n"))
        self.assertTrue(self.check(prefix + "kepler\n"))

    def test_pinned_aems_delegation(self):
        body = ("jobs:\n  crypto:\n    uses: "
                "dlworrell/AEMS/.github/workflows/aes-sec-001-distributed.yml"
                "@4df3a37418536d97865c9774701d910eb6f11bdb\n")
        self.assertFalse(self.check(body))
        # A reusable call must not override runner selection, even with a hosted label.
        self.assertTrue(self.check(body + "    runs-on: ubuntu-latest\n"))

    def test_unreviewed_aems_revisions(self):
        for revision in ("1111111111111111111111111111111111111111",
                         "scope-adoption", "v1", "${{ inputs.ref }}"):
            with self.subTest(revision=revision):
                body = ("jobs:\n  crypto:\n    uses: "
                        "dlworrell/AEMS/.github/workflows/aes-sec-001-distributed.yml@"
                        + revision + "\n")
                self.assertTrue(self.check(body))

    def test_unreviewed_reusable_and_invalid_yaml(self):
        self.assertTrue(self.check("jobs:\n  test:\n    uses: other/lab/.github/workflows/run.yml@main\n"))
        self.assertTrue(self.check("jobs: ["))


if __name__ == "__main__":
    unittest.main()
