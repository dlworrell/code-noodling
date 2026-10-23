# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:7cff81d3-f059-4195-94ab-9796422628db

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from election_reporting.ci.publish_reports import (
    CREATE_COMMIT_MUTATION,
    GitHub,
    PublishError,
    ReportChanges,
    build_commit_input,
    build_graphql_request,
    detect_changes,
    parse_name_status,
    publish,
    restrict_to,
)

REPO = "owner/repo"
BRANCH = "automation/election-reports"


class FakeGh:
    """Records ``gh api`` invocations and replays canned responses."""

    def __init__(self, responses: dict[tuple[str, str], object]):
        self.responses = responses
        self.calls: list[tuple[str, str, object]] = []

    def __call__(self, args, input=None, capture_output=True, check=False):
        assert args[:3] == ["gh", "api", "--method"], args
        method, endpoint = args[3], args[4]
        payload = json.loads(input) if input is not None else None
        self.calls.append((method, endpoint, payload))
        key = (method, endpoint)
        if key not in self.responses:
            return subprocess.CompletedProcess(args, 1, b"", f"HTTP 404 {key}".encode())
        body = self.responses[key]
        if isinstance(body, Exception):
            return subprocess.CompletedProcess(args, 1, b"", str(body).encode())
        out = b"" if body is None else json.dumps(body).encode()
        return subprocess.CompletedProcess(args, 0, out, b"")

    def endpoints(self) -> list[tuple[str, str]]:
        return [(method, endpoint) for method, endpoint, _ in self.calls]


def run_git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class ParseNameStatusTests(unittest.TestCase):
    def test_empty_output_is_no_change(self) -> None:
        self.assertTrue(parse_name_status(b"").empty)

    def test_classifies_statuses(self) -> None:
        raw = b"M\0r/b.json\0A\0r/a.md\0D\0r/old.txt\0T\0r/c\0"
        changes = parse_name_status(raw)
        self.assertEqual(changes.additions, ("r/a.md", "r/b.json", "r/c"))
        self.assertEqual(changes.deletions, ("r/old.txt",))

    def test_paths_with_spaces_and_unicode(self) -> None:
        raw = "A\0r/sp ace/ü.md\0".encode()
        self.assertEqual(parse_name_status(raw).additions, ("r/sp ace/ü.md",))

    def test_rejects_rename_and_malformed(self) -> None:
        with self.assertRaises(PublishError):
            parse_name_status(b"R100\0a\0")
        with self.assertRaises(PublishError):
            parse_name_status(b"M\0")

    def test_restrict_to_rejects_outside_paths(self) -> None:
        restrict_to(ReportChanges(("election-analysis/reports/a",), ()), "election-analysis/reports")
        with self.assertRaises(PublishError):
            restrict_to(ReportChanges(("README.md",), ()), "election-analysis/reports")
        with self.assertRaises(PublishError):
            restrict_to(ReportChanges((), ("election-analysis/reports/../x",)), "election-analysis/reports")


class PayloadTests(unittest.TestCase):
    def test_binary_safe_base64_and_deletions(self) -> None:
        blob = bytes(range(256)) * 3
        files = {"r/bin.dat": blob, "r/t.md": "naïve\n".encode()}
        commit_input = build_commit_input(
            repository=REPO,
            branch=BRANCH,
            expected_head_oid="a" * 40,
            headline="reports: update",
            body="",
            changes=ReportChanges(("r/bin.dat", "r/t.md"), ("r/gone.json",)),
            read_bytes=files.__getitem__,
        )
        self.assertEqual(
            commit_input["branch"],
            {"repositoryNameWithOwner": REPO, "branchName": BRANCH},
        )
        self.assertEqual(commit_input["expectedHeadOid"], "a" * 40)
        self.assertEqual(commit_input["message"], {"headline": "reports: update"})
        additions = commit_input["fileChanges"]["additions"]
        self.assertEqual([a["path"] for a in additions], ["r/bin.dat", "r/t.md"])
        self.assertEqual(base64.b64decode(additions[0]["contents"]), blob)
        self.assertEqual(commit_input["fileChanges"]["deletions"], [{"path": "r/gone.json"}])
        request = build_graphql_request(commit_input)
        self.assertEqual(request["query"], CREATE_COMMIT_MUTATION)
        self.assertIs(request["variables"]["input"], commit_input)
        json.dumps(request)  # must be JSON-serializable

    def test_many_files_and_no_skip_ci(self) -> None:
        paths = tuple(f"r/f{i:04}.json" for i in range(500))
        commit_input = build_commit_input(
            repository=REPO,
            branch=BRANCH,
            expected_head_oid="b" * 40,
            headline="reports: update election result analysis",
            body="Regenerated.",
            changes=ReportChanges(paths, ()),
            read_bytes=lambda p: p.encode(),
        )
        self.assertEqual(len(commit_input["fileChanges"]["additions"]), 500)
        self.assertNotIn("deletions", commit_input["fileChanges"])
        self.assertNotIn("[skip ci]", json.dumps(commit_input["message"]))


class PublishFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        run_git(self.root, "init", "-q")
        reports = self.root / "election-analysis" / "reports"
        (reports / "edt").mkdir(parents=True)
        (reports / "analysis.json").write_text('{"v": 1}\n')
        (reports / "edt" / "old.md").write_text("old\n")
        (self.root / "README.md").write_text("readme\n")
        run_git(self.root, "add", "-A")
        run_git(self.root, "commit", "-q", "-m", "init")
        self.base_sha = run_git(self.root, "rev-parse", "HEAD")
        self.logs: list[str] = []

    def regenerate(self) -> None:
        reports = self.root / "election-analysis" / "reports"
        (reports / "analysis.json").write_text('{"v": 2}\n')
        (reports / "edt" / "old.md").unlink()
        (reports / "edt" / "new.bin").write_bytes(b"\x00\xff\x10")
        (self.root / "README.md").write_text("unrelated local edit\n")

    def run_publish(self, fake: FakeGh):
        return publish(
            github=GitHub(REPO, runner=fake),
            repo_root=self.root,
            report_path="election-analysis/reports",
            branch=BRANCH,
            base="main",
            base_sha=self.base_sha,
            title="reports: update election result analysis",
            run_url="https://example.invalid/run/1",
            log=self.logs.append,
        )

    def test_detect_changes_does_not_touch_real_index(self) -> None:
        self.regenerate()
        changes, tree = detect_changes(self.root, "election-analysis/reports")
        self.assertEqual(
            changes.additions,
            ("election-analysis/reports/analysis.json", "election-analysis/reports/edt/new.bin"),
        )
        self.assertEqual(changes.deletions, ("election-analysis/reports/edt/old.md",))
        self.assertRegex(tree, r"^[0-9a-f]{40}$")
        self.assertEqual(run_git(self.root, "diff", "--cached", "--name-only"), "")

    def test_no_changes_makes_no_api_calls(self) -> None:
        fake = FakeGh({})
        self.assertIsNone(self.run_publish(fake))
        self.assertEqual(fake.calls, [])

    def test_creates_branch_commit_and_pull_request(self) -> None:
        self.regenerate()
        fake = FakeGh(
            {
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [
                    {"ref": f"refs/heads/{BRANCH}-other", "object": {"sha": "f" * 40}}
                ],
                ("POST", f"repos/{REPO}/git/refs"): {},
                ("POST", "graphql"): {
                    "data": {"createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}
                },
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}&base=main"): [],
                ("POST", f"repos/{REPO}/pulls"): {"html_url": "https://pr/1", "number": 1},
            }
        )
        self.assertEqual(self.run_publish(fake), "https://pr/1")
        self.assertEqual(
            fake.endpoints(),
            [
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"),
                ("POST", f"repos/{REPO}/git/refs"),
                ("POST", "graphql"),
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}&base=main"),
                ("POST", f"repos/{REPO}/pulls"),
            ],
        )
        self.assertEqual(
            fake.calls[1][2], {"ref": f"refs/heads/{BRANCH}", "sha": self.base_sha}
        )
        commit_input = fake.calls[2][2]["variables"]["input"]
        self.assertEqual(commit_input["expectedHeadOid"], self.base_sha)
        files = commit_input["fileChanges"]
        self.assertEqual(
            [a["path"] for a in files["additions"]],
            ["election-analysis/reports/analysis.json", "election-analysis/reports/edt/new.bin"],
        )
        self.assertEqual(base64.b64decode(files["additions"][1]["contents"]), b"\x00\xff\x10")
        self.assertEqual(files["deletions"], [{"path": "election-analysis/reports/edt/old.md"}])
        pr = fake.calls[4][2]
        self.assertEqual((pr["head"], pr["base"]), (BRANCH, "main"))
        # Nothing ever targets the base branch ref.
        for method, endpoint, _ in fake.calls:
            self.assertNotIn("heads/main", endpoint)

    def test_existing_branch_is_force_reset_and_pr_updated(self) -> None:
        self.regenerate()
        old_head = "e" * 40
        fake = FakeGh(
            {
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [
                    {"ref": f"refs/heads/{BRANCH}", "object": {"sha": old_head}}
                ],
                ("GET", f"repos/{REPO}/git/commits/{old_head}"): {
                    "tree": {"sha": "0" * 40},
                    "parents": [{"sha": "9" * 40}],
                },
                ("PATCH", f"repos/{REPO}/git/refs/heads/{BRANCH}"): {},
                ("POST", "graphql"): {
                    "data": {"createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}
                },
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}&base=main"): [
                    {"number": 7, "html_url": "https://pr/7"}
                ],
                ("PATCH", f"repos/{REPO}/pulls/7"): {},
            }
        )
        self.assertEqual(self.run_publish(fake), "https://pr/7")
        self.assertEqual(fake.calls[2][2], {"sha": self.base_sha, "force": True})
        self.assertNotIn(("POST", f"repos/{REPO}/pulls"), fake.endpoints())

    def test_identical_branch_content_skips_commit(self) -> None:
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        head = "d" * 40
        fake = FakeGh(
            {
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [
                    {"ref": f"refs/heads/{BRANCH}", "object": {"sha": head}}
                ],
                ("GET", f"repos/{REPO}/git/commits/{head}"): {
                    "tree": {"sha": tree},
                    "parents": [{"sha": self.base_sha}],
                },
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}&base=main"): [
                    {"number": 7, "html_url": "https://pr/7"}
                ],
                ("PATCH", f"repos/{REPO}/pulls/7"): {},
            }
        )
        self.assertEqual(self.run_publish(fake), "https://pr/7")
        self.assertNotIn(("POST", "graphql"), fake.endpoints())

    def test_pull_request_failure_names_required_setting(self) -> None:
        self.regenerate()
        fake = FakeGh(
            {
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
                ("POST", f"repos/{REPO}/git/refs"): {},
                ("POST", "graphql"): {
                    "data": {"createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}
                },
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}&base=main"): [],
                ("POST", f"repos/{REPO}/pulls"): RuntimeError(
                    "GitHub Actions is not permitted to create or approve pull requests (HTTP 403)"
                ),
            }
        )
        with self.assertRaisesRegex(PublishError, "Allow GitHub Actions to create"):
            self.run_publish(fake)

    def test_graphql_errors_fail(self) -> None:
        self.regenerate()
        fake = FakeGh(
            {
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
                ("POST", f"repos/{REPO}/git/refs"): {},
                ("POST", "graphql"): {"errors": [{"message": "expectedHeadOid mismatch"}]},
            }
        )
        with self.assertRaises(PublishError):
            self.run_publish(fake)


if __name__ == "__main__":
    unittest.main()
