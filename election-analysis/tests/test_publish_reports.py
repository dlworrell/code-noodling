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
from unittest import mock

from election_reporting.ci.publish_reports import (
    CREATE_COMMIT_MUTATION,
    GitHub,
    PublishError,
    ReportChanges,
    batch_changes,
    build_commit_input,
    build_graphql_request,
    encoded_size,
    detect_changes,
    parse_name_status,
    publish,
    restrict_to,
)

REPO = "owner/repo"
BRANCH = "automation/election-reports"
STAGING = BRANCH + "-staging"


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
        if callable(body):
            body = body(payload)
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


class BatchTests(unittest.TestCase):
    SIZES = {"r/a": 300, "r/b": 300, "r/c": 900, "r/d": 30}

    def test_b1_small_change_is_one_batch(self) -> None:
        changes = ReportChanges(("r/a", "r/b"), ("r/x",))
        self.assertEqual(batch_changes(changes, self.SIZES.__getitem__, limit=10_000), [changes])

    def test_b2_packs_within_limit_in_path_order_with_deletions_first(self) -> None:
        changes = ReportChanges(("r/a", "r/b", "r/c", "r/d"), ("r/x",))
        batches = batch_changes(changes, self.SIZES.__getitem__, limit=encoded_size(600))
        self.assertEqual([b.additions for b in batches], [("r/a", "r/b"), ("r/c",), ("r/d",)])
        self.assertEqual([b.deletions for b in batches], [("r/x",), (), ()])
        self.assertEqual(sum((b.additions for b in batches), ()), changes.additions)

    def test_b3_deletions_only_is_one_batch(self) -> None:
        changes = ReportChanges((), ("r/x", "r/y"))
        self.assertEqual(batch_changes(changes, self.SIZES.__getitem__), [changes])

    def test_b4_a_file_too_large_for_any_request_is_refused(self) -> None:
        with self.assertRaisesRegex(PublishError, "too large"):
            batch_changes(ReportChanges(("r/huge",), ()), lambda _: 40 * 1024 * 1024)


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
        self.base = "dev-daily"
        self.logs: list[str] = []

    def regenerate(self) -> None:
        reports = self.root / "election-analysis" / "reports"
        (reports / "analysis.json").write_text('{"v": 2}\n')
        (reports / "edt" / "old.md").unlink()
        (reports / "edt" / "new.bin").write_bytes(b"\x00\xff\x10")
        (self.root / "README.md").write_text("unrelated local edit\n")

    def run_publish(self, fake: FakeGh):
        # Successful API fixtures include the separate, observed signature evidence.
        _, tree = detect_changes(self.root, "election-analysis/reports")
        fake.responses.setdefault(("GET", f"repos/{REPO}"),
                                  {"node_id": "REPO", "default_branch": "main"})
        fake.responses.setdefault(("GET", f"repos/{REPO}/compare/{self.base_sha}...dev-daily"),
                                  {"status": "identical"})
        fake.responses.setdefault(("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"), [])
        fake.responses.setdefault(("GET", f"repos/{REPO}/git/matching-refs/heads/{STAGING}"), [])
        fake.responses.setdefault(("POST", f"repos/{REPO}/git/refs"), {})
        fake.responses.setdefault(("DELETE", f"repos/{REPO}/git/refs/heads/{STAGING}"), None)
        fake.responses.setdefault(("GET", f"repos/{REPO}/git/commits/{'c' * 40}"), {
            "sha": "c" * 40, "tree": {"sha": tree}, "parents": [{"sha": self.base_sha}],
            "verification": {"verified": True, "reason": "valid"},
        })
        return publish(
            github=GitHub(REPO, runner=fake),
            repo_root=self.root,
            report_path="election-analysis/reports",
            branch=BRANCH,
            base=self.base,
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
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"): [],
                ("POST", f"repos/{REPO}/pulls"): {"html_url": "https://pr/1", "number": 1},
            }
        )
        self.assertEqual(self.run_publish(fake), "https://pr/1")
        self.assertEqual(
            fake.endpoints(),
            [
                ("GET", f"repos/{REPO}/compare/{self.base_sha}...dev-daily"),
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"),
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"),
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{STAGING}"),
                ("GET", f"repos/{REPO}"),
                ("POST", f"repos/{REPO}/git/refs"),
                ("POST", "graphql"),
                ("GET", f"repos/{REPO}/git/commits/{'c' * 40}"),
                ("GET", f"repos/{REPO}"),
                ("POST", f"repos/{REPO}/git/refs"),
                ("GET", f"repos/{REPO}"),
                ("DELETE", f"repos/{REPO}/git/refs/heads/{STAGING}"),
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"),
                ("POST", f"repos/{REPO}/pulls"),
            ],
        )
        # The commits are built on the staging branch; the new automation
        # branch is created directly at the last commit, never at the base.
        self.assertEqual(
            fake.calls[5][2], {"ref": f"refs/heads/{STAGING}", "sha": self.base_sha}
        )
        self.assertEqual(
            fake.calls[9][2], {"ref": f"refs/heads/{BRANCH}", "sha": "c" * 40}
        )
        commit_input = fake.calls[6][2]["variables"]["input"]
        self.assertEqual(commit_input["branch"]["branchName"], STAGING)
        self.assertEqual(commit_input["expectedHeadOid"], self.base_sha)
        files = commit_input["fileChanges"]
        self.assertEqual(
            [a["path"] for a in files["additions"]],
            ["election-analysis/reports/analysis.json", "election-analysis/reports/edt/new.bin"],
        )
        self.assertEqual(base64.b64decode(files["additions"][1]["contents"]), b"\x00\xff\x10")
        self.assertEqual(files["deletions"], [{"path": "election-analysis/reports/edt/old.md"}])
        pr = fake.calls[13][2]
        self.assertEqual((pr["head"], pr["base"]), (BRANCH, "dev-daily"))
        # Nothing ever targets a base branch ref.
        for method, endpoint, _ in fake.calls:
            self.assertNotIn("heads/main", endpoint)
            self.assertNotIn("heads/dev-daily", endpoint)

    def test_existing_branch_moves_to_new_commit_and_pr_stays_open(self) -> None:
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
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"): [
                    {"number": 7, "html_url": "https://pr/7", "base": {"ref": "dev-daily"}}
                ],
                ("PATCH", f"repos/{REPO}/pulls/7"): {},
            }
        )
        self.assertEqual(self.run_publish(fake), "https://pr/7")
        lease = next(payload["variables"]["input"] for _, _, payload in fake.calls
                     if payload and "UpdateRefsInput" in payload.get("query", ""))
        # One leased move from the old head straight to the new commit: the
        # branch never equals the base, so GitHub keeps the pull request open.
        self.assertEqual(lease["refUpdates"], [{"name": f"refs/heads/{BRANCH}",
                         "beforeOid": old_head, "afterOid": "c" * 40, "force": True}])
        for _, _, payload in fake.calls:
            if payload and "UpdateRefsInput" in payload.get("query", ""):
                for update in payload["variables"]["input"]["refUpdates"]:
                    if update["name"] == f"refs/heads/{BRANCH}":
                        self.assertNotEqual(update["afterOid"], self.base_sha)
        self.assertNotIn(("POST", f"repos/{REPO}/pulls"), fake.endpoints())
        self.assertIn(("DELETE", f"repos/{REPO}/git/refs/heads/{STAGING}"), fake.endpoints())

    def test_s1_leftover_staging_branch_is_reset_with_a_lease(self) -> None:
        self.regenerate()
        leftover = "a" * 40
        old = "e" * 40
        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"):
                [{"ref": f"refs/heads/{BRANCH}", "object": {"sha": old}}],
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{STAGING}"):
                [{"ref": f"refs/heads/{STAGING}", "object": {"sha": leftover}}],
            ("GET", f"repos/{REPO}/git/commits/{old}"): {"tree": {"sha": "0" * 40}, "parents": []},
            ("POST", "graphql"): {"data": {"updateRefs": {"clientMutationId": None},
                "createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}},
            ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"):
                [{"number": 7, "html_url": "https://pr/7", "base": {"ref": "dev-daily"}}],
            ("PATCH", f"repos/{REPO}/pulls/7"): {},
        })
        self.assertEqual(self.run_publish(fake), "https://pr/7")
        moves = [p["variables"]["input"]["refUpdates"][0] for _, e, p in fake.calls
                 if e == "graphql" and "UpdateRefsInput" in p["query"]]
        self.assertEqual(moves, [
            {"name": f"refs/heads/{STAGING}", "beforeOid": leftover, "afterOid": self.base_sha, "force": True},
            {"name": f"refs/heads/{BRANCH}", "beforeOid": old, "afterOid": "c" * 40, "force": True},
        ])

    def test_s2_staging_cleanup_failure_is_only_a_warning(self) -> None:
        self.regenerate()
        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
            ("POST", "graphql"): {"data": {"createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}},
            ("DELETE", f"repos/{REPO}/git/refs/heads/{STAGING}"): RuntimeError("HTTP 422 Reference does not exist"),
            ("POST", f"repos/{REPO}/pulls"): {"html_url": "https://pr/1", "number": 1},
        })
        self.assertEqual(self.run_publish(fake), "https://pr/1")
        self.assertTrue(any("could not delete" in line for line in self.logs))

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
                    "sha": head, "tree": {"sha": tree},
                    "parents": [{"sha": self.base_sha}],
                    "verification": {"verified": True, "reason": "valid"},
                },
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"): [
                    {"number": 7, "html_url": "https://pr/7", "base": {"ref": "dev-daily"}}
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
                ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"): [],
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


    def test_unverified_new_commit_never_opens_a_pr(self):
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        for verification in ({}, {"verified": False, "reason": "unsigned"},
                             {"verified": True, "reason": "unknown_key"},
                             {"verified": 1, "reason": "valid"}):
            fake = FakeGh({
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
                ("POST", f"repos/{REPO}/git/refs"): {},
                ("POST", "graphql"): {"data": {"createCommitOnBranch": {
                    "commit": {"oid": "c"*40, "url": "u"}}}},
                ("GET", f"repos/{REPO}/git/commits/{'c'*40}"): {
                    "sha": "c"*40, "tree": {"sha": tree},
                    "parents": [{"sha": self.base_sha}], "verification": verification},
            })
            with self.subTest(verification=verification), self.assertRaisesRegex(PublishError, "did not verify"):
                self.run_publish(fake)
            # The pre-write ambiguity check reads the PR list; nothing may change a PR.
            self.assertFalse(any("/pulls" in endpoint and method != "GET" for method, endpoint, _ in fake.calls))

    def test_unsigned_identical_content_is_replaced(self):
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        old = "e"*40
        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"):
                [{"ref": f"refs/heads/{BRANCH}", "object": {"sha": old}}],
            ("GET", f"repos/{REPO}/git/commits/{old}"): {
                "sha": old, "tree": {"sha": tree}, "parents": [{"sha": self.base_sha}],
                "verification": {"verified": False, "reason": "unsigned"}},
            ("POST", "graphql"): {"data": {"updateRefs": {"clientMutationId": None},
                "createCommitOnBranch": {"commit": {"oid": "c"*40, "url": "u"}}}},
            ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"):
                [{"number": 7, "html_url": "https://pr/7", "base": {"ref": "dev-daily"}}],
            ("PATCH", f"repos/{REPO}/pulls/7"): {},
        })
        self.assertEqual(self.run_publish(fake), "https://pr/7")
        mutations = [p["query"] for _, e, p in fake.calls if e == "graphql"]
        self.assertTrue(any("UpdateRefsInput" in q for q in mutations))
        self.assertTrue(any("CreateCommitOnBranchInput" in q for q in mutations))

    def test_concurrent_branch_update_aborts_before_any_pull_request_change(self):
        self.regenerate()
        old = "e"*40

        def graphql(payload):
            if "UpdateRefsInput" in payload["query"]:
                return {"errors": [{"message": "beforeOid mismatch"}]}
            return {"data": {"createCommitOnBranch": {"commit": {"oid": "c"*40, "url": "u"}}}}

        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"):
                [{"ref": f"refs/heads/{BRANCH}", "object": {"sha": old}}],
            ("GET", f"repos/{REPO}/git/commits/{old}"): {"tree": {"sha": "0"*40}, "parents": []},
            ("POST", "graphql"): graphql,
        })
        with self.assertRaisesRegex(PublishError, "leased"):
            self.run_publish(fake)
        moves = [p for _, e, p in fake.calls if e == "graphql" and "UpdateRefsInput" in p["query"]]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["variables"]["input"]["refUpdates"][0]["beforeOid"], old)
        self.assertFalse(any("/pulls" in e and m != "GET" for m, e, _ in fake.calls))

    def test_verification_binds_tree_parent_and_identity(self):
        oid = "c"*40
        good = {"sha": oid, "tree": {"sha": "d"*40}, "parents": [{"sha": self.base_sha}],
                "verification": {"verified": True, "reason": "valid"}}
        for field, value in (("sha", "e"*40), ("tree", {"sha": "e"*40}), ("parents", [])):
            observed = {**good, field: value}
            fake = FakeGh({("GET", f"repos/{REPO}/git/commits/{oid}"): observed})
            with self.subTest(field=field), self.assertRaises(PublishError):
                GitHub(REPO, runner=fake).require_verified_commit(oid, self.base_sha, "d"*40)

    def test_default_branch_and_pull_request_target_fail_closed(self):
        fake = FakeGh({("GET", f"repos/{REPO}"): {"node_id": "REPO", "default_branch": "main"}})
        with self.assertRaisesRegex(PublishError, "default branch"):
            GitHub(REPO, runner=fake).point_branch("main", self.base_sha, True, self.base_sha)
        with mock.patch.dict("os.environ", {"GITHUB_EVENT_NAME": "pull_request_target"}):
            with self.assertRaisesRegex(PublishError, "pull_request_target"):
                self.run_publish(FakeGh({}))


    # ---------------------------------------------------------------- route (ADR-0013)
    # Acceptance cases, written before the change:
    # R1 publication into dev-daily works (test_creates_branch_commit_and_pull_request);
    # R2 main, fitment or any other base is refused before any GitHub call;
    # R3 a generating commit that is not on dev-daily is refused before any write;
    # R4 an open PR from the automation branch to another base (#91) is retargeted, not duplicated;
    # R5 several open PRs from the automation branch are refused before any write at all;
    # R6 a rerun with identical content leaves the commit alone and does not retarget again.

    def writes(self, fake: FakeGh) -> list[tuple[str, str]]:
        return [(m, e) for m, e in fake.endpoints() if m in ("POST", "PATCH", "PUT", "DELETE")]

    def test_r2_other_bases_are_refused_before_any_call(self) -> None:
        self.regenerate()
        for base in ("main", "fitment", "release", "automation/election-reports-2"):
            fake = FakeGh({})
            self.base = base
            with self.subTest(base=base), self.assertRaisesRegex(PublishError, "dev-daily only"):
                self.run_publish(fake)
            self.assertEqual(fake.calls, [])

    def test_r3_generating_commit_off_base_is_refused_before_writes(self) -> None:
        self.regenerate()
        for status in ("behind", "diverged", None):
            fake = FakeGh({
                ("GET", f"repos/{REPO}/compare/{self.base_sha}...dev-daily"): {"status": status},
                ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
            })
            with self.subTest(status=status), self.assertRaisesRegex(PublishError, "not on dev-daily"):
                self.run_publish(fake)
            self.assertEqual(self.writes(fake), [])
            self.assertEqual(fake.endpoints(), [("GET", f"repos/{REPO}/compare/{self.base_sha}...dev-daily")])

    def test_r3_unreadable_comparison_is_refused(self) -> None:
        """The compare call itself fails (HTTP 404): publish stops there, fail closed.

        run_publish only defaults the compare response when none is given, so
        configuring the failure here makes this test exercise exactly the
        unreadable-comparison path (Em's rereview of 8c9a7c0).
        """
        self.regenerate()
        compare = ("GET", f"repos/{REPO}/compare/{self.base_sha}...dev-daily")
        fake = FakeGh({compare: RuntimeError("HTTP 404 Not Found")})
        with self.assertRaisesRegex(PublishError, "compare"):
            self.run_publish(fake)
        # The compare call is the only call made: nothing read after it, nothing written.
        self.assertEqual(fake.endpoints(), [compare])
        self.assertEqual(self.writes(fake), [])

    def test_r4_wrong_base_pull_request_is_migrated(self) -> None:
        self.regenerate()
        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"):
                [{"ref": f"refs/heads/{BRANCH}", "object": {"sha": "e" * 40}}],
            ("GET", f"repos/{REPO}/git/commits/{'e' * 40}"): {"tree": {"sha": "0" * 40}, "parents": []},
            ("POST", "graphql"): {"data": {"updateRefs": {"clientMutationId": None},
                                  "createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}},
            ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"):
                [{"number": 91, "html_url": "https://pr/91", "base": {"ref": "main"}}],
            ("PATCH", f"repos/{REPO}/pulls/91"): {},
        })
        self.assertEqual(self.run_publish(fake), "https://pr/91")
        patch = next(p for m, e, p in fake.calls if (m, e) == ("PATCH", f"repos/{REPO}/pulls/91"))
        self.assertEqual(patch["base"], "dev-daily")
        self.assertNotIn(("POST", f"repos/{REPO}/pulls"), fake.endpoints())
        self.assertIn("Migrating pull request #91 from base main to dev-daily.", self.logs)

    def test_r5_several_open_pull_requests_are_refused(self) -> None:
        self.regenerate()
        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
            ("POST", f"repos/{REPO}/git/refs"): {},
            ("POST", "graphql"): {"data": {"createCommitOnBranch": {"commit": {"oid": "c" * 40, "url": "u"}}}},
            ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"):
                [{"number": 78, "base": {"ref": "main"}}, {"number": 91, "base": {"ref": "main"}}],
        })
        with self.assertRaisesRegex(PublishError, "several open pull requests"):
            self.run_publish(fake)
        # Refused before ANY write: no branch reset, no commit, no PR change.
        self.assertEqual(self.writes(fake), [])

    def chain_fake(self, tree: str, verified_first: bool = True) -> FakeGh:
        first, second = "1" * 40, "2" * 40
        commits = {first: (self.base_sha, None), second: (first, tree)}

        def create(payload):
            parent = payload["variables"]["input"]["expectedHeadOid"]
            oid = first if parent == self.base_sha else second
            return {"data": {"createCommitOnBranch": {"commit": {"oid": oid, "url": "u"}}}}

        responses = {
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"): [],
            ("POST", f"repos/{REPO}/git/refs"): {},
            ("POST", "graphql"): create,
            ("POST", f"repos/{REPO}/pulls"): {"html_url": "https://pr/2", "number": 2},
        }
        for oid, (parent, commit_tree) in commits.items():
            ok = verified_first or oid != first
            responses[("GET", f"repos/{REPO}/git/commits/{oid}")] = {
                "sha": oid, "tree": {"sha": commit_tree or "9" * 40}, "parents": [{"sha": parent}],
                "verification": {"verified": ok, "reason": "valid" if ok else "unsigned"}}
        return FakeGh(responses)

    def test_b5_large_change_is_published_as_a_verified_chain(self) -> None:
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        fake = self.chain_fake(tree)
        with mock.patch("election_reporting.ci.publish_reports.MAX_BATCH_BYTES", 4):
            self.assertEqual(self.run_publish(fake), "https://pr/2")
        inputs = [p["variables"]["input"] for _, e, p in fake.calls if e == "graphql"]
        self.assertEqual([i["expectedHeadOid"] for i in inputs], [self.base_sha, "1" * 40])
        self.assertEqual([i["message"]["headline"][-6:] for i in inputs], [" (1/2)", " (2/2)"])
        paths = [[a["path"] for a in i["fileChanges"].get("additions", [])] for i in inputs]
        self.assertEqual(paths, [["election-analysis/reports/analysis.json"],
                                 ["election-analysis/reports/edt/new.bin"]])
        self.assertEqual(inputs[0]["fileChanges"]["deletions"],
                         [{"path": "election-analysis/reports/edt/old.md"}])
        self.assertNotIn("deletions", inputs[1]["fileChanges"])

    def test_b6_unverified_commit_in_a_chain_stops_before_the_next(self) -> None:
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        fake = self.chain_fake(tree, verified_first=False)
        with mock.patch("election_reporting.ci.publish_reports.MAX_BATCH_BYTES", 4), \
                self.assertRaisesRegex(PublishError, "did not verify"):
            self.run_publish(fake)
        self.assertEqual(len([e for _, e, _ in fake.calls if e == "graphql"]), 1)
        self.assertFalse(any("/pulls" in e and m != "GET" for m, e, _ in fake.calls))

    def test_b7_rerun_on_an_existing_chain_is_idempotent(self) -> None:
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        fake = self.chain_fake(tree)
        fake.responses[("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}")] = [
            {"ref": f"refs/heads/{BRANCH}", "object": {"sha": "2" * 40}}]
        fake.responses[("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}")] = [
            {"number": 2, "html_url": "https://pr/2", "base": {"ref": "dev-daily"}}]
        fake.responses[("PATCH", f"repos/{REPO}/pulls/2")] = {}
        with mock.patch("election_reporting.ci.publish_reports.MAX_BATCH_BYTES", 4):
            self.assertEqual(self.run_publish(fake), "https://pr/2")
        self.assertNotIn(("POST", "graphql"), fake.endpoints())

    def test_r6_rerun_is_idempotent(self) -> None:
        self.regenerate()
        _, tree = detect_changes(self.root, "election-analysis/reports")
        head = "d" * 40
        fake = FakeGh({
            ("GET", f"repos/{REPO}/git/matching-refs/heads/{BRANCH}"):
                [{"ref": f"refs/heads/{BRANCH}", "object": {"sha": head}}],
            ("GET", f"repos/{REPO}/git/commits/{head}"): {
                "sha": head, "tree": {"sha": tree}, "parents": [{"sha": self.base_sha}],
                "verification": {"verified": True, "reason": "valid"}},
            ("GET", f"repos/{REPO}/pulls?state=open&head=owner:{BRANCH}"):
                [{"number": 91, "html_url": "https://pr/91", "base": {"ref": "dev-daily"}}],
            ("PATCH", f"repos/{REPO}/pulls/91"): {},
        })
        self.assertEqual(self.run_publish(fake), "https://pr/91")
        self.assertEqual(self.writes(fake), [("PATCH", f"repos/{REPO}/pulls/91")])
        patch = fake.calls[-1][2]
        self.assertNotIn("base", patch)


class WorkflowRouteTests(unittest.TestCase):
    """The workflows publish into dev-daily, from a dev-daily checkout, never the default branch."""

    ROOT = Path(__file__).resolve().parents[2]

    def read(self, path: str) -> str:
        return (self.ROOT / path).read_text(encoding="utf-8")

    def test_automation_publishes_as_ring0_with_fallback(self) -> None:
        # GitHub does not run workflows on commits or PRs made with the workflow's
        # own GITHUB_TOKEN, so automation PRs never got required checks (#91).
        # Both publishers mint a catylist-ring-0 token (Catylist ADR-0012) and fall
        # back to github.token only when the key secret is absent.
        for path in (".github/workflows/election-result-reports.yml",
                     ".github/workflows/election-source-watch.yml"):
            text = self.read(path)
            self.assertIn("actions/create-github-app-token@64ab729a6294000f5aa4a4bc0b73d1f32bc45510", text, path)
            self.assertIn("app-id: ${{ secrets.RING0_APP_ID || '5474337' }}", text, path)
            self.assertIn("environment: RING0", text, path)
            if path.endswith("election-source-watch.yml"):
                # Publishing runs even when a state's fetch failed (!cancelled()),
                # so the ring-0 key and token steps must too, or the publish falls
                # back to github.token and the snapshot PR gets no CI (run 42259780114).
                key_step = text.split("- name: Check for the catylist-ring-0 key", 1)[1].split("- name:", 1)[0]
                self.assertIn("if: ${{ !cancelled() }}", key_step, path)
                self.assertIn("if: ${{ !cancelled() && steps.ring0_key.outputs.present == 'true' }}", text, path)
                publish_step = text.split("- name: Publish verified commit and pull request", 1)[1].split("- name:", 1)[0]
                self.assertIn("if: ${{ !cancelled() }}", publish_step, path)
            else:
                self.assertIn("if: ${{ steps.ring0_key.outputs.present == 'true' }}", text, path)
            token_step = text.split("- name: Create catylist-ring-0 token", 1)[1].split("- name:", 1)[0]
            self.assertIn("continue-on-error: true", token_step, path)  # a bad key never stops publication
            self.assertIn("GH_TOKEN: ${{ steps.ring0.outputs.token || github.token }}", text, path)
            publish = text.split("- name: Publish verified commit and pull request", 1)[1]
            self.assertNotIn("GH_TOKEN: ${{ github.token }}", publish.split("run:", 1)[0], path)

    def test_reports_workflow_publishes_from_and_into_dev_daily(self) -> None:
        text = self.read(".github/workflows/election-result-reports.yml")
        self.assertIn("github.ref == 'refs/heads/dev-daily'", text)
        self.assertIn("BASE_BRANCH: dev-daily", text)
        self.assertNotIn("BASE_BRANCH: ${{ github.event.repository.default_branch }}", text)

    def test_source_watch_checks_out_and_publishes_into_dev_daily(self) -> None:
        text = self.read(".github/workflows/election-source-watch.yml")
        self.assertIn("ref: dev-daily", text)
        self.assertIn("BASE_BRANCH: dev-daily", text)
        self.assertNotIn("BASE_BRANCH: ${{ github.event.repository.default_branch }}", text)

    def test_scripts_publish_against_the_checked_out_commit(self) -> None:
        for script in ("election-analysis/ci/reports.sh", "election-analysis/ci/watch.sh"):
            with self.subTest(script=script):
                text = self.read(script)
                self.assertIn('--base-sha "$(git rev-parse HEAD)"', text)
                self.assertNotIn('--base-sha "${GITHUB_SHA:?}"', text)


if __name__ == "__main__":
    unittest.main()
