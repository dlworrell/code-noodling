"""Publish regenerated election reports as a GitHub-verified commit on a pull request.

ADR-0006 (AES-DEV-001-R003/R005) forbids automation from pushing to the default
branch and requires automation-produced commits to be GitHub-verified. A plain
``git push`` from a runner yields an unsigned commit, so this script instead:

1. Detects which files under the report directory differ from the analyzed commit
   (``git add -A`` + ``git diff --cached --name-status``, staged in the local
   index only; nothing is committed or pushed locally).
2. Creates the automation branch from the analyzed commit, or force-resets it there.
   Only the workflow-owned automation branch is ever force-updated, never the base.
3. Commits the changes through GraphQL ``createCommitOnBranch`` using the
   workflow's ``GITHUB_TOKEN``. GitHub signs such commits, so they show Verified.
4. Opens a pull request from the automation branch to the base branch, or updates
   the existing one.

If nothing changed, it does nothing. If the automation branch already carries
exactly this content on top of the analyzed commit, it only ensures the PR exists.

All GitHub calls go through ``gh api`` (preinstalled on hosted runners) with the
token in ``GH_TOKEN``. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_BRANCH = "automation/election-reports"
DEFAULT_PATH = "reports/elections"
DEFAULT_TITLE = "reports: update election result analysis"

CREATE_COMMIT_MUTATION = """
mutation($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) {
    commit { oid url }
  }
}
""".strip()


class PublishError(RuntimeError):
    """Raised when publishing cannot complete."""


@dataclass(frozen=True)
class ReportChanges:
    """Repository-relative paths to add/replace and to delete."""

    additions: tuple[str, ...] = ()
    deletions: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.additions and not self.deletions


# --------------------------------------------------------------------------- pure


def parse_name_status(output: bytes) -> ReportChanges:
    """Parse ``git diff --name-status -z --no-renames`` output.

    Entries are NUL-separated ``STATUS\\0PATH\\0`` pairs. With ``--no-renames``
    (and no copy detection) every entry has exactly one path. ``D`` is a deletion;
    ``A``, ``M`` and ``T`` (type change) are additions in ``createCommitOnBranch``
    terms because a ``FileAddition`` replaces whatever is at that path.
    """
    parts = output.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    if len(parts) % 2:
        raise PublishError(f"unexpected git name-status output: {output!r}")
    additions: list[str] = []
    deletions: list[str] = []
    for status_raw, path_raw in zip(parts[0::2], parts[1::2]):
        status = status_raw.decode("ascii")
        path = path_raw.decode("utf-8")
        code = status[:1]
        if code == "D":
            deletions.append(path)
        elif code in {"A", "M", "T"}:
            additions.append(path)
        else:
            raise PublishError(f"unsupported change status {status!r} for {path}")
    return ReportChanges(tuple(sorted(additions)), tuple(sorted(deletions)))


def restrict_to(changes: ReportChanges, prefix: str) -> ReportChanges:
    """Fail closed if a change escapes the report directory."""
    root = prefix.rstrip("/") + "/"
    for path in (*changes.additions, *changes.deletions):
        if not path.startswith(root) or ".." in Path(path).parts:
            raise PublishError(f"refusing to publish path outside {root}: {path}")
    return changes


def build_commit_input(
    *,
    repository: str,
    branch: str,
    expected_head_oid: str,
    headline: str,
    body: str,
    changes: ReportChanges,
    read_bytes: Callable[[str], bytes],
) -> dict:
    """Build the ``CreateCommitOnBranchInput`` value.

    File contents are read as bytes and base64-encoded, so binary files are safe.
    """
    file_changes: dict = {}
    if changes.additions:
        file_changes["additions"] = [
            {"path": path, "contents": base64.b64encode(read_bytes(path)).decode("ascii")}
            for path in changes.additions
        ]
    if changes.deletions:
        file_changes["deletions"] = [{"path": path} for path in changes.deletions]
    message: dict = {"headline": headline}
    if body:
        message["body"] = body
    return {
        "branch": {"repositoryNameWithOwner": repository, "branchName": branch},
        "expectedHeadOid": expected_head_oid,
        "message": message,
        "fileChanges": file_changes,
    }


def build_graphql_request(commit_input: dict) -> dict:
    return {"query": CREATE_COMMIT_MUTATION, "variables": {"input": commit_input}}


DEFAULT_SUMMARY = (
    "Automated regeneration of `reports/elections` by the "
    "**Election Result Reports** workflow."
)


def pull_request_body(
    changes: ReportChanges, base_sha: str, run_url: str, summary: str = DEFAULT_SUMMARY
) -> str:
    lines = [
        summary,
        "",
        f"- Generated from commit `{base_sha}`.",
    ]
    if run_url:
        lines.append(f"- Workflow run: {run_url}")
    lines += [
        f"- Files added or updated: {len(changes.additions)}",
        f"- Files deleted: {len(changes.deletions)}",
        "",
        "The commit was created through the GitHub GraphQL API "
        "(`createCommitOnBranch`) and is signed by GitHub. This branch is owned by "
        "the workflow and is force-reset on each run; do not push to it by hand.",
        "",
        "Per ADR-0006, automation never pushes to the default branch; review and "
        "merge this pull request to publish the changes.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- I/O


Runner = Callable[..., subprocess.CompletedProcess]


@dataclass
class GitHub:
    """Thin wrapper around ``gh api``. ``runner`` is injectable for tests."""

    repository: str
    runner: Runner = field(default=subprocess.run)

    def api(self, method: str, endpoint: str, payload: dict | None = None) -> object:
        args = ["gh", "api", "--method", method, endpoint]
        stdin = None
        if payload is not None:
            args += ["--input", "-"]
            stdin = json.dumps(payload).encode("utf-8")
        result = self.runner(args, input=stdin, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", "replace").strip()
            raise PublishError(f"gh api {method} {endpoint} failed: {stderr}")
        text = result.stdout.decode("utf-8").strip()
        return json.loads(text) if text else None

    def branch_head(self, branch: str) -> str | None:
        refs = self.api("GET", f"repos/{self.repository}/git/matching-refs/heads/{branch}")
        for ref in refs or []:
            if ref.get("ref") == f"refs/heads/{branch}":
                return ref["object"]["sha"]
        return None

    def commit(self, sha: str) -> dict:
        return self.api("GET", f"repos/{self.repository}/git/commits/{sha}")

    def point_branch(self, branch: str, sha: str, exists: bool) -> None:
        if exists:
            self.api(
                "PATCH",
                f"repos/{self.repository}/git/refs/heads/{branch}",
                {"sha": sha, "force": True},
            )
        else:
            self.api(
                "POST",
                f"repos/{self.repository}/git/refs",
                {"ref": f"refs/heads/{branch}", "sha": sha},
            )

    def create_commit(self, request: dict) -> dict:
        data = self.api("POST", "graphql", request)
        if not isinstance(data, dict) or data.get("errors"):
            raise PublishError(f"createCommitOnBranch failed: {data!r}")
        return data["data"]["createCommitOnBranch"]["commit"]

    def open_pull_request(self, branch: str, base: str) -> dict | None:
        owner = self.repository.split("/", 1)[0]
        pulls = self.api(
            "GET",
            f"repos/{self.repository}/pulls?state=open&head={owner}:{branch}&base={base}",
        )
        return pulls[0] if pulls else None

    def upsert_pull_request(self, branch: str, base: str, title: str, body: str) -> str:
        existing = self.open_pull_request(branch, base)
        if existing:
            self.api(
                "PATCH",
                f"repos/{self.repository}/pulls/{existing['number']}",
                {"title": title, "body": body},
            )
            return existing["html_url"]
        try:
            created = self.api(
                "POST",
                f"repos/{self.repository}/pulls",
                {"title": title, "head": branch, "base": base, "body": body},
            )
        except PublishError as error:
            raise PublishError(
                f"{error} -- the verified commit is on {branch}, but the pull request "
                "could not be opened. A maintainer must enable Settings > Actions > "
                "General > 'Allow GitHub Actions to create and approve pull requests', "
                f"or open the pull request from {branch} manually."
            ) from error
        return created["html_url"]


def git(args: Sequence[str], cwd: Path, env: dict | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise PublishError(
            f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', 'replace')}"
        )
    return result.stdout


def detect_changes(repo_root: Path, report_path: str) -> tuple[ReportChanges, str]:
    """Return report changes relative to HEAD and the resulting tree id.

    Uses a throwaway index file so the checkout's real index is not touched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))
        git(["read-tree", "HEAD"], repo_root, env)
        git(["add", "-A", "--", report_path], repo_root, env)
        raw = git(
            ["diff", "--cached", "--name-status", "-z", "--no-renames", "HEAD", "--", report_path],
            repo_root,
            env,
        )
        tree = git(["write-tree"], repo_root, env).decode("ascii").strip()
    return restrict_to(parse_name_status(raw), report_path), tree


def publish(
    *,
    github: GitHub,
    repo_root: Path,
    report_path: str,
    branch: str,
    base: str,
    base_sha: str,
    title: str,
    run_url: str,
    summary: str = DEFAULT_SUMMARY,
    log: Callable[[str], None] = print,
) -> str | None:
    changes, tree = detect_changes(repo_root, report_path)
    if changes.empty:
        log("Reports are already current; nothing to publish.")
        return None
    log(f"{len(changes.additions)} file(s) to add/update, {len(changes.deletions)} to delete.")
    body = pull_request_body(changes, base_sha, run_url, summary)

    head = github.branch_head(branch)
    if head is not None:
        current = github.commit(head)
        parents = [p["sha"] for p in current.get("parents", [])]
        if parents == [base_sha] and current["tree"]["sha"] == tree:
            log(f"{branch} already holds these reports on top of {base_sha}.")
            return github.upsert_pull_request(branch, base, title, body)

    github.point_branch(branch, base_sha, exists=head is not None)
    commit_input = build_commit_input(
        repository=github.repository,
        branch=branch,
        expected_head_oid=base_sha,
        headline=title,
        body=f"Regenerated from {base_sha}.",
        changes=changes,
        read_bytes=lambda path: (repo_root / path).read_bytes(),
    )
    commit = github.create_commit(build_graphql_request(commit_input))
    log(f"Created verified commit {commit['oid']} on {branch}.")
    url = github.upsert_pull_request(branch, base, title, body)
    log(f"Pull request: {url}")
    return url


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--base", required=True, help="pull request base branch")
    parser.add_argument("--base-sha", required=True, help="commit the reports were generated from")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="first line of the PR body")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if not args.repository:
        parser.error("--repository or GITHUB_REPOSITORY is required")
    if args.branch == args.base:
        parser.error("the automation branch must differ from the base branch")
    try:
        url = publish(
            github=GitHub(args.repository),
            repo_root=args.repo_root,
            report_path=args.path,
            branch=args.branch,
            base=args.base,
            base_sha=args.base_sha,
            title=args.title,
            run_url=args.run_url,
            summary=args.summary,
        )
    except PublishError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"pull_request_url={url or ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
