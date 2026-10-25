# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:f096fb5d-1b0a-4350-8fa6-e2ae954ad481

"""Report-only provenance checks for the election-analysis scope (election slice, step 3).

Checks, each with exit code 0 (pass), 1 (violation) or 2 (indeterminate):

- reuse: `reuse lint` with election-analysis/ as the REUSE root. License texts
  for publishers whose data has not arrived yet (PENDING_LICENSES) may be
  unused; any other non-compliance is a violation;
- file-ids: every authored file has exactly one Catalyst-File-ID, a version 4
  urn:uuid, in its header, unique within the scope. Files that automation
  writes (data snapshots, provenance records, reports) are licensed by
  REUSE.toml and identified by SHA-256 instead (baseline finding E2);
- manifest: scope-manifest.json with the commit and, for every tracked file,
  its path, kind, File-ID (authored files) and SHA-256.

Unlike crypto_pipeline, there is no Doxygen check: the scope is Python.

`all` runs every check, writes results.json, scope-manifest.json and
summary.md to --output, and exits with the worst status: 1 if any check found
a violation, otherwise 2 if any was indeterminate, otherwise 0. The calling
workflow records that status without failing the pull request.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

PASS, VIOLATION, INDETERMINATE = 0, 1, 2
SCOPE = Path(__file__).resolve().parents[1]
SCOPE_NAME = "election-analysis/"
FILE_ID = re.compile(r"Catalyst-File-ID:\s*(\S+)")
UUID4 = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
HEADER_LINES = 20  # File-IDs live in the leading comment block

# Written by the source watcher or the report workflow; licensed by REUSE.toml
# path annotations and identified by the SHA-256 in SOURCES.json and
# analysis.json, so they carry no File-ID. Order matters: first match wins.
AUTOMATION = [
    ("reports/*", "generated-report"),
    ("data/elections/*/*/input/*", "source-data"),
    ("data/elections/*/*/raw/*", "source-data"),
    ("data/elections/*/*/areas/*", "derived-data"),
    ("data/elections/*/*/SOURCES.json", "provenance-record"),
    ("data/crosswalks/*/*.csv", "derived-data"),
    ("docs/engineering/*.json", "engineering-record"),
    ("tests/fixtures/*.json", "regression-evidence"),
    ("tests/fixtures/replay-2024/*", "frozen-source-regression"),
]

# LicenseRef texts drafted for publishers whose snapshots arrive from the
# 2026-11-03 election on. Until then REUSE reports them as unused; once their
# data is committed they are used and this list no longer matters.
PENDING_LICENSES = {
    "LicenseRef-ChelanCounty-PublicData",
    "LicenseRef-KittitasCounty-PublicData",
    "LicenseRef-ME-SOS-PublicData",
    "LicenseRef-PierceCounty-PublicData",
    "LicenseRef-SnohomishCounty-PublicData",
}


def governed_files(scope: Path) -> list[str]:
    """Files under the scope, relative to it: tracked files in a git checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "."], cwd=scope, check=True, capture_output=True
        ).stdout.decode()
        files = [name for name in out.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        files = [str(p.relative_to(scope)) for p in scope.rglob("*") if p.is_file()]
    return sorted(name for name in files if (scope / name).is_file())


def kind(name: str) -> str:
    if name.startswith("LICENSES/"):
        return "license-text"
    if name == "REUSE.toml":
        return "reuse-config"
    for pattern, label in AUTOMATION:
        if fnmatch.fnmatch(name, pattern):
            return label
    return "authored"


def ids_in(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = [next(handle, "") for _ in range(HEADER_LINES)]
    except OSError:
        return []
    return [match.group(1) for line in head for match in FILE_ID.finditer(line)]


def file_ids(scope: Path) -> tuple[int, dict]:
    found: dict[str, str] = {}
    problems: list[str] = []
    exempt = 0
    for name in governed_files(scope):
        if kind(name) != "authored":
            exempt += 1
            continue
        ids = ids_in(scope / name)
        if not ids:
            problems.append(f"{name}: no Catalyst-File-ID in its header")
        elif len(ids) > 1:
            problems.append(f"{name}: {len(ids)} Catalyst-File-IDs; expected exactly one")
        else:
            if not UUID4.match(ids[0]):
                problems.append(f"{name}: {ids[0]!r} is not a version 4 urn:uuid")
            found[name] = ids[0]
    seen: dict[str, str] = {}
    for name, value in sorted(found.items()):
        if value in seen:
            problems.append(f"{name}: Catalyst-File-ID {value} is also used by {seen[value]}")
        seen.setdefault(value, name)
    details = {"files_with_id": len(found), "exempt": exempt, "violations": problems, "ids": found}
    return (VIOLATION if problems else PASS), details


def reuse_lint(scope: Path) -> tuple[int, dict]:
    if shutil.which("reuse") is None:
        return INDETERMINATE, {"reason": "reuse is not installed"}
    run = subprocess.run(["reuse", "--root", str(scope), "lint", "--json"], capture_output=True, text=True)
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        return INDETERMINATE, {"reason": "reuse lint did not return JSON", "stderr": run.stderr[-2000:]}
    summary = report.get("summary", {})
    non_compliant = {key: value for key, value in report.get("non_compliant", {}).items() if value}
    unused = set(non_compliant.pop("unused_licenses", []) or [])
    pending = sorted(unused & PENDING_LICENSES)
    unexpected = sorted(unused - PENDING_LICENSES)
    if unexpected:
        non_compliant["unused_licenses"] = unexpected
    details = {
        "reuse_version": report.get("reuse_tool_version"),
        "spec_version": report.get("reuse_spec_version"),
        "compliant": summary.get("compliant"),
        "files_total": summary.get("files_total"),
        "files_with_copyright_info": summary.get("files_with_copyright_info"),
        "files_with_licensing_info": summary.get("files_with_licensing_info"),
        "pending_licenses": pending,
        "non_compliant": non_compliant,
    }
    return (VIOLATION if non_compliant else PASS), details


def commit(scope: Path) -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=scope, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def manifest(scope: Path, ids: dict[str, str]) -> tuple[int, dict]:
    sha = commit(scope)
    entries = []
    for name in governed_files(scope):
        entries.append(
            {
                "path": f"{SCOPE_NAME}{name}",
                "kind": kind(name),
                "file_id": ids.get(name),
                "sha256": hashlib.sha256((scope / name).read_bytes()).hexdigest(),
            }
        )
    document = {
        "schema": "https://catalyst.dlworrell.dev/scope-manifest/v1-draft",
        "repository": os.environ.get("GITHUB_REPOSITORY", "dlworrell/code-noodling"),
        "scope": SCOPE_NAME,
        "commit": sha,
        "files": entries,
    }
    return (PASS if sha else INDETERMINATE), document


def write_summary(output: Path, results: dict, overall: int) -> str:
    words = {PASS: "pass", VIOLATION: "violation", INDETERMINATE: "indeterminate"}
    rows = [
        "## election-analysis provenance (report-only)",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, (status, details) in results.items():
        if name == "reuse":
            if "files_total" in details:
                detail = f"{details.get('files_with_licensing_info')}/{details.get('files_total')} files licensed"
                if details.get("pending_licenses"):
                    detail += f"; {len(details['pending_licenses'])} license text(s) awaiting their first data"
            else:
                detail = details.get("reason", "")
        elif name == "file-ids":
            detail = (f"{details['files_with_id']} authored files with IDs, {details['exempt']} exempt; "
                      f"{len(details['violations'])} problem(s)")
        else:
            detail = f"{len(details.get('files', []))} files at {details.get('commit', '')[:12]}"
        rows.append(f"| {name} | {status} ({words[status]}) | {detail} |")
    rows += ["", f"Overall: {overall} ({words[overall]}). Report-only: this does not fail the pull request.", ""]
    text = "\n".join(rows)
    (output / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=("reuse", "file-ids", "manifest", "all"))
    parser.add_argument("--output", type=Path, default=Path("build/provenance"))
    parser.add_argument("--scope", type=Path, default=SCOPE)
    args = parser.parse_args(argv)
    scope = args.scope.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    ids_status, ids_details = file_ids(scope)
    checks = {
        "reuse": lambda: reuse_lint(scope),
        "file-ids": lambda: (ids_status, ids_details),
        "manifest": lambda: manifest(scope, ids_details["ids"]),
    }
    selected = list(checks) if args.check == "all" else [args.check]
    results = {name: checks[name]() for name in selected}

    if "manifest" in results:
        text = json.dumps(results["manifest"][1], indent=2) + "\n"
        (args.output / "scope-manifest.json").write_text(text, encoding="utf-8")
    report = {
        name: {"status": status, **({k: v for k, v in details.items() if k != "ids"} if name != "manifest" else {"commit": details.get("commit"), "files": len(details.get("files", []))})}
        for name, (status, details) in results.items()
    }
    statuses = [status for status, _ in results.values()]
    overall = VIOLATION if VIOLATION in statuses else INDETERMINATE if INDETERMINATE in statuses else PASS
    report["overall"] = overall
    report["commit"] = commit(scope)
    report["scope"] = SCOPE_NAME
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = write_summary(args.output, results, overall)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
