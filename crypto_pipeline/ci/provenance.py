# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:1a228748-c8e0-4388-9df5-9b9291af72cd

"""Report-only provenance checks for the crypto_pipeline scope (PRM-0012 step 3).

Checks, each with exit code 0 (pass), 1 (violation) or 2 (indeterminate):

- reuse: `reuse lint` with crypto_pipeline/ as the REUSE root;
- file-ids: every governed file has exactly one Catalyst-File-ID, a version 4
  urn:uuid, in its header or its .license sidecar, unique within the scope;
- doxygen: count of undocumented C items (Doxygen WARN_IF_UNDOCUMENTED);
- manifest: scope-manifest.json with the commit and, for every governed file,
  its path, File-ID and SHA-256.

`all` runs every check, writes results.json, scope-manifest.json and
summary.md to --output, and exits with the worst status: 1 if any check found
a violation, otherwise 2 if any was indeterminate, otherwise 0. The calling
workflow records that status without failing the pull request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PASS, VIOLATION, INDETERMINATE = 0, 1, 2
SCOPE = Path(__file__).resolve().parents[1]
FILE_ID = re.compile(r"Catalyst-File-ID:\s*(\S+)")
UUID4 = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
HEADER_LINES = 20  # File-IDs live in the leading comment block or a sidecar
UNDOCUMENTED = re.compile(r"warning: .*is not documented")


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
    if name.endswith(".license"):
        return "sidecar"
    return "file"


def ids_in(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = [next(handle, "") for _ in range(HEADER_LINES)]
    except OSError:
        return []
    return [match.group(1) for line in head for match in FILE_ID.finditer(line)]


def file_ids(scope: Path) -> tuple[int, dict]:
    files = governed_files(scope)
    names = set(files)
    found: dict[str, str] = {}
    problems: list[str] = []
    for name in files:
        if kind(name) == "license-text":
            continue
        if kind(name) == "sidecar":
            if name[: -len(".license")] not in names:
                problems.append(f"{name}: sidecar for a file that is not in the scope")
            continue
        ids = ids_in(scope / name)
        sidecar = name + ".license"
        if sidecar in names:
            ids += ids_in(scope / sidecar)
        if not ids:
            problems.append(f"{name}: no Catalyst-File-ID in its header or a .license sidecar")
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
    return (VIOLATION if problems else PASS), {"files_with_id": len(found), "violations": problems, "ids": found}


def reuse_lint(scope: Path) -> tuple[int, dict]:
    if shutil.which("reuse") is None:
        return INDETERMINATE, {"reason": "reuse is not installed"}
    run = subprocess.run(["reuse", "--root", str(scope), "lint", "--json"], capture_output=True, text=True)
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        return INDETERMINATE, {"reason": "reuse lint did not return JSON", "stderr": run.stderr[-2000:]}
    summary = report.get("summary", {})
    details = {
        "reuse_version": report.get("reuse_tool_version"),
        "spec_version": report.get("reuse_spec_version"),
        "compliant": summary.get("compliant"),
        "files_total": summary.get("files_total"),
        "files_with_copyright_info": summary.get("files_with_copyright_info"),
        "files_with_licensing_info": summary.get("files_with_licensing_info"),
        "non_compliant": report.get("non_compliant", {}),
        "recommendations": report.get("recommendations", []),
    }
    return (PASS if summary.get("compliant") else VIOLATION), details


def doxygen(scope: Path) -> tuple[int, dict]:
    if shutil.which("doxygen") is None:
        return INDETERMINATE, {"reason": "doxygen is not installed"}
    with tempfile.TemporaryDirectory() as work:
        log = Path(work) / "warnings.log"
        config = "\n".join(
            [
                f'INPUT = "{scope}"',
                "FILE_PATTERNS = *.c *.h",
                "RECURSIVE = YES",
                f'OUTPUT_DIRECTORY = "{work}"',
                "EXTRACT_ALL = NO",
                "WARN_IF_UNDOCUMENTED = YES",
                "WARN_NO_PARAMDOC = NO",
                "GENERATE_HTML = NO",
                "GENERATE_LATEX = NO",
                "QUIET = YES",
                f'WARN_LOGFILE = "{log}"',
                "",
            ]
        )
        run = subprocess.run(["doxygen", "-"], input=config, capture_output=True, text=True)
        if run.returncode != 0 or not log.exists():
            return INDETERMINATE, {"reason": "doxygen failed", "stderr": run.stderr[-2000:]}
        items = sorted(
            {
                line.replace(str(scope) + os.sep, "")
                for line in log.read_text(errors="replace").splitlines()
                if UNDOCUMENTED.search(line)
            }
        )
    version = subprocess.run(["doxygen", "--version"], capture_output=True, text=True).stdout.strip()
    return (VIOLATION if items else PASS), {"doxygen_version": version, "undocumented": len(items), "items": items}


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
        data = (scope / name).read_bytes()
        entries.append(
            {
                "path": f"crypto_pipeline/{name}",
                "kind": kind(name),
                "file_id": ids.get(name),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    document = {
        "schema": "https://catalyst.dlworrell.dev/scope-manifest/v1-draft",
        "repository": os.environ.get("GITHUB_REPOSITORY", "dlworrell/code-noodling"),
        "scope": "crypto_pipeline/",
        "commit": sha,
        "files": entries,
    }
    return (PASS if sha else INDETERMINATE), document


def write_summary(output: Path, results: dict, overall: int) -> str:
    words = {PASS: "pass", VIOLATION: "violation", INDETERMINATE: "indeterminate"}
    rows = [
        "## crypto_pipeline provenance (report-only)",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, (status, details) in results.items():
        if name == "reuse":
            detail = f"{details.get('files_with_licensing_info')}/{details.get('files_total')} files licensed" if "files_total" in details else details.get("reason", "")
        elif name == "file-ids":
            detail = f"{details['files_with_id']} files with IDs; {len(details['violations'])} problem(s)"
        elif name == "doxygen":
            detail = f"{details['undocumented']} undocumented C items" if "undocumented" in details else details.get("reason", "")
        else:
            detail = f"{len(details.get('files', []))} files at {details.get('commit', '')[:12]}"
        rows.append(f"| {name} | {status} ({words[status]}) | {detail} |")
    rows += ["", f"Overall: {overall} ({words[overall]}). Report-only: this does not fail the pull request.", ""]
    text = "\n".join(rows)
    (output / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=("reuse", "file-ids", "doxygen", "manifest", "all"))
    parser.add_argument("--output", type=Path, default=Path("build/provenance"))
    parser.add_argument("--scope", type=Path, default=SCOPE)
    args = parser.parse_args(argv)
    scope = args.scope.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    ids_status, ids_details = file_ids(scope)
    checks = {
        "reuse": lambda: reuse_lint(scope),
        "file-ids": lambda: (ids_status, ids_details),
        "doxygen": lambda: doxygen(scope),
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
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = write_summary(args.output, results, overall)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return overall


if __name__ == "__main__":
    sys.exit(main())
