# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:e1c33314-9616-4f73-9f97-85f4c11db80e

"""Reproduce OC v1 from its exact Git snapshot and installed pinned runtime.

Does not install packages, fetch Git objects, retune, or alter the first seal.
Current CI dependencies may advance independently of a historical experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "09d825d81083049903bd544e550adacd915a2c4a"
PREFIX = "election-analysis/"
STEM = "oc-official-"
MODEL = "retrospective_official_queue_scenarios"


class ReproductionError(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout


def safe_path(name):
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
        raise ReproductionError("frozen path must be a relative POSIX path")
    return path


def verify_frozen_tree(repo, commit, model_files):
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not model_files or len(model_files) > 200:
        raise ReproductionError("exact source commit and bounded frozen inventory required")
    checked = {}
    for name, expected in sorted(model_files.items()):
        safe_path(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ReproductionError("invalid frozen SHA-256")
        actual = sha(git(repo, "show", commit+":"+PREFIX+name))
        if actual != expected:
            raise ReproductionError("source differs from frozen protocol: "+name)
        checked[name] = actual
    inventory = git(repo, "ls-tree", "-r", "--name-only", commit, "--", PREFIX+"election_reporting").decode().splitlines()
    actual_py = {p.removeprefix(PREFIX) for p in inventory if p.endswith(".py")}
    expected_py = {p for p in model_files if p.startswith("election_reporting/") and p.endswith(".py")}
    if actual_py != expected_py:
        raise ReproductionError("frozen package Python inventory differs")
    return checked


def extract_snapshot(raw, destination):
    if len(raw) > 64_000_000:
        raise ReproductionError("Git archive exceeds byte budget")
    target = Path(destination)
    target.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        if (len(members) > 1000 or sum(m.size for m in members) > 64_000_000
                or any(m.size < 0 or m.size > 16_000_000 for m in members)
                or len({m.name for m in members}) != len(members)):
            raise ReproductionError("Git archive inventory/expanded-size budget exceeded")
        for member in members:
            rel = safe_path(member.name)
            if rel.parts[0] != "election-analysis" or not (member.isfile() or member.isdir()):
                raise ReproductionError("nonregular or out-of-scope Git archive member")
        for member in members:
            path = target/safe_path(member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    return target/"election-analysis"


def pinned_versions(raw):
    pins = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([A-Za-z0-9.+_-]+)", line)
        if not match or match[1] in pins:
            raise ReproductionError("unrecognized or duplicate frozen runtime pin")
        pins[match[1]] = match[2]
    if not pins:
        raise ReproductionError("frozen runtime pins missing")
    return pins


def require_runtime(pins, observed):
    if any(observed.get(name) != version for name, version in pins.items()):
        raise ReproductionError(f"installed runtime differs from frozen pins: expected {pins}, observed {observed}; use a separate matching interpreter; no packages were installed")


def semantic(value, *, scoring=False):
    ignored = {"generated_at", "prediction_generated_at"}
    if scoring:
        ignored.add("sealed_predictions_sha256")
    if isinstance(value, dict):
        return {k: semantic(v, scoring=scoring) for k, v in value.items() if k not in ignored}
    return [semantic(v, scoring=scoring) for v in value] if isinstance(value, list) else value


def reproduce(repo, output, python=sys.executable, source_commit=SOURCE):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ReproductionError("exact 40-hex source commit required before Git access")
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ReproductionError("output must be empty; existing seals are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    docs = PREFIX+"docs/engineering/"
    protocol_raw = git(repo, "show", source_commit+":"+docs+STEM+"protocol-v1.json")
    protocol = json.loads(protocol_raw)
    if protocol.get("schema_version") != 3 or protocol.get("mode") != MODEL:
        raise ReproductionError("only the explicit OC v1 experiment is supported")
    checked = verify_frozen_tree(repo, source_commit, protocol["model_files"])
    pins = pinned_versions(git(repo, "show", source_commit+":"+PREFIX+"ci/requirements-provenance.txt"))
    code = "import importlib.metadata as m,json,sys\ndef v(n):\n try:return m.version(n)\n except m.PackageNotFoundError:return None\nprint(json.dumps({'python':sys.version.split()[0],'packages':{n:v(n) for n in sys.argv[1:]}}))"
    runtime = json.loads(subprocess.run([python, "-I", "-c", code, *pins], check=True, capture_output=True, text=True, timeout=30).stdout)
    require_runtime(pins, runtime["packages"])
    with tempfile.TemporaryDirectory() as td:
        scope = extract_snapshot(git(repo, "archive", "--format=tar", source_commit, PREFIX), Path(td)/"source")
        original_docs = scope/"docs/engineering"
        original_predictions = json.loads((original_docs/(STEM+"predictions-v1.json")).read_text())
        original_scores = json.loads((original_docs/(STEM+"scores-v1.json")).read_text())
        original_seal = json.loads((original_docs/(STEM+"seal-v1.json")).read_text())
        if sha((original_docs/(STEM+"predictions-v1.json")).read_bytes()) != original_seal["predictions_sha256"]:
            raise ReproductionError("original first seal does not match original prediction bytes")
        launch = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv[0]='election_reporting.official_validation';runpy.run_module('election_reporting.official_validation',run_name='__main__')"
        common = ["--protocol", str(original_docs/(STEM+"protocol-v1.json")), "--fixtures", str(scope/"tests/fixtures")]
        def run(phase, extra):
            result = subprocess.run([python, "-I", "-c", launch, str(scope), phase, *common, *extra], cwd=scope, check=True, capture_output=True, text=True, timeout=120)
            (output/(phase+".log")).write_text(result.stdout+result.stderr)
        predictions_path, scores_path = output/"predictions.json", output/"scores.json"
        run("predict", ["--output", str(predictions_path)])
        predictions = json.loads(predictions_path.read_text())
        if semantic(predictions) != semantic(original_predictions):
            raise ReproductionError("prediction semantics differ; first seal retained and scoring stopped")
        seal = {"schema_version": 1, "sealed_at": datetime.now(timezone.utc).isoformat(),
                "predictions_sha256": sha(predictions_path.read_bytes()), "original_first_seal": original_seal["predictions_sha256"],
                "protocol_sha256": predictions["protocol_sha256"], "reserved_outcomes_read": predictions["reserved_outcomes_read"]}
        (output/"reproduction-seal.json").write_text(json.dumps(seal, indent=2)+"\n")
        run("score", ["--predictions", str(predictions_path), "--predictions-sha256", seal["predictions_sha256"], "--output", str(scores_path)])
        scores = json.loads(scores_path.read_text())
        if semantic(scores, scoring=True) != semantic(original_scores, scoring=True):
            raise ReproductionError("score/ledger semantics differ from the first run")
    verification = {"schema_version": 1, "status": "MATCH", "source_commit": source_commit,
        "source_tree": git(repo, "rev-parse", source_commit+"^{tree}").decode().strip(),
        "caller_checkout_commit": git(repo, "rev-parse", "HEAD").decode().strip(),
        "protocol_file_sha256": sha(protocol_raw), "model_files_verified": checked, "runtime": runtime, "frozen_runtime_pins": pins,
        "prediction_rows": len(predictions["rows"]), "base_contests": scores["scored_base_rows"], "reserved_outcomes_read_in_prediction": predictions["reserved_outcomes_read"],
        "original_first_seal_preserved": True, "prediction_semantics_match": True, "score_and_ledger_semantics_match": True,
        "normalized_keys": ["generated_at", "prediction_generated_at", "scoring-only sealed_predictions_sha256"],
        "output_sha256": {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()},
        "qualification": "Reproduces the original conditional experiment from its frozen code/runtime. No new data, cap tuning, historical-publication authentication, calibrated forecast or independent final re-audit is inferred."}
    (output/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-commit", default=SOURCE)
    args = parser.parse_args()
    try:
        result = reproduce(args.repo, args.output, args.python, args.source_commit)
    except (ReproductionError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, str(exc)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "source_commit", "prediction_rows", "base_contests", "original_first_seal_preserved")}))


if __name__ == "__main__":
    main()
