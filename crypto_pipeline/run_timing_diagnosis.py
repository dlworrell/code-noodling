#!/usr/bin/env python3
"""Retain a predeclared timing/control matrix, including failures. Linux/x86 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = "dc269651fb2567e46755cfb2a13d3875592968b5"
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dudect", type=Path, default=ROOT / "vendor/dudect")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing evidence is never overwritten")
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--optimization", choices=["O2", "O3"], default="O2")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--batches", type=int, default=101)
    args = parser.parse_args()
    if args.batches < 2 or args.cpu not in os.sched_getaffinity(0):
        parser.error("Need >=2 batches and a CPU in the current affinity mask")
    vendor = args.dudect.resolve()
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        parser.error("Unexpected dudect revision")
    subprocess.run(["git", "-C", str(vendor), "diff", "--exit-code", "HEAD", "--", "src/dudect.h"], check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    flags = ["-" + args.optimization, "-fno-lto", "-fPIC", "-Wall", "-Wextra"]
    # Declare the full matrix before measurements; do not select reruns by outcome.
    modes = [("leaky-control", "CT_TIMING_NEGATIVE_CONTROL"),
             ("public-baseline", "CT_TIMING_BASELINE"),
             ("input-only", "CT_TIMING_INPUT_ONLY"),
             ("random-labels", "CT_TIMING_RANDOM_LABELS"),
             ("candidate", None)]
    matrix = [(name, macro, kind) for name, macro in modes for kind in (0, 1)]
    report = {"dudect": PIN, "compiler": args.compiler, "flags": flags,
              "cpu": args.cpu, "batches": args.batches, "limbs": 16,
              "security_status": "NOT CLEARED FOR SECRET USE",
              "plan": [{"mode": n, "fixed_class": k} for n, _, k in matrix],
              "results": [], "hashes": {}}

    def save():
        for file in sorted(output.iterdir()):
            if file.is_file() and file.name != "report.json":
                report["hashes"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for file in (ROOT / "crypto_montgomery_ct.c", ROOT / "crypto_montgomery_ct.h",
                     ROOT / "dudect_montgomery.c", Path(__file__), vendor / "src/dudect.h"):
            report["hashes"][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with (output / "environment.txt").open("w") as log:
            for command in ([args.compiler, "--version"], ["uname", "-a"], ["lscpu"]):
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            for pattern in ("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor" % args.cpu,
                            "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % args.cpu):
                file = Path(pattern)
                log.write(pattern + ": " + (file.read_text() if file.exists() else "unavailable\n"))
        obj = output / "candidate.o"
        subprocess.run([args.compiler, "-std=c11", *flags, "-c", str(ROOT / "crypto_montgomery_ct.c"), "-o", str(obj)], check=True)
        with (output / "candidate.asm").open("w") as log:
            subprocess.run(["objdump", "-drwC", str(obj)], stdout=log, check=True)
        for name, macro, kind in matrix:
            binary = output / f"{name}-{kind}"
            command = [args.compiler, "-std=gnu11", *flags, "-I" + str(vendor / "src"),
                       "-DCT_LIMBS=16", f"-DCT_FIXED_CLASS={kind}", f"-DCT_BATCHES={args.batches}"]
            if macro:
                command.append("-D" + macro)
            command += [str(ROOT / "dudect_montgomery.c"), str(obj), "-lm", "-o", str(binary)]
            subprocess.run(command, check=True)
            with binary.with_suffix(".log").open("w") as log:
                result = subprocess.run(["taskset", "-c", str(args.cpu), str(binary)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=300)
            code = result.returncode
            outcome = "threshold_crossed" if code == 1 else "no_crossing" if code == 0 else "execution_error"
            report["results"].append({"mode": name, "fixed_class": kind,
                                      "returncode": code, "outcome": outcome, "command": command})
            print(f"{name}-{kind}: {outcome}", flush=True)
            save()
        # Controls must work; a clean bounded run never clears the historical signal.
        failed = any(r["returncode"] != (1 if r["mode"] == "leaky-control" else 0)
                     for r in report["results"])
        report["matrix_completed"] = True
        report["diagnostic_failure"] = failed
        return int(failed)
    finally:
        save()


if __name__ == "__main__":
    sys.exit(main())
