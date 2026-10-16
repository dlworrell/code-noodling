# Timing runner and setup integration, 2026-10-14

The existing run_timing_diagnosis.py remains the authoritative runner. It now
supports --limbs 4/16/32 (all three when omitted), preserves both fixed classes
and all five original modes, and keeps --output as a required fresh directory.
No historical input distributions or timing signals have been replaced.

Each run copies and hashes the required C sources, headers, runner and pinned
clean dudect header before building. Builds use that snapshot. Source drift
observed at checkpoints is sticky and fails the diagnostic; changes and reversions
between checkpoints are not detectable, but cannot change the copied build inputs.
Compiler metadata, build/run argv, diagnostics, disassembly, pending/in-flight/
terminal states, and artifact hashes are retained through atomic report replacement.
This is atomic visibility, not a power-loss durability guarantee. A process forcibly
killed before its finally block runs leaves the last checkpoint, not a final report.
Runtime exit codes 0/1 additionally require the harness completion marker, so a
failed taskset invocation cannot pass as a detected negative control.

The manual-only timing workflow has six independent jobs: O2/O3 x 4/16/32 limbs.
Each attempts ten configurations, with 400-second execution and 60-second build
limits and a 90-minute job budget. Shared-runner timing remains exploratory;
affinity does not isolate interrupts, competing tasks or SMT siblings.

Setup retains universal six-step inversion and adds rejection of overlap under a
flat-address model. Its exact AST specialization and typed-field error model were
updated; the latter trusts the overlap-predicate mapping. This does not resolve
the earlier direct byte-copy verifier discrepancy or establish machine-code timing.

Validation:
- Seven orchestration fault tests pass: expected outcomes, base/harness compilation
  failure, execution timeout, launch failure, taskset error, and deleted/restored source.
- Four Montgomery oracle/setup tests pass, including 1024 limbs, both overlap
  orderings, exact adjacency, and preservation on errors.
- All eight scoped proof stages pass; governance gate and ratchet pass.
- Real GCC O2 smoke: ten 4-limb binaries built against pinned dudect; all ten were
  intentionally timed out at 0.01 seconds. Outcomes survived and diagnostic failed.
  This validates orchestration, not absence/presence of statistical leakage.
