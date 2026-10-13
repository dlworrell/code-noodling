# Montgomery constant-flow candidate: scope and evidence

Status: experimental candidate; NOT cleared for secret use. A local extended
16-limb timing run crossed dudect's threshold. This signal remains unresolved.
A green deterministic CI job is not permission to ignore the timing evidence.

## Threat model and API

`crypto_montgomery_ctx_init` accepts a PUBLIC odd modulus >1 and public limb
count (1..1024). Setup may branch on that public data. The initialized context
must remain immutable. This API does not cover a secret modulus such as a private
RSA CRT prime. Addresses, array capacities, and limb counts are also public.

`crypto_montgomery_mul_ct` accepts secret a,b with 0<=a,b<m and computes
`a*b*R^-1 mod m`, R=2^(64*n), in n output limbs. Capacity/pointer errors are
checked before touching secret data. Reducedness is a caller precondition and
is NOT checked with a secret-dependent branch or error. If conversion/reduction
is required upstream it must independently satisfy the same security model.
The old dispatcher and variable-time inverse are not part of this path.

The C candidate uses fixed-count product and REDC loops, a retained top carry,
widened borrow arithmetic, and masked final selection. Addresses depend only on
public loop indices. It wipes its t and difference arrays with volatile stores;
this does not guarantee elimination of every register spill or caller copy.
It has no secret-dependent allocation, division, or table lookup in the source.
This is a control-flow discipline, not a portable machine-level timing proof.

Three memory primitives are reviewed: initialize the exact context object,
copy n<=1024 public modulus limbs into its 1024-limb array, and initialize an
exact 2*n-limb local product array. All caller arrays must satisfy their actual
allocation contracts; a supplied capacity cannot prove that allocation exists.

## Deterministic checks

`test_montgomery_ct.py` compares 600 reduced-operand cases against Python integer
Montgomery arithmetic, including small moduli in wide arrays, full-width moduli,
carry cases, odd lengths, and 1024 limbs. It also checks setup-error preservation.

`test_montgomery_ct.c` exercises buffer guards, public errors, known products,
and sizes 1,2,4,16,32,64,1024. With CT_TAINT_TEST, both secret operand arrays are
marked undefined using Valgrind client requests before the candidate call.
Results are declassified only after it returns. The harness asserts that it is
actually running under Valgrind. A deliberately secret-dependent loop must
produce Valgrind exit code 99; an unrelated crash does not count as detection.

`run_constant_flow.sh` checks GCC/Clang at O2/O3 in GitHub Actions. It retains
compiler/CPU metadata, object disassembly, binary hashes, oracle/Valgrind logs,
and sanitizer results as workflow artifacts. Taint analysis detects exercised
secret-dependent branches/addresses; it does not model power, EM, speculative
execution, operand-dependent instruction latency, or all possible paths.

## Timing experiments

`dudect_montgomery.c` uses upstream dudect pinned to
`dc269651fb2567e46755cfb2a13d3875592968b5`. Each batch requests 10000 measurements;
the first batch warms up. Input classes are randomized. The public modulus is
R-1, and fixed inputs are either zero or m-1, compared with random reduced inputs
whose low bit is cleared. This is not exhaustive input-distribution coverage.
The callback observes a result byte to retain the computation.

`run_dudect.sh` runs 101 batches per class/size on one CPU, stops with a nonzero
exit code on detected leakage, and validates a deliberate timing-leak control.
It is available through manual workflow_dispatch. Shared-runner measurements
are exploratory evidence; repeat on controlled target hardware before drawing
security conclusions. The default deterministic CI matrix does not run timing.

Initial local GCC 13.3, O2, x86-64 Xeon Platinum 8370C, virtualized-runner results:

| Limbs | Fixed class | Batch budget | Final reported abs(t) | Outcome |
|---:|---|---:|---:|---|
| 4 | zero | 21 | 2.06 | No threshold crossing |
| 4 | m-1 | 21 | 2.09 | No threshold crossing |
| 16 | zero | 21 | 8.04 | Elevated, below threshold |
| 16 | m-1 | 21 | 8.55 | Elevated, below threshold |
| 32 | zero | 21 | 2.22 | No threshold crossing |
| 32 | m-1 | 21 | 3.00 | No threshold crossing |
| 16 | zero, CPU 0 | 101 | 5.11 | No threshold crossing |
| 16 | m-1, CPU 0 | 101 | 10.11 | LEAKAGE EVIDENCE at batch 77 |
| 4 | deliberate leak, CPU 0 | 21 | 162.66 | Control detected at batch 3 |
| 16 | secret-independent baseline, CPU 0 | 101 | 1.62 | No threshold crossing |

Raw local timing logs, compiler/CPU metadata, and code hashes are retained in
[evidence](evidence/montgomery-local-20261013.json). The reviewed object
[disassembly](evidence/montgomery-gcc13-O2.asm) is also retained.

Pinned dudect uses a moderate threshold of 10. Passing means only no crossing
in the bounded run. Do not discard the failed run or retry until green. The secret-independent baseline always multiplies public 1 by public 1,
regardless of the sampled secret class; its clean result does not exonerate the
secret-input candidate. The
16-limb result needs diagnosis with controls and independent hardware; it may
reflect a real hardware-level effect or an experimental confounder. It is not
justified to attribute it to noise without further evidence.

## Assembly review and remaining work

The local GCC 13.3 O2 object uses integer mul/imul and add/adc for arithmetic,
setb/setae and bitwise operations for final subtraction/selection. Reviewed
conditional branches concern public pointer/capacity validation, loop indices,
stack probes, and stack canary checks. No division or secret-indexed addressing
was observed. This manual inspection is limited to that exact build, not all
compiler versions or architectures. CI retains disassembly for additional builds.

Next: resolve the timing signal on controlled hardware, review each target's
machine code/instruction timing, and obtain independent review. Constant-time
inversion is a separate milestone; the existing binary-GCD inverse remains
variable-time. No whole-library, whole-protocol, or physical side-channel
security claim is made by this candidate.

References:
- https://github.com/oreparaz/dudect
- https://github.com/agl/ctgrind
- https://mbed-tls.readthedocs.io/en/latest/kb/testing/testing-constant-flow/
- https://bearssl.org/constanttime.html
