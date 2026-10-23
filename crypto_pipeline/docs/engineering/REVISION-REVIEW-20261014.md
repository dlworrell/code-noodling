<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:a81e81d3-394d-48ef-a5f6-559033acbec1
-->

# Review of the October 13 submitted revisions

The latest non-withdrawn submissions were reviewed together against PR #9 at
`2e47782b6d47312bb5ce2bfa393a751776f0c3a6`. They are proposals, not verified
replacements. Some excerpts have transcription damage; other changes alter the
mathematics, API contract or strength of verification. No supplied pass report,
hash, statistical result or claimed root cause was accepted without evidence.
Security status remains **NOT CLEARED FOR SECRET USE**.

## Blocking arithmetic and memory findings

| Submission | Finding and disposition |
|---|---|
| Latest `crypto_montgomery_ct.c` | `t` has 2048 elements. At n=1024 the last reduction row updates `t[2048]`, and final selection reads it again. A reconstructed executable using the submitted operations fails UBSan with `index 2048 out of bounds for type limb_t [2048]`. At smaller n, `t[2*n]` lies in the allocation but is not initialized by the `2*n`-word memset. Keep the existing separately retained high carry. |
| Candidate carry propagation | Updating only the next limb cannot propagate a carry through a run of all-one limbs. The preceding version's direct addition into `high` gives an intermediate overflow the wrong radix weight. Keep propagation through all remaining words. |
| Latest core `secure_montgomery_reduction` | Reintroduces that direct-to-high error and removes the `product < modulus*R` validation. For n=2, m=2^128-1, a=b=m-1, the one-step version returns `0xffffffffffffffff0000000000000001`, while the correct Montgomery result and existing implementation return 1. It also fails at n=16. |
| Candidate borrow | `(sub >> 63) & 1` reads bit 63 of a 128-bit unsigned subtraction, not its underflow. For t=2^63, modulus limb=0, incoming borrow=0, it reports a borrow although the difference is positive. Keep widened comparison against modulus limb plus incoming borrow. |
| Candidate selection | For high=0 and borrow=1, `-(((borrow-high)>>63)^1)` produces all ones and chooses the difference, when the unchanged upper words are required. Keep the proven selection formula and its high/borrow preconditions. |
| Product row carry | The existing `t[i+n] = carry` is correct: earlier row writes stop at i+n-1. The added accumulation is not a fix for this algorithm. |
| Latest `test_bounds.c` | `factor_scratch` has 1026 entries but `reset_output` writes 2050 entries into it. Separate loops must use each array's own size. Removing SIZE_MAX cases hides a contract regression; validation must precede allocation. Keep existing maximum-size coverage. |
| Latest `test_arithmetic.py` | Reduces Montgomery coverage to 128 limbs, ordinary products to 64 and inverse to 16; masks away the upper output limbs, drops rejection cases and wrapper coverage. Its invalid-domain calls use a 2-limb result with n=2 although the multiplication API needs 4. Retain full allocations and existing high-limb/1024-limb assertions. |

## Integration and proof findings

| Submission | Finding and disposition |
|---|---|
| Latest `crypto_multiplier.h` | Contains types only; deletes public prototypes and API contracts. Retain the complete repository header, including capacities, overlap restrictions and variable-time warnings. |
| Latest core C text | `subtract_mod(... const limb_t m, ...)` and `execute_blinded_multiplication(... const limb_t a, ...)` lose pointer stars. Two comments begin `/` instead of `/*`. These are compilation blockers, not numerical errors. The preceding draft also references undeclared `m` in `crypto_blind`. |
| Fixed maximum arrays | Bounded allocations are not a stack-safety theorem. Proposed NTT arrays reserve 160 KiB per call; length is already at most 8192 for n<=1024. Maximum-sized recursive Karatsuba arrays can multiply stack costs. Keep current allocation policy pending measured stack budgets or an explicit workspace API. |
| `subtract_mod` rationale | In-place subtraction supports r==a. When x<y, the n-word subtraction is R+x-y; adding m intentionally drops the carry R, leaving x-y+m in [0,m). Discarding that final carry is correct, unlike the carry retained for `(x+m)/2`. |
| Latest `check_initialization.py` | Broken quotes/indentation and `2n` mutation anchors. Its UnaryOp handler rejects every expression except `sizeof(limb_t)`, then tries to evaluate that type as a variable. It must return 8 for sizeof and separately handle ordinary unary operations. It also rejects maximum-sized arrays (>32), and silently permits absent scratch objects in the final wipe check. Retain the working interpreter. |
| Latest `prove_setup_inverse.py` | Matches every top-level `for` whose counter is i, including the public modulus scan. Normalizing both loop bounds changes the wrong loop and fails the schema. Match the Newton recurrence structurally and uniquely; do not update hashes simply to accept a changed program. |
| Latest `prove_loop_composition.py` | Damaged ending (`not_claimed` quote, indentation, `if name == 'main'`). Uses Python `if z >= M` on a symbolic Boolean; this raises instead of generating a solver condition. `choose_sub` is absent and its required equation is silently replaced by True. Retain explicit `z3.If` and the congruence witness premise. |
| Proposed induction template | Compares a 64-bit extract with a 128-bit constant, causing a sort error. Even with widths fixed, MAC carry is not generally <2: all operands B-1 yield carry B-1. A separate addition-only propagation step has a bit carry. Widen to 129 bits or use integer bounds to establish absence of 128-bit overflow; bit-vector extraction alone does not prove that. |
| Latest context proof | Removes the source digest prerequisite check. A passing certificate for stale source must not discharge a dependency. Fresh Solver objects already isolate assertions; reusing symbolic names across separate solvers is not cross-contamination. Retain source binding and all error-path/mutation checks. |
| Local arithmetic/schema edits | Canonical AST generation already normalizes source formatting. Whitespace-stripped alternate hashes do not establish correspondence of new statements to old equations. Explicit quotient helpers can be valid but division by constant B was not nonlinear division, and adding tautological equalities does not strengthen a proof. |
| Latest source-flow checker and restricted syntax | Additional secret labels largely duplicate the existing nonempty-dependency rule. Regex mutation anchors need a unique match and syntax preservation. Struct validation must check `->`, not merely object/field names. Expanding allowed syntax requires matching interpreter and flow semantics; unsupported operations should continue to fail closed. |
| Latest `memory16.c` | One contiguous allocation with fixed offsets does not explore arbitrary alignment. Object-level checks can miss crossing logical subarray boundaries inside the same pool. Oddness excludes modulus 0 but not 1. Capacity error paths are useful additional checks; they should not replace the original separate-object, arbitrary-payload memory harness. |
| Alignment patch | `Select(array,i)>=0` constrains stored data, not pointer alignment (and is signed for bit-vectors). `i>=total_limbs` is unreachable inside `range(total_limbs)`. It does not resolve the earlier context-copy discrepancy. |
| Unwind claims | 34 covers the original n=16 explicit loops only if unwinding assertions pass. Full maximum-array wipes add loops of 1024/2048 iterations. There is no universal two-iteration safety rule that proves new loop bounds. |
| Latest runner | Explicit timeout reporting is useful and has been integrated. Keep hashing memory16.c, run_proofs.py, requirements and installer as well as other source; the proposed whitelist omits these. Missing evidence files must not be silently ignored. File writes are not atomic merely because they occur in finally. |

## CI, timing and evidence findings

| Submission | Finding and disposition |
|---|---|
| Latest Makefile | Refers to unsupplied `test_multiplier.c`, omits the candidate from the shared library, and builds the same harness for smoke/bounds. `bounds-sanitize` compiles without running, unlike the existing target. Keep the existing makefile and distinct harnesses; mark shared phony. |
| Latest validation workflow | `make clean` before sanitizers deletes the shared library and it is not rebuilt before Python loads it. Whole-library division rejection conflicts with intentionally variable-time NTT/modular operations. Grep absence is not a constant-time proof. Taint driver/control coverage must use the candidate harness and verify the deliberate control, not a missing test_multiplier.c. |
| Candidate C test and dudect driver | Duplicated context structs/prototypes should use crypto_montgomery_ct.h. Calls inside assert disappear with NDEBUG. The latest taint test correctly returns to a public setup modulus; earlier versions wrongly tainted public modulus and left the copied context tainted. Keep n=1 zero-capacity tests. |
| Latest smoke test | Oversized buffers hide potential overrun evidence and printed results are not assertions. Integrated exact independent expected values, upper-half zero checks, errno and inverse identity checks into the existing smoke test; reject NDEBUG for this target. |
| Python binding | Existing ordinary modular semantics and signed-input reduction are sound. Packing changes do not establish constant time or remove all Python allocation overhead. Keep explicit ctypes signatures and a real wrapper import test; do not swallow ImportError. |
| Timing driver | `% (UINT64_MAX-1)` biases the input distribution; changing fixed/random classes changes the experiment. Repeated UINT64_MAX-1 limbs are not the integer m-1 except at n=1. Chunk-size multiples do not alone prove base-pointer alignment. Keep established experiment classes for comparisons; treat new classes as separately named experiments. |
| Timing runner | CPU affinity is not physical-core isolation. Default CPU 2 may be unavailable. Restore dirty-vendor checks, minimum batch checks, compiler version, incremental reports, exact object linkage and timeout outcomes. The latest 10x450s limit exceeds a 45-minute job in the worst case. A threshold non-crossing is not a security pass. |
| Shell timing runner | `[ -f /sys/.../cpu*/... ]` can expand to multiple arguments. Inspect selected CPU or iterate paths. set -e exits on the first detected signal, preventing later matrix evidence/hash collection. Keep the diagnosis runner that preserves outcomes. |
| Valgrind suppressions | No matching observed external-library defect was provided. Do not add blanket suppression patterns to make taint tests pass; negative control must remain detected for the intended reason. Python suppressions are unrelated to the native C test executable. |
| Specialized modulus | Supplied limbs do equal 2^1024-10933. The supplied inverse `0xD40051833AFB3935` is wrong: low-word product is `0x0e66d68a14fddb87`, not UINT64_MAX. Correct negative inverse is `0x8128b823676cf39d`. No primality proof was supplied; this number is not Proth form (its m-1 has only one factor of two). Specializing one modulus narrows the theorem and does not resolve a byte-model discrepancy for arbitrary moduli. |
| Supplied JSON report/audit prose | Claimed source hashes, timings and passes are not execution evidence. Prior phase-3 composition has 60 positive obligations plus two controls; 14 describes separate arithmetic lemmas. UNKNOWN, exhausted resources and non-reproduced counterexamples do not establish alignment as a cause or a verifier bug. Retain the unresolved status. |
| One-command gateway | Bash triple-quoted prose is a command, not a Python docstring. Path assumptions, actual script names, mandatory tools, distinct artifact directories and failure recording must be resolved before adoption. |

## Accepted changes and validation

- Added wiping of the active temporary subtraction limbs in the existing core.
- Strengthened the existing smoke test with exact dense/sparse oracle results,
  upper-half zero checks, errno and modular inverse identity assertions.
- Marked shared as phony without deleting evidence from clean.
- Recorded proof subprocess timeouts as explicit failures; preserved hashes and
  unique output directories.
- Kept working carry propagation, public-context contract, full-size tests,
  schema/dependency checks and separate C memory objects.

Local GCC build and smoke test pass. Existing nine arithmetic tests and two
candidate tests pass, including their 1024-limb cases. ASan/UBSan boundary tests
pass with `ASAN_OPTIONS=detect_leaks=0`: the environment's LeakSanitizer fails
while inspecting `/proc/.../task`, so no local leak-sanitizer pass is claimed.
The proposed candidate reconstruction fails UBSan at n=1024 as described above.
Full formal package and governance results are recorded in the accompanying
review evidence. No new timing clearance or independent C refinement is claimed.
