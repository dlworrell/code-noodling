<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:d25da695-2ac0-4f8d-94e8-dbf5b0faebae
-->

# Scoped Montgomery proof work: 16 limbs

**Status: proof fragments and source checks, not an end-to-end verified
implementation. NOT CLEARED FOR SECRET USE.** The recorded physical timing
signal remains unresolved. Passing this workflow does not change that status.

## Contract and intended theorem

Let B=2^64, n=16 and R=B^16. Public inputs are an immutable initialized context,
its odd modulus 1<m<R, its inverse satisfying m[0]*inverse = -1 mod B, array
addresses, allocation capacities, and alias relationships. Secret a,b satisfy
0<=a,b<m. Arrays are valid, aligned and initialized; out is disjoint from a,b and
ctx. The read-only inputs a,b may alias. Stack space is sufficient.

The eventual combined theorem is: execution is defined, terminates, produces
0<=z<m with z*R congruent to a*b modulo m, and produces the same leakage trace
for every pair of valid secret operands with identical public state. A trace
contains source control decisions and logical memory addresses/access widths,
not secret loaded/stored values. No secret modulus, speculative, physical power,
EM, hardware instruction-latency, or whole-protocol claim is included.

## What runs now

| Check | Result established | Trusted boundary / limitation |
|---|---|---|
| CBMC `memory16.c` | Bounds, pointer validity/overflow, signed overflow, shift/division checks, return/errno, complete loop unwinding at n=16 | Actual production C is included directly; CBMC/GCC preprocessing/library models are trusted; this does not separately prove every read initialized |
| Source information-flow checker | Every branch, short circuit, loop bound, array index/base, local array size, and memset argument is public under its restricted C model | Custom checker and pycparser are trusted, not formally verified; assumes defined execution; no compiler or machine-code theorem |
| Z3 arithmetic obligations | 14 universal word/packed-state/REDC/subtraction lemmas | Standalone mathematical lemmas; correspondence to every C step and complete invariant composition remain open |
| Initialization abstract execution | 1,512 array reads are initialized; all 16 output limbs written; 32+16 scratch limbs zero on return | Actual C AST, secret values abstracted to unknown; trusted restricted interpreter; valid 16-limb public path only |
| Setup inverse lifting | Six recognized C iterations establish the negative inverse modulo 2^64 | Eight SMT obligations plus an exact AST specialization; context copy/error effects remain open |
| Local C arithmetic refinement | Five source-bound word-step obligations: multiply/split, carry propagation, borrow/narrowing, mask/selection | Exact kernel AST must match; whole-loop induction is still open |
| Extracted REDC carry block | CBMC checks the 1025-bit suffix identity, unchanged lower words, terminal carry bit, and safety/unwinding at all 16 offsets; three carry-loss mutations fail only the arithmetic property | Actual adjacent AST statements are emitted under trusted extraction; incoming carry is an arbitrary word and initial high is zero; MAC entry conditions and existing outer high remain composition obligations |
| Negative controls | Four flow leaks, five initialization/wipe defects, two broken setup recurrences, a stale arithmetic mapping, and the dropped-high-carry arithmetic counterexample are detected | Validates specific checker behaviors, not soundness of the checkers themselves |

The memory harness permits arbitrary modulus, inverse and operand bits, a
stronger input domain for these memory properties than the arithmetic contract.
It fixes count/capacities at 16 and uses nondeterministic independent operands,
with a public nondeterministic choice of a==b. It does not cover invalid-pointer
error paths or every other limb count. `--unwind 34` covers the longest explicit
32-iteration loop, and **unwinding assertions must succeed**. No assumptions
about operand values are used. Unsigned wraparound is intentional and is not
forbidden by an unsigned-overflow check.

The source checker computes a conservative fixed point of secret dependencies
through scalar assignments. All operand/output/scratch-array contents are
classified secret, even when a particular element happens to be zero. Context
contents and pointer identities are public. It rejects secret dependencies at
all observable control/address uses. Unsupported syntax, calls, pointer
operations, context writes, and declassification are rejected. This is a
structural information-flow argument, not a sampling test: with equal public
state, each accepted operation preserves equality of public state and the next
control/address event. It deliberately loses precision rather than guessing
that secret-derived data is public. C undefined behavior and compilation need
separate justification before composing this result into a security theorem.

The arithmetic file negates each stated lemma and requires Z3 to return UNSAT;
SAT or UNKNOWN fails. It emits each SMT-LIB query for inspection/replay. The
negative control must instead return SAT. The quotient lemma establishes the
explicit witness `z*R-T = m*Q`, which implies congruence without asking the solver
to reason about a variable-modulus remainder. The full proof still needs the
C refinement and induction linking packed states to these equations.

## Second milestone: initialization and source-bound refinements

`check_initialization.py` executes the restricted C AST at n=16, representing
all input/context payload words as initialized unknown values. Each scratch
slot starts uninitialized; `memset` initializes exactly the represented byte
range. Array reads check the actual public index, writes mark that slot, and
pointer p follows its actual scratch-array target. Any unknown control/index,
unsupported construct, out-of-range public arithmetic or out-of-bounds access
fails. Return stops execution. Final checks require every output slot initialized
and both scratch arrays zero. Caller buffers/context must themselves be valid
and initialized. This models source array erasure, not register spills or an
optimizer's treatment of secret copies.

Five mutations remove product initialization, leave difference elements unwritten,
leave an output unwritten, remove carry initialization, or truncate a wipe. Each
must fail for an initialization/wipe reason. This checker is a trusted abstract
interpreter, not an independently verified C semantics implementation. The
experimental CBMC uninitialized-local pass did not understand these array writes
and memset well enough to establish this property; it is not credited as a proof.

`prove_setup_inverse.py` matches the full setup function's canonical AST to an
explicit schema, with only initial inverse and iteration count interpreted as
parameters. For odd m[0], the initial inverse 1 is correct modulo 2. Each source
iteration doubles precision: 1,2,4,8,16,32,64 bits. The SMT equations include
separate integer wrap quotients for the unsigned 64-bit product, subtraction,
result product, and final negation. No claim that C executes in unbounded integers
is made. Each lift supplies an explicit divisibility witness. Five iterations
are rejected for insufficient precision; initial inverse zero fails the base
case. Other source-schema changes require re-review. These eight obligations
prove the recognized recurrence and stored inverse, not every setup field effect.

`prove_c_steps.py` similarly binds five local obligations to the exact kernel AST
and records the corresponding C loop bodies. A changed shift is rejected as a
stale mapping. The obligations establish that the multiply-accumulate fits 128
bits, that storing/shifting splits it exactly, that carry propagation yields a
bit, that 128-to-64-bit borrow narrowing matches modular subtraction, and that
the mask selects the intended limb. Their preconditions, including the incoming
carry range and final high/borrow bits, still need to be carried through the
whole loops. A matching source hash alone is not that composition proof.

Direct byte-level context-effects verification remains open. Direct and inverse-abstracted CBMC
attempts exhausted local resources. Z3's array backend returned UNKNOWN; an
alternative CBMC array encoding produced a modulus-copy counterexample. Replaying
that concrete modulus (limb 0 = 1, limb 15 = 128) against the native GCC build did
not reproduce the copy failure. This does not resolve the verifier discrepancy.
Those runs are retained as failed/incomplete attempts in the second milestone's
evidence and are not part of the passing CI claim.

## Third milestone: conditional composition and field contracts

`prove_loop_composition.py` binds the exact kernel AST to packed-state equations.
It checks telescoping product/carry/subtraction rows, all 16 REDC offsets and
coefficient bounds, retained-high bounds, canonical selection, and the final
Montgomery congruence witness. It requires the local arithmetic, inverse, and
initialization and extracted carry certificates. Dropping the final row carry and skipping final
subtraction each produce a concrete counterexample. Static footprint checks
cover the initially zero carry destinations and preservation of cancelled limbs.

These are checked algebraic composition obligations under a **trusted manual
C-to-equation specialization**. Exact AST matching prevents silent source drift;
it does not independently prove the mapping or discharge every C invariant
precondition. This is not an end-to-end C functional correctness theorem.

`prove_context_contract.py` checks nine typed-field properties under explicit
word contracts for zeroing and nonoverlapping copying. Success is scoped to
16 limbs; null pointers, zero/oversized counts and invalid 16-limb moduli cover
error paths. It checks copying, zero tail, stored length/inverse, return/errno,
and unchanged fields on errors. Four model mutations produce counterexamples.
Padding bytes and independent byte-level C semantics are excluded. An additional
explicit-word CBMC attempt exhausted memory; no failed attempt counts as proof.

The runner executes both models in CI and retains their SMT inputs, certificates,
and dependency hashes. Cryptographic security status remains unchanged.

## Fourth milestone: extracted suffix carry contract

`verify_carry_loop.py` reuses the restricted parser for the real
`crypto_montgomery_mul_ct` function and requires its complete canonical AST to
match the reviewed kernel schema. It finds exactly one adjacent `k` propagation
loop and `high += carry` update inside an outer `i` loop. It emits those AST
nodes as a compound, preserving statement terminators and identifiers. The
production C is not changed or replaced with a handwritten loop. The source,
actual headers, generated harnesses, extracted statements, commands, property
results, and negative-control traces are retained with hashes.

For public n=16, each offset i=0..15 is checked separately. Together these
finite cases cover every permitted REDC offset, with every scratch word and
the full-word incoming carry unconstrained. Let s=i+16 and L=16-i. The checked
integer equation, represented without wrap in unsigned 1025-bit arithmetic, is

```
pack(t_after[s:32]) + B^L * high = pack(t_before[s:32]) + carry_in
```

The largest right side is `(2^1024-1)+(2^64-1)`, which fits 1025 bits.
An arbitrary lower-region index establishes preservation of every word below s.
The final carry must be 0 or 1. This includes all-ones suffixes with incoming
carry 1 and the maximum full-word input carry. The harness starts high at zero
and hence measures the newly generated terminal carry; it does not establish
the range or non-wrapping accumulation of the production outer high state.

Unlike the whole-kernel memory harness, these addition-only queries also enable
unsigned-overflow checks: every addition and packed shift in this local contract
must fit its type. Full unwinding uses 33 for the 32-word initialization loop;
bounds, pointer, pointer-overflow, signed-overflow, shift and division checks are
also enabled. Unsigned wrap in other production operations remains intentional.

All sixteen positive queries must return exit 0, JSON `cProverStatus=success`,
and SUCCESS for every reported property. Three controls at row zero drop the
incoming carry, drop the terminal update, or truncate the propagation loop.
Each must return exit 10 and a counterexample trace for exactly
`MATH_INVARIANT`, with all other properties SUCCESS. Error text, malformed JSON,
missing/duplicate property identifiers, unresolved properties, and unrelated
safety/unwinding failures are rejected. A changed source shift is rejected as
a stale mapping. Fourteen driver tests exercise evidence rejection, extraction, timeout retention,
and refusal to overwrite evidence.

The composition stage requires a current passing certificate containing exactly
these sixteen offsets and three controls. The remaining connection from each
MAC iteration into the suffix and from the suffix into the retained outer high
state is still trusted proof glue. This is a local C-block contract at n=16,
not an independently verified frontend or a complete inductive kernel theorem.

Run it independently with pinned tools and a new output directory:

```sh
.tools/proof-venv/bin/python crypto_pipeline/proof/verify_carry_loop.py \
  --cbmc .tools/cbmc/usr/bin/cbmc --output build/carry-16-run-01
.tools/proof-venv/bin/python -m unittest discover \
  -s crypto_pipeline/proof -p test_carry_driver.py
```

The master runner now executes nine proof/check stages; the formal workflow
also executes the driver tests. No new physical timing measurements are implied.

## Reproduce

On Linux x86-64 with Python 3.12, GCC preprocessing, curl and dpkg-deb:

```sh
python3 -m venv .tools/proof-venv
.tools/proof-venv/bin/python -m pip install -r crypto_pipeline/proof/requirements.txt
bash crypto_pipeline/proof/install_cbmc.sh .tools/cbmc
.tools/proof-venv/bin/python crypto_pipeline/proof/run_proofs.py \
  --cbmc .tools/cbmc/usr/bin/cbmc --output build/formal-16-run-01
```

CBMC 6.8.0 is downloaded from its upstream release and SHA-256 checked before
extraction; pycparser 2.23 and z3-solver 4.15.4.0 are pinned. Existing result
directories are refused. The runner records exact commands, source hashes,
solver input hashes and logs; exceptions/timeout/nonzero/UNKNOWN never count as
proof. A completed report has `complete: true` and
`all_scoped_checks_passed: true`, but still lists the open obligations.

GitHub's `Montgomery Scoped Formal Checks` workflow repeats these checks and
retains artifacts even after failure. The source-level model excludes compiler
optimizations; existing GCC/Clang dynamic taint checks remain complementary.

## Remaining proof work

1. Independently verify the C-to-equation specialization and standard memory primitive contracts; audit the initialization and inverse models.
2. Replace trusted composition/footprint glue with independently checked C refinement, including invariant preconditions across iterations.
3. Resolve the direct context byte-array verifier discrepancy; the field model does not resolve it.
4. Establish preservation to the selected executable/compiler configuration.
5. Resolve the empirical timing signal under an appropriate hardware model.

An exploratory CBMC self-composition recorded complete traces, then arbitrary
trace witnesses. It exceeded practical local resources; a multiplication
abstraction attempt also exposed a missing nondeterministic-function model.
These attempts are **not accepted proofs** and are not the CI gate. The smaller
source checker above is the current explicit trusted component. No failed or
incomplete paired-execution check is represented as successful.

References:
- https://www.cprover.org/cbmc/
- https://model-checking.github.io/cbmc-training/cbmc/overview/loop-unwinding.html
- https://github.com/Z3Prover/z3
- https://github.com/eliben/pycparser

### Setup overlap rejection (2026-10-14)

Setup now rejects intersecting context/modulus ranges before reading modulus
payloads, using ordered unsigned address differences on the supported flat-address
target. The caller must still supply valid aligned objects of sufficient size.
The six Newton iterations and the multiplication kernel are unchanged. The exact
setup AST mapping was reviewed and updated for the five inserted statements.
The field-contract model includes an abstract public overlap error predicate;
its correspondence to pointer-to-integer conversion and interval arithmetic is
trusted, not an independently checked byte-memory theorem. Native tests cover
both address orderings, exact adjacency, and unchanged context on rejection.
