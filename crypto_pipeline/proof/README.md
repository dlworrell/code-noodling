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
| Negative controls | Four source leaks rejected and a dropped-high-carry arithmetic counterexample found | Validates specific checker behaviors, not soundness of the checkers themselves |

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

1. Prove initialization completeness and context setup/inverse correctness.
2. Connect actual limb arrays and C word operations to packed integer state;
   instantiate and compose row/carry/REDC loop invariants through all 16 limbs.
3. Derive the functional output theorem from that refinement, rather than
   treating independently proved arithmetic lemmas as a proof of the C.
4. Independently audit or replace the custom source checker, and establish
   preservation to the selected executable/compiler configuration.
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
