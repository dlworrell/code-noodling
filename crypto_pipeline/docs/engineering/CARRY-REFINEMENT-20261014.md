# REDC suffix carry refinement at 16 limbs

The proof package now checks the actual AST-extracted suffix carry loop and
adjacent high update, eliminating a handwritten copy of the loop. The complete
recognized kernel must match the reviewed schema. Generated harnesses use the
real headers with explicit width/unsignedness assertions and an unsigned
1025-bit packed representation.

For each of the sixteen REDC offsets, CBMC 6.8.0 checks all initialized scratch
word values and every full-word incoming carry. The packed suffix plus terminal
carry equals the original suffix plus the incoming carry. Every lower word is
preserved, the final carry is a bit, all safety checks pass, and unwinding is
complete. Each positive query has 46 successful properties, including explicit
unsigned-overflow checks for this addition-only block.

Three mutations at row zero drop the incoming carry, drop the terminal update,
or truncate propagation after one limb. Each yields exit 10 with a concrete
counterexample for MATH_INVARIANT only; all other properties succeed. The compact
counterexamples in evidence/carry-refinement-20261014.json were replayed using
independent Python integer arithmetic and match the CBMC expected/actual values.
The workflow retains full traces and generated inputs as artifacts.

The driver rejects malformed JSON, old result schemas, wrong exits, missing or
duplicate properties, unresolved results, missing traces, and unrelated safety
failures. Fourteen unit tests pass, including timeout retention and refusal to
overwrite evidence. All nine integrated proof/check stages and governance pass
locally. Final retained certificates were checked against their recorded hashes.

The composition stage requires the current passing carry certificate and all
sixteen positive offsets plus three negative controls. This discharges a local
conditional suffix contract with initial high=0. It does not yet establish that
actual MAC iterations supply the incoming carry precondition, or connect the
new terminal carry to the existing outer high state. The extractor, harness and
remaining C-to-equation composition remain trusted. No production arithmetic was
changed. The context byte-copy discrepancy and hardware timing signal remain
open; no new timing experiment was run.
