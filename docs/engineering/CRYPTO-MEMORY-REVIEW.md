# Cryptographic experiment memory review

Reviewed 2026-10-13 by Codex (automated code review); owner: dlworrell.
Scope: sixteen memory primitive calls in crypto_pipeline/crypto_multiplier.c.
This review concerns buffer bounds, not timing or cryptographic certification.
The prior twelve-call review is preserved in repository history; this revision
reassesses calls whose source fingerprints or ordinal identities have changed.

## Preconditions and bounds

Public array entry points validate non-NULL pointers and counts in 1..1024
before any VLA or memory access. Private helpers receive validated bounded
counts. The header specifies exact array capacities and disjoint output/input
storage; read-only inputs can alias each other. Pointer-only APIs cannot verify
allocation size. Supplying a short buffer or forbidden overlap violates the
caller contract. In-place reduction is explicitly documented.

The ordinary product needs 2*n limbs. Odd Karatsuba calls n>16 pad to n+1, at
most 1024, in private pa/pb/pr arrays. Even calls split at h=n/2, with h+1-limb
midpoint sums and 2*(h+1)-limb cross product. Every recursive operand is fully
initialized. Reconstruction accesses only the 2*n-limb result. Recursion shrinks
after any one-limb padding step.

NTT uses d=4*n<=4096 digits and power-of-two L>=2*d, L<=8192. Its fa/fb arrays
hold L uint32_t elements and first holds 2*d. Transform arrays require up to
96 KiB of stack, plus caller and function overhead. Recursive arithmetic and
other local arrays also need sufficient stack; this code targets normal hosted
thread stacks, not tiny embedded stacks.

All memset calls below initialize ordinary working storage. None is relied upon
for secret erasure. Explicit wiping uses a volatile byte loop, but not all
multiplication or transform intermediates are erased; see the arithmetic notes.

## Reviewed primitive calls

| Site | Bound / overlap invariant |
|---|---|
| Comba output initialization | The validated output contract provides 2*n limbs, matching the fill count. Internal calls provide equivalent bounded local segments. |
| REDC memmove | Source [n,2*n) and destination [0,n) lie within the 2*n-limb product allocation. memmove permits overlap. |
| REDC upper-half zeroing | product+n has n remaining limbs, exactly the fill length. |
| Odd Karatsuba pa copy | pa has n+1 limbs; n source limbs are copied, then the final limb is explicitly zeroed. |
| Odd Karatsuba pb copy | Same bound for pb; local destination is distinct from caller source. |
| Odd Karatsuba result copy | pr has 2*(n+1) initialized limbs; exactly 2*n go to the contracted output. |
| NTT output initialization | Output has 2*n limbs and the fill matches. |
| NTT residue copy | first has 2*d uint32_t elements; fa has L>=2*d initialized elements; arrays are distinct. |
| Long-division remainder initialization | r has n limbs, passed either as a private n-limb array or a contracted output; fill is n limbs. |
| Inverse modulus copy | v is an n-limb VLA; mod contract supplies n readable limbs, distinct from local v. |
| Inverse x1 initialization | x1 is n limbs, matching the fill. |
| Inverse x2 initialization | x2 is n limbs, matching the fill. |
| Inverse result copy | Either selected local coefficient has n initialized limbs; output contract supplies n writable limbs. No output copy occurs on failure. |
| Blinding factor initialization | k is n limbs; zero-fill initializes also the unused high limbs when the modulus has leading zero limbs. |
| Blinded-value copy | value and caller blinded output each have n limbs; value is initialized by successful ordinary modular multiplication. |
| Factor copy | k and caller factor output each have n limbs and are disjoint. Both output copies occur only after successful sampling. |

## Regression evidence

`make -C crypto_pipeline bounds-sanitize` runs ASan/UBSan with recovery disabled,
invalid input/count checks, boundary lengths, output sentinels, and selected
mathematical regressions. `python3 crypto_pipeline/test_arithmetic.py` compares
products/reductions/inverses with independent Python integers, including maximum
sizes. GitHub CI runs these alongside clang-tidy and Valgrind smoke execution.
These checks complement the per-call invariants; they are not exhaustive proofs.

See [arithmetic notes](CRYPTO-ARITHMETIC.md) for repaired mathematical defects,
API migration, NTT coefficient bounds, benchmarks, and remaining security work.
