# Cryptographic prototype memory review

Reviewed 2026-10-13 by Codex (automated code review); owner: dlworrell.
Scope: the twelve memory primitive calls in crypto_pipeline/crypto_multiplier.c.
This is not a cryptographic correctness or side-channel certification.

## Enforced and caller obligations

Every limb-array entry point rejects NULL arrays and counts outside 1..1024
before allocating a VLA or accessing array storage. Invalid inputs leave output
unchanged and set errno to EINVAL (the blinding function also returns -1).
The cap bounds all byte counts: n*8 <= 8192, 2*n*8 <= 16384, and
4*n*8 <= 32768 bytes. Recursive scratch plus nested local arrays require adequate
stack space; applications with small thread stacks must supply a larger stack.

The legacy C interfaces cannot infer allocation capacity from a pointer. Callers
must provide valid aligned arrays of the lengths documented in the header.
Outputs must be disjoint from input arrays; read-only inputs may alias each other.
The explicitly in-place reduction operations modify their own product/r array.
These requirements also apply to FFI callers. An undersized allocation or
forbidden overlap remains a caller contract violation, not a checked error.

## Per-call review

All memset calls are ordinary initialization, not secret erasure.

| Call | Bound / overlap argument |
|---|---|
| Comba result initialization | Caller supplies 2*n output limbs. Validated n bounds the identical zero-fill byte count. Recursive callers provide 2*n limbs in result/scratch segments. |
| Reduction memmove | Caller supplies 2*n product limbs. Source [n,2*n) and destination [0,n) are in that allocation. memmove permits overlap. |
| Reduction upper-half initialization | Destination [n,2*n) is the upper half of that same product allocation. |
| Inverse input to u | u is n limbs; input is contractually n readable limbs; u is a distinct local array. |
| Inverse modulus to v | v is n limbs; mod is contractually n readable limbs; v is a distinct local array. |
| Inverse x1 initialization | x1 is exactly n limbs; n was checked before VLA creation. |
| Inverse x2 initialization | x2 is exactly n limbs; n was checked before VLA creation. |
| Inverse x1 to result_inv | Source is local n-limb x1; caller supplies n writable limbs disjoint from locals. This is memory-safe under the contract but the inverse algorithm is unfinished. |
| Unblinding final copy | Source intermediate_product is 2*n limbs; only its low n limbs are copied to the n-limb output. |
| Blinding final copy | Source intermediate is 2*n limbs; only n limbs are copied to the n-limb output. Truncation is an arithmetic issue, not an out-of-bounds copy. |
| Karatsuba scratch initialization | scratch is 4*n limbs and memset uses sizeof(scratch). Recursive helper is now private. For even n>16 it reserves 2*n limbs and recurses at n/2, so S(n)=2*n+S(n/2)<4*n; odd sizes use Comba instead of dropping a high limb. |
| Dispatcher temporary to result | temp_prod is 2*n limbs; result contract requires 2*n writable limbs; the source is a distinct local array. |

## Evidence and limits

Run `make -C crypto_pipeline bounds-sanitize` for instrumented regression tests.
They cover rejected counts 0, 1025, SIZE_MAX; representative NULL arguments;
and sizes 1,16,17,31,32,33,63,64,65,1023,1024 through all limb-array entry points.
Output sentinels supplement AddressSanitizer/UndefinedBehaviorSanitizer checks.
The workflow runs these tests with sanitizer recovery disabled.
Existing four-limb smoke cases also run under Valgrind.

Approval of a memory primitive's invariant does not approve its surrounding
arithmetic. Known open correctness/security problems include:

- Multiprecision inverse is a placeholder returning 1, so unblinding is unfinished.
- Karatsuba drops midpoint-addition carries; borrow and carry propagation need
  mathematical repair and independent big-integer oracle tests.
- Montgomery reduction can drop a carry beyond 2*n limbs and its final
  subtraction can overflow mod[i]+borrow. Its carry loop is data-dependent.
- The NTT fallback branch does not perform modular reduction.
- Blinding discards the upper product and does not return the blinding factor.
- The sample Python binding encodes only two of its four limbs.

These are open defects, not waived or resolved by this ledger. The prototype
must not be described as production-ready or side-channel safe. A future checked
API should take explicit capacities and return status instead of relying on errno.
