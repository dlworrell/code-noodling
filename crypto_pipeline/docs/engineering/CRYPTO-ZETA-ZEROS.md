<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:69b21e2b-f4e3-4b4c-ab25-5a34a78de4b6
-->

# Zeros of the Riemann zeta function on the critical line, to T = 10^4

Status: Experimental (slice 2: Gabcke's bound confirmed)
Scope: `crypto_pipeline/` (governed project scope, Catylist ADR-0004)
Files: `crypto_zeta_ball.h`, `crypto_zeta_ball.c`, `crypto_zeta.h`,
`crypto_zeta.c`, `zeta_verify.c`, `tests/test_zeta.c`, `tests/test_zeta.py`

## Result

`./zeta_verify 10000 100 1000` proves, using only the sources marked
*confirmed* below, that every zero of ζ(s) with 0 < Im s ≤ 10000.88 (the Gram
point g₁₀₁₄₃) lies on the critical line Re s = ½ and is simple, and that

| T | N(T) | Published (OEIS A072080) |
|---|---:|---:|
| 100 | 29 | 29 |
| 1000 | 649 | 649 |
| 10⁴ | 10142 | 10142 |

The run uses the Euler–Maclaurin formula for every value of ζ (method `em`,
13,045 evaluations, about 4 minutes on one core of this container). With
`--rs`, the Riemann–Siegel formula with Gabcke's remainder bound (confirmed
from his dissertation, see [Sources](#sources-and-their-status)) proves the
same result in about 8 s. The two methods are independent ways to the same
counts.

This is a proof only up to T = 10⁴. It is a first slice: a small,
independent, rigorous checker.

## What is proved, and how

Hardy's function Z(t) = e^{iθ(t)} ζ(½ + it) is real for real t, with
|Z(t)| = |ζ(½ + it)|. If Z(a) and Z(b) have opposite signs, ζ has a zero on
the critical line between a and b. Counting sign changes gives a lower bound
on the number of zeros on the line; Turing's method gives an upper bound on
the number of all zeros, N(t₀). When the two agree, every zero up to t₀ is on
the line and simple.

### 1. Every value is an enclosure

`crypto_zeta_ball.c` implements midpoint–radius ("ball") arithmetic on MPFR.
A ball [m ± r] stands for every x with |x − m| ≤ r, and each operation
returns a ball that contains f(x) for every x in its input balls:

- *propagation*: an explicit derivative bound over the input ball, e.g.
  |xy − m_x m_y| ≤ |m_x| r_y + |m_y| r_x + r_x r_y,
  |log x − log m| ≤ r / (m − r), |e^x − e^m| ≤ e^m (e^r − 1),
  |sin x − sin m| ≤ r;
- *rounding*: MPFR rounds every result correctly (to nearest) and reports
  through its ternary return value whether the result is exact (MPFR
  manual). When it is not, one unit in the
  last place of the rounded midpoint is added, which bounds the error even
  when rounding crosses a power of two.

Radii are 64-bit MPFR numbers rounded upward; a quantity that is subtracted
in radius arithmetic is first bounded in the safe direction. A sign is used
only when the ball excludes 0. Doubles appear only as exact sample points
and in heuristics that choose where to sample, which cannot make a wrong
result look proven.

### 2. θ(t) from Stirling's series

θ(t) = Im log Γ(¼ + it/2) − (t/2) log π, with log Γ the branch that is real
on the positive axis (θ(0) = 0). With z = ¼ + it/2:

- if t < 80, shift: log Γ(z) = log Γ(z + 40) − Σ_{k<40} log(z + k), where
  Im log(z + k) = atan((t/2)/(k + ¼)) (principal branch, Re > 0);
- log Γ(w) = (w − ½) log w − w + ½ log 2π + Σ_{k=1}^{19} B₂ₖ / (2k(2k−1) w^{2k−1}) + R
  (DLMF 5.11.1), with |R| at most sec^{40}(½ ph w) times the first neglected
  term, |B₄₀| / (40·39·|w|³⁹) (DLMF 5.11(ii)). For Re w > 0,
  sec²(½ ph w) = 2|w| / (|w| + Re w), which the code bounds from the balls.

Bernoulli numbers are computed exactly as GMP rationals (Akiyama–Tanigawa).
At |w| ≥ 40 the remainder is below 10⁻⁴⁰. θ is evaluated on balls, so it can
also be enclosed over a whole interval of t (used by Turing's method below).

The asymptotic expansion θ(t) ≈ (t/2) log(t/2π) − t/2 − π/8 + 1/(48t) +
7/(5760t³) is **not** used in any proof; it places approximate Gram points (a
heuristic). Gabcke (1979, Einleitung (4), (5), p. 2) states it with one more
term and an explicit remainder: θ(t) = (t/2) log(t/2π) − t/2 − π/8 + 1/(48t) +
7/(5760t³) + 31/(80640t⁵) + Rθ(t) with |Rθ(t)| < 1/(3322 t⁷) for t ≥ 10.
`test_theta` checks the Stirling value against that, an independent route, at
t = 10, 20, 50, 200, 1000, 7005 and 10⁴ (agreement within 1/(3322 t⁷) plus the
radius), and checks that dropping the 31/(80640t⁵) term breaks the agreement.

### 3. ζ(½ + it) by Euler–Maclaurin

With f(x) = x^{−s} in DLMF 2.10.1 (a = N, upper limit → ∞, first for
Re s > 1, then by analytic continuation to Re s > 1 − 2M):

ζ(s) = Σ_{n<N} n^{−s} + N^{1−s}/(s−1) + N^{−s}/2
      + Σ_{k=1}^{M} B₂ₖ/(2k)! · s(s+1)⋯(s+2k−2) · N^{−s−2k+1} + E,

E = −∫_N^∞ B̃₂ₘ(x)/(2M)! · s(s+1)⋯(s+2M−1) · x^{−s−2M} dx.

The periodic Bernoulli function satisfies |B̃₂ₘ(x)| ≤ |B₂ₘ| (DLMF 24.2.11
and 24.9.1), so with σ = ½

|E| ≤ |B₂ₘ|/(2M)! · |s(s+1)⋯(s+2M−1)| · N^{½−2M} / (2M − ½).

The code takes N = ⌊t/π⌋ + 10 and adds terms until this bound is below 2⁻⁸⁰
(it fails if the bound never falls below 2⁻⁴⁰). Then Z(t) = Re(e^{iθ}ζ), and
the enclosure of Im(e^{iθ}ζ), which is exactly 0, must contain 0; otherwise
the evaluation fails with `ZZ_ERR_INCONSISTENT`. Since ζ is computed without
θ, this check catches an error in θ (or ζ) larger than the enclosure radius.

### 4. Z(t) by Riemann–Siegel with Gabcke's bound

For t ≥ 200, with τ = t/2π, N = ⌊√τ⌋ and p = √τ − N:

Z(t) = 2 Σ_{n≤N} n^{−½} cos(θ(t) − t log n) + (−1)^{N−1} τ^{−¼} C₀(p) + R₀(t),
C₀(p) = cos(2π(p² − p − 1/16)) / cos(2πp),

with |R₀(t)| < 0.127 t^{−¾} for t ≥ 200: Gabcke (1979), Satz 3.2.2 (b),
p. 55, also stated in the Einleitung, (6) and (8), p. 3. The bound is in t,
not τ. Gabcke writes the formula in D. H. Lehmer's form (Einleitung (1), (2),
p. 2) with a = √(t/2π) (so a^{−½} = τ^{−¼}), N = ⌊a⌋, z = 1 − 2(a − N) =
1 − 2p and C₀(z) = F(z) = cos(π/2·(z² + ¾)) / cos(πz). Substituting z = 1 − 2p
gives π/2·(z² + ¾) = 2π(p² − p − 1/16) + π and πz = π − 2πp, so numerator and
denominator each change sign and F(1 − 2p) is exactly the C₀(p) above. The
code adds (127/1000) t^{−¾}, evaluated at the smallest t of the input ball and
rounded up.

Slice 1 used 0.127 τ^{−¾} before the thesis could be read; that is
(2π)^{¾} ≈ 3.96 times Gabcke's bound, so it was sound but looser, and its
results were reported as conditional. With the exact bound, Riemann–Siegel
falls back to Euler–Maclaurin at 2 points up to 10⁴ instead of 6.
`zz_z_rs` refuses t < 200, a ball for √τ that straddles an integer, and a C₀
denominator ball that contains 0 (p near ¼ or ¾); the verifier then falls back
to Euler–Maclaurin. Every 64th Gram sample at t ≥ 200 is also evaluated by
Euler–Maclaurin: its Im check must pass and the two enclosures of Z must
overlap. This spot check guards against defects in the implementation (a
wrong θ or a dropped term), which Gabcke's bound by itself cannot catch.

### 5. Isolating sign changes on Gram intervals

Gram points g_n solve θ(g_n) = nπ. The verifier samples Z at approximate
Gram points g̃₋₁ ≈ 9.67, g̃₀, g̃₁, … (doubles from the heuristic expansion;
a point whose sign cannot be decided is nudged by 10⁻⁴ of the spacing). It
calls g̃_n *good* when (−1)^n Z(g̃_n) > 0 (Gram's law). Between consecutive
good points g̃_a < g̃_b (a Gram block) Rosser's rule predicts b − a zeros; if
fewer sign changes are found, every subinterval of the block is bisected,
round by round (up to 12 rounds), until the block has at least b − a. Every
subinterval is split, not only same-sign pairs: one sign change can hide
three zeros, and an early version that split only same-sign pairs failed at
t ≈ 2667. The checkpoints (and T itself) are sampled exactly, so N(h) is the
number of sign changes ending at or before h. Rosser's rule only guides the
search; the proof is the count in step 6.

`zz_gram_point` separately encloses g_n with a certificate
θ(lo) < nπ < θ(hi), and `zz_z_sign_interval` proves the sign of Z on that
whole enclosure. `test_gram_law` uses them to show Gram's law holds for
n = −1, …, 125 and first fails at n = 126 (g₁₂₆ ≈ 282.45; OEIS A114856 and
A002505).

### 6. Turing's method with Trudgian's constants

Let S(t) = π⁻¹ arg ζ(½ + it), by continuous variation along the segments
2 → 2 + it → ½ + it (Trudgian 2011, (1.1)). Then

N(t) = θ(t)/π + 1 + S(t).                                            (∗)

(∗) is the Riemann–von Mangoldt formula in exact form. *Derivation:* ξ(s) =
½ s(s−1) π^{−s/2} Γ(s/2) ζ(s) is entire, its zeros are the non-trivial zeros,
and ξ(s) = ξ(1 − s) with ξ real on the real axis. By the argument principle
on the rectangle with corners −1, 2, 2 + it, −1 + it and that symmetry,
N(t) = π⁻¹ Δ arg ξ along 2 → 2 + it → ½ + it. Along that path s(s−1)
contributes π, π^{−s/2} Γ(s/2) contributes θ(t), and ζ contributes π S(t).
This derivation was written for this note; it was not compared against a
textbook statement.

Trudgian (2011), Theorem 2.2 (confirmed): for t₂ > t₁ > 168π,
∫_{t₁}^{t₂} S(t) dt ≤ 2.067 + 0.059 log t₂. Only this upper bound is used.

Take t₀ > 168π a good Gram sample at or above T, and t₁ > t₀ (40 Gram
intervals later, ending at a good Gram point). If L(t) is the number of sign
changes found in (t₀, t], then N(t) ≥ N(t₀) + L(t) for t ≥ t₀, so by (∗)
S(t) ≥ S(t₀) + L(t) − (θ(t) − θ(t₀))/π. Integrating over [t₀, t₁] and using
Trudgian's bound:

N(t₀) ≤ 1 + [2.067 + 0.059 log t₁ − ∫L + π⁻¹ ∫_{t₀}^{t₁} θ(t) dt] / (t₁ − t₀).

(The θ(t₀) terms cancel.) The code bounds ∫L from below by Σ (t₁ − hi) over
the sign changes (lo, hi) in [t₀, t₁], and ∫θ from above by evaluating θ on
balls covering pieces of length 1/64 and summing (piece length) × (upper end).
All arithmetic is rounded in the safe direction.

If ⌊bound⌋ equals the number of sign changes found in (0, t₀], every zero
with 0 < γ ≤ t₀ is accounted for by a distinct sign change of odd
multiplicity, so each is on the line and simple. If the count is lower, the
check fails (`ZZ_ERR_TURING`); if higher, something is inconsistent
(`ZZ_ERR_INCONSISTENT`). For T = 10⁴ the bound is 10144.57, so N(g₁₀₁₄₃) ≤
10144, matching the 10144 sign changes found.

## Sources and their status

| Item | Used for | Status |
|---|---|---|
| T. Trudgian, "Improvements to Turing's method", Math. Comp. 80 (2011) 2259–2279, Theorem 2.2: ∫ S ≤ 2.067 + 0.059 log t₂ for t₂ > t₁ > 168π | Turing's bound | Confirmed from arXiv:0903.1885v3 (§2.1). Its footnote 3 notes 168π could be lowered; the code keeps t₀ > 168π. |
| DLMF 5.11.1 and §5.11(ii) (Stirling's series; complex remainder at most sec^{2n}(½ ph z) times the first neglected term; DLMF cites Olver 1997) | θ(t) | Confirmed from dlmf.nist.gov/5.11 |
| DLMF 2.10.1 (Euler–Maclaurin), 24.2.11 and 24.9.1 (\|B₂ₙ(x)\| < \|B₂ₙ\| on (0, 1)) | ζ and its remainder | Confirmed from dlmf.nist.gov; the remainder bound above is derived from them here |
| Riemann–von Mangoldt identity (∗) | Turing's bound | Derived here (argument principle); not compared against a textbook |
| W. Gabcke, *Neue Herleitung und explizite Restabschätzung der Riemann-Siegel-Formel*, Dissertation, Göttingen 1979, Satz 3.2.2 (b): \|R₀(t)\| < 0.127 t^{−¾} for t ≥ 200, for the formula with C₀ only | `--rs` | Confirmed from the dissertation PDF (2015 typesetting of the 1979 thesis, 133 pages; printed p. 55 = PDF p. 63; restated in the Einleitung, (6) and (8), printed p. 3), read 2026-11-23. Formula and C₀: Einleitung (1), (2), printed p. 2. Not committed. |
| Gabcke, Einleitung (4), (5): θ expansion with \|Rθ(t)\| < 1/(3322 t⁷), t ≥ 10 | A test cross-check; heuristic sample placement | Confirmed from the same PDF (printed p. 2); not used in proofs |
| MPFR correct rounding and ternary values | Every enclosure | MPFR documentation |
| OEIS A072080, A114856, A013629, A002505, A058303, A065434, A065452, A065453, A192492, A305741–A305744, A306004 | Test expectations | Retrieved 2026-11-23 |

## Evidence

- Full run, method `em`: `status 0`, N(10⁴) = 10142, N(1000) = 649,
  N(100) = 29, t₀ = g₁₀₁₄₃ = 10000.8818, t₁ = 10034.9635, Turing bound
  10144.567, 10144 sign changes up to t₀, 13,045 evaluations, all by
  Euler–Maclaurin.
- Method `rs` (Gabcke's bound): the same counts and the same 840 bad Gram
  points; 13,363 evaluations, of which 241 by Euler–Maclaurin (spot checks,
  t < 200, and 2 fallbacks), about 8 s.
- Compared locally with the full OEIS b-files (not committed: OEIS data is
  CC BY-SA 4.0):
  - the 840 bad Gram points up to n = 10143 equal OEIS A114856 exactly;
  - each of the first 10,000 sign-change brackets meets [⌊γ_k⌋, ⌊γ_k⌋ + 1)
    for the k-th zero in OEIS A013629, including Lehmer's pair (zeros 6709
    and 6710, brackets [7005.0623, 7005.0903] and [7005.0903, 7005.1182]);
  - N(600) = 341 matches the count in A013629.
- θ and Z agree with mpmath 1.4.1 (used during development only, not in CI)
  to 30 digits at t = 1, γ₁, 20, 100, 282.45, 1000, 7005.08 and 10⁴.

### Tests

`tests/test_zeta.c` (`make all`, Valgrind in CI, ASan/UBSan in
`make bounds-sanitize`):

- ball arithmetic: enclosures of e, 1/3, π/4 and log 2 / 2 checked against
  512-bit values; propagation through sin; refusals for log, sqrt and 1/x
  of balls reaching 0, and complex log off the right half-plane;
- θ against Gabcke's expansion within 1/(3322 t⁷) at seven heights from 10
  to 10⁴, failing without its last term; a θ offset of 0.01 visible;
- the first ten published zeros (OEIS): Z changes sign across γ_k ± 10⁻²⁰;
- Gram's law holds for n = −1, …, 125 and fails at 126 and 134, on
  certified enclosures of g_n;
- Lehmer's pair: sign changes in (7005.04, 7005.08) and (7005.08, 7005.12),
  and Riemann–Siegel decides the middle sign;
- Riemann–Siegel and Euler–Maclaurin enclosures overlap at 49 points in
  [200, 2000] (at least 40 must be decidable by Riemann–Siegel), and stop
  overlapping when the n = N term is dropped;
- verification to T = 600 with both methods, N(100) = 29 and N(600) = 341;
- argument and workspace checks, and outputs with small capacities.

`tests/test_zeta.py` (CI `differential` step) runs `zeta_verify` to T = 10⁴
with both methods and checks N(100), N(1000), N(10⁴), the Turing equality,
the first eight bad Gram points, agreement between the methods, the first
two zeros and Lehmer's pair against the brackets.

### Negative cases

Each defect is injected through `zz_faults` (tests only) and must make the
verification fail:

| Defect | Method | Caught by |
|---|---|---|
| Sign change in the Gram interval of 14.13… ([g₋₁, g₀)) not counted | em | Turing: bound exceeds count (`ZZ_ERR_TURING`) |
| θ + 10⁻⁶ everywhere | em | Im(e^{iθ}ζ) check |
| θ + 10⁻⁶, + 0.05 or + π at t ≥ 200 only | rs | Euler–Maclaurin spot check (Im check) |
| k = 1 Euler–Maclaurin term dropped | em | Im(e^{iθ}ζ) check |
| n = N Riemann–Siegel term dropped, t ≥ 200 | rs | Count: a Gram block near t = 230 loses its sign changes (`ZZ_ERR_ROSSER`); the spot check at t = 285.75 would also fail (unit-tested) |

Two findings from these tests shaped the design:

- Before the spot checks existed, RS mode with θ + 0.05 above t = 200
  **passed** at T = 600 with the correct count, 341: a small θ error kept
  every sign. Riemann–Siegel uses θ directly and has no internal check, so
  the spot checks were added.
- A sign change skipped inside Turing's range (beyond t₀) does not fail the
  proof when the bound has slack: it only weakens ∫L, and the count up to t₀
  is still complete. The test records this (g₄₂₀ at T = 600 still proves).

## AES-SEC-001

- New code is in new files; no existing file's lines moved.
- No `memcpy`, `memset` or `memmove`, and no other review-required
  primitive: an early version used `malloc`, `realloc`, `free` and
  `snprintf`, which the strict scan reported as 34 new findings. The final
  code allocates nothing itself: the context and every array (workspace,
  outputs) belong to the caller, `log n` and `n^{−½}` are computed when
  needed instead of tabulated, and failure reasons are static strings. MPFR
  and GMP manage the storage of their own numbers.
- The pinned AEMS scan (`aes_sec_001_scan.py crypto_pipeline
  --include-dangerous-primitives --strict --strict-review-ratchet`, AEMS
  4df3a37) reports no new findings; the ledger is unchanged (19 reviewed).
- Workspace bounds: `refine_tail` checks `n + (2·len − 1) ≤ cap` before using
  scratch space at the end of the workspace, and every push checks capacity;
  running out fails with `ZZ_ERR_ARGS` ("workspace too small").
  `zz_verify_work_size` gives a size that suffices unless a block needs
  unusually deep refinement.

## Build dependency

MPFR (with GMP) is a new build dependency of this scope: `libmpfr-dev` is
added to `ci/validate.sh install-tools`. Only `test_zeta` and `zeta_verify`
link it (`-lmpfr -lgmp -lm`); `libcryptomult.so` and the other tests do not.
Tested with MPFR 4.2.1 and GMP 6.3.0 (Ubuntu 24.04.5 packages).

## Limits

- Soundness rests on MPFR's documented correct rounding, the cited theorems,
  and (∗) as derived here. None of it is machine-checked; the C code is
  tested, not proved.
- Gabcke's bound is taken from the thesis as stated; its proof (Chapters 3
  and 4) was not re-checked here. The RS–EM spot checks and the 49-point
  overlap test are consistent with it but are not a proof of it.
- Variable-time, public inputs; research prototype. Heights up to 10⁶ are
  accepted; Euler–Maclaurin cost grows like T² overall, so large heights
  should use `--rs`.
- Not included: higher Riemann–Siegel corrections (C₁, …), Lean
  formalization of the Turing step, and a mixed method that trusts
  Riemann–Siegel only where Euler–Maclaurin confirms it.

## References

- T. Trudgian, "Improvements to Turing's method", Math. Comp. 80 (2011),
  2259–2279; arXiv:0903.1885.
- A. M. Turing, "Some calculations of the Riemann zeta-function" (1953),
  as cited and described by Trudgian (not consulted).
- NIST Digital Library of Mathematical Functions, §§2.10, 5.11, 24.2, 24.9
  (dlmf.nist.gov), citing F. W. J. Olver, *Asymptotics and Special
  Functions*, A K Peters, 1997.
- W. Gabcke, *Neue Herleitung und explizite Restabschätzung der
  Riemann-Siegel-Formel*, Dissertation, Göttingen, 1979,
  hdl:11858/00-1735-0000-0022-6013-8 (Satz 3.2.2; Einleitung (1)–(8)).
- M. Rubinstein, "Computational methods and experiments in analytic number
  theory", arXiv:math/0412181 (2005).
- The MPFR library manual, mpfr.org.
- OEIS, oeis.org (sequences listed above).
