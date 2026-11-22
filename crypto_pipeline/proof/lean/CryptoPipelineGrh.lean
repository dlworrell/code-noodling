/-
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:e5551d08-93b6-44d7-9a6b-7005f63b4e81
-/
import Mathlib

/-!
# Formal link for the GRH-conditional primality test

`crypto_prime.c` decides primality of an odd `N` by running the strong
(Miller–Rabin) test for every base `a` with `2 ≤ a ≤ min(B, N - 2)`, where
`B = crypto_grh_witness_bound(N) = ⌈2 · (L · 0.69315)²⌉` and `L` is the bit
length of `N`.

This file proves, in Lean with Mathlib:

1. `cBound_numerator_lt`: the C computation of `B` cannot overflow 64 bits.
2. `two_log_sq_le_cBound`: `B ≥ 2 · (ln N)²`, the bound in Bach's theorem.
3. `prime_millerAllBases`: unconditionally, every prime passes every base, so
   the test never calls a prime composite.
4. `millerAllBases_prime`: assuming Bach's theorem (`BachWitnessBound`, an
   explicit hypothesis, not an axiom), every odd number that passes all bases
   up to `B` is prime.

Bach's theorem itself (every odd composite `N` has a strong witness below
`2 (ln N)²`, assuming the extended Riemann hypothesis) is NOT proved here.
Nor is the correspondence between this model and the C source: `cBound` is
the C formula transcribed by hand, and the tests check them against each
other on sample values.
-/

namespace CryptoPipeline

/-- `crypto_grh_witness_bound` for a candidate of bit length `L`:
`⌈2 · L² · 13863² / 20000²⌉`, written exactly as the C integer expression. -/
def cBound (L : ℕ) : ℕ :=
  (2 * L * L * 13863 * 13863 + (20000 * 20000 - 1)) / (20000 * 20000)

/-- Values the C tests also check. -/
theorem cBound_61 : cBound 61 = 3576 := by norm_num [cBound]
theorem cBound_127 : cBound 127 = 15499 := by norm_num [cBound]

/-- No overflow: for every supported size (at most 1024 limbs, so `L ≤ 65536`)
the C numerator stays below `2^61`, far inside `uint64_t`. -/
theorem cBound_numerator_lt {L : ℕ} (h : L ≤ 65536) :
    2 * L * L * 13863 * 13863 + (20000 * 20000 - 1) < 2 ^ 61 := by
  have hsq : L * L ≤ 65536 * 65536 := Nat.mul_le_mul h h
  have e : 2 * L * L * 13863 * 13863 = 384365538 * (L * L) := by ring
  rw [e]
  have h2 : 384365538 * (L * L) ≤ 384365538 * (65536 * 65536) :=
    Nat.mul_le_mul_left _ hsq
  norm_num at h2 ⊢
  omega

/-- The integer ceiling is at least the real quantity `2 · L² · 0.69315²`. -/
theorem real_le_cBound (L : ℕ) :
    2 * (L : ℝ) ^ 2 * (13863 / 20000) ^ 2 ≤ (cBound L : ℝ) := by
  have hd : (0 : ℕ) < 20000 * 20000 := by norm_num
  have key : 2 * L * L * 13863 * 13863 ≤ cBound L * (20000 * 20000) := by
    have h := Nat.lt_div_mul_add (a := 2 * L * L * 13863 * 13863 + (20000 * 20000 - 1)) hd
    unfold cBound
    omega
  have keyR : (2 * L * L * 13863 * 13863 : ℝ) ≤ (cBound L : ℝ) * (20000 * 20000) := by
    exact_mod_cast key
  nlinarith [keyR]

/-- `ln N < L · ln 2` where `L = N.size` is the bit length of `N`. -/
theorem log_lt_size_mul_log_two (N : ℕ) (hN : 0 < N) :
    Real.log N < N.size * Real.log 2 := by
  have h1 : (N : ℝ) < 2 ^ N.size := by exact_mod_cast Nat.lt_size_self N
  have h0 : (0 : ℝ) < N := by exact_mod_cast hN
  have h := Real.log_lt_log h0 h1
  rwa [Real.log_pow] at h

/-- The C bound covers Bach's bound: `2 · (ln N)² ≤ crypto_grh_witness_bound(N)`. -/
theorem two_log_sq_le_cBound (N : ℕ) (hN : 0 < N) :
    2 * (Real.log N) ^ 2 ≤ (cBound N.size : ℝ) := by
  have hone : (1 : ℝ) ≤ N := by exact_mod_cast hN
  have hlog0 : 0 ≤ Real.log N := Real.log_nonneg hone
  have hlt := log_lt_size_mul_log_two N hN
  have hl2 : Real.log 2 < 13863 / 20000 := by
    have := Real.log_two_lt_d9
    norm_num at this ⊢
    linarith
  have hL : (0 : ℝ) ≤ N.size := Nat.cast_nonneg _
  have hmul : (N.size : ℝ) * Real.log 2 ≤ N.size * (13863 / 20000) :=
    mul_le_mul_of_nonneg_left hl2.le hL
  have hle : Real.log N ≤ N.size * (13863 / 20000) := by linarith
  have hsq : (Real.log N) ^ 2 ≤ (N.size * (13863 / 20000 : ℝ)) ^ 2 :=
    pow_le_pow_left₀ hlog0 hle 2
  have hc := real_le_cBound N.size
  nlinarith [hsq, hc]

/-- Strong probable prime to base `a`: for every decomposition `N - 1 = 2^s · d`
with `d` odd, `a^d = 1` or `(a^d)^(2^r) = -1` for some `r < s`, in `ZMod N`.
For odd `N ≥ 3` the decomposition is unique; it is the one the C code computes. -/
def StrongProbablePrime (N a : ℕ) : Prop :=
  ∀ s d : ℕ, N - 1 = 2 ^ s * d → Odd d →
    ((a : ZMod N) ^ d = 1 ∨ ∃ r < s, ((a : ZMod N) ^ d) ^ (2 ^ r) = -1)

/-- In a ring without zero divisors, `x^(2^s) = 1` forces `x = 1` or some
`x^(2^r) = -1` with `r < s`. -/
theorem pow_two_pow_eq_one {R : Type*} [CommRing R] [NoZeroDivisors R] (x : R) :
    ∀ s : ℕ, x ^ (2 ^ s) = 1 → x = 1 ∨ ∃ r < s, x ^ (2 ^ r) = -1
  | 0, h => Or.inl (by simpa using h)
  | s + 1, h => by
    have e : 2 ^ (s + 1) = 2 ^ s + 2 ^ s := by ring
    rw [e, pow_add] at h
    rcases mul_self_eq_one_iff.mp h with h1 | h1
    · rcases pow_two_pow_eq_one x s h1 with h2 | ⟨r, hr, h2⟩
      · exact Or.inl h2
      · exact Or.inr ⟨r, by omega, h2⟩
    · exact Or.inr ⟨s, by omega, h1⟩

/-- Every base `0 < a < p` passes the strong test for a prime `p`. -/
theorem prime_strongProbablePrime {p a : ℕ} (hp : p.Prime) (ha0 : 0 < a) (hap : a < p) :
    StrongProbablePrime p a := by
  intro s d hpd _hd
  haveI := Fact.mk hp
  have hne : (a : ZMod p) ≠ 0 := by
    intro h
    rw [ZMod.natCast_zmod_eq_zero_iff_dvd] at h
    have := Nat.le_of_dvd ha0 h
    omega
  have hferm : (a : ZMod p) ^ (p - 1) = 1 := ZMod.pow_card_sub_one_eq_one hne
  rw [hpd, mul_comm, pow_mul] at hferm
  exact pow_two_pow_eq_one _ s hferm

/-- `N - 1` is never a witness for odd `N ≥ 3`, which is why the C loop can
stop at `N - 2`. -/
theorem strongProbablePrime_pred {N : ℕ} (hodd : Odd N) (h3 : 3 ≤ N) :
    StrongProbablePrime N (N - 1) := by
  intro s d hNd hd
  have hcast : ((N - 1 : ℕ) : ZMod N) = -1 := by
    rw [Nat.cast_sub (by omega), ZMod.natCast_self, Nat.cast_one, zero_sub]
  have hs : 0 < s := by
    rcases Nat.eq_zero_or_pos s with rfl | h
    · obtain ⟨k, hk⟩ := hodd
      obtain ⟨j, hj⟩ := hd
      simp at hNd
      omega
    · exact h
  right
  refine ⟨0, hs, ?_⟩
  rw [hcast, pow_zero, pow_one, Odd.neg_one_pow hd]

/-- What the C loop checks: every base from 2 to `min(B, N - 2)` passes. -/
def MillerAllBases (N : ℕ) : Prop :=
  ∀ a, 2 ≤ a → a ≤ min (cBound N.size) (N - 2) → StrongProbablePrime N a

/-- Bach's theorem, as an explicit hypothesis: every odd composite `N ≥ 3`
has a strong witness `a` with `2 ≤ a < N` and `a < 2 (ln N)²`. Bach (1990)
proved this assuming the extended Riemann hypothesis. -/
def BachWitnessBound : Prop :=
  ∀ N : ℕ, Odd N → 3 ≤ N → ¬ N.Prime →
    ∃ a : ℕ, 2 ≤ a ∧ a < N ∧ (a : ℝ) < 2 * (Real.log N) ^ 2 ∧ ¬ StrongProbablePrime N a

/-- Unconditional: a prime always passes the C loop. -/
theorem prime_millerAllBases {N : ℕ} (hp : N.Prime) : MillerAllBases N := by
  intro a ha2 hale
  have hle : a ≤ N - 2 := le_trans hale (min_le_right _ _)
  exact prime_strongProbablePrime hp (by omega) (by omega)

/-- Conditional on Bach's theorem: an odd `N ≥ 3` that passes the C loop is prime. -/
theorem millerAllBases_prime (hBach : BachWitnessBound) {N : ℕ} (hodd : Odd N)
    (h3 : 3 ≤ N) (hall : MillerAllBases N) : N.Prime := by
  by_contra hnp
  obtain ⟨a, ha2, haN, halog, hw⟩ := hBach N hodd h3 hnp
  have hboundR : (a : ℝ) ≤ cBound N.size :=
    le_trans halog.le (two_log_sq_le_cBound N (by omega))
  have hbound : a ≤ cBound N.size := by exact_mod_cast hboundR
  have hneq : a ≠ N - 1 := by
    rintro rfl
    exact hw (strongProbablePrime_pred hodd h3)
  exact hw (hall a ha2 (le_min hbound (by omega)))

/-- Together: under Bach's theorem the C loop decides primality of odd `N ≥ 3`. -/
theorem millerAllBases_iff_prime (hBach : BachWitnessBound) {N : ℕ} (hodd : Odd N)
    (h3 : 3 ≤ N) : MillerAllBases N ↔ N.Prime :=
  ⟨millerAllBases_prime hBach hodd h3, prime_millerAllBases⟩

end CryptoPipeline
