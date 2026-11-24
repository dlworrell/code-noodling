/-
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:1b085b81-2d68-4a85-a3f7-37153b107481
-/
import Mathlib

/-!
# Formal link for the Turing step of the zeta-zeros verification

`crypto_zeta.c` (`turing_bound`, `zz_verify`) proves that every zero of ζ up
to a Gram point `t₀` is on the critical line by comparing the number `c` of
sign changes of Hardy's `Z` found in `(0, t₀]` with an upper bound for `N(t₀)`
computed as

  `U = 1 + (A − Σᵢ (t₁ − hᵢ) + Iθ / π) / (t₁ − t₀)`,

where `A = 2.067 + 0.059 log t₁` (Trudgian 2011, Theorem 2.2), the `hᵢ` are
the right ends of the sign changes found in `[t₀, t₁]`, and `Iθ` is an upper
bound for `∫_{t₀}^{t₁} θ`. If `⌊U⌋ = c`, the verification is complete.
See `docs/engineering/CRYPTO-ZETA-ZEROS.md`, §6.

This file proves, in Lean with Mathlib:

1. `integral_step`: the exact integral of one step `t ↦ [h < t]`.
2. `turing_upper_bound`: `N(t₀) ≤ U`, from three hypotheses (below).
3. `turing_count_exact`: if `N(t₀)` is a natural number `n₀ ≥ c` and
   `⌊U⌋₊ = c`, then `n₀ = c`.

The hypotheses of `turing_upper_bound` are NOT proved here:

- `hTrud`: `∫_{t₀}^{t₁} (N − θ/π − 1) ≤ A`. This is Trudgian's Theorem 2.2
  (`∫ S ≤ A`) rewritten with the identity `N = θ/π + 1 + S`; both are
  analytic number theory that Mathlib does not contain.
- `hcount`: `N(t₀) + L(t) ≤ N(t)` on `[t₀, t₁]`, where `L` counts the `hᵢ`
  below `t`. This holds because each sign change of the real function `Z`
  marks a zero of ζ on the critical line (intermediate value theorem), and
  the C code proves each sign by interval arithmetic.
- `hIθ`: `∫ θ ≤ Iθ`, which the C code bounds by evaluating `θ` on balls.

As in the GRH file, these enter as hypotheses of the theorems, never as
axioms, and the correspondence between this model and the C source is
tested, not proved.
-/

namespace CryptoPipeline.Turing

open MeasureTheory Set

/-- One step: `1` once `t` has passed `h`, else `0`. -/
noncomputable def step (h t : ℝ) : ℝ := if h < t then 1 else 0

/-- `L(t)`: how many of the right ends `hs` lie below `t`. -/
noncomputable def stepCount (hs : List ℝ) (t : ℝ) : ℝ :=
  (hs.map fun h => step h t).sum

theorem step_eq_indicator (h : ℝ) : step h = (Ioi h).indicator (fun _ => (1 : ℝ)) := by
  funext t
  simp [step, Set.indicator, Set.mem_Ioi]

theorem step_monotone (h : ℝ) : Monotone (step h) := by
  intro a b hab
  by_cases h1 : h < a
  · have h2 : h < b := lt_of_lt_of_le h1 hab
    simp [step, h1, h2]
  · by_cases h2 : h < b <;> simp [step, h1, h2]

theorem stepCount_cons (h : ℝ) (hs : List ℝ) (t : ℝ) :
    stepCount (h :: hs) t = step h t + stepCount hs t := by
  simp [stepCount]

theorem stepCount_monotone (hs : List ℝ) : Monotone (stepCount hs) := by
  induction hs with
  | nil => intro a b _; simp [stepCount]
  | cons h hs ih =>
    intro a b hab
    rw [stepCount_cons, stepCount_cons]
    exact add_le_add (step_monotone h hab) (ih hab)

/-- The exact integral of one step over `[t₀, t₁]` when `t₀ ≤ h ≤ t₁`. -/
theorem integral_step {t₀ t₁ h : ℝ} (h0 : t₀ ≤ h) (h1 : h ≤ t₁) :
    ∫ t in t₀..t₁, step h t = t₁ - h := by
  rw [intervalIntegral.integral_of_le (h0.trans h1), step_eq_indicator,
    setIntegral_indicator measurableSet_Ioi, Set.Ioc_inter_Ioi, max_eq_right h0,
    setIntegral_const, Real.volume_Ioc, ENNReal.toReal_ofReal (sub_nonneg.mpr h1),
    smul_eq_mul, mul_one]

/-- `∫ L = Σᵢ (t₁ − hᵢ)` when every `hᵢ` lies in `[t₀, t₁]`. -/
theorem integral_stepCount {t₀ t₁ : ℝ} (hs : List ℝ) (hhs : ∀ h ∈ hs, t₀ ≤ h ∧ h ≤ t₁) :
    ∫ t in t₀..t₁, stepCount hs t = (hs.map fun h => t₁ - h).sum := by
  induction hs with
  | nil => simp [stepCount]
  | cons h hs ih =>
    have hh := hhs h (List.mem_cons_self h hs)
    have hrest : ∀ g ∈ hs, t₀ ≤ g ∧ g ≤ t₁ := fun g hg => hhs g (List.mem_cons_of_mem h hg)
    calc ∫ t in t₀..t₁, stepCount (h :: hs) t
        = ∫ t in t₀..t₁, (step h t + stepCount hs t) := by simp only [stepCount_cons]
      _ = (∫ t in t₀..t₁, step h t) + ∫ t in t₀..t₁, stepCount hs t :=
          intervalIntegral.integral_add (step_monotone h).intervalIntegrable
            (stepCount_monotone hs).intervalIntegrable
      _ = ((h :: hs).map fun g => t₁ - g).sum := by
          rw [integral_step hh.1 hh.2, ih hrest, List.map_cons, List.sum_cons]

/-- The Turing upper bound computed by `turing_bound` in `crypto_zeta.c`. -/
theorem turing_upper_bound {N θ : ℝ → ℝ} {t₀ t₁ A Iθ : ℝ} (hs : List ℝ)
    (ht : t₀ < t₁)
    (hN : IntervalIntegrable N volume t₀ t₁)
    (hθ : IntervalIntegrable θ volume t₀ t₁)
    (hTrud : ∫ t in t₀..t₁, (N t - θ t / Real.pi - 1) ≤ A)
    (hhs : ∀ h ∈ hs, t₀ ≤ h ∧ h ≤ t₁)
    (hcount : ∀ t ∈ Icc t₀ t₁, N t₀ + stepCount hs t ≤ N t)
    (hIθ : ∫ t in t₀..t₁, θ t ≤ Iθ) :
    N t₀ ≤ 1 + (A - (hs.map fun h => t₁ - h).sum + Iθ / Real.pi) / (t₁ - t₀) := by
  have hlen : 0 < t₁ - t₀ := sub_pos.mpr ht
  have hL : IntervalIntegrable (stepCount hs) volume t₀ t₁ :=
    (stepCount_monotone hs).intervalIntegrable
  -- ∫ (N t₀ + L) ≤ ∫ N.
  have hmono : ∫ t in t₀..t₁, (N t₀ + stepCount hs t) ≤ ∫ t in t₀..t₁, N t :=
    intervalIntegral.integral_mono_on ht.le (intervalIntegrable_const.add hL) hN hcount
  rw [intervalIntegral.integral_add intervalIntegrable_const hL,
    intervalIntegral.integral_const, integral_stepCount hs hhs, smul_eq_mul] at hmono
  -- ∫ (N − θ/π − 1) = ∫ N − (∫ θ)/π − (t₁ − t₀).
  have hsplit : ∫ t in t₀..t₁, (N t - θ t / Real.pi - 1) =
      (∫ t in t₀..t₁, N t) - (∫ t in t₀..t₁, θ t) / Real.pi - (t₁ - t₀) := by
    rw [intervalIntegral.integral_sub (hN.sub (hθ.div_const _)) intervalIntegrable_const,
      intervalIntegral.integral_sub hN (hθ.div_const _), intervalIntegral.integral_div,
      intervalIntegral.integral_const, smul_eq_mul, mul_one]
  rw [hsplit] at hTrud
  have hpi : 0 < Real.pi := Real.pi_pos
  have hθdiv : (∫ t in t₀..t₁, θ t) / Real.pi ≤ Iθ / Real.pi :=
    div_le_div_of_nonneg_right hIθ hpi.le
  -- (t₁ − t₀)(N t₀ − 1) ≤ A − Σ + Iθ/π, then divide by t₁ − t₀ > 0.
  have key : (t₁ - t₀) * (N t₀ - 1) ≤ A - (hs.map fun h => t₁ - h).sum + Iθ / Real.pi := by
    nlinarith
  have hdiv : N t₀ - 1 ≤ (A - (hs.map fun h => t₁ - h).sum + Iθ / Real.pi) / (t₁ - t₀) := by
    rw [le_div_iff₀ hlen]
    linarith
  linarith

/-- The check `zz_verify` performs: if `N(t₀) = n₀` with `c ≤ n₀` (each of the
`c` sign changes counts a zero) and the floor of the upper bound is `c`, then
`n₀ = c`, so every zero up to `t₀` is one of the counted sign changes. -/
theorem turing_count_exact {n₀ c : ℕ} {U : ℝ} (hlow : c ≤ n₀) (hup : (n₀ : ℝ) ≤ U)
    (hfloor : ⌊U⌋₊ = c) : n₀ = c := by
  have hlt : U < (⌊U⌋₊ : ℝ) + 1 := Nat.lt_floor_add_one U
  rw [hfloor] at hlt
  have : (n₀ : ℝ) < (c : ℝ) + 1 := lt_of_le_of_lt hup hlt
  have : n₀ < c + 1 := by exact_mod_cast this
  omega

/-- `turing_upper_bound` and `turing_count_exact` together, in the form the C
code uses: the count of sign changes up to `t₀` equals `N(t₀)`. -/
theorem turing_verifies {N θ : ℝ → ℝ} {t₀ t₁ A Iθ : ℝ} (hs : List ℝ) {n₀ c : ℕ}
    (ht : t₀ < t₁)
    (hN : IntervalIntegrable N volume t₀ t₁)
    (hθ : IntervalIntegrable θ volume t₀ t₁)
    (hTrud : ∫ t in t₀..t₁, (N t - θ t / Real.pi - 1) ≤ A)
    (hhs : ∀ h ∈ hs, t₀ ≤ h ∧ h ≤ t₁)
    (hcount : ∀ t ∈ Icc t₀ t₁, N t₀ + stepCount hs t ≤ N t)
    (hIθ : ∫ t in t₀..t₁, θ t ≤ Iθ)
    (hn₀ : N t₀ = n₀) (hlow : c ≤ n₀)
    (hfloor : ⌊1 + (A - (hs.map fun h => t₁ - h).sum + Iθ / Real.pi) / (t₁ - t₀)⌋₊ = c) :
    n₀ = c := by
  have hU := turing_upper_bound hs ht hN hθ hTrud hhs hcount hIθ
  rw [hn₀] at hU
  exact turing_count_exact hlow hU hfloor

end CryptoPipeline.Turing
