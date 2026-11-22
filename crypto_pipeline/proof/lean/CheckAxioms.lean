/-
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:6fdd56c1-ddb7-4240-852b-1f88e3e96a10
-/
import CryptoPipelineGrh

/-! Run by CI with `lake env lean CheckAxioms.lean`. The workflow fails unless
every theorem below depends only on `propext`, `Classical.choice` and
`Quot.sound` (no `sorryAx`, no added axioms). Bach's theorem enters only as
the hypothesis `BachWitnessBound`, never as an axiom. -/

#print axioms CryptoPipeline.cBound_numerator_lt
#print axioms CryptoPipeline.two_log_sq_le_cBound
#print axioms CryptoPipeline.prime_millerAllBases
#print axioms CryptoPipeline.millerAllBases_prime
#print axioms CryptoPipeline.millerAllBases_iff_prime
