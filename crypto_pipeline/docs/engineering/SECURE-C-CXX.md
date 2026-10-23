<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:f1c5e354-da28-46d7-b0c8-bc97b86af739
-->

# Secure C/C++ Profile: crypto_pipeline scope

Status: Adopted
Repository: dlworrell/code-noodling
Scope: `crypto_pipeline/` (governed project scope, Catylist ADR-0004)
Inherited Standard: AES-SEC-001
Repository profile: [`docs/engineering/SECURE-C-CXX.md`](../../../docs/engineering/SECURE-C-CXX.md)

## Policy

This scope inherits `AES-SEC-001: Secure C and C++ Coding Rules` for the files
it owns: everything under `crypto_pipeline/`. It applies the repository
profile without relaxation and adds the scope-specific requirements below.

The scope's C sources are `crypto_multiplier.c`, `crypto_montgomery_ct.c`,
their headers, the timing harness `dudect_montgomery.c`, the tests under
`tests/`, and the formal harness `proof/memory16.c`. The analysis
configuration is the repository root `.clang-tidy`.

## Scope requirements

In addition to the repository profile, code in this scope must:

- keep secret-dependent control flow and memory access out of the
  constant-flow candidate (`crypto_montgomery_ct.c`), as checked by
  `run_constant_flow.sh` and the formal checks under `proof/`;
- treat timing measurements as evidence, not as constant-time certification;
- record every reviewed use of a review-required primitive in this scope's
  review-disposition ledger with its bounds invariant.

## Evidence

| Evidence | Location |
|---|---|
| Review-disposition ledger | [`AES-SEC-001-review-dispositions.json`](AES-SEC-001-review-dispositions.json) |
| Waiver log | [`AES-SEC-001-waivers.md`](AES-SEC-001-waivers.md) |
| Arithmetic contracts | [`CRYPTO-ARITHMETIC.md`](CRYPTO-ARITHMETIC.md) |
| Memory review | [`CRYPTO-MEMORY-REVIEW.md`](CRYPTO-MEMORY-REVIEW.md) |
| Constant-flow scope and evidence | [`CRYPTO-CONSTANT-FLOW.md`](CRYPTO-CONSTANT-FLOW.md) |
| Captured evidence records | [`evidence/`](evidence/) |

## Transition

AEMS does not yet scan governed project scopes separately (Catylist #24).
Until it does, AES-SEC-001 requires the repository-wide scan and the
repository-level ledger to remain authoritative for these files. This scope's
ledger carries the same dispositions re-keyed to scope-relative paths, ready
for the scope-rooted scan, and is supplementary until then.

## Waivers

A waiver for a file in this scope must be recorded in this scope's waiver log.
It must identify the rule, file path, reason, safety invariant, owner, review
date, and supporting test evidence.
