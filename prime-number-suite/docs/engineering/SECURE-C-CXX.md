<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:3234f38d-9327-4e3e-9cf7-bb71a78cbc9c
-->

# Secure C/C++ Profile: Prime Number Suite

AES-SEC-001 applies without relaxation to every file owned by the
`prime-number-suite/` governed scope, including nested `dice/`, optional
PhysX and CUDA code, and retained legacy sources. See the
[repository profile](../../../docs/engineering/SECURE-C-CXX.md). The
repository-root `.clang-tidy` is the shared scanner configuration.

Check bounds, integer conversions, and allocation-size arithmetic for sieve
ranges, input counts and generated buffers. Check allocation failures and
preserve ownership on reallocation. For GPU code, cover host-side bounds and
device transfer lengths, and distinguish compile-only evidence from hardware
execution. The scope does not imply that experimental or legacy code is
production-ready.

The suite CMake build compiles `dice/src/dice_cpu.cpp` and tests the CPU dice
engine with CTest. The same source is separately built, tested, and distribution-
checked by the standalone Dice Autotools workflow; this governance registration
does not change either workflow's triggers. See the [build and CI
documentation](../../README.md).

Review-required primitives and their approved invariants are recorded in the
[review dispositions](AES-SEC-001-review-dispositions.json). Existing baseline
entries and dispositions were rekeyed from repository-relative identities to
scope-relative identities with the pinned AEMS migration tool without making
new approvals: all 3 baseline entries and 3 dispositions moved, and none remain
in the root ledger. The identity map is recorded in
[`scope-ledger-rekey-20261124.json`](evidence/scope-ledger-rekey-20261124.json). Approved exceptions, if any, belong only in the [scope waiver
log](AES-SEC-001-waivers.md).
