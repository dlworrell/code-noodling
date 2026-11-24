<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:95d4f9a0-9b2d-4fb7-88ff-4c04036c8b8e
-->

# Fuzzing and exhaustive checks of the certificate checkers

Status: Track A hardening of the public-value certificate checkers
(`crypto_pocklington_verify`, `crypto_ecpp_verify`, `crypto_mixed_verify`).
These checks look for crashes, undefined behaviour, contract violations and
accepted composites. Passing them is evidence, not proof; the checkers remain
experimental until the machine-checked proofs, independent cross-checks and
review in the Track A plan are done.

Files: `tests/fuzz_certificates.c` (harness), `tests/fuzz_corpus.py` (seeds),
`tests/test_cert_exhaustive.c` (exhaustive small-number check), `ci/fuzz.sh`
(runner) and `.github/workflows/crypto_fuzz.yml` (CI).

## What is checked

**The harness** decodes each input into one small certificate (up to six
nodes or four ECPP steps, six factors per node, three limbs per value) and runs
the matching checker under ASan and UBSan. For every input it requires:

- no memory error or undefined behaviour;
- the error contract: `-1` with `EINVAL`, `EDOM` or `ERANGE` and a failed-node
  index inside the certificate, or `0` with `errno` cleared and the index left
  unchanged;
- soundness where it is cheap to decide: an accepted certificate whose root N
  is below 2^64 must prove a prime (`crypto_is_prime_u64` is exact there).

It builds two ways. With libFuzzer it is coverage-guided. With
`-DFUZZ_STANDALONE` it replays the seeds, requires each to be accepted (a
rejected seed means the corpus and the byte format disagree), then runs
deterministic mutations, so it works with any compiler.

**The seeds** are valid certificates in the harness's byte format: the
shared-node Pocklington tree from the depth tests, single Pocklington and ECPP
nodes, the p127 ECPP chain, generated 128- and 160-bit mixed certificates, and
four Pocklington certificates for N < 256 written with one-byte values so that
mutations stay among small numbers.

**The exhaustive check** enumerates every odd N in [5, 512), every
certificate of one or two prime powers dividing N − 1 (64-bit leaves) and every
base in [2, N − 2], and requires that only primes are accepted: 491,916
certificates in about 1.4 s. It also requires every odd prime in the range to
be proven by some enumerated certificate, so the search is not vacuous.

## What it found and what it would find

On the current checkers: nothing. Locally, 20,000 standalone mutations, a
60-second libFuzzer run (80,687 executions, coverage 449) and the exhaustive
check all passed.

The checks were tested against two planted bugs in a scratch copy of
`crypto_cert.c`:

| Planted bug | Exhaustive check | Fuzzing (3 minutes, about 500,000 runs) |
|---|---|---|
| F^2 > N condition removed | caught: N = 49, 3^1, a = 18 | not caught |
| gcd(a^((N−1)/q) − 1, N) = 1 removed | caught: N = 15, 2 · 7, a = 4 | not caught |

Fed the N = 49 input directly, the harness aborts with
`SOUNDNESS: accepted composite N = 49`, so its oracle works; random mutation
rarely makes the three or four coordinated edits needed to reach such inputs.
**Fuzzing here is evidence of memory safety and contract adherence; soundness
evidence comes from the exhaustive check** and, later, the Lean proofs and
cross-checks. The exhaustive check covers single Pocklington nodes on small
numbers only; it does not cover ECPP steps, multi-node linking or large
values.

## Running it

From `crypto_pipeline/`:

```sh
bash ci/fuzz.sh exhaustive
bash ci/fuzz.sh smoke                    # FUZZ_ITERATIONS=20000 by default
FUZZ_SECONDS=60 bash ci/fuzz.sh libfuzzer   # needs clang and libclang-rt
```

Builds go to a temporary directory outside the repository. Set
`FUZZ_OUTPUT=<dir>` to keep the corpus, logs and any crash input; CI uploads
them as an artifact.

## Next steps

- An exhaustive or sampled check for small ECPP steps (curves over N < 2^10).
- Exhaustive checks of linking in mixed certificates with small nodes.
- A persistent corpus and longer scheduled runs.
