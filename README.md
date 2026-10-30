# Code Noodling

`code-noodling` is an experimental monorepo with three codebases:

- CPU and multi-GPU (Tesla K80) prime sieves and an unbiased dice engine;
- multi-precision integer multiplication research for cryptography, with
  constant-flow and formal checks;
- Washington, Alaska and Maine election-result watching with
  change-probability reports.

Each codebase keeps its source, build configuration, tests, and detailed
documentation in its own directory. The crypto and election codebases are
governed scopes: REUSE-licensed, with Sigstore-attested provenance.

## Codebases

| Codebase | Location | Purpose | Documentation |
|---|---|---|---|
| Prime Number and Dice Suite | [`prime-number-suite/`](prime-number-suite/) | CPU and multi-GPU (Tesla K80) prime sieves, an unbiased prime-seeded dice engine, and PhysX/Maya dice experiments | [`prime-number-suite/README.md`](prime-number-suite/README.md) |
| Crypto Pipeline | [`crypto_pipeline/`](crypto_pipeline/) | Multi-precision integer multiplication experiments (research prototype) | [`crypto_pipeline/README.md`](crypto_pipeline/README.md) |
| Election Result Reporting | [`election-analysis/`](election-analysis/) | Washington, Alaska and Maine election result watching, snapshot comparison, and change-probability reports | [`election-analysis/README.md`](election-analysis/README.md) |

## Repository Layout

```text
code-noodling/
├── README.md
├── .clang-tidy              repository C/C++ static-analysis policy
├── aes-manifest.yaml        repository manifest and governed project scopes
├── .github/                 workflows and contribution templates
├── crypto_pipeline/         integer multiplication experiments (governed scope)
├── docs/                    repository engineering standards
├── election-analysis/       election result reporting (governed scope)
├── prime-number-suite/      prime generators and dice programs
├── scripts/, tests/         repository CI checks and their tests
```

## Quick Start

Prime generators and CPU dice (CMake):

```bash
cmake -S prime-number-suite -B build \
  -DBUILD_CUDA_SIEVE_MGPU=OFF -DBUILD_PHYSX_DICE=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Dice engine as a standalone Autotools package:

```bash
cd prime-number-suite/dice
autoreconf -fi && ./configure && make && make check
```

The CUDA sieve defaults to the Tesla K80 (`sm_37`), which needs CUDA 11.x and
the 470 driver. See the suite README's
[Legacy GPU support](prime-number-suite/README.md#legacy-gpu-support-tesla-k80)
section.

## Continuous Integration

| Workflow | Scope |
|---|---|
| Dice Autotools | `prime-number-suite/dice/`: build, `make check`, `make distcheck` |
| Prime Suite | rest of `prime-number-suite/`: CPU build and tests, CUDA 11.4 `sm_37` compile (required), CUDA 12 compile (informational) |
| K80 Hardware Check | blocked on GitHub-hosted infrastructure; use [local hardware validation](docs/engineering/HARDWARE_EXECUTION_SECURITY.md) |
| Cryptographic Wrapper Validation Pipeline | `crypto_pipeline/`: build, clang-tidy, Valgrind, sanitizers, differential tests |
| Montgomery Constant-Flow Candidate | `crypto_pipeline/`: oracle, taint, negative control, timing (manual) |
| Montgomery Scoped Formal Checks | `crypto_pipeline/proof/`: source checks, lemmas, CBMC contracts |
| Election Result Reports | election data and reporting code; report updates arrive as a verified automation pull request |
| election-analysis Provenance (report-only) | `election-analysis/`: REUSE, Catalyst-File-ID and scope-manifest checks and test results, never failing the pull request; on pushes to main, signs and verifies the scope manifest and test results |
| AES-SEC-001 Governance | repository-wide C/C++ secure-coding scan |

## Engineering Governance

C and C++ changes inherit the AES-SEC-001 secure-coding profile. The profile,
review dispositions, and waiver log are in [`docs/engineering/`](docs/engineering/).

[`aes-manifest.yaml`](aes-manifest.yaml) declares governed project scopes
(Catylist ADR-0004). Each scope keeps its own `README.md`, `ci/` scripts,
`tests/` and adoption evidence under `<scope>/docs/engineering/`; the root
workflows only choose triggers and call the scope's `ci/` scripts.

| Scope | Path | Adoption evidence |
|---|---|---|
| crypto-pipeline | [`crypto_pipeline/`](crypto_pipeline/) | [`crypto_pipeline/docs/engineering/`](crypto_pipeline/docs/engineering/) |
| election-analysis | [`election-analysis/`](election-analysis/) | None needed: AES-SEC-001 is declared not applicable (no C or C++) |

Everything else is the repository-level remainder until it is migrated.
The pinned AEMS governance workflow scans crypto with its own profile and
ledger, verifies the election scope remains non-native, and scans the remainder
with declared scopes excluded. It retains the tracked-file coverage plan.

---

## Crypto Pipeline: Integer Multiplication Experiments

`crypto_pipeline/` compares exact full-width multiplication algorithms and their
practical crossover costs, inspired by the Scientific American article archived in
that directory. Modular arithmetic is a separate layer on top.

> **Research prototype, not production cryptography.** No API is certified
> constant-time or side-channel resistant, and legacy `secure_`/`ct_` names do not
> establish those properties.

| Algorithm | Notes |
|-----------|-------|
| **Comba** | Base-2^64 column multiplication; automatic dispatch uses it below 64 limbs |
| **Karatsuba** | Recursive, carry-preserving, odd sizes zero-padded; dispatched from 64 limbs |
| **NTT** | Two-prime radix-2 transform with CRT reconstruction; directly callable, slower than Karatsuba in local benchmarks |
| **Montgomery REDC** | `a*b*R^-1 mod m` for odd moduli |
| **Modular multiply / inverse / blinding** | Ordinary `a*b mod m`, variable-time binary extended-GCD inverse, and blinding that returns its factor |

Inputs are 1 to 1024 little-endian 64-bit limbs. A ctypes binding is in
`linking_ctypes.py`.

```bash
cd crypto_pipeline
make clean && make shared all
./test_suite                 # functional smoke test
python3 tests/test_arithmetic.py   # differential tests against Python integers
make bounds-sanitize         # boundary tests under ASan/UBSan
python3 benchmark_multiply.py
```

CI (**Cryptographic Wrapper Validation Pipeline**) runs clang-tidy, Valgrind, the
sanitizer bounds tests and the Python oracle. The **AES-SEC-001 Governance**
workflow scans for review-required memory primitives. Each one needs a recorded
disposition in `docs/engineering/AES-SEC-001-review-dispositions.json`.

A separate Montgomery constant-flow candidate is under review in PR #9. It is
**not cleared for secret use**, because a local timing test crossed the dudect
threshold. The scoped formal-verification scripts in `crypto_pipeline/proof/`
target that candidate. They become runnable once PR #9 lands.

See the [crypto pipeline README](crypto_pipeline/README.md),
[mathematical notes](crypto_pipeline/docs/engineering/CRYPTO-ARITHMETIC.md) and
[memory review](crypto_pipeline/docs/engineering/CRYPTO-MEMORY-REVIEW.md).

---

## Election Result Reporting

The repository also contains a standard-library Python workflow for comparing dated
Washington, Alaska and Maine election-result snapshots. The experimental analyzer
preserves descriptive tallies and separate lifecycle/tabulation stages. Conditional
count scenarios require dated counting evidence and an explicit target and
information time; historical, legacy and unsupported RCV records withhold forecasts.
Predictive calibration remains unestablished; see the
[election audit tracker](https://github.com/dlworrell/code-noodling/issues/45).

Pick the states and counties to follow in
[`election-analysis/data/watch.toml`](election-analysis/data/watch.toml): Washington (King County and
its neighbors), Alaska (statewide and every borough and census area) and Maine
(statewide and every county). From polls closing until certification, the **Election Source
Watch** workflow fetches each new official result snapshot, converts it to
plain-text CSV, records its source and checksum, and opens a pull request. Merging it
runs **Election Result Reports**, which uses the Engineering Documents Toolkit (EDT) to
publish and validate each election's report under `election-analysis/reports/<state>/<election day>/`.
Snapshots can also be added by hand under `election-analysis/data/elections/<state>/<election day>/input/`.
The generated `facebook-summary.txt` is a concise, copy-ready social post with the
leading change probabilities, reliability scores, and an explicit model caveat.

See [Election Result Reporting](election-analysis/docs/election-reporting.md) for input requirements,
formulas, risk-band definitions, scope controls, and local execution.

---

## Licensing

Status: decided (PRM-0011 Slice 0) and applied, following the REUSE
Specification (Catylist ADR-0007 §2.4), to the governed scopes
`crypto_pipeline/` and `election-analysis/`: each has its own `LICENSES/`, and
every file carries an SPDX header or is covered by the scope's `REUSE.toml`.
Elsewhere in the repository the [License](#license) notice below still
applies until each project adopts it.

These decisions are recorded here because AES-003 does not yet define a
`licensing` field for `aes-manifest.yaml`. A proposal to add one is pending in
`dlworrell/AES`. When it is adopted, this table moves into the manifest.

| Path | Rights holder | Copyright text | SPDX identifier |
|---|---|---|---|
| Repository default (everything not listed below; as for all public `dlworrell` repositories unless otherwise noted) | Donovan Worrell | `2026 Donovan Worrell` | `Apache-2.0` |
| `election-analysis/data/elections/washington/*/input/king-county/` | King County Elections (third party) | `King County Elections` | `LicenseRef-KingCounty-PublicData` |
| `election-analysis/data/elections/washington/*/{input,raw}/{snohomish,pierce,kittitas,chelan}-county/` | Each county's Auditor (Elections) (third party) | `<County> County Auditor (Elections)` | `LicenseRef-<County>County-PublicData` |
| `election-analysis/data/elections/washington/*/{input,raw}/washington/` | Washington Secretary of State (third party) | `Washington Secretary of State` | `LicenseRef-WA-SOS-PublicData` |
| `election-analysis/data/elections/washington/*/areas/washington/` (derived from the Secretary of State's precinct workbook) | Washington Secretary of State (third party) | `Washington Secretary of State` | `LicenseRef-WA-SOS-PublicData` |
| `election-analysis/data/elections/alaska/*/{input,raw}/` and `election-analysis/data/crosswalks/alaska/*.csv` (the crosswalk derived from the Division's precinct layer and Census blocks) | State of Alaska, Division of Elections (third party) | `State of Alaska, Division of Elections` | `LicenseRef-AK-DOE-PublicData` |
| `election-analysis/data/elections/maine/*/{input,raw}/` | Maine Secretary of State (third party) | `Maine Secretary of State` | `LicenseRef-ME-SOS-PublicData` |
| `election-analysis/reports/` (generated election reports) | Donovan Worrell | `2026 Donovan Worrell` | `CC-BY-4.0` |

- The `LicenseRef-` identifiers cover public datasets whose publishers state no
  license (terms UNKNOWN where not reviewed). Their texts are in
  [`election-analysis/LICENSES/`](election-analysis/LICENSES/); per-snapshot
  source URLs, retrieval times and SHA-256 are in each election's
  `SOURCES.json`. The county, Alaska and Maine texts cover data arriving from
  the 2026-11-03 election on.
- Election source data moved to `election-analysis/data/` (formerly `election-data/`);
  these rows moved with it.
- Files derived from public data take the license of their source documents
  (decided 2026-10-23).
- Data from other publishers added later (other Washington counties, the
  Alaska Division of Elections, the Maine Secretary of State) takes the same
  form: the publisher as rights holder and a matching `LicenseRef-`
  identifier.

---

## Author

**Donovan Worrell**  
Seattle, WA, USA  
donovan.worrell@gmail.com  

---

## License

MIT License © 2025 Donovan Worrell  
Permission is granted to use, modify, and distribute this software with attribution.

This MIT notice remains the stated license until the license files and headers
recorded under [Licensing](#licensing) are added. The repository default then
becomes Apache-2.0.

## Releasing

Changes move `dev-daily` → `fitment` → `main`, and a release is published only after the owner approves it. See [docs/RELEASING.md](docs/RELEASING.md).
