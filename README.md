# Code Noodling

`code-noodling` is an experimental monorepo for small engineering and analysis
projects. Each codebase keeps its source, build configuration, tests, and
detailed documentation in its own directory.

## Codebases

| Codebase | Location | Purpose | Documentation |
|---|---|---|---|
| Prime Number and Dice Suite | [`prime-number-suite/`](prime-number-suite/) | CPU and multi-GPU (Tesla K80) prime sieves, an unbiased prime-seeded dice engine, and PhysX/Maya dice experiments | [`prime-number-suite/README.md`](prime-number-suite/README.md) |
| Crypto Pipeline | [`crypto_pipeline/`](crypto_pipeline/) | Multi-precision integer multiplication experiments (research prototype) | [`crypto_pipeline/README.md`](crypto_pipeline/README.md) |
| Election Result Reporting | [`election_reporting/`](election_reporting/), [`election-data/`](election-data/) | Washington election snapshot comparison and change-probability reports | [`docs/election-reporting.md`](docs/election-reporting.md) |

## Repository Layout

```text
code-noodling/
├── README.md
├── .clang-tidy              repository C/C++ static-analysis policy
├── aes-manifest.yaml        repository manifest and governed project scopes
├── .github/                 workflows and contribution templates
├── crypto_pipeline/         integer multiplication experiments (governed scope)
├── docs/                    engineering standards and election docs
├── election-data/           election configuration and source snapshots
├── election_reporting/      election-analysis Python package
├── prime-number-suite/      prime generators and dice programs
├── reports/                 generated election reports
└── tests/                   election-reporting tests
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

Everything else is the repository-level remainder until it is migrated. Until
AEMS can scan scopes separately, the repository-wide AES-SEC-001 scan and
ledger stay authoritative for every file.

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
Washington election-result snapshots and estimating the statistical probability that
each current winner, top-two cutoff, or judicial majority status changes before
certification. Every probability carries a separate 0–100 evidence-reliability score.

Drop official county CSV exports or Washington `All Results` XLSX workbooks into
`election-data/input/<source>/` with a date in each filename, then run the GitHub
Actions workflow named **Election Result Reports**. It uses the Engineering Documents
Toolkit (EDT) to publish and validate the report under `reports/elections/`.
The generated `facebook-summary.txt` is a concise, copy-ready social post with the
leading change probabilities, reliability scores, and an explicit model caveat.

See [Election Result Reporting](docs/election-reporting.md) for input requirements,
formulas, risk-band definitions, scope controls, and local execution.

---

## Licensing

Status: decided, not yet applied (PRM-0011 Slice 0). No license files or file
headers have been added yet; Slice 1 adds them following the REUSE
Specification (Catylist ADR-0007 §2.4). Until then, the [License](#license)
notice below still applies.

These decisions are recorded here because AES-003 does not yet define a
`licensing` field for `aes-manifest.yaml`. A proposal to add one is pending in
`dlworrell/AES`. When it is adopted, this table moves into the manifest.

| Path | Rights holder | Copyright text | SPDX identifier |
|---|---|---|---|
| Repository default (all code and documentation not listed below) | Donovan Worrell | `2026 Donovan Worrell` | `Apache-2.0` |
| `election-data/input/king-county/` | King County Elections (third party) | `King County Elections` | `LicenseRef-KingCounty-PublicData` |
| `election-data/input/washington/` | Washington Secretary of State (third party) | `Washington Secretary of State` | `LicenseRef-WA-SOS-PublicData` |
| `reports/elections/` (generated election reports) | Donovan Worrell | `2026 Donovan Worrell` | `CC-BY-4.0` |

- The two `LicenseRef-` identifiers cover public datasets published with no
  license stated. Their drafts in [`LICENSES/`](LICENSES/) list each file's
  source URL, retrieval date (UNKNOWN where none was recorded) and SHA-256.
- Election source data moves to `election-analysis/data/input/` in a later
  change; these rows move with it.
- Data from other Washington counties added later takes the same form: the
  county elections office as rights holder and a matching `LicenseRef-`
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
