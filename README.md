# Code Noodling

`code-noodling` is an experimental monorepo for small engineering and analysis
projects. Each codebase owns its source, build configuration, tests, generated
outputs, and detailed documentation in a dedicated directory. `README.md` is
the only tracked file stored directly at the repository root.

## Codebases

| Codebase | Location | Purpose | Entry documentation |
|---|---|---|---|
| Prime Number and Dice Suite | [`prime-number-suite/`](prime-number-suite/) | CPU/CUDA prime generation, deterministic prime-seeded dice, and PhysX/Maya experiments | [`prime-number-suite/README.md`](prime-number-suite/README.md) |
| Election Result Reporting | [`election_reporting/`](election_reporting/) and [`election-data/`](election-data/) | Parse Washington election snapshots and estimate the probability that reported decision boundaries change | [`docs/election-reporting.md`](docs/election-reporting.md) |

## Repository Layout

```text
code-noodling/
├── README.md                   repository index (only loose root file)
├── .github/                    GitHub workflows and contribution templates
├── docs/                       shared and election-reporting documentation
├── election-data/              election configuration and source snapshots
├── election_reporting/         election-analysis Python package
├── prime-number-suite/         prime generators and prime-seeded dice code
├── reports/                    generated election reports and EDT evidence
└── tests/                      election-reporting tests
```

The `.github/` directory remains at repository level because GitHub requires
that location for Actions workflows and contribution templates. Existing
election-reporting directories retain their established paths.

## Quick Start

For the prime and dice programs:

```bash
cmake -S prime-number-suite -B prime-number-suite/build \
  -DBUILD_CUDA_SIEVE_MGPU=OFF \
  -DBUILD_PHYSX_DICE=OFF
cmake --build prime-number-suite/build --parallel
ctest --test-dir prime-number-suite/build --output-on-failure
```

For election reporting:

```bash
python -m unittest discover -s tests -p 'test_election_reporting.py' -v
python -m election_reporting \
  --config election-data/election.toml \
  --output reports/elections
```

See each codebase's documentation for prerequisites, optional hardware support,
input formats, model limitations, and review status.

## Engineering Governance

C and C++ changes inherit the repository's AES-SEC-001 secure-coding profile.
The local profile and waiver log live under [`docs/engineering/`](docs/engineering/).
The prime/dice suite's compiler-analysis configuration now lives beside the
code it governs at [`prime-number-suite/.clang-tidy`](prime-number-suite/.clang-tidy).

## Author and License

Donovan Worrell — Seattle, Washington, USA

MIT License © 2025 Donovan Worrell. Permission is granted to use, modify, and
distribute this software with attribution.
