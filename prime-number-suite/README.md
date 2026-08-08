# Prime Number and Dice Suite

This directory collects every prime-number generator and every dice codebase
that creates or consumes the generators' prime-list output. The maintained
portable path is a C11 CPU sieve feeding a C++17 mathematical dice engine;
CUDA, PhysX, and Maya components extend the experiments when their proprietary
or hardware-specific dependencies are available.

Prime values are used only to make pseudo-random test runs reproducible. They
do not add entropy, make the generators cryptographically secure, or substitute
for an operating-system randomness source.

## Component Status

| Component | Source | Status | Verification path |
|---|---|---|---|
| CPU segmented 6-wheel sieve | [`generators/ose.c`](generators/ose.c) | Maintained | Warning-clean C11 build and black-box correctness tests |
| CUDA multi-GPU odd-only sieve | [`generators/cuda_sieve_mgpu.cu`](generators/cuda_sieve_mgpu.cu) | Maintained, hardware-dependent | CUDA build; compare JSON output with `ose_cpu` |
| CPU dice engine | [`dice/dice_cpu.cc`](dice/dice_cpu.cc) | Maintained | Warning-clean C++17 build and deterministic integration tests |
| Multi-die PhysX simulator | [`dice/physx_dice_multi.cpp`](dice/physx_dice_multi.cpp) | Maintained experiment, SDK-dependent | Optional CMake target when a complete PhysX SDK is found |
| PhysX mesh helpers | [`dice/die_mesh.h`](dice/die_mesh.h) | Maintained with PhysX simulator | Compiled through `physx_dice_multi` |
| Maya mesh generator | [`dice/maya_dice.py`](dice/maya_dice.py) | Maya-only prototype | Manual execution inside Autodesk Maya |
| Earlier PhysX prototypes | [`dice/legacy/`](dice/legacy/) | Reference-only | Excluded from CMake; limitations documented in its README |

## Directory Layout

```text
prime-number-suite/
├── README.md
├── README.legacy.txt          archived pre-reorganization documentation
├── CMakeLists.txt
├── .clang-tidy
├── .gitignore
├── generators/
│   ├── ose.c
│   └── cuda_sieve_mgpu.cu
├── dice/
│   ├── dice_cpu.cc
│   ├── die_mesh.h
│   ├── maya_dice.py
│   ├── physx_dice_multi.cpp
│   └── legacy/
│       ├── README.md
│       ├── dice_roll_improved.cpp
│       ├── dice_roll_with_physx.cpp
│       └── physx_dice.cpp
└── tests/
    ├── test_dice_cpu.py
    └── test_ose.py
```

Generated JSON, CSV, logs, compiler products, and build trees are ignored only
within this suite, leaving the election-reporting codebase's versioned evidence
rules unaffected.

## Portable Build and Tests

From the repository root:

```bash
cmake -S prime-number-suite -B prime-number-suite/build \
  -DBUILD_CUDA_SIEVE_MGPU=OFF \
  -DBUILD_PHYSX_DICE=OFF
cmake --build prime-number-suite/build --parallel
ctest --test-dir prime-number-suite/build --output-on-failure
```

The two portable programs can also be reviewed without CMake:

```bash
cc -O2 -std=c11 -Wall -Wextra -Wpedantic \
  prime-number-suite/generators/ose.c -lm -o ose_cpu

c++ -O2 -std=c++17 -Wall -Wextra -Wpedantic \
  prime-number-suite/dice/dice_cpu.cc -o dice_cpu
```

The CMake options are:

- `BUILD_CUDA_SIEVE_MGPU`: attempts the CUDA target; defaults to `ON` and skips
  it with a warning if no CUDA compiler is present.
- `BUILD_PHYSX_DICE`: attempts the maintained PhysX target; defaults to `ON`
  and skips it if the complete SDK is not found.
- `PHYSX_ROOT`: optional path to the PhysX SDK.
- `CMAKE_CUDA_ARCHITECTURES`: defaults to `37` for the original Tesla K80
  target and can be overridden for newer GPUs.

## Prime Generation

Generate primes in an inclusive CPU range:

```bash
./prime-number-suite/build/ose_cpu 2 50000000 \
  --json prime-number-suite/primes_50M.json \
  --no-list
```

Generate every prime from 2 through an upper bound on CUDA devices:

```bash
./prime-number-suite/build/cuda_sieve_mgpu 50000000 \
  --gpus 4 \
  --seg 128M \
  --json prime-number-suite/primes_50M.json
```

Both generators emit this stable schema:

```json
{
  "range": {"start": 2, "end": 11},
  "count": 5,
  "primes": [2, 3, 5, 7, 11]
}
```

The CPU generator accepts any inclusive `START END` range. The CUDA generator
accepts an upper bound and therefore always starts at 2. `--json -` writes only
JSON to standard output so its output can be piped safely.

## Prime-Seeded Dice

Run a deterministic D20 experiment using the generated prime list:

```bash
./prime-number-suite/build/dice_cpu \
  --faces 20 \
  --count 20000 \
  --use-prime-seeds prime-number-suite/primes_50M.json \
  --csv prime-number-suite/d20.csv \
  --chi
```

Run an `NdM+K` expression:

```bash
./prime-number-suite/build/dice_cpu \
  --spec '3d6+2' \
  --count 5000 \
  --use-prime-seeds prime-number-suite/primes_50M.json \
  --log-json prime-number-suite/rolls.json \
  --chi
```

`--seed-per-roll` is the default. `--seed-per-bundle` uses one prime-derived
seed for each complete `NdM+K` bundle. Each seed mixes the prime with its
logical roll index, so wrapping a short prime list does not repeat an identical
RNG state. Runs without a prime file use a fixed documented seed and are also
reproducible.

## PhysX and Maya Notes

`physx_dice_multi` supports D6, D8, D12, and D20 physical simulations plus a
virtual fallback for other face counts. It can create a chute with `--chute`
and write aggregate JSON/CSV output. Its geometry, face mapping, settling
thresholds, and statistical behavior require review on an installed PhysX SDK;
the portable CI path cannot validate those proprietary runtime behaviors.

`dice/maya_dice.py` is guarded against execution on import but must still run
inside Maya. It is a mesh-generation prototype, not part of the statistical
test path.

## Review Evidence

The integration tests cover:

- inclusive prime boundaries, negative starts, empty ranges, and a range near
  one million;
- valid JSON to standard output and files;
- rejection of reversed ranges and invalid column counts;
- reproducible prime-seeded D20 output;
- sequence-index mixing when a short prime list wraps;
- `3d6+2` bounds and JSON/CSV record counts.

CUDA and PhysX remain explicit hardware/SDK review items. Their source headers
document partitioning, memory bounds, ownership, seeding, and known dependency
constraints so a reviewer can evaluate those paths without relying on the old
root-level layout.
