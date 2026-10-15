# Prime Number and Dice Suite

This directory collects the prime-number generators and the dice programs that
create or consume their prime-list output. The maintained portable path is a
C11 CPU sieve feeding a C++17 mathematical dice engine. CUDA, PhysX, and Maya
components extend the experiments when their hardware or proprietary
dependencies are available.

Prime values are used only to make pseudo-random runs reproducible. They do not
add entropy, make the generators cryptographically secure, or substitute for an
operating-system randomness source.

## Layout

```text
prime-number-suite/
├── README.md
├── README.legacy.txt            archived pre-reorganization readme
├── CMakeLists.txt               generators, CPU dice, PhysX, CTest
├── generators/
│   ├── ose.c                    CPU segmented 6-wheel sieve (C11)
│   └── cuda_sieve_mgpu.cu       multi-GPU odd-only sieve (CUDA, sm_37 default)
├── dice/                        self-contained Autotools package
│   ├── configure.ac
│   ├── Makefile.am
│   ├── README.md
│   ├── src/dice_cpu.cpp         unbiased CPU dice engine (C++17)
│   └── tests/                   make check shell tests
├── physx/
│   ├── physx_dice_multi.cpp     multi-die PhysX simulator
│   ├── die_mesh.hpp             PhysX mesh and face-normal helpers
│   └── maya_dice.py             Maya-only mesh generator prototype
├── legacy/                      reference-only prototypes (not built)
│   ├── README.md
│   ├── dice_roll_improved.cpp
│   ├── dice_roll_with_physX.cpp
│   └── physx_dice.cpp
└── tests/                       CTest integration tests (Python)
    ├── test_dice_cpu.py
    └── test_ose.py
```

## Component status

| Component | Source | Build | Status |
|---|---|---|---|
| CPU sieve | [`generators/ose.c`](generators/ose.c) | CMake | Maintained; black-box tests in CTest |
| CUDA multi-GPU sieve | [`generators/cuda_sieve_mgpu.cu`](generators/cuda_sieve_mgpu.cu) | CMake | Maintained; compile-checked in CI, runtime needs a GPU |
| CPU dice engine | [`dice/src/dice_cpu.cpp`](dice/src/dice_cpu.cpp) | Autotools (also CMake) | Maintained; `make check` and CTest |
| PhysX simulator | [`physx/physx_dice_multi.cpp`](physx/physx_dice_multi.cpp) | CMake (optional) | Maintained experiment; needs the PhysX SDK |
| Maya mesh generator | [`physx/maya_dice.py`](physx/maya_dice.py) | none | Run inside Autodesk Maya only |
| Earlier PhysX prototypes | [`legacy/`](legacy/) | none | Reference-only; see [`legacy/README.md`](legacy/README.md) |

## Building

### Generators and PhysX (CMake)

From the repository root, the portable CPU targets (`ose_cpu`, `dice_cpu`) and
their tests:

```bash
cmake -S prime-number-suite -B build \
  -DBUILD_CUDA_SIEVE_MGPU=OFF \
  -DBUILD_PHYSX_DICE=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

With CUDA and PhysX:

```bash
cmake -S prime-number-suite -B build \
  -DBUILD_CUDA_SIEVE_MGPU=ON \
  -DBUILD_PHYSX_DICE=ON \
  -DPHYSX_ROOT=/opt/physx
cmake --build build --parallel
```

| Option | Default | Effect |
|---|---|---|
| `BUILD_CUDA_SIEVE_MGPU` | `ON` | Build `cuda_sieve_mgpu`; skipped with a warning if no CUDA compiler is found |
| `BUILD_PHYSX_DICE` | `ON` | Build `physx_dice_multi`; skipped with a warning if the PhysX SDK is incomplete |
| `PHYSX_ROOT` | empty | PhysX SDK location when it is not in a standard path |
| `CMAKE_CUDA_ARCHITECTURES` | `37` | GPU targets; `37` is the Tesla K80. Use for example `"75;86"` for newer GPUs |
| `NATIVE_ARCH` | `OFF` | Add `-march=native` to host code. Leave off for CI or binaries copied to other machines |

Non-Debug builds compile host code with `-O3`. Debug builds keep the compiler's
debug defaults so sanitizers and debuggers behave normally.

### Dice engine (Autotools)

`dice/` is a standalone GNU Autotools package. See
[`dice/README.md`](dice/README.md) for details.

```bash
cd prime-number-suite/dice
autoreconf -fi
./configure
make
make check
make distcheck
```

This needs `autoconf`, `automake`, `autoconf-archive`, and a C++17 compiler
with `__uint128_t` (GCC or Clang on a 64-bit target).

## Usage

Generate primes in an inclusive range on the CPU, or up to `N` on CUDA devices:

```bash
./build/ose_cpu 2 50000000 --json primes_50M.json --no-list
./build/cuda_sieve_mgpu 50000000 --gpus 8 --seg 128M --json primes_50M.json
```

Both generators write the same JSON schema:

```json
{
  "range": {"start": 2, "end": 11},
  "count": 5,
  "primes": [2, 3, 5, 7, 11]
}
```

Roll prime-seeded dice from that list:

```bash
./build/dice_cpu --faces 20 --count 20000 --use-prime-seeds primes_50M.json --chi
./build/dice_cpu --spec '3d6+2' --count 5000 --use-prime-seeds primes_50M.json --log-json rolls.json
```

Run the physical simulator (requires PhysX; start the PhysX Visual Debugger
first to watch live):

```bash
./build/physx_dice_multi --spec 3d6+2 --spec 1d20 --trials 20000 \
  --use-prime-seeds primes_50M.json --chute --pvd 127.0.0.1:5425 --chi
```

## Continuous integration

| Workflow | Triggers | What it checks |
|---|---|---|
| [`dice-autotools.yml`](../.github/workflows/dice-autotools.yml) | changes to `dice/` | `autoreconf`, `configure`, `make`, `make check`, `make distcheck` |
| [`prime-suite.yml`](../.github/workflows/prime-suite.yml) | changes to this suite outside `dice/` | CPU build and CTest (also with ASan/UBSan); CUDA compile-only builds |
| [`k80-hardware.yml`](../.github/workflows/k80-hardware.yml) | manual only | sm_37 build and GPU-vs-CPU prime comparison on a self-hosted K80 runner |

The CUDA job in `prime-suite.yml` has two compile-only builds, because hosted
runners have no GPU:

- **legacy (required):** `nvidia/cuda:11.4.3-devel-ubuntu20.04` (GCC 9),
  `CMAKE_CUDA_ARCHITECTURES=37`. This build must stay green.
- **modern (informational):** CUDA 12.6 for sm_75 and sm_86. It may fail
  without blocking, and it never replaces the sm_37 build.

## Legacy GPU support (Tesla K80)

The primary GPU target is 4x Tesla K80 in a Cubix PCIe expander. K80 support is
a hard requirement: do not drop `sm_37` or require CUDA 12.

- **Architecture:** the K80 is Kepler, compute capability 3.7 (`sm_37`).
- **CUDA version:** CUDA 12 removed `sm_37`, so builds for the K80 need
  CUDA 11.x or older. This project targets CUDA 11.4.
- **Driver:** use the NVIDIA 470 driver branch, the last branch that supports
  Kepler. It supports up to CUDA 11.4.
- **Host compiler:** CUDA 11.4's `nvcc` supports GCC 10 or older. Ubuntu 20.04's
  default GCC 9 works. On newer distributions, install `gcc-10`/`g++-10` and
  pass `-DCMAKE_CUDA_HOST_COMPILER=g++-10`.
- **Device count:** each K80 board carries two GK210 GPUs, so four boards
  appear as 8 CUDA devices. Use `--gpus 8` to spread the sieve across all of
  them.
- **Warnings:** CUDA 11.x warns that `sm_37` is deprecated. CI passes
  `-Wno-deprecated-gpu-targets` to silence it.

## Self-hosted K80 runner

`k80-hardware.yml` stays inert until a runner labeled `k80` is registered on
the K80 machine. To set one up:

1. Install the NVIDIA 470 driver, the CUDA 11.4 toolkit, GCC 10 or older,
   CMake 3.20 or newer, `git`, and Python 3. Confirm that `nvidia-smi` lists
   8 devices and that `nvcc --version` reports 11.4.
2. On GitHub, open the repository's **Settings → Actions → Runners → New
   self-hosted runner** and choose Linux x64. Follow the download steps shown
   there, then register the runner with the extra label `k80`:

   ```bash
   ./config.sh --url https://github.com/dlworrell/code-noodling \
     --token <TOKEN-FROM-THAT-PAGE> --labels k80
   ```

3. Install and start it as a service so it survives reboots:

   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

4. Run the workflow from **Actions → K80 Hardware Check → Run workflow**. It
   builds `cuda_sieve_mgpu` for `sm_37`, sieves up to the requested limit on
   the selected GPUs, and checks that the primes match `ose_cpu`.

The workflow is `workflow_dispatch` only. Do not add `push` or `pull_request`
triggers: jobs from pull requests must never run on self-hosted hardware.

## Notes

- `physx_dice_multi` geometry, face mapping, settling thresholds, and
  statistics need review on an installed PhysX SDK. CI cannot build or run it.
- `physx/maya_dice.py` is guarded against running on import but must still be
  run inside Maya.
- The dice parser accepts trailing characters after a valid spec (`2d6*` reads
  as `2d6`). This is long-standing behavior.
- `README.legacy.txt` is the original readme, kept for history. Its paths
  describe the old root-level layout.
