# Code Noodling  
### Prime Generation + Physical Dice Simulation Suite  

An experimental C/C++/CUDA project exploring:  
- **Optimized prime sieving** (CPU + multi-GPU CUDA)  
- **Accurate dice randomness** for tabletop and RPG mechanics  
- **Physically simulated dice rolls** using NVIDIA PhysX  
- **Auto-tuned performance** across CPU/GPU hardware  
- **Live visualization** with the PhysX Visual Debugger (PVD)
- **Multi-precision integer arithmetic** experiments (Comba, Karatsuba, NTT, Montgomery) in `crypto_pipeline/`

---

## Features

| Component | Description |
|------------|-------------|
| **OSE / CUDA Sieve** | Optimized Sieve of Eratosthenes segmented, odd-only, GPU-parallel, JSON export |
| **Dice Engine (CPU)** | Pure math engine for `NdM+K` expressions unbiased, optional prime-seeded randomness, chi-square validation |
| **PhysX Dice Simulator** | Fully physical D6, D8, D12, D20 dice with realistic collisions, ramp chute, multi-threading, and GPU acceleration |
| **Visualization** | Optional live streaming via `--pvd` to PhysX Visual Debugger or NVIDIA Omniverse View |
| **Data Outputs** | JSON and CSV result logging, with chi-square test of fairness and probability distribution graphs |
| **Crypto Pipeline** | Experimental 64-bit-limb multiplication and modular arithmetic in C, checked against Python integers; research prototype, not production cryptography |

---

## Build Instructions

### Prerequisites
- **CMake 3.20**
- **C++17 compiler** (clang / gcc / MSVC)
- *(Optional)* **CUDA Toolkit** (for GPU sieve)
- *(Optional)* **NVIDIA PhysX SDK** (for physics dice)
- *(Optional)* **PhysX Visual Debugger (PVD)** to visualize rolls live

### Build All Targets

```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA_SIEVE_MGPU=ON -DBUILD_PHYSX_DICE=ON
cmake --build . -j
```

> **Tesla K80 users:** CUDA architecture defaults to `sm_37`.  
> Override with:  
> `-DCMAKE_CUDA_ARCHITECTURES=37` *(or `80` for A100s, etc.)*

---

## CPU Dice Engine

```bash
# Fair D6, 10k rolls, chi-square test
./dice_cpu --faces 6 --count 10000 --chi

# D20, 20k rolls, prime-seeded, CSV output
./dice_cpu --faces 20 --count 20000 --use-prime-seeds primes_50M.json --csv d20.csv --chi

# RPG Expression Example
./dice_cpu --spec "3d6+2" --count 5000 --use-prime-seeds primes_50M.json --log-json rolls.json --chi
```

**Flags**
- `--spec NdM+K` Dice expression (can repeat)  
- `--count N` Number of rolls per set  
- `--chi` Enable chi-square fairness check  
- `--csv` / `--log-json` โ Export results  

---

## CUDA Multi-GPU Prime Generator

```bash
./cuda_sieve_mgpu 50000000 --gpus 4 --seg 256M --json primes_50M.json
```

Generates all primes 50 M using up to four Tesla K80 GPUs, writing them to `primes_50M.json`.

---

## PhysX Dice Simulator (D6/D8/D12/D20)

```bash
# Basic D6 simulation with 50k rolls and chi-square test
./physx_dice_multi --spec 1d6 --trials 50000 --chi

# Complex run with multiple dice, chute ramp, PVD visualization, and JSON/CSV output
./physx_dice_multi   --spec 3d6+2 --spec 1d8 --spec 1d12 --spec 1d20   --trials 20000   --use-prime-seeds primes_50M.json   --chute   --pvd 127.0.0.1:5425   --json physx_runs.json --csv physx_counts.csv --chi
```

**Flags**
- `--chute` Adds an angled ramp, side walls, and backstop for realism  
- `--pvd [host:port]` Stream live to PhysX Visual Debugger *(default: 127.0.0.1 : 5425)*  
- `--use-prime-seeds` Deterministic seed list from JSON primes  
- `--json` / `--csv` Write results to disk  

>  Make sure PVD is running **before** launching your sim to see live dice tumbling!

---

## Repository Layout

```text
code-noodling/
CMakeLists.txt
.gitignore
cuda_sieve_mgpu.cu
dice_cpu.cpp
physx_dice_multi.cpp
die_mesh.h
OSE.c                      (optional legacy)
OSE_CUDA.cc                (optional legacy)
README.md
primes_50M.json            (generated at runtime)
```

---

## Technical Notes

- **Hardware autodetect:** automatically uses all CPU threads and available CUDA devices.  
- **Deterministic seeding:** optional uses primes for stable pseudo-random sequences.  
- **PhysX cooking:** convex meshes generated internally for D8, D12, D20 via `die_mesh.h`.  
- **Chute geometry:** procedural ramp (30° tilt) + sidewalls + backstop.  
- **Visualization:** controlled via `--pvd`; debug draw enabled for collision shapes & contacts.  
- **Data outputs:** JSON + CSV + chi-square test summary.  

---

##  Example Workflow

```bash
# Step 1 โ Generate prime seeds
./cuda_sieve_mgpu 50000000 --json primes_50M.json

# Step 2 โ Simulate physical dice using those seeds
./physx_dice_multi --spec 2d6+3 --trials 10000 --use-prime-seeds primes_50M.json --chute --pvd --chi
```

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
python3 test_arithmetic.py   # differential tests against Python integers
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

See the [crypto pipeline README](crypto_pipeline/README.MD),
[mathematical notes](docs/engineering/CRYPTO-ARITHMETIC.md) and
[memory review](docs/engineering/CRYPTO-MEMORY-REVIEW.md).

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

## Author

**Donovan Worrell**  
Seattle, WA, USA  
donovan.worrell@gmail.com  

---

## License

MIT License ยฉ 2025 Donovan Worrell  
Permission is granted to use, modify, and distribute this software with attribution.
