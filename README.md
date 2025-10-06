# ๐ฒ Code Noodling  
### Prime Generation + Physical Dice Simulation Suite  

An experimental C/C++/CUDA project exploring:  
- **Optimized prime sieving** (CPU + multi-GPU CUDA)  
- **Accurate dice randomness** for tabletop and RPG mechanics  
- **Physically simulated dice rolls** using NVIDIA PhysX  
- **Auto-tuned performance** across CPU/GPU hardware  
- **Live visualization** with the PhysX Visual Debugger (PVD)

---

## ๐ Features

| Component | Description |
|------------|-------------|
| ๐งฎ **OSE / CUDA Sieve** | Optimized Sieve of Eratosthenes โ segmented, odd-only, GPU-parallel, JSON export |
| โ๏ธ **Dice Engine (CPU)** | Pure math engine for `NdM+K` expressions โ unbiased, optional prime-seeded randomness, chi-square validation |
| ๐ง **PhysX Dice Simulator** | Fully physical D6, D8, D12, D20 dice with realistic collisions, ramp chute, multi-threading, and GPU acceleration |
| ๐ญ **Visualization** | Optional live streaming via `--pvd` to PhysX Visual Debugger or NVIDIA Omniverse View |
| ๐ **Data Outputs** | JSON and CSV result logging, with chi-square test of fairness and probability distribution graphs |

---

## ๐๏ธ Build Instructions

### Prerequisites
- **CMake โฅ 3.20**
- **C++17 compiler** (clang / gcc / MSVC)
- *(Optional)* **CUDA Toolkit** (for GPU sieve)
- *(Optional)* **NVIDIA PhysX SDK** (for physics dice)
- *(Optional)* **PhysX Visual Debugger (PVD)** โ to visualize rolls live

### ๐งฉ Build All Targets

```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA_SIEVE_MGPU=ON -DBUILD_PHYSX_DICE=ON
cmake --build . -j
```

> ๐ง  **Tesla K80 users:** CUDA architecture defaults to `sm_37`.  
> Override with:  
> `-DCMAKE_CUDA_ARCHITECTURES=37` *(or `80` for A100s, etc.)*

---

## ๐ป CPU Dice Engine

```bash
# Fair D6, 10k rolls, chi-square test
./dice_cpu --faces 6 --count 10000 --chi

# D20, 20k rolls, prime-seeded, CSV output
./dice_cpu --faces 20 --count 20000 --use-prime-seeds primes_50M.json --csv d20.csv --chi

# RPG Expression Example
./dice_cpu --spec "3d6+2" --count 5000 --use-prime-seeds primes_50M.json --log-json rolls.json --chi
```

**Flags**
- `--spec NdM+K` โ Dice expression (can repeat)  
- `--count N` โ Number of rolls per set  
- `--chi` โ Enable chi-square fairness check  
- `--csv` / `--log-json` โ Export results  

---

## ๐งฎ CUDA Multi-GPU Prime Generator

```bash
./cuda_sieve_mgpu 50000000 --gpus 4 --seg 256M --json primes_50M.json
```

Generates all primes โค 50 M using up to four Tesla K80 GPUs, writing them to `primes_50M.json`.

---

## ๐ง PhysX Dice Simulator (D6/D8/D12/D20)

```bash
# Basic D6 simulation with 50k rolls and chi-square test
./physx_dice_multi --spec 1d6 --trials 50000 --chi

# Complex run with multiple dice, chute ramp, PVD visualization, and JSON/CSV output
./physx_dice_multi   --spec 3d6+2 --spec 1d8 --spec 1d12 --spec 1d20   --trials 20000   --use-prime-seeds primes_50M.json   --chute   --pvd 127.0.0.1:5425   --json physx_runs.json --csv physx_counts.csv --chi
```

**Flags**
- `--chute` โ Adds an angled ramp, side walls, and backstop for realism  
- `--pvd [host:port]` โ Stream live to PhysX Visual Debugger *(default: 127.0.0.1 : 5425)*  
- `--use-prime-seeds` โ Deterministic seed list from JSON primes  
- `--json` / `--csv` โ Write results to disk  

> ๐ก Make sure PVD is running **before** launching your sim to see live dice tumbling!

---

## ๐ Repository Layout

```text
code-noodling/
โโโ CMakeLists.txt
โโโ .gitignore
โโโ cuda_sieve_mgpu.cu
โโโ dice_cpu.cpp
โโโ physx_dice_multi.cpp
โโโ die_mesh.h
โโโ OSE.c                      (optional legacy)
โโโ OSE_CUDA.cc                (optional legacy)
โโโ README.md
โโโ primes_50M.json            (generated at runtime)
```

---

## ๐งฐ Technical Notes

- **Hardware autodetect:** automatically uses all CPU threads and available CUDA devices.  
- **Deterministic seeding:** optional โ uses primes for stable pseudo-random sequences.  
- **PhysX cooking:** convex meshes generated internally for D8, D12, D20 via `die_mesh.h`.  
- **Chute geometry:** procedural โ ramp (30ยฐ tilt) + sidewalls + backstop.  
- **Visualization:** controlled via `--pvd`; debug draw enabled for collision shapes & contacts.  
- **Data outputs:** JSON + CSV + chi-square test summary.  

---

## ๐งช Example Workflow

```bash
# Step 1 โ Generate prime seeds
./cuda_sieve_mgpu 50000000 --json primes_50M.json

# Step 2 โ Simulate physical dice using those seeds
./physx_dice_multi --spec 2d6+3 --trials 10000 --use-prime-seeds primes_50M.json --chute --pvd --chi
```

---

## ๐งโ๐ป Author

**Donovan Worrell**  
Seattle, WA, USA  
๐ง donovan.worrell@gmail.com  

---

## ๐ชช License

MIT License ยฉ 2025 Donovan Worrell  
Permission is granted to use, modify, and distribute this software with attribution.
