# 🎲 Code Noodling  
### Prime Generation + Physical Dice Simulation Suite  

An experimental C/C++/CUDA project exploring:  
- **Optimized prime sieving** (CPU + multi-GPU CUDA)  
- **Accurate dice randomness** for tabletop and RPG mechanics  
- **Physically simulated dice rolls** using NVIDIA PhysX  
- **Auto-tuned performance** across CPU/GPU hardware  
- **Live visualization** with the PhysX Visual Debugger (PVD)

---

## 🚀 Features

| Component | Description |
|------------|-------------|
| 🧮 **OSE / CUDA Sieve** | Optimized Sieve of Eratosthenes — segmented, odd-only, GPU-parallel, JSON export |
| ⚙️ **Dice Engine (CPU)** | Pure math engine for `NdM+K` expressions — unbiased, optional prime-seeded randomness, chi-square validation |
| 🧊 **PhysX Dice Simulator** | Fully physical D6, D8, D12, D20 dice with realistic collisions, ramp chute, multi-threading, and GPU acceleration |
| 🔭 **Visualization** | Optional live streaming via `--pvd` to PhysX Visual Debugger or NVIDIA Omniverse View |
| 📊 **Data Outputs** | JSON and CSV result logging, with chi-square test of fairness and probability distribution graphs |

---

## 🏗️ Build Instructions

### Prerequisites
- **CMake ≥ 3.20**
- **C++17 compiler** (clang / gcc / MSVC)
- *(Optional)* **CUDA Toolkit** (for GPU sieve)
- *(Optional)* **NVIDIA PhysX SDK** (for physics dice)
- *(Optional)* **PhysX Visual Debugger (PVD)** — to visualize rolls live

### 🧩 Build All Targets

```bash
mkdir build && cd build
cmake .. -DBUILD_CUDA_SIEVE_MGPU=ON -DBUILD_PHYSX_DICE=ON
cmake --build . -j
```
##🧠 Tesla K80 users: CUDA architecture defaults to sm_37.
#Override with: -DCMAKE_CUDA_ARCHITECTURES=37 (or 80 for A100s, etc.)

### 💻 CPU Dice Engine

```bash

# Fair D6, 10k rolls, chi-square test
./dice_cpu --faces 6 --count 10000 --chi

# D20, 20k rolls, prime-seeded, CSV output
./dice_cpu --faces 20 --count 20000 --use-prime-seeds primes_50M.json --csv d20.csv --chi

# #RPG Expression Example
./dice_cpu --spec "3d6+2" --count 5000 --use-prime-seeds primes_50M.json --log-json rolls.json --chi
```
# Flags:
	•	--spec NdM+K → Dice expression (can repeat)
	•	--count N → Number of rolls per set
	•	--chi → Enable chi-square fairness check
	•	--csv / --log-json → Export results

###🧮 CUDA Multi-GPU Prime Generator

```bash
‘./cuda_sieve_mgpu 50000000 --gpus 4 --seg 256M --json primes_50M.json

# Generates all primes ≤ 50 M using up to 4 Tesla K80 GPUs, writing them to primes_50M.json.

###🧊 PhysX Dice Simulator (D6/D8/D12/D20)

```bash
# Basic D6 simulation with 50k rolls and chi-square test
./physx_dice_multi --spec 1d6 --trials 50000 --chi

# Complex run with multiple dice, chute ramp, PVD visualsation, and JSON/CSV output
./physx_dice_multi \
  --spec 3d6+2 --spec 1d8 --spec 1d12 --spec 1d20 \
  --trials 20000 \
  --use-prime-seeds primes_50M.json \
  --chute \
  --pvd 127.0.0.1:5425 \
  --json physx_runs.json --csv physx_counts.csv --chi
```
### Flags:
	•	--chute → Adds an angled ramp, side walls, and backstop for realism
	•	--pvd [host:port] → Stream live to PhysX Visual Debugger (default 127.0.0.1:5425)
	•	--use-prime-seeds → Deterministic seed list from JSON primes
	•	--json / --csv → Write results to disk

# 💡 Make sure PVD is running before launching your sim to see live dice tumbling!

### 📁 Repository Layout

###🧰 Technical Notes
	•	Hardware autodetect: automatically uses all CPU threads and available CUDA devices.
	•	Deterministic seeding: optional — uses primes for stable pseudo-random sequences.
	•	PhysX cooking: convex meshes generated internally for D8, D12, D20 via die_mesh.h.
	•	Chute geometry: procedural — ramp (30° tilt) + sidewalls + backstop.
	•	Visualization: controlled via --pvd; debug draw enabled for collision shapes & contacts.
	•	Data outputs: JSON + CSV + chi-square test summary.

### 🧪 Example Workflow

# Step 1: Generate prime seeds
./cuda_sieve_mgpu 50000000 --json primes_50M.json

# Step 2: Simulate physical dice using those seeds
./physx_dice_multi --spec 2d6+3 --trials 10000 --use-prime-seeds primes_50M.json --chute --pvd --chi

###🧑‍💻 Author

Donovan Worrell
Seattle, WA, USA
#📧 donovan.worrell@gmail.com

###🪪 License

MIT License © 2025 Donovan Worrell
Permission is granted to use, modify, and distribute this software with attribution


