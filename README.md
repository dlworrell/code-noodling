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