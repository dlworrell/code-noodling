> **Experimental prototype:** the inverse routine is unfinished and other arithmetic
> and timing defects remain. Do not use this code to protect secrets. CI checks
> compilation and memory behavior, not cryptographic security. See
> [the memory review and open defects](../docs/engineering/CRYPTO-MEMORY-REVIEW.md).
> Array counts are limited to 1..1024 limbs; buffer contracts are in the header.

# Secure Cryptographic Multiplication Wrapper Engine

![Build Status](https://github.com)

An optimized, side-channel-resistant arbitrary-precision multiplication wrapper for high-assurance cryptographic applications.

## Key Features
* **Adaptive Dispatching:** Dynamically switches algorithms (Comba, Karatsuba, NTT) depending on integer operand limb dimension thresholds.
* **Side-Channel Antidote Protection:** Implements strict constant-time conditional selection pipelines and branchless Montgomery reduction to isolate against timing and simple power analysis (SPA) leaks.
* **Hardware Blinding Vectors:** Integrates a platform-independent CSPRNG entropy capture channel (`/dev/urandom` & Windows `BCryptGenRandom`) to process data blinding and prevent Differential Power Analysis (DPA).

## Getting Started

### Local Compilation
To compile the standalone test suite on your local development machine:
```bash
make clean
make all
./test_suite
```

### Building the Dynamic Shared Library
To compile the shared object target for high-level language binding hooks (e.g., Python `ctypes` or Rust FFI bridges):
```bash
make shared
```
