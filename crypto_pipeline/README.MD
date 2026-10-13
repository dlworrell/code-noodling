# Integer multiplication experiments

Inspired by Jack Murtagh's Scientific American article, *Mathematicians still
 don't know the fastest way to multiply numbers* (October 2026), archived in
this directory. The central experiment compares exact full-width multiplication
algorithms and their practical crossover costs.

**Research prototype, not production cryptography.** Arithmetic is checked against
Python integers, but no API is certified constant-time or side-channel resistant.
Legacy `secure_`, `ct_`, and `is_constant_time` names do not establish those properties.

## Implemented algorithms

- Comba: base-2^64 column multiplication with a three-limb accumulator.
- Karatsuba: recursive three-product identity; midpoint sums retain their carry
  limb, odd sizes are zero-padded, and reconstruction propagates all carries.
- NTT: radix-2 transforms modulo 998244353 and 1004535809, base-2^16 digits,
  exact CRT reconstruction, and carry propagation back to base-2^64.
- Montgomery REDC: preserves the extra high carry and consistently returns
  `T * R^-1 mod m`, where `R = 2^(64*n)`.
- Ordinary modular multiplication: exact product followed by binary long division.
- Modular inverse: variable-time binary extended GCD for odd moduli.
- Blinding: rejection-sampled invertible factor `k`, `a*k mod m`, and ordinary
  modular unblinding. Both the blinded value and `k` are returned.

The bounded NTT is not the Harvey–van der Hoeven algorithm discussed in the
article. Its two-prime reconstruction is valid for this implementation's maximum
of 1024 limbs. It must not be generalized to larger lengths without revisiting
transform order and coefficient bounds.

## Build and verification

Run from `crypto_pipeline/` on Linux with GCC or Clang and Python 3:

```sh
make clean
make shared all
./test_suite
python3 test_arithmetic.py
make bounds-sanitize
python3 benchmark_multiply.py
```

The makefile's default optimization flags target GCC. GitHub CI additionally
runs clang-tidy, Valgrind, and sanitizer boundary tests. This project's platform
validation currently covers Linux; the Windows entropy branch is not CI-tested.

`test_arithmetic.py` checks thousands of small inverse cases, seeded random and
adversarial full products, Montgomery carry/borrow regressions, composite
moduli, domain errors, blinding round-trips, and the Python binding. It includes
sizes at recursion/dispatch boundaries and the 1024-limb maximum.

## API contracts and migration

See `crypto_multiplier.h` for capacities, overlap restrictions, and errors.
All limb-array APIs accept 1..1024 little-endian 64-bit limbs. Callers must
provide correctly sized buffers and adequate stack space (the NTT alone uses
up to 96 KiB of transform arrays). Pointer-only APIs cannot verify capacity.

- `crypto_multiply_reduce` retains **Montgomery** semantics, `a*b*R^-1 mod m`.
  Inputs must be reduced and the modulus must be odd and greater than 1.
  Its result buffer holds 2*n limbs; the upper half is cleared.
- `crypto_mod_multiply` computes **ordinary** `a*b mod m`, including even moduli
  greater than 1, with an n-limb output. Inputs need not be reduced.
- `crypto_mod_inverse` returns 0/-1 and sets errno; noninvertible inputs fail
  with EDOM without changing output. The old `ct_mod_inverse_multiprecision`
  entry point delegates to it and is explicitly variable-time.
- `crypto_blind` returns both outputs. The old three-argument
  `execute_blinded_multiplication` fails with ENOTSUP because that ABI cannot
  provide the modulus or return the necessary factor. Migrate callers.
- `execute_unblinding` uses ordinary modular arithmetic, matching `crypto_blind`.
- `linking_ctypes.secure_multiply_wrapper` retains its legacy Python name but
  computes ordinary `(a*b)%m`, encodes every limb, and checks errors.

Automatic dispatch uses Comba below 64 limbs and Karatsuba otherwise. A nonzero
legacy mode flag always selects Comba. NTT remains directly callable for
experiments: the local benchmark found no winning NTT crossover within the
supported range. These thresholds are provisional and should be remeasured on
the target hardware. This is an empirical comparison, not an optimality claim.

See [mathematical notes](../docs/engineering/CRYPTO-ARITHMETIC.md) and
[memory review](../docs/engineering/CRYPTO-MEMORY-REVIEW.md).
