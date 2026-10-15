# dice-cpu

`dice_cpu` is the unbiased CPU dice engine for the prime-number suite. It
supports `--faces M` single-die rolls and `--spec NdM+K` expressions, with
optional prime-seeded determinism, JSON/CSV logs, and a chi-square fairness
check. The source is [`src/dice_cpu.cpp`](src/dice_cpu.cpp).

This directory is a self-contained GNU Autotools package. The same source is
also built by the suite's CMake project (`../CMakeLists.txt`).

## Build

Requirements: a C++17 compiler with `__uint128_t` (GCC or Clang on a 64-bit
target), Autoconf, Automake, and `autoconf-archive` (for
`AX_CXX_COMPILE_STDCXX`).

```sh
# Debian/Ubuntu
sudo apt-get install autoconf automake autoconf-archive g++

autoreconf -fi        # generates configure and Makefile.in (not committed)
./configure
make
make check            # spec parsing, determinism, range, chi-square tests
make distcheck        # builds and tests the release tarball
```

`configure` stops with a clear error if the compiler lacks `__uint128_t`,
because the unbiased Lemire sampler needs a 128-bit product.

## Usage

```sh
./dice_cpu --faces 6 --count 10000 --chi
./dice_cpu --spec "3d6+2" --count 5000 --use-prime-seeds primes.json --chi
./dice_cpu --faces 20 --count 20000 --csv rolls.csv
./dice_cpu --spec "4d8-1" --count 1000 --log-json rolls.json
```

| Flag | Meaning |
|------|---------|
| `--faces M` | Roll one M-sided die per trial (default 6) |
| `--spec NdM+K` | Roll an `NdM+K` bundle per trial; prints the total |
| `--count N` | Number of trials (default 100) |
| `--use-prime-seeds FILE` | Seed from a `{"primes": [...]}` JSON list |
| `--seed-per-roll` / `--seed-per-bundle` | Reseed for each die (default) or each bundle |
| `--chi` | Print a chi-square p-value and counts to stderr |
| `--csv FILE` / `--log-json FILE` | Write per-roll results |

Without a primes file, `dice_cpu` uses a fixed built-in seed, so repeated runs
print identical output. Invalid specs and unknown flags exit with status 2.

## Tests

`make check` runs the shell tests in `tests/`:

| Test | Checks |
|------|--------|
| `spec_parsing.sh` | `d6`, `3d6+2`, `4d8-1` succeed; malformed specs and unknown flags exit 2 |
| `determinism.sh` | Two runs with the same fixed seed (built-in or primes file) match |
| `range.sh` | Every `--faces 20` roll is in 1..20 |
| `chi_square.sh` | `--faces 6 --count 60000 --chi` gives p > 1e-6 (loose, to avoid flaky CI) |

The parser accepts trailing characters after a valid spec (for example `2d6*`
is read as `2d6`). That is existing behavior and is not covered as an error.
