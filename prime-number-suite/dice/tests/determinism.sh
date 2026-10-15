#!/bin/sh
# Two runs with the same fixed seed produce identical output. dice_cpu has two
# fixed-seed paths: its built-in default seed (no primes file) and a fixed
# prime-seed list passed with --use-prime-seeds.
. "$(dirname "$0")/common.sh"

primes="$WORKDIR/primes.json"
printf '{"primes": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]}\n' > "$primes"

check_same() {
  "$DICE_CPU" "$@" > "$WORKDIR/a.out" || fail "run 1 failed: $*"
  "$DICE_CPU" "$@" > "$WORKDIR/b.out" || fail "run 2 failed: $*"
  [ -s "$WORKDIR/a.out" ] || fail "no output: $*"
  cmp -s "$WORKDIR/a.out" "$WORKDIR/b.out" || fail "outputs differ: $*"
}

check_same --faces 6 --count 500
check_same --spec 3d6+2 --count 500
check_same --faces 20 --count 500 --use-prime-seeds "$primes"
check_same --spec 4d8-1 --count 500 --use-prime-seeds "$primes"
check_same --spec 4d8-1 --count 500 --use-prime-seeds "$primes" --seed-per-bundle

echo "PASS: determinism"
