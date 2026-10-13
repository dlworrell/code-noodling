#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
compiler=${CT_CC:-gcc}
optimization=${CT_OPT:--O2}
dudect_dir=${DUDECT_DIR:-vendor/dudect}
expected=dc269651fb2567e46755cfb2a13d3875592968b5
[[ $(git -C "$dudect_dir" rev-parse HEAD) == "$expected" ]] || { echo 'Unexpected dudect revision' >&2; exit 2; }
git -C "$dudect_dir" diff --quiet HEAD -- src/dudect.h
artifact_dir=${CT_ARTIFACT_DIR:-../build/ct-timing}
mkdir -p "$artifact_dir"
artifact_dir=$(cd "$artifact_dir" && pwd)
cpu=${CT_CPU:-$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')}
{
  "$compiler" --version
  printf 'flags: -std=c11 %s -fno-lto -fPIC -Wall -Wextra\n' "$optimization"
  printf 'cpu affinity: %s\ndudect: %s\n' "$cpu" "$expected"
  uname -a
  lscpu
} > "$artifact_dir/environment.txt"
"$compiler" -std=c11 "$optimization" -fno-lto -fPIC -Wall -Wextra \
  -c crypto_montgomery_ct.c -o "$artifact_dir/candidate.o"
objdump -drwC "$artifact_dir/candidate.o" > "$artifact_dir/candidate.asm"
"$compiler" -std=gnu11 "$optimization" -fno-lto -I"$dudect_dir/src" \
  -DCT_TIMING_NEGATIVE_CONTROL dudect_montgomery.c "$artifact_dir/candidate.o" \
  -lm -o "$artifact_dir/negative-control"
if taskset -c "$cpu" "$artifact_dir/negative-control" > "$artifact_dir/negative-control.log"; then
  echo 'ERROR: timing negative control not detected' >&2; exit 1
else
  [[ $? == 1 ]] || { echo 'ERROR: unexpected negative-control error' >&2; exit 1; }
fi
"$compiler" -std=gnu11 "$optimization" -fno-lto -I"$dudect_dir/src" \
  -DCT_LIMBS=16 -DCT_FIXED_CLASS=1 -DCT_BATCHES=101 -DCT_TIMING_BASELINE \
  dudect_montgomery.c "$artifact_dir/candidate.o" -lm -o "$artifact_dir/baseline"
taskset -c "$cpu" "$artifact_dir/baseline" > "$artifact_dir/baseline.log"
for n in 4 16 32; do
  for kind in 0 1; do
    binary="$artifact_dir/timing-$n-$kind"
    "$compiler" -std=gnu11 "$optimization" -fno-lto -I"$dudect_dir/src" \
      -DCT_LIMBS="$n" -DCT_FIXED_CLASS="$kind" -DCT_BATCHES=101 \
      dudect_montgomery.c "$artifact_dir/candidate.o" -lm -o "$binary"
    taskset -c "$cpu" "$binary" > "$binary.log"
    tail -2 "$binary.log"
  done
done
sha256sum crypto_montgomery_ct.c crypto_montgomery_ct.h "$dudect_dir/src/dudect.h" \
  "$artifact_dir/candidate.o" "$artifact_dir"/timing-*-* > "$artifact_dir/sha256.txt"
echo 'Bounded timing runs completed. No security certification is implied.'
