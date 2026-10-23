#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:470d4f8e-51c1-4907-8a67-6f8eebb9d582

set -euo pipefail
cd "$(dirname "$0")"
compiler=${CT_CC:-gcc}
optimization=${CT_OPT:--O2}
artifact_dir=${CT_ARTIFACT_DIR:-../build/constant-flow}
mkdir -p "$artifact_dir"
artifact_dir=$(cd "$artifact_dir" && pwd)
flags=(-std=c11 "$optimization" -fno-lto -fPIC -Wall -Wextra)
{
  "$compiler" --version
  printf 'flags:'; printf ' %q' "${flags[@]}"; printf '\n'
  uname -a
  lscpu
  valgrind --version
} > "$artifact_dir/environment.txt"
"$compiler" "${flags[@]}" -c crypto_montgomery_ct.c -o "$artifact_dir/candidate.o"
objdump -drwC "$artifact_dir/candidate.o" > "$artifact_dir/candidate.asm"
"$compiler" -shared "$artifact_dir/candidate.o" -o "$artifact_dir/libct.so"
CT_LIBRARY="$artifact_dir/libct.so" python3 tests/test_montgomery_ct.py > "$artifact_dir/oracle.log" 2>&1
"$compiler" "${flags[@]}" -I. -DCT_TAINT_TEST tests/test_montgomery_ct.c "$artifact_dir/candidate.o" -o "$artifact_dir/taint"
valgrind --tool=memcheck --error-exitcode=99 --track-origins=yes \
  "$artifact_dir/taint" > "$artifact_dir/taint.log" 2>&1
"$compiler" "${flags[@]}" -I. -DCT_TAINT_TEST -DCT_NEGATIVE_CONTROL tests/test_montgomery_ct.c \
  "$artifact_dir/candidate.o" -o "$artifact_dir/negative-control"
if valgrind --tool=memcheck --error-exitcode=99 "$artifact_dir/negative-control" \
    > "$artifact_dir/negative-control.log" 2>&1; then
  echo 'ERROR: deliberately leaky control was not detected' >&2
  exit 1
else
  status=$?
  if [[ $status != 99 ]]; then
    echo "ERROR: negative control failed for an unexpected reason: $status" >&2
    exit 1
  fi
fi
"$compiler" -std=c11 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
  -fno-sanitize-recover=all -I. crypto_montgomery_ct.c tests/test_montgomery_ct.c -o "$artifact_dir/sanitized"
"$artifact_dir/sanitized" > "$artifact_dir/sanitizer.log" 2>&1
sha256sum crypto_montgomery_ct.c crypto_montgomery_ct.h "$artifact_dir/candidate.o" \
  "$artifact_dir/libct.so" "$artifact_dir/taint" > "$artifact_dir/sha256.txt"
printf '%s\n' 'Oracle, sanitizer, and constant-flow instrumentation checks passed; negative control detected.'
printf '%s\n' 'This is bounded implementation evidence, not constant-time certification.'
