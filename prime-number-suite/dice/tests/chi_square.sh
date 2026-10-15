#!/bin/sh
# A fair d6 over 60000 rolls yields a chi-square p-value that is not absurdly
# small. The threshold is deliberately loose (p > 1e-6) so CI is not flaky.
. "$(dirname "$0")/common.sh"

"$DICE_CPU" --faces 6 --count 60000 --chi > /dev/null 2> "$WORKDIR/chi" ||
  fail "dice_cpu exited with status $?"

p=$(sed -n 's/.*\[chi\].*p=\([^ ]*\).*/\1/p' "$WORKDIR/chi")
[ -n "$p" ] || fail "no p-value in output: $(cat "$WORKDIR/chi")"

awk -v p="$p" 'BEGIN { exit !(p + 0 > 1e-6) }' ||
  fail "chi-square p-value $p is not above 1e-6"

echo "PASS: chi-square (p=$p)"
