#!/bin/sh
# Every roll for --faces 20 is an integer in 1..20, and one roll is printed
# per trial.
. "$(dirname "$0")/common.sh"

count=20000
"$DICE_CPU" --faces 20 --count "$count" > "$WORKDIR/rolls" ||
  fail "dice_cpu exited with status $?"

tr ' ' '\n' < "$WORKDIR/rolls" | awk -v want="$count" '
  NF == 0 { next }
  $0 !~ /^[0-9]+$/ || $0 < 1 || $0 > 20 { bad++; if (bad <= 5) print "out of range: " $0 }
  { n++ }
  END {
    if (n != want) { print "expected " want " rolls, got " n; exit 1 }
    exit (bad > 0)
  }' >&2 || fail "range check failed"

echo "PASS: range"
