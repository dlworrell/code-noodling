#!/bin/sh
# Valid NdM+K specs succeed with one total per trial; malformed specs and
# unknown flags exit with status 2.
. "$(dirname "$0")/common.sh"

for spec in d6 3d6+2 4d8-1; do
  out=$("$DICE_CPU" --spec "$spec" --count 5) ||
    fail "valid spec '$spec' exited with status $?"
  n=$(printf '%s\n' "$out" | wc -w)
  [ "$n" -eq 5 ] || fail "spec '$spec' printed $n totals, expected 5"
done

for spec in x 3x6 d 3d 0d6 d1 3d6+ d-6; do
  status=0
  "$DICE_CPU" --spec "$spec" --count 5 >/dev/null 2>&1 || status=$?
  [ "$status" -eq 2 ] ||
    fail "invalid spec '$spec' exited with status $status, expected 2"
done

status=0
"$DICE_CPU" --no-such-flag >/dev/null 2>&1 || status=$?
[ "$status" -eq 2 ] || fail "unknown flag exited with status $status, expected 2"

echo "PASS: spec parsing"
