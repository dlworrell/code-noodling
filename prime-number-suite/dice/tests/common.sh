# Shared helpers for the dice_cpu shell tests. Sourced, not executed.

: "${DICE_CPU:?DICE_CPU must point at the dice_cpu binary (set by make check)}"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# Scratch files go to a private temporary directory, never the source tree,
# because make distcheck runs the tests against a read-only source copy.
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/dice-test.XXXXXX") || fail "mktemp failed"
trap 'rm -rf "$WORKDIR"' EXIT
