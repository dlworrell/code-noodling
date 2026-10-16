#!/usr/bin/env bash
# Montgomery scoped formal checks, run by .github/workflows/crypto_formal.yml.
# Usage: ci/formal.sh <step>
# Run from the repository root: tool and output paths (.tools/, build/formal)
# are relative to the working directory, matching the workflow's artifacts.
set -euo pipefail
proof="$(cd "$(dirname "$0")/../proof" && pwd)"

case "${1:-}" in
  install-tools)
    python -m pip install -r "$proof/requirements.txt"
    bash "$proof/install_cbmc.sh" .tools/cbmc
    ;;
  syntax)
    python -m compileall -q "$proof"
    ;;
  proofs)
    python "$proof/run_proofs.py" --cbmc .tools/cbmc/usr/bin/cbmc --output build/formal
    ;;
  carry-tests)
    python -m unittest discover -s "$proof" -p test_carry_driver.py
    ;;
  *)
    echo "usage: $0 {install-tools|syntax|proofs|carry-tests}" >&2
    exit 2
    ;;
esac
