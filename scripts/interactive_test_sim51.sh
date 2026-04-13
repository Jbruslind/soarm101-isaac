#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Isaac Sim 5.1 is now the default stack used by interactive_test.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[info] scripts/interactive_test_sim51.sh is deprecated; forwarding to scripts/interactive_test.sh"
exec "$SCRIPT_DIR/interactive_test.sh" "$@"
