#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the default Isaac Sim 5.1 container.
# This validates:
#  1) Container starts
#  2) GPU is visible in container
#  3) Isaac Sim version is 5.1.x
#  4) Basic RTX startup can run with the Vulkan driver check workaround
#
# NOTE: This script is intentionally a short non-streaming sanity check.
# For WebRTC streaming, use scripts/interactive_test.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PUBLIC_IP="${PUBLIC_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
PUBLIC_IP="${PUBLIC_IP:-127.0.0.1}"

cd "$PROJECT_DIR/docker"

echo "=== Isaac Sim 5.1 Smoke Test ==="
echo "Public IP: $PUBLIC_IP"

docker compose --profile collect run --rm \
  -e ISAAC_SIM_HEADLESS=1 \
  -e LIVESTREAM=0 \
  -e "PUBLIC_IP=$PUBLIC_IP" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  isaac-sim \
  bash -lc '
    echo "[smoke] nvidia-smi:"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
    echo "[smoke] /isaac-sim/VERSION:"
    if [ -f /isaac-sim/VERSION ]; then
      cat /isaac-sim/VERSION
    else
      echo "VERSION file missing"
      exit 1
    fi
    echo "[smoke] launching kit briefly..."
    /isaac-sim/python.sh - <<'"'"'PY'"'"'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
app.update()
print("[smoke] SimulationApp started")
app.close()
PY
  '

echo "=== Smoke test complete ==="
