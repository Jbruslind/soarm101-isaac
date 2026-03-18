#!/usr/bin/env bash
# Run the SO-ARM101 reach environment with a GUI so you can see the robot via the third-person viewport camera.
# Uses livestream so you connect with the Isaac Sim WebRTC client to view the simulation.
#
# Usage: ./scripts/play_reach.sh
# Then open the Isaac Sim WebRTC streaming client, enter this host's IP, and connect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/docker"

docker compose --profile collect run --rm \
    -e LIVESTREAM=2 \
    -e PYTHONUNBUFFERED=1 \
    isaac-sim \
    /isaac-sim/python.sh /isaac_envs/play_reach.py
