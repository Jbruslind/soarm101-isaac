#!/usr/bin/env bash
# Run the SO-ARM101 reach environment on Isaac Sim 6.2 with a GUI so you can see the robot via the third-person viewport camera.
# Uses livestream so you connect with the Isaac Sim WebRTC client to view the simulation.
#
# Usage: ./scripts/play_reach.sh          # WebRTC livestream
#        ./scripts/play_reach.sh --gui    # window on the Sunshine display (watch via Moonlight)
# For WebRTC, open the Isaac Sim WebRTC streaming client, enter this host's IP, and connect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=_gui.sh
source "$SCRIPT_DIR/_gui.sh"

cd "$PROJECT_DIR/docker"

STREAM_ENV=(-e LIVESTREAM=2)
if [[ "${1:-}" == "--gui" ]]; then
    gui_enable "$PROJECT_DIR/docker"
    STREAM_ENV=()
fi

docker compose --profile collect run --rm \
    "${STREAM_ENV[@]}" \
    -e PYTHONUNBUFFERED=1 \
    isaac-sim \
    /isaac-sim/python.sh /isaac_envs/play_reach.py
