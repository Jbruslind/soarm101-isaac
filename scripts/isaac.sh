#!/usr/bin/env bash
# Run any command in the Isaac Sim / Isaac Lab container, headless or with a GUI.
#
#   --headless (default)  No window, no streaming: fastest for training / data generation.
#   --gui                 Normal Isaac Sim window on the Sunshine display; watch it live
#                         from any Moonlight client on the LAN (./scripts/sunshine.sh up first).
#
# Usage:
#   ./scripts/isaac.sh --gui                                   # full Isaac Sim editor
#   ./scripts/isaac.sh --gui /isaac-sim/python.sh /isaac_envs/play_reach.py
#   ./scripts/isaac.sh --headless --cameras -- /isaac-sim/python.sh /isaac_envs/sim_data_collector.py --num-episodes 10
#   ./scripts/isaac.sh --headless /workspace/isaaclab/isaaclab.sh -p \
#       /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-Franka-v0
#   ./scripts/isaac.sh bash                                    # shell (headless env)
#
# Isaac Lab scripts using AppLauncher honour HEADLESS=0/1, which this sets per mode.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=_gui.sh
source "$SCRIPT_DIR/_gui.sh"

MODE="headless"
EXTRA_ENV=(-e PYTHONUNBUFFERED=1)

while [[ $# -gt 0 ]]; do
    case $1 in
        --gui) MODE="gui"; shift ;;
        --headless) MODE="headless"; shift ;;
        --cameras) EXTRA_ENV+=(-e ENABLE_CAMERAS=1); shift ;;
        --) shift; break ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) break ;;
    esac
done

cd "$PROJECT_DIR/docker"

if [[ "$MODE" == "gui" ]]; then
    gui_enable "$PROJECT_DIR/docker"
    if [[ $# -eq 0 ]]; then
        set -- /isaac-sim/isaac-sim.sh --allow-root
    fi
else
    EXTRA_ENV+=(-e HEADLESS=1 -e ISAAC_SIM_HEADLESS=1 -e LIVESTREAM=0)
    if [[ $# -eq 0 ]]; then
        set -- bash
    fi
fi

exec docker compose --profile collect run --rm "${EXTRA_ENV[@]}" isaac-sim "$@"
