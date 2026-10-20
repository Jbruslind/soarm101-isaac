#!/usr/bin/env bash
# Copy a robot USD made by Isaac Sim's URDF importer (GUI) to the path the envs load:
#   robot_description/usd/soarm101/soarm101.usda (+ payloads/)
#
# Usage: ./scripts/copy_urdf_import_to_usd.sh [path/to/imported/<robot>.usda]
# Default source: robot_description/urdf/soarm101_isaacsim/soarm101_isaacsim.usda
#
# Prefer ./scripts/setup_robot_usd.sh: it runs the same importer (3.0) headless and also
# moves the articulation root to the robot prim so PhysX builds a fixed-base articulation
# (see docker/isaac-sim/convert_urdf.py). A GUI import copied with this script lacks that
# fix and the envs will see a floating base held by a world joint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC="${1:-$PROJECT_DIR/robot_description/urdf/soarm101_isaacsim/soarm101_isaacsim.usda}"
DEST_DIR="$PROJECT_DIR/robot_description/usd/soarm101"

if [[ ! -f "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  echo "Import the URDF in Isaac Sim first (File > Import > soarm101_isaacsim.urdf)," >&2
  echo "or run ./scripts/setup_robot_usd.sh instead." >&2
  exit 1
fi
SRC_DIR="$(dirname "$SRC")"
if [[ ! -d "$SRC_DIR/payloads" ]]; then
  echo "No payloads/ folder next to $SRC; expected URDF importer 3.0 output." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
rm -rf "$DEST_DIR/payloads"
cp -r "$SRC_DIR/payloads" "$DEST_DIR/payloads"
cp -f "$SRC" "$DEST_DIR/soarm101.usda"
echo "Copied to $DEST_DIR/soarm101.usda (+ payloads/)"
echo "WARNING: GUI imports lack convert_urdf.py's articulation-root fix; prefer ./scripts/setup_robot_usd.sh."
