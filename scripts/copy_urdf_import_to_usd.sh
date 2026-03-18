#!/usr/bin/env bash
# Copy the USD file produced by Isaac Sim's URDF import (default path) to the
# path expected by this project (robot_description/usd/soarm101.usd).
# Run this after importing soarm101_isaacsim.urdf in the Isaac Sim GUI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$PROJECT_DIR/robot_description/urdf/soarm101_isaacsim/soarm101_isaacsim.usd"
DEST="$PROJECT_DIR/robot_description/usd/soarm101.usd"

if [[ ! -f "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  echo "Import the URDF in Isaac Sim first (File > Import > soarm101_isaacsim.urdf)." >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
cp -f "$SRC" "$DEST"
echo "Copied to $DEST"
