#!/usr/bin/env bash
# Create the directory that Isaac Sim's URDF importer writes to by default.
# The importer derives the output path from the URDF path and writes to
#   <urdf_dir>/<urdf_basename>/<urdf_basename>.usd
# e.g. robot_description/urdf/soarm101_isaacsim/soarm101_isaacsim.usd
# Run this before importing soarm101_isaacsim.urdf in the Isaac Sim GUI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMPORTER_DIR="$PROJECT_DIR/robot_description/urdf/soarm101_isaacsim"
USD_DIR="$PROJECT_DIR/robot_description/usd"

mkdir -p "$IMPORTER_DIR" "$USD_DIR"
# Container (root or uid 1234) must create files and subdirs (e.g. configuration/) here.
# Make robot_description writable by all so the URDF importer can write.
chmod -R a+rwX "$PROJECT_DIR/robot_description"
echo "Created $IMPORTER_DIR and made robot_description writable by container."
echo "After importing in the GUI, run: ./scripts/copy_urdf_import_to_usd.sh"
