#!/usr/bin/env bash
# Download SO-ARM101 URDF + meshes from TheRobotStudio/SO-ARM100 and
# convert URDF to USD for Isaac Sim using the Isaac Sim container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REPO_BASE="https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"
URDF_DIR="$PROJECT_DIR/robot_description/urdf"
MESH_DIR="$PROJECT_DIR/robot_description/meshes"
USD_DIR="$PROJECT_DIR/robot_description/usd"

STLS=(
  base_motor_holder_so101_v1.stl
  base_so101_v2.stl
  motor_holder_so101_base_v1.stl
  motor_holder_so101_wrist_v1.stl
  moving_jaw_so101_v1.stl
  rotation_pitch_so101_v1.stl
  sts3215_03a_no_horn_v1.stl
  sts3215_03a_v1.stl
  under_arm_so101_v1.stl
  upper_arm_so101_v1.stl
  waveshare_mounting_plate_so101_v2.stl
  wrist_roll_follower_so101_v1.stl
  wrist_roll_pitch_so101_v2.stl
)

mkdir -p "$URDF_DIR" "$MESH_DIR" "$USD_DIR"

# --- Download URDF ---
if [ ! -f "$URDF_DIR/soarm101.urdf" ]; then
  echo "[1/3] Downloading URDF..."
  curl -sL "$REPO_BASE/so101_new_calib.urdf" -o "$URDF_DIR/soarm101_raw.urdf"
  sed 's|filename="assets/|filename="package://soarm_description/meshes/|g' \
    "$URDF_DIR/soarm101_raw.urdf" > "$URDF_DIR/soarm101.urdf"
  sed 's|filename="assets/|filename="/robot_description/meshes/|g' \
    "$URDF_DIR/soarm101_raw.urdf" > "$URDF_DIR/soarm101_isaacsim.urdf"
  rm "$URDF_DIR/soarm101_raw.urdf"
else
  echo "[1/3] URDF already exists, skipping download."
fi

# --- Download STL meshes ---
echo "[2/3] Downloading STL meshes..."
for stl in "${STLS[@]}"; do
  if [ ! -f "$MESH_DIR/$stl" ]; then
    echo "  -> $stl"
    curl -sL "$REPO_BASE/assets/$stl" -o "$MESH_DIR/$stl"
  fi
done
echo "  $(ls "$MESH_DIR"/*.stl 2>/dev/null | wc -l) meshes present."

# --- Convert URDF to USD (requires Isaac Sim container) ---
echo "[3/3] Converting URDF to USD..."
if [ -f "$USD_DIR/soarm101.usd" ]; then
  echo "  USD already exists. Delete robot_description/usd/soarm101.usd to regenerate."
  exit 0
fi

echo "  Launching Isaac Sim container for URDF -> USD conversion..."
docker run --rm --runtime=nvidia \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$PROJECT_DIR/robot_description:/robot_description" \
  nvcr.io/nvidia/isaac-sim:5.1.0 \
  bash -c '
    python3 -c "
import omni.kit.app
from isaacsim.asset.importer.urdf import import_urdf, ImportConfig

config = ImportConfig()
config.fix_base = True
config.make_instanceable = True
config.merge_fixed_joints = True
config.default_drive_type = 1  # position

import_urdf(
    \"/robot_description/urdf/soarm101_isaacsim.urdf\",
    config,
    \"/robot_description/usd/soarm101.usd\",
)
print(\"USD saved to /robot_description/usd/soarm101.usd\")
"
  ' 2>&1 || {
    echo "  WARNING: Automated USD conversion failed."
    echo "  You can convert manually inside Isaac Sim:"
    echo "    File > Import > select soarm101_isaacsim.urdf"
    echo "    Enable: Static Base, Allow Self-Collision"
    echo "    Save as: robot_description/usd/soarm101.usd"
  }

echo "Done."
