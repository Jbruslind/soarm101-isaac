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
# URDF importer 3.0 writes a root layer plus a payloads/ folder; both go into USD_OUT_DIR
# (default robot_description/usd/soarm101/). Set USD_OUT_DIR to convert somewhere else,
# e.g. USD_OUT_DIR=/tmp/soarm_usd ./scripts/setup_robot_usd.sh to compare with the committed asset.
USD_OUT_DIR="${USD_OUT_DIR:-$USD_DIR/soarm101}"
USD_OUT_NAME="soarm101.usda"
echo "[3/3] Converting URDF to USD..."
if [ -f "$USD_OUT_DIR/$USD_OUT_NAME" ] && [ -z "${FORCE:-}" ]; then
  echo "  $USD_OUT_DIR/$USD_OUT_NAME already exists. Re-run with FORCE=1 to regenerate."
  exit 0
fi
mkdir -p "$USD_OUT_DIR"
USD_OUT_DIR="$(cd "$USD_OUT_DIR" && pwd)"

echo "  Launching the soarm-isaac-sim container for URDF -> USD conversion..."
# Uses the project image and this repo's convert_urdf.py (mounted, so no rebuild is needed
# after editing it), which wraps the Isaac Sim 6.x URDF importer via Isaac Lab's UrdfConverter.
(cd "$PROJECT_DIR/docker" && docker compose --profile collect run --rm \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$USD_OUT_DIR:/usd_out" \
  -v "$PROJECT_DIR/docker/isaac-sim/convert_urdf.py:/usr/local/bin/convert_urdf.py:ro" \
  isaac-sim \
  /isaac-sim/python.sh /usr/local/bin/convert_urdf.py \
    --urdf /robot_description/urdf/soarm101_isaacsim.urdf \
    --out "/usd_out/$USD_OUT_NAME") 2>&1 || {
    echo "  WARNING: Automated USD conversion failed."
    echo "  You can convert manually inside Isaac Sim:"
    echo "    File > Import > select soarm101_isaacsim.urdf"
    echo "    Enable: Static Base, Allow Self-Collision"
    echo "    Then run ./scripts/copy_urdf_import_to_usd.sh <imported root .usda>"
    exit 1
  }

echo "Done: $USD_OUT_DIR/$USD_OUT_NAME"
