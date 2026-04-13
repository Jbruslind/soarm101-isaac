#!/usr/bin/env bash
# Run the Isaac Sim container (GUI or headless streaming) with cache and robot_description
# mounted so you can run headless and import URDF. Runs as the image's default user so
# bundled scripts are executable; host dirs are made writable via chmod so the container
# can write material cache and URDF import output.
#
# Usage: ./scripts/run_isaac_sim_gui.sh
# Then inside the container: bash runheadless.sh -v
# Optional: ISAAC_SIM_TAG=5.1.0 (default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${ISAAC_SIM_CACHE_DIR:-$PROJECT_DIR/.cache/isaac-sim}"
ISAAC_TAG="${ISAAC_SIM_TAG:-5.1.0}"

# Official layout: cache/main (ov, warp), computecache, config, data, logs, pkg (see install_container.html).
mkdir -p "$CACHE_DIR"/cache/main/{ov,warp} \
         "$CACHE_DIR"/cache/computecache \
         "$CACHE_DIR"/config \
         "$CACHE_DIR"/data/{documents,Kit} \
         "$CACHE_DIR"/logs \
         "$CACHE_DIR"/pkg \
         "$CACHE_DIR"/kit \
         "$CACHE_DIR/cache/main/ov/Kit/107.3/69cbf6ad"

# Container runs as root or uid 1234; host dirs are owned by you. Make cache and robot_description
# writable by the container so Isaac Sim can write material_cache.json and URDF import output.
chmod -R a+rwX "$CACHE_DIR" 2>/dev/null || true
"$SCRIPT_DIR/prepare_urdf_import_dir.sh"

# Run as the image's default user so bundled scripts (runheadless.sh, etc.) are executable.
docker run --name isaac-sim --entrypoint bash -it --rm \
  --runtime=nvidia --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$PROJECT_DIR/robot_description:/robot_description" \
  -v "$CACHE_DIR/cache/main:/isaac-sim/.cache:rw" \
  -v "$CACHE_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$CACHE_DIR/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "$CACHE_DIR/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$CACHE_DIR/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE_DIR/pkg:/root/.local/share/ov/pkg:rw" \
  -v "$CACHE_DIR/kit:/isaac-sim/kit/cache:rw" \
  nvcr.io/nvidia/isaac-sim:"$ISAAC_TAG"
