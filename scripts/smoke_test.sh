#!/usr/bin/env bash
# End-to-end smoke test for the VLA Robot Full Stack.
#
# Validates:
#   1. Project structure and all required files exist
#   2. Docker Compose configs parse correctly
#   3. Python modules import without errors
#   4. Generates synthetic test episodes (no GPU needed)
#   5. Computes normalization statistics
#   6. Validates LeRobot v3.0 output format
#
# For the full GPU pipeline (Isaac Sim + OpenPi), run:
#   ./scripts/collect_sim_data.sh --episodes 5
#   ./scripts/train.sh
#   ./scripts/eval_sim.sh
#
# Usage:
#   ./scripts/smoke_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  [PASS] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== VLA Robot Full Stack Smoke Test ==="
echo ""

# --- 1. Project Structure ---
echo "1. Checking project structure..."
check "README.md exists" test -f "$PROJECT_DIR/README.md"
check "docker-compose.yml exists" test -f "$PROJECT_DIR/docker/docker-compose.yml"
check "docker-compose.cloud.yml exists" test -f "$PROJECT_DIR/docker/docker-compose.cloud.yml"
check ".env exists" test -f "$PROJECT_DIR/docker/.env"
check ".env.cloud exists" test -f "$PROJECT_DIR/docker/.env.cloud"
check "Isaac Sim Dockerfile" test -f "$PROJECT_DIR/docker/isaac-sim/Dockerfile"
check "OpenPi Dockerfile" test -f "$PROJECT_DIR/docker/openpi-server/Dockerfile"
check "ROS2 Bridge Dockerfile" test -f "$PROJECT_DIR/docker/ros2-bridge/Dockerfile"
check "Training Dockerfile" test -f "$PROJECT_DIR/docker/training/Dockerfile"
check "Caddyfile" test -f "$PROJECT_DIR/docker/openpi-server/Caddyfile"
check "SO-ARM101 URDF" test -f "$PROJECT_DIR/robot_description/urdf/soarm101.urdf"
check "Isaac Sim URDF variant" test -f "$PROJECT_DIR/robot_description/urdf/soarm101_isaacsim.urdf"
check "STL meshes (13 files)" test "$(ls "$PROJECT_DIR/robot_description/meshes/"*.stl 2>/dev/null | wc -l)" -eq 13
check "Isaac Lab reach env" test -f "$PROJECT_DIR/isaac_envs/soarm_reach_env.py"
check "Isaac Lab pick env" test -f "$PROJECT_DIR/isaac_envs/soarm_pick_env.py"
check "Sim data collector" test -f "$PROJECT_DIR/isaac_envs/sim_data_collector.py"
check "VLA bridge node" test -f "$PROJECT_DIR/ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py"
check "Observation builder" test -f "$PROJECT_DIR/ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/observation_builder.py"
check "SoARM training config" test -f "$PROJECT_DIR/training/configs/soarm_config.py"
check "Norm stats script" test -f "$PROJECT_DIR/training/scripts/compute_norm_stats.py"
check "Train LoRA script" test -f "$PROJECT_DIR/training/scripts/train_lora.py"

# --- 2. Script permissions ---
echo ""
echo "2. Checking script permissions..."
for script in setup_robot_usd.sh collect_sim_data.sh train.sh eval_sim.sh \
              deploy_real.sh deploy_cloud.sh sync_data.sh smoke_test.sh; do
    check "$script is executable" test -x "$PROJECT_DIR/scripts/$script"
done

# --- 3. Docker Compose validation ---
echo ""
echo "3. Validating Docker Compose configs..."
if command -v docker &> /dev/null; then
    check "docker-compose.yml parses" docker compose -f "$PROJECT_DIR/docker/docker-compose.yml" config --quiet
    check "docker-compose.cloud.yml parses" docker compose -f "$PROJECT_DIR/docker/docker-compose.cloud.yml" config --quiet
else
    echo "  [SKIP] Docker not available"
fi

# --- 4. URDF validation ---
echo ""
echo "4. Validating URDF..."
check "URDF has 6 revolute joints" test "$(grep -c 'type="revolute"' "$PROJECT_DIR/robot_description/urdf/soarm101.urdf")" -eq 6
check "URDF references package:// meshes" grep -q 'package://soarm_description/meshes/' "$PROJECT_DIR/robot_description/urdf/soarm101.urdf"
check "Isaac URDF references /robot_description/" grep -q '/robot_description/meshes/' "$PROJECT_DIR/robot_description/urdf/soarm101_isaacsim.urdf"

# --- 5. Generate synthetic test data ---
echo ""
echo "5. Generating synthetic test episodes..."
TEST_DATA_DIR=$(mktemp -d)
python3 -c "
import sys, json, os
sys.path.insert(0, '$PROJECT_DIR/isaac_envs')
from sim_data_collector import LeRobotWriter
import numpy as np

writer = LeRobotWriter('$TEST_DATA_DIR', fps=30)
for ep in range(5):
    writer.start_episode(task='smoke_test_reach')
    for step in range(30):
        state = np.random.randn(6).astype(np.float32) * 0.5
        action = np.random.randn(6).astype(np.float32) * 0.1
        writer.add_frame(state=state, action=action)
    writer.end_episode()
writer.save()
print('OK')
" 2>&1 && {
    check "5 episodes generated" test "$(ls "$TEST_DATA_DIR/meta/episodes/"*.json 2>/dev/null | wc -l)" -eq 5
    if python3 -c "import pyarrow" 2>/dev/null; then
        check "Parquet data file" test -f "$TEST_DATA_DIR/data/train-00000-of-00001.parquet"
    else
        echo "  [SKIP] Parquet data file (pyarrow not installed on host -- available in Docker)"
    fi
    check "info.json metadata" test -f "$TEST_DATA_DIR/meta/info.json"
    check "stats.json normalization" test -f "$TEST_DATA_DIR/meta/stats.json"

    # Validate info.json
    check "info.json has v3.0" grep -q '"v3.0"' "$TEST_DATA_DIR/meta/info.json"
    check "info.json has so_arm101" grep -q '"so_arm101"' "$TEST_DATA_DIR/meta/info.json"
} || {
    echo "  [FAIL] Synthetic data generation failed"
    FAIL=$((FAIL + 1))
}

# --- 6. Compute norm stats on test data ---
echo ""
echo "6. Computing normalization statistics..."
python3 "$PROJECT_DIR/training/scripts/compute_norm_stats.py" \
    --data-dir "$TEST_DATA_DIR" 2>&1 && {
    check "stats.json updated" test -f "$TEST_DATA_DIR/meta/stats.json"
    check "Stats has observation.state" grep -q '"observation.state"' "$TEST_DATA_DIR/meta/stats.json"
    check "Stats has action" grep -q '"action"' "$TEST_DATA_DIR/meta/stats.json"
} || {
    echo "  [FAIL] Norm stats computation failed"
    FAIL=$((FAIL + 1))
}

# --- 7. Python imports ---
echo ""
echo "7. Checking Python imports..."
check "observation_builder imports" python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/ros2_ws/src/soarm_vla_bridge')
from soarm_vla_bridge.observation_builder import build_observation, SOARM_JOINT_NAMES
import numpy as np
obs = build_observation({'shoulder_pan': 0.1}, np.zeros((224,224,3), dtype=np.uint8))
assert 'state' in obs and 'images' in obs
"

# Cleanup
rm -rf "$TEST_DATA_DIR"

# --- Summary ---
echo ""
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Some checks failed. Review the output above."
    exit 1
else
    echo ""
    echo "All checks passed!"
    echo ""
    echo "Next steps for full GPU pipeline:"
    echo "  1. docker login nvcr.io   (NGC credentials for Isaac Sim)"
    echo "  2. ./scripts/setup_robot_usd.sh"
    echo "  3. ./scripts/collect_sim_data.sh --episodes 50"
    echo "  4. ./scripts/train.sh"
    echo "  5. ./scripts/eval_sim.sh"
    exit 0
fi
