#!/bin/bash
set -e

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Build workspace if src/ directory is present and not yet built
if [ -d "/ros2_ws/src" ] && [ ! -d "/ros2_ws/install" ]; then
    echo "Building ROS2 workspace..."
    cd /ros2_ws
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 || true
fi

# Source workspace overlay
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Rebuild the bridge package each start so source-mounted changes are active.
if [ -d "/ros2_ws/src/soarm_vla_bridge" ]; then
    echo "Rebuilding soarm_vla_bridge overlay..."
    cd /ros2_ws
    colcon build --symlink-install --packages-select soarm_vla_bridge --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 || true
    if [ -f "/ros2_ws/install/setup.bash" ]; then
        source /ros2_ws/install/setup.bash
    fi
fi

exec "$@"
