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

exec "$@"
