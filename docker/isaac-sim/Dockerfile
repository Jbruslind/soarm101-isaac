FROM nvcr.io/nvidia/isaac-lab:2.3.1

# The Isaac Lab pre-built image already ships:
#   - Isaac Sim 5.1.x (Python at /isaac-sim/python.sh)
#   - Isaac Lab 2.3.1 (isaaclab.* modules pre-installed)
# We add ROS 2 Jazzy runtime libraries (Ubuntu 24.04 / Noble) that the
# Isaac Sim 5.1 `isaacsim.ros2.bridge` extension links against at runtime.
#
# Reference: https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html

USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl gnupg lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu noble main" \
       | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       ros-jazzy-ros-base \
       ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*
# Jazzy debs install Python 3.12 message bindings under /opt/ros/jazzy/lib/python3.12/.
# Isaac Sim uses Python 3.11 + bridge-bundled rclpy/rosidl — importing both causes ABI
# asserts (e.g. rcl_interfaces__msg__parameter_event__convert_from_py). Keep only native .so.
RUN bash -c 'shopt -s nullglob; for _pydir in /opt/ros/*/lib/python3.*; do rm -rf "$_pydir"; done'

# ROS-related LD_LIBRARY_PATH ordering is set in entrypoint.sh (Isaac bridge *.so before /opt/ros).

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    gymnasium \
    tensorboard \
    pyarrow \
    av \
    pandas

# Copy helper script for URDF -> USD conversion
COPY convert_urdf.py /usr/local/bin/convert_urdf.py

# Entrypoint that sources ROS2 before running any command
COPY entrypoint.sh /isaac-sim-entrypoint.sh
RUN chmod +x /isaac-sim-entrypoint.sh

WORKDIR /isaac_envs

ENTRYPOINT ["/isaac-sim-entrypoint.sh"]
CMD ["bash"]
