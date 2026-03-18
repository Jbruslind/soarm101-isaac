FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ROS2 packages for robot control
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-trajectory-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-pip python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the VLA bridge (openpi-client + image processing)
RUN pip3 install --no-cache-dir \
    openpi-client \
    numpy \
    opencv-python-headless \
    Pillow

# Build the ROS2 workspace at container start (source-mounted)
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

WORKDIR /ros2_ws

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
