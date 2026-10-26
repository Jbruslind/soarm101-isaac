# Isaac Sim + Isaac Lab image for the SO-ARM101 stack.
#
# The whole Isaac stack is pinned by two build args, set from docker/.env
# (ISAACSIM_VERSION / ISAACLAB_REF) so there is one place to bump it:
#   - ISAACSIM_VERSION: nvcr.io/nvidia/isaac-sim tag (Python 3.12, Ubuntu 24.04)
#   - ISAACLAB_REF:     Isaac Lab git commit, installed the same way as Isaac Lab's
#                       own docker/Dockerfile.base. release/3.0.0 targets Isaac Sim 6.2;
#                       the prebuilt nvcr.io/nvidia/isaac-lab:3.0.0-rc1 image still
#                       bundles a 6.1 release candidate.
#
# Isaac Sim 5.1 (Isaac Lab 2.x) crashes in librtx.scenedb on NVIDIA drivers newer
# than the 580 branch; 6.2 runs on current drivers. See docs/ISAAC_VERSIONS.md.
ARG ISAACSIM_VERSION=6.2.0.1
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG ISAACLAB_REF=70ab24fec3eabf51222817db0cdb5a4e7133bd7e
ENV ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    DOCKER_USER_HOME=/root \
    HOME=/root \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_RETRIES=12

# The 6.x images default to the unprivileged isaac-sim user; the stack (cache
# volumes under /root, Isaac Lab's own image) runs as root.
USER root
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl wget ca-certificates gnupg lsb-release \
        libglib2.0-0 ncurses-term \
    && rm -rf /var/lib/apt/lists/*

# The 6.2 image deduplicates Kit's Python with symlinks, e.g. pip/_vendor/urllib3/* ->
# extscache/omni.kit.pip_archive-*/pip_prebundle/urllib3/*. Isaac Lab's install upgrades
# urllib3, pip uninstalls that prebundled copy, and pip's own vendored files vanish
# ("No module named 'pip._vendor.urllib3.util.retry'"). Make them real files first.
RUN find /isaac-sim/kit/python/lib/python3.12 -type l -exec sh -c \
        'for l; do t=$(readlink -f "$l") && [ -f "$t" ] && cp --remove-destination "$t" "$l"; done' _ {} + \
    && /isaac-sim/python.sh -m pip --version

# --- Isaac Lab (pinned commit) -------------------------------------------------
RUN mkdir -p "$ISAACLAB_PATH" && cd "$ISAACLAB_PATH" \
    && git init -q && git remote add origin https://github.com/isaac-sim/IsaacLab.git \
    && git fetch -q --depth 1 origin "$ISAACLAB_REF" && git checkout -q FETCH_HEAD \
    && rm -rf .git \
    && chmod +x isaaclab.sh \
    && ln -sf "$ISAACSIM_ROOT_PATH" "$ISAACLAB_PATH/_isaac_sim" \
    && bash docker/scripts/install_carb_env_shim.sh "$ISAACSIM_ROOT_PATH"

RUN "$ISAACLAB_PATH/isaaclab.sh" -p -m pip install --no-cache-dir toml pytest \
    && "$ISAACLAB_PATH/isaaclab.sh" -p "$ISAACLAB_PATH/tools/install_deps.py" apt "$ISAACLAB_PATH/source" \
    && apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "$ISAACSIM_ROOT_PATH/kit/cache" /root/.cache/ov /root/.cache/pip \
        /root/.cache/nvidia/GLCache /root/.nv/ComputeCache /root/.nvidia-omniverse/logs \
        /root/.local/share/ov/data /root/Documents

RUN "$ISAACLAB_PATH/isaaclab.sh" --install && rm -rf /root/.cache/pip/*
# Same post-install cleanup as Isaac Lab's Dockerfile.base (packages may be absent).
RUN "$ISAACLAB_PATH/isaaclab.sh" -p -m pip uninstall -y imageio-ffmpeg quadprog || true

# --- ROS 2 Jazzy runtime libraries -----------------------------------------------
# isaacsim.ros2.core ships its own Jazzy rclpy/rosidl; the system debs provide extra
# native message libraries. Isaac Sim 6.x uses Python 3.12 like Jazzy, but keep only
# the native .so from /opt/ros so Python always uses the bridge-bundled bindings
# (mixing both causes rosidl ABI asserts). LD_LIBRARY_PATH order: entrypoint.sh.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*
RUN bash -c 'shopt -s nullglob; for _pydir in /opt/ros/*/lib/python3.*; do rm -rf "$_pydir"; done'

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    gymnasium \
    tensorboard \
    pyarrow \
    av \
    pandas

# msgpack: the policy-server wire protocol (isaac_envs/soarm_common/wire.py) for the closed-loop
# policy client (tutorials/fasteners/part1/07_run_policy.py, #36). websockets ships with Isaac Sim.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir "msgpack==1.2.3"

# Copy helper script for URDF -> USD conversion
COPY convert_urdf.py /usr/local/bin/convert_urdf.py

# Entrypoint that sets the ROS 2 library order before running any command
COPY entrypoint.sh /isaac-sim-entrypoint.sh
RUN chmod +x /isaac-sim-entrypoint.sh

WORKDIR /isaac_envs

ENTRYPOINT ["/isaac-sim-entrypoint.sh"]
CMD ["bash"]
