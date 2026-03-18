FROM nvcr.io/nvidia/isaac-lab:2.1.0

# The Isaac Lab pre-built image already ships:
#   - Isaac Sim 5.1.0 (Python at /isaac-sim/python.sh)
#   - Isaac Lab 2.1.0 (isaaclab.* modules pre-installed)
# We only add packages needed for data collection and training support.
#
# Reference: https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html

USER root

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    gymnasium \
    tensorboard \
    pyarrow \
    av \
    pandas

# Copy helper script for URDF -> USD conversion
COPY convert_urdf.py /usr/local/bin/convert_urdf.py

WORKDIR /isaac_envs

CMD ["bash"]
