FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pin NumPy<2 for OpenCV wheels. Use a constraints file so opencv cannot pull numpy 2.x.
# Phospho-bridge decodes sensor_msgs/Image with numpy+cv2 only (no ros-humble-cv-bridge).
RUN printf '%s\n' "numpy>=1.23.5,<2" >/tmp/pip-constraints.txt && \
    pip3 install --no-cache-dir -c /tmp/pip-constraints.txt \
        fastapi \
        "uvicorn[standard]" \
        opencv-python-headless && \
    pip3 install --no-cache-dir --force-reinstall -c /tmp/pip-constraints.txt "numpy>=1.23.5,<2" && \
    python3 -c "import numpy as np; assert int(np.__version__.split('.')[0])<2, np.__version__; import cv2; print('numpy', np.__version__, 'cv2', cv2.__version__)"

COPY app /app

WORKDIR /app

EXPOSE 8080

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 main.py"]
