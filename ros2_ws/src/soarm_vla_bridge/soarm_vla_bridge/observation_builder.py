"""Build observation dicts from ROS2 messages for the OpenPi client.

Handles image resizing / uint8 conversion and joint-state packing in
the format expected by the SO-ARM101 OpenPi policy config.
"""

from __future__ import annotations

import numpy as np

SOARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

IMAGE_SIZE = 224


def resize_image(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize to (size, size, 3) uint8 using nearest-neighbor (fast, no OpenCV dep)."""
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    if h == size and w == size:
        return img.astype(np.uint8)

    try:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    except ImportError:
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        return np.array(pil_img, dtype=np.uint8)


def build_observation(
    joint_positions: dict[str, float],
    wrist_image: np.ndarray | None = None,
    overhead_image: np.ndarray | None = None,
    prompt: str = "move the robot arm",
) -> dict:
    """Construct the observation dict expected by OpenPi's WebsocketClientPolicy.

    Returns a dict with keys: "state", "images", "prompt".
    The openpi-client handles image preprocessing (resize + uint8) but we
    do it here too for explicit control.
    """
    state = np.array(
        [joint_positions.get(name, 0.0) for name in SOARM_JOINT_NAMES],
        dtype=np.float32,
    )

    images = {}
    if wrist_image is not None:
        images["cam_wrist"] = resize_image(wrist_image)
    if overhead_image is not None:
        images["cam_overhead"] = resize_image(overhead_image)

    return {
        "state": state,
        "images": images,
        "prompt": prompt,
    }
