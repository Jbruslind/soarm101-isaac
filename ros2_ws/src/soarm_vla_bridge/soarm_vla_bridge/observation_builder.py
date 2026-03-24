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
POLICY_MODE_SOARM = "soarm"
POLICY_MODE_DROID = "droid"


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
    policy_mode: str = POLICY_MODE_SOARM,
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
    wrist_resized = resize_image(wrist_image) if wrist_image is not None else None
    overhead_resized = (
        resize_image(overhead_image) if overhead_image is not None else None
    )

    if wrist_resized is not None:
        images["cam_wrist"] = wrist_resized
    if overhead_resized is not None:
        images["cam_overhead"] = overhead_resized

    # DROID fallback compatibility:
    # If the OpenPi server is running the DROID policy, it expects LeRobot-style
    # keys under "observation/*" (see OpenPi `DroidInputs` transform).
    #
    # We map SOARM -> DROID as follows:
    # - wrist camera   -> observation/wrist_image_left
    # - overhead camera-> observation/exterior_image_1_left
    # - joint state    -> observation/joint_position (7D) + observation/gripper_position (1D)
    #
    # Note: SOARM has 5 arm joints + 1 gripper, while DROID expects 7 arm joints + 1 gripper.
    # We pad the missing two arm joints with zeros to satisfy the shape contract.
    droid_joint_position = np.zeros((7,), dtype=np.float32)
    arm_joint_names = SOARM_JOINT_NAMES[:5]
    droid_joint_position[: len(arm_joint_names)] = np.array(
        [joint_positions.get(n, 0.0) for n in arm_joint_names], dtype=np.float32
    )
    droid_gripper_position = np.array(
        [float(np.clip(joint_positions.get("gripper", 0.0), -1.0, 1.0))],
        dtype=np.float32,
    )

    zero_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    droid_wrist_image_left = wrist_resized if wrist_resized is not None else zero_img
    droid_exterior_image_1_left = (
        overhead_resized if overhead_resized is not None else zero_img
    )

    soarm_obs = {
        # SOARM / pi0-style inputs
        "state": state,
        "images": images,
        "prompt": prompt,
    }

    droid_obs = {
        # DROID fallback inputs
        "observation/joint_position": droid_joint_position,
        "observation/gripper_position": droid_gripper_position,
        "observation/wrist_image_left": droid_wrist_image_left,
        "observation/exterior_image_1_left": droid_exterior_image_1_left,
    }

    if policy_mode == POLICY_MODE_DROID:
        # DROID policies consume LeRobot-style observation/* keys.
        return {
            **droid_obs,
            # Keep prompt for policy configs that also consume language.
            "prompt": prompt,
        }

    # SOARM-native policies consume compact state/images keys.
    return soarm_obs
