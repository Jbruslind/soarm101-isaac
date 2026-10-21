"""Layout of the fastener demo HDF5 files, with a reader, a validator and an action swapper.

Plain Python (``h5py`` + ``numpy``; no Isaac Lab, no GPU), so it also runs in the LeRobot
conversion (#34) and in the tests. The files are Isaac Lab's ``HDF5DatasetFileHandler`` format
(``record_demos.py``, ``replay_demos.py``, Mimic), written by ``soarm_tasks.recorders``::

    /data                         attrs: env_args (JSON, Isaac Lab), total, soarm_* (file_attrs)
      /demo_<k>                   attrs: num_samples, seed, success, + episode_attrs()
        actions          (T, A)   env action as sent to env.step; replay_demos.py replays this
                                  IK-Abs: [pos(3), quat xyzw(4), gripper(1)]; IK-Rel: [dpos(3), drot(3), gripper +-1]
        ee_pose_actions  (T, 8)   EE target [pos, quat xyzw] in the robot base frame + gripper target (rad),
                                  as executed. IK-Abs: equal to actions.
        joint_pos_targets (T, 6)  absolute joint-position targets applied to the drives (rad, contract
                                  JOINT_NAMES order). This is the contract / LeRobot ``action``.
        processed_actions (T, 8)  Isaac Lab's processed action terms (= ee_pose_actions here)
        obs/joint_pos    (T, 6)   joint state before the action (rad); also joint_vel, ee_pose,
                                  gripper_pos, fastener_pose, bin_pos, actions (= last action)
        obs/wrist_rgb    (T, 224, 224, 3) uint8   contract camera images, before the action
        obs/overhead_rgb (T, 224, 224, 3) uint8
        initial_state/...         scene state after the reset (replay starts here)
        states/...                scene state after each step

Step ``t`` pairs the observation before the action with that action, as LeRobot does.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from soarm_common import contract as C

# [start:keys]
SCHEMA_VERSION = 1
ACTIONS = "actions"
EE_POSE_ACTIONS = "ee_pose_actions"
JOINT_POS_TARGETS = "joint_pos_targets"
JOINT_POS = "obs/joint_pos"
IMAGE_KEYS = tuple(f"obs/{cam.obs_key}" for cam in C.CAMERAS)  # obs/wrist_rgb, obs/overhead_rgb
EE_POSE_ACTION_DIM = 8
EPISODE_ATTRS = ("seed", "success", "fastener", "prompt", "source", "env_id", "reset_index", "num_envs")
# [end:keys]


def file_attrs() -> dict:
    """Attributes of the ``data`` group, next to Isaac Lab's ``env_args``."""
    return {
        "soarm_schema_version": SCHEMA_VERSION,
        "soarm_contract_version": C.CONTRACT_VERSION,
        "soarm_joint_names": json.dumps(list(C.JOINT_NAMES)),
        "soarm_control_hz": C.CONTROL_HZ,
        "soarm_image_keys": json.dumps(list(IMAGE_KEYS)),
    }


def episode_attrs(*, fastener: str, prompt: str, source: str, env_id: int, reset_index: int,
                  num_envs: int) -> dict:
    """Attributes of one ``demo_<k>`` group (``seed`` and ``success`` are Isaac Lab's own).

    ``seed`` is the env seed of the run. With the same seed and ``num_envs``, (``env_id``,
    ``reset_index``) names the same episode again; ``initial_state`` restores its start exactly.
    """
    return {"fastener": fastener, "prompt": prompt, "source": source, "env_id": env_id,
            "reset_index": reset_index, "num_envs": num_envs}


def _flatten(group, prefix: str = "") -> dict:
    import h5py

    out = {}
    for key, item in group.items():
        path = f"{prefix}{key}"
        if isinstance(item, h5py.Group):
            out.update(_flatten(item, path + "/"))
        else:
            out[path] = item
    return out


# [start:reader]
def episode_names(path: str) -> list[str]:
    """Demo group names in index order (demo_0, demo_1, ..., demo_10)."""
    import h5py

    with h5py.File(path, "r") as f:
        return sorted(f["data"].keys(), key=lambda n: int(n.rsplit("_", 1)[1]))


def load_episode(path: str, name: str, keys: tuple[str, ...] | None = None) -> tuple[dict, dict]:
    """(arrays, attrs) of one demo. Arrays are flat-keyed (``"obs/wrist_rgb"``); ``keys`` limits
    which are read (images are large)."""
    import h5py

    with h5py.File(path, "r") as f:
        group = f["data"][name]
        items = _flatten(group)
        arrays = {k: np.asarray(v) for k, v in items.items() if keys is None or k in keys}
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in group.attrs.items()}
    return arrays, attrs
# [end:reader]


def file_info(path: str) -> dict:
    """Env name, schema attributes, demo count, steps and successes of a file."""
    import h5py

    with h5py.File(path, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs.get("env_args", "{}"))
        demos = [data[n] for n in data]
        return {
            "env_name": env_args.get("env_name"),
            "schema_version": int(data.attrs.get("soarm_schema_version", 0)),
            "demos": len(demos),
            "steps": int(sum(int(d.attrs.get("num_samples", 0)) for d in demos)),
            "successes": int(sum(bool(d.attrs.get("success", False)) for d in demos)),
            "fasteners": sorted({str(d.attrs.get("fastener", "")) for d in demos}),
            "sources": sorted({str(d.attrs.get("source", "")) for d in demos}),
            "bytes": os.path.getsize(path),
        }


# [start:validate]
def validate(path: str, require_images: bool = True, require_ee: bool = True) -> list[str]:
    """Problems found in a demo file (empty list = valid)."""
    import h5py

    problems = []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return ["no /data group"]
        data = f["data"]
        if int(data.attrs.get("soarm_schema_version", -1)) != SCHEMA_VERSION:
            problems.append(f"soarm_schema_version != {SCHEMA_VERSION}")
        if json.loads(data.attrs.get("soarm_joint_names", "[]")) != list(C.JOINT_NAMES):
            problems.append("soarm_joint_names differ from the contract")
        if not json.loads(data.attrs.get("env_args", "{}")).get("env_name"):
            problems.append("env_args has no env_name")
        for name in data:
            demo = data[name]
            items = _flatten(demo)
            where = f"{name}:"
            for attr in EPISODE_ATTRS:
                if attr not in demo.attrs:
                    problems.append(f"{where} missing attr {attr!r}")
            required = [ACTIONS, JOINT_POS_TARGETS, JOINT_POS]
            required += [EE_POSE_ACTIONS] if require_ee else []
            required += list(IMAGE_KEYS) if require_images else []
            missing = [k for k in required if k not in items]
            problems += [f"{where} missing dataset {k!r}" for k in missing]
            if not any(k.startswith("initial_state/") for k in items):
                problems.append(f"{where} missing initial_state")
            if missing:
                continue
            t = int(demo.attrs.get("num_samples", -1))
            for k in required:
                if items[k].shape[0] != t:
                    problems.append(f"{where} {k} has {items[k].shape[0]} steps, num_samples={t}")
            if items[JOINT_POS_TARGETS].shape[1:] != (C.NUM_JOINTS,):
                problems.append(f"{where} {JOINT_POS_TARGETS} shape {items[JOINT_POS_TARGETS].shape}")
            if items[JOINT_POS].shape[1:] != (C.NUM_JOINTS,):
                problems.append(f"{where} {JOINT_POS} shape {items[JOINT_POS].shape}")
            if require_ee:
                ee = items[EE_POSE_ACTIONS]
                if ee.shape[1:] != (EE_POSE_ACTION_DIM,):
                    problems.append(f"{where} {EE_POSE_ACTIONS} shape {ee.shape}")
                elif t > 0 and not np.allclose(np.linalg.norm(ee[:, 3:7], axis=-1), 1.0, atol=1e-3):
                    problems.append(f"{where} {EE_POSE_ACTIONS} quaternions are not unit")
            for k in IMAGE_KEYS if require_images else ():
                img = items[k]
                if img.dtype != np.uint8 or img.shape[1:] != (C.IMAGE_HEIGHT, C.IMAGE_WIDTH, 3):
                    problems.append(f"{where} {k} is {img.dtype} {img.shape[1:]}, want uint8 "
                                    f"({C.IMAGE_HEIGHT}, {C.IMAGE_WIDTH}, 3)")
    return problems
# [end:validate]


# [start:swap]
def copy_with_actions(src: str, dst: str, action_key: str, env_name: str) -> int:
    """Copy ``src`` to ``dst`` with every demo's ``actions`` replaced by ``action_key`` and the
    file's env name set to ``env_name``, so ``replay_demos.py`` replays that action instead.

    E.g. ``joint_pos_targets`` on ``Soarm-PickPlace-Fastener-JointAbs-v0`` replays the contract
    action, ``ee_pose_actions`` on ``...-IK-Abs-v0`` replays teleop demos as absolute poses.
    Returns the number of demos.
    """
    import h5py

    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        data = f["data"]
        for name in data:
            demo = data[name]
            if action_key not in demo:
                raise KeyError(f"{name} has no {action_key!r}")
            values = np.asarray(demo[action_key])
            del demo[ACTIONS]
            demo.create_dataset(ACTIONS, data=values, compression="gzip", compression_opts=2)
        env_args = json.loads(data.attrs.get("env_args", "{}"))
        env_args["env_name"] = env_name
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["soarm_actions_from"] = action_key
        return len(data)
# [end:swap]
