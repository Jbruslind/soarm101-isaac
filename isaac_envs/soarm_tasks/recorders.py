"""Demo recording for the fastener tasks: Isaac Lab recorder terms and HDF5 file handler.

Isaac Lab's ``RecorderManager`` collects one ``EpisodeData`` per env and writes it to HDF5 with
``HDF5DatasetFileHandler`` (the format of ``record_demos.py``, ``replay_demos.py`` and Mimic's
``annotate_demos.py``). :class:`DemoRecorderManagerCfg` keeps Isaac Lab's default terms
(``ActionStateRecorderManagerCfg``: initial state, states, actions, policy obs, processed actions)
and adds the SO-ARM101 contract data. The resulting layout is ``soarm_tasks.demo_schema``.

The tasks register this cfg as ``demo_recorder_cfg_entry_point``, so ``record_demos.py`` uses it.
``scripts/generate_demos.py`` sets it on the env cfg directly. Pure configuration at import time
(no Kit needed); the terms only run inside an env.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from soarm_common import contract as C

from . import demo_schema as S


# ═══════════════════════════════════════════════════════════════════════════
# Recorder terms
# ═══════════════════════════════════════════════════════════════════════════

# [start:contract_recorder]
class ContractActionsRecorder(RecorderTerm):
    """Per step: the EE-pose action as executed (pre-step) and the absolute joint-position targets
    the drives received (post-step), in contract joint order."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self._joint_ids, names = robot.find_joints(list(C.JOINT_NAMES), preserve_order=True)
        assert tuple(names) == C.JOINT_NAMES, names

    def record_pre_step(self):
        # EE target [pos, quat xyzw] of the arm term (IK-Abs: the action itself; teleop: the
        # integrated target), then the gripper joint target (rad).
        am = self._env.action_manager
        if "arm_action" not in am.active_terms or "gripper_action" not in am.active_terms:
            return None, None
        arm = am.get_term("arm_action").processed_actions
        if arm.shape[-1] != 7:  # a joint-space arm action has no EE pose
            return None, None
        return S.EE_POSE_ACTIONS, torch.cat([arm, am.get_term("gripper_action").processed_actions], dim=-1)

    def record_post_step(self):
        target = self._env.scene["robot"].data.joint_pos_target.torch[:, self._joint_ids]
        return S.JOINT_POS_TARGETS, target
# [end:contract_recorder]


class CameraImagesRecorder(RecorderTerm):
    """Per step (pre-step, like the policy obs): the contract camera images, uint8 (H, W, 3), under
    ``obs/<CameraSpec.obs_key>``. Kept in host memory until the episode is written."""

    def record_pre_step(self):
        images = self._env.obs_buf.get("images")
        if not images:
            return None, None
        return "obs", {k: v.cpu() for k, v in images.items()}


class EpisodeMetaRecorder(RecorderTerm):
    """After each reset: attach seed, fastener, prompt and episode bookkeeping to the new episode
    (written as HDF5 attributes by :class:`DemoDatasetFileHandler`)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._resets = torch.zeros(env.num_envs, dtype=torch.long)
        self.source = getattr(cfg, "source", "unknown")
        self._names = self._prompts = None  # per env; read from the stage once (assets never change)

    def record_post_reset(self, env_ids: Sequence[int] | None):
        from .mdp import fastener_names, language_instructions

        env = self._env
        if self._names is None:
            self._names, self._prompts = fastener_names(env), language_instructions(env)
        names, prompts = self._names, self._prompts
        ids = range(env.num_envs) if env_ids is None else [int(i) for i in env_ids]
        for i in ids:
            ep = env.recorder_manager.get_episode(i)
            ep.seed = env.cfg.seed if env.cfg.seed is not None else -1
            ep.soarm_meta = S.episode_attrs(
                fastener=names[i] or env.cfg.fastener_names[0], prompt=prompts[i], source=self.source,
                env_id=i, reset_index=int(self._resets[i]), num_envs=env.num_envs,
            )
            self._resets[i] += 1
        return None, None


@configclass
class ContractActionsRecorderCfg(RecorderTermCfg):
    class_type: type = ContractActionsRecorder


@configclass
class CameraImagesRecorderCfg(RecorderTermCfg):
    class_type: type = CameraImagesRecorder


@configclass
class EpisodeMetaRecorderCfg(RecorderTermCfg):
    class_type: type = EpisodeMetaRecorder
    source: str = "unknown"
    """Where the demos come from: ``"scripted"`` or ``"teleop"``."""


# ═══════════════════════════════════════════════════════════════════════════
# File handler and manager cfg
# ═══════════════════════════════════════════════════════════════════════════

# [start:file_handler]
class DemoDatasetFileHandler(HDF5DatasetFileHandler):
    """Isaac Lab's HDF5 handler, plus the file-level schema attributes and the per-episode
    attributes from :class:`EpisodeMetaRecorder` (``EpisodeData.soarm_meta``)."""

    def create(self, file_path: str, env_name: str | None = None):
        super().create(file_path, env_name)
        for key, value in S.file_attrs().items():
            self._hdf5_data_group.attrs[key] = value

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None, dataset_compression: bool = True):
        name = f"demo_{self._demo_count if demo_id is None else demo_id}"
        super().write_episode(episode, demo_id, dataset_compression)
        if name in self._hdf5_data_group:
            for key, value in getattr(episode, "soarm_meta", {}).items():
                self._hdf5_data_group[name].attrs[key] = value
# [end:file_handler]


# [start:recorder_cfg]
@configclass
class DemoRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Isaac Lab's action/state recorders + contract actions, camera images and episode metadata."""

    dataset_file_handler_class_type: type = DemoDatasetFileHandler
    record_contract_actions = ContractActionsRecorderCfg()
    record_camera_images = CameraImagesRecorderCfg()
    record_episode_meta = EpisodeMetaRecorderCfg(source="scripted")


@configclass
class TeleopDemoRecorderManagerCfg(DemoRecorderManagerCfg):
    record_episode_meta = EpisodeMetaRecorderCfg(source="teleop")
# [end:recorder_cfg]
