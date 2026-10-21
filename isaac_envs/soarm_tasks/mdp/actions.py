"""Live action term of the teleop variant (loaded through ``IntegratedPoseIKActionCfg.class_type``).

Imported only after Kit has started: the parent term imports ``pxr``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action_term import SO101PoseIKAction

from ..expert import top_down_quat_from_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import IntegratedPoseIKActionCfg


class IntegratedPoseIKAction(SO101PoseIKAction):
    """Integrates delta-pose actions into an absolute EE target and tracks it with pose IK.

    ``processed_actions`` is the absolute target ``[pos(3), quat xyzw(4)]`` in the robot base
    frame: the same quantity the IK-Abs task takes as its action. The demo recorder stores it, so
    teleop demos can be replayed on the IK-Abs task too.
    """

    cfg: IntegratedPoseIKActionCfg

    def __init__(self, cfg: IntegratedPoseIKActionCfg, env: ManagerBasedEnv):
        if cfg.controller.use_relative_mode:
            raise ValueError("IntegratedPoseIKAction integrates deltas itself: set controller.use_relative_mode=False")
        super().__init__(cfg, env)
        n = self.num_envs
        self._target_pos = torch.zeros(n, 3, device=self.device)
        self._target_quat = torch.zeros(n, 4, device=self.device)
        self._target_quat[:, 3] = 1.0
        self._target_yaw = torch.zeros(n, device=self.device)
        self._needs_sync = torch.ones(n, dtype=torch.bool, device=self.device)
        self._abs_command = torch.zeros(n, 7, device=self.device)
        ws = torch.tensor(cfg.workspace, device=self.device)
        self._ws_lo, self._ws_hi = ws[:, 0], ws[:, 1]

    @property
    def action_dim(self) -> int:
        return 6  # [dx, dy, dz, rx, ry, rz]

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._abs_command

    @property
    def target_pose(self) -> torch.Tensor:
        """(N, 7) absolute EE target [pos, quat xyzw] in the robot base frame."""
        return self._abs_command

    @property
    def needs_sync(self) -> torch.Tensor:
        """(N,) True for envs whose target is re-read from the EE at the next action."""
        return self._needs_sync

    # [start:integrate]
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        delta = actions * self._scale
        ee_pos, ee_quat = self._compute_frame_pose()
        # after a reset, start from the ready pose (the IK moves the arm there)
        sync = self._needs_sync
        if sync.any():
            self._target_pos[sync] = torch.tensor(self.cfg.start_pos, device=self.device)
            self._target_yaw[sync] = self.cfg.start_yaw
            self._needs_sync[:] = False
        # position: integrate the delta. Keys may not push the target further than max_lead from
        # the EE (or further than it already is, while the arm is still moving there), nor out of
        # the workspace (widened to contain the current target, so the target never jumps).
        prev = self._target_pos
        pos = prev + delta[:, :3]
        lead = pos - ee_pos
        limit = (prev - ee_pos).norm(dim=-1, keepdim=True).clamp_min(self.cfg.max_lead)
        lead = lead * torch.clamp(limit / lead.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        lo, hi = torch.minimum(self._ws_lo, prev), torch.maximum(self._ws_hi, prev)
        self._target_pos = torch.maximum(torch.minimum(ee_pos + lead, hi), lo)
        # orientation: top-down with an integrated yaw, or the full rotation vector
        if self.cfg.top_down:
            self._target_yaw += delta[:, 5]
            self._target_quat = top_down_quat_from_yaw(self._target_yaw)
        else:
            angle = delta[:, 3:6].norm(dim=-1)
            axis = delta[:, 3:6] / angle.clamp_min(1e-9).unsqueeze(-1)
            dq = math_utils.quat_from_angle_axis(angle, axis)
            self._target_quat = math_utils.normalize(math_utils.quat_mul(dq, self._target_quat))
        self._abs_command = torch.cat([self._target_pos, self._target_quat], dim=-1)
        self._ik_controller.set_command(self._abs_command, ee_pos, ee_quat)
    # [end:integrate]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_sync[:] = True
        else:
            self._needs_sync[env_ids] = True
