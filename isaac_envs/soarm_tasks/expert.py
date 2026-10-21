"""Scripted IK expert for ``Soarm-PickPlace-Fastener-IK-Abs-v0`` (batched over envs).

The same waypoint state machine as ``sim_data_collector.ScriptedPick`` (the 20/20 cube pick),
adapted to a bolt lying on its side and vectorized over ``num_envs``:

    settle -> hover -> descend -> close -> lift -> carry -> lower -> release -> retreat

* The gripper is held top-down (gripper_frame_link +z down). Its x axis, the direction the jaws
  close along, is turned (``dyaw``) to lie across the bolt axis; of the two directions across
  the axis, the one nearer ScriptedPick's default "x towards the base" is used.
* The grasp point is on the bolt axis, ``grasp_axis_offset`` from the head's bearing face
  towards the tip (the shank) or negative (the head). The jaws close on it with the fixed jaw on
  the +x side, so the EE goal is the grasp point shifted by ``grasp_offset`` along +x.
* The commanded EE goal moves at most ``max_speed`` per step; the action term's IK tracks it.

``act`` returns IK-Abs actions: ``[pos(3), quat xyzw(4)]`` of gripper_frame_link in the robot base
frame, then the gripper joint target (rad).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply, quat_from_matrix


@dataclass
class ExpertParams:
    # [start:params]
    hover_z: float = 0.08            # m, EE height while moving
    grasp_z: float = 0.007           # m, EE height when closing (jaw tips reach ~7 mm lower)
    place_z: float = 0.06            # m, EE height when releasing over the bin (bolt drops ~2-3 cm)
    grasp_axis_offset: float = 0.008  # m along the bolt axis from the bearing face to the shank
    grasp_offset: float = 0.004      # m, grasp point at x = -grasp_offset in the gripper frame
    open: float = 1.0                # rad, gripper while approaching / releasing
    closed: float = -0.15            # rad, gripper goal while holding (below contact: squeeze)
    max_speed: float = 0.006         # m per control step for the commanded EE goal (0.01: 86%, see docs)
    carry_speed: float = 0.006       # m per step while the bolt is held (lift, carry, lower)
    hover_tol: float = 0.004         # m, EE tolerance above the grasp point
    tol: float = 0.01                # m, EE tolerance in the other phases
    settle_steps: int = 15           # steps the bolt gets to settle before the grasp is planned
    # [end:params]


# (name, gripper ("open"/"closed"), steps to hold once within tol, max steps in phase)
PHASES = (
    ("settle", "open", 0, 0),        # max steps = params.settle_steps
    ("hover", "open", 3, 90),
    ("descend", "open", 5, 45),
    ("close", "closed", 30, 30),
    ("lift", "closed", 0, 40),
    ("carry", "closed", 0, 80),
    ("lower", "closed", 5, 40),
    ("release", "open", 15, 15),
    ("retreat", "open", 10**9, 10**9),
)
PHASE_NAMES = tuple(p[0] for p in PHASES)
P = {name: i for i, name in enumerate(PHASE_NAMES)}


def wrap_half_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to [-pi/2, pi/2): directions across an axis repeat every pi."""
    return torch.remainder(a + torch.pi / 2, torch.pi) - torch.pi / 2


def x_yaw(pos: torch.Tensor, dyaw: torch.Tensor) -> torch.Tensor:
    """ScriptedPick's gripper x yaw: towards the base, turned by ``dyaw``."""
    return torch.atan2(pos[:, 1], pos[:, 0]) + torch.pi + dyaw


def top_down_quat(pos: torch.Tensor, dyaw: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) with gripper z down and x at :func:`x_yaw`."""
    yaw = x_yaw(pos, dyaw)
    c, s, zero = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)
    x_axis = torch.stack([c, s, zero], -1)
    y_axis = torch.stack([s, -c, zero], -1)
    z_axis = torch.stack([zero, zero, -torch.ones_like(yaw)], -1)
    return quat_from_matrix(torch.stack([x_axis, y_axis, z_axis], -1))


class FastenerPickExpert:
    """Batched pick-and-place expert. Call :meth:`reset` for envs that were reset."""

    def __init__(self, num_envs: int, device, bin_center: tuple[float, float], params: ExpertParams | None = None):
        self.n, self.device, self.p = num_envs, device, params or ExpertParams()
        self.bin = torch.tensor([bin_center[0], bin_center[1]], device=device)
        z = lambda *s: torch.zeros(num_envs, *s, device=device)  # noqa: E731
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t = torch.zeros_like(self.phase)
        self.settled = torch.zeros_like(self.phase)
        self.cmd, self.grasp_pos, self.target = z(3), z(3), z(3)
        self.dyaw = z()
        self.has_cmd = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.max_phase = torch.zeros_like(self.phase)  # furthest phase reached (diagnostics)

    def reset(self, env_ids: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = 0
        self.t[ids] = 0
        self.settled[ids] = 0
        self.max_phase[ids] = 0
        self.has_cmd[ids] = False

    # [start:plan]
    def plan(self, fastener_pose_b: torch.Tensor, ids: torch.Tensor):
        """Grasp point, EE goal above it and wrist yaw from the fastener pose (base frame)."""
        pos, quat = fastener_pose_b[ids, :3], fastener_pose_b[ids, 3:7]
        axis = quat_apply(quat, torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(len(ids), -1))
        grasp = pos - self.p.grasp_axis_offset * axis
        axis_yaw = torch.atan2(axis[:, 1], axis[:, 0])
        # Jaws close along gripper x: put x across the bolt axis, nearest to the default yaw. The
        # default yaw depends on where the EE is, so solve at the goal (a few fixed-point passes).
        goal = grasp.clone()
        for _ in range(3):
            dyaw = wrap_half_pi(axis_yaw + torch.pi / 2 - x_yaw(goal, torch.zeros_like(axis_yaw)))
            yaw = x_yaw(goal, dyaw)
            goal = grasp.clone()
            goal[:, :2] += self.p.grasp_offset * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
        self.dyaw[ids] = wrap_half_pi(axis_yaw + torch.pi / 2 - x_yaw(goal, torch.zeros_like(axis_yaw)))
        self.target[ids] = goal
    # [end:plan]

    def _goals(self, ee: torch.Tensor) -> torch.Tensor:
        p, ph = self.p, self.phase
        goal = self.target.clone()
        goal[:, 2] = p.hover_z
        at_grasp = (ph == P["descend"])
        goal[at_grasp, 2] = p.grasp_z
        # close / lift: hold the EE where the descent ended, then lift straight up
        holding = (ph == P["close"]) | (ph == P["lift"])
        goal[holding] = self.grasp_pos[holding]
        goal[ph == P["lift"], 2] = p.hover_z
        # over the bin: shift by the same grasp offset so the bolt, not the EE, is centred
        placing = (ph >= P["carry"])
        if placing.any():
            place = torch.zeros(int(placing.sum()), 3, device=self.device)
            place[:, :2] = self.bin
            yaw = x_yaw(place, self.dyaw[placing])
            place[:, :2] += p.grasp_offset * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
            place[:, 2] = p.hover_z
            lowered = ((ph == P["lower"]) | (ph == P["release"]))[placing]
            place[lowered, 2] = p.place_z
            goal[placing] = place
        return goal

    def act(self, ee_pos_b: torch.Tensor, fastener_pose_b: torch.Tensor) -> torch.Tensor:
        p = self.p
        new = ~self.has_cmd
        self.cmd[new] = ee_pos_b[new]
        self.has_cmd[:] = True
        settling = self.phase == P["settle"]
        if settling.any():
            self.plan(fastener_pose_b, settling.nonzero().squeeze(-1))
        # remember where the descent ended when the close phase starts
        goal = self._goals(ee_pos_b)
        dist = torch.linalg.vector_norm(ee_pos_b - goal, dim=-1)
        tol = torch.where(self.phase == P["hover"], p.hover_tol, p.tol)
        self.settled += (dist < tol).long()
        self.t += 1
        hold = torch.tensor([ph[2] for ph in PHASES], device=self.device)[self.phase]
        max_steps = torch.tensor([p.settle_steps] + [ph[3] for ph in PHASES[1:]], device=self.device)[self.phase]
        advance = ((self.settled > hold) | (self.t >= max_steps)) & (self.phase < len(PHASES) - 1)
        advance[settling] = self.t[settling] >= p.settle_steps
        entering_close = advance & (self.phase == P["descend"])
        self.grasp_pos[entering_close] = ee_pos_b[entering_close]
        self.phase += advance.long()
        self.t[advance] = 0
        self.settled[advance] = 0
        self.max_phase = torch.maximum(self.max_phase, self.phase)
        # bounded-speed command towards the (possibly new) goal
        goal = self._goals(ee_pos_b)
        holding = ((self.phase >= P["lift"]) & (self.phase <= P["lower"])).unsqueeze(-1)
        speed = torch.where(holding, p.carry_speed, p.max_speed)
        delta = goal - self.cmd
        self.cmd = self.cmd + delta * (speed / torch.maximum(delta.norm(dim=-1, keepdim=True), speed))
        grip_closed = torch.tensor([ph[1] == "closed" for ph in PHASES], device=self.device)[self.phase]
        gripper = torch.where(grip_closed, p.closed, p.open).unsqueeze(-1)
        return torch.cat([self.cmd, top_down_quat(self.cmd, self.dyaw), gripper], dim=-1)

    def phase_names(self) -> list[str]:
        return [PHASE_NAMES[i] for i in self.phase.tolist()]
