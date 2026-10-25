"""Tests for isaac_envs/soarm_tasks (GitHub issue #32). No GPU and no Kit needed.

The plain-Python parts (work cell, language, contract camera pose) run anywhere; the gym
registration and env-cfg tests need Isaac Lab importable (inside the Isaac image:
``/isaac-sim/python.sh -m pytest /isaac_envs/tests``) and are skipped elsewhere.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # isaac_envs/, like the env scripts

from soarm_common import contract as C  # noqa: E402
from soarm_tasks import language, work_cell  # noqa: E402

TASKS = (
    "Soarm-PickPlace-Fastener-JointAbs-v0",
    "Soarm-PickPlace-Fastener-IK-Abs-v0",
    "Soarm-PickPlace-Fastener-State-v0",
)


# --- plain Python ------------------------------------------------------------------------------


def test_instruction_from_fastener_name():
    assert language.instruction_for("hex_bolt_M6x20") == "pick up the M6 hex bolt and place it in the bin"
    assert language.fastener_phrase("socket_cap_M8x40") == "M8 socket head cap screw"
    assert language.fastener_phrase("hex_nut_M5") == "M5 hex nut"
    assert language.fastener_name_from_attrs("hex_bolt", "M6", 20.0) == "hex_bolt_M6x20"
    with pytest.raises(ValueError):
        language.fastener_phrase("rivet_M6")


def test_bin_parts_enclose_inner_box():
    b = work_cell.BIN
    parts = {name: (size, center) for name, size, center in b.parts()}
    assert set(parts) == {"floor", "wall_xp", "wall_xn", "wall_yp", "wall_yn"}
    size, center = parts["wall_xp"]
    assert center[0] - size[0] / 2 == pytest.approx(b.inner_size[0] / 2)  # inner face of +x wall
    size, center = parts["wall_yn"]
    assert center[1] + size[1] / 2 == pytest.approx(-b.inner_size[1] / 2)
    assert parts["floor"][1][2] + parts["floor"][0][2] / 2 == pytest.approx(b.floor_top)


def test_spawn_area_clear_of_bin_and_reachable():
    b = work_cell.BIN
    gap = (b.center[1] - b.inner_size[1] / 2 - b.wall_thickness) - work_cell.FASTENER_SPAWN_Y[1]
    assert gap >= 0.05
    far = math.hypot(work_cell.FASTENER_SPAWN_X[1], work_cell.FASTENER_SPAWN_Y[0])
    assert far < 0.30 and work_cell.FASTENER_SPAWN_X[0] >= 0.12


def _project(spec, point):
    """Pixel of an env-frame point in a world-mounted contract camera ("world" convention)."""
    x, y, z, w = spec.rot
    # rotation matrix of (x, y, z, w); columns = camera axes (forward, left, up) in the env frame
    r = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    d = [p - e for p, e in zip(point, spec.pos)]
    fwd, left, up = (sum(r[i][k] * d[i] for i in range(3)) for k in range(3))
    f_px = spec.width * spec.focal_length / spec.horizontal_aperture
    return spec.width / 2 - f_px * left / fwd, spec.height / 2 - f_px * up / fwd, fwd


def test_overhead_camera_sees_work_cell():
    """Spawn-area corners and bin corners land inside the 224 px overhead image."""
    cam = C.OVERHEAD_CAMERA
    b = work_cell.BIN
    corners = [(x, y, 0.0) for x in work_cell.FASTENER_SPAWN_X for y in work_cell.FASTENER_SPAWN_Y]
    corners += [(b.center[0] + sx * b.inner_size[0] / 2, b.center[1] + sy * b.inner_size[1] / 2, b.wall_height)
                for sx in (-1, 1) for sy in (-1, 1)]
    for p in corners:
        u, v, depth = _project(cam, p)
        assert depth > 0 and 0 <= u < cam.width and 0 <= v < cam.height, (p, u, v)


def test_overhead_resolves_m6_bolt():
    """A 24 mm long M6x20 bolt across the view at the spawn-area centre is >= 10 px long."""
    cx, cy = sum(work_cell.FASTENER_SPAWN_X) / 2, sum(work_cell.FASTENER_SPAWN_Y) / 2
    u0, v0, _ = _project(C.OVERHEAD_CAMERA, (cx, cy - 0.012, 0.004))
    u1, v1, _ = _project(C.OVERHEAD_CAMERA, (cx, cy + 0.012, 0.004))
    assert math.hypot(u1 - u0, v1 - v0) >= 10.0


# --- Isaac Lab (cfg only, no Kit) ---------------------------------------------------------------

isaaclab = pytest.importorskip("isaaclab", reason="needs Isaac Lab (run inside the Isaac image)")


def test_tasks_registered():
    import gymnasium as gym

    import soarm_tasks

    for task in TASKS:
        assert task in gym.registry
        spec = gym.spec(task)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"].startswith("soarm_tasks.pick_place_env_cfg:")
    assert set(soarm_tasks.TASK_IDS) == set(TASKS)


@pytest.mark.parametrize("task", TASKS)
def test_env_cfg_builds(task):
    from isaaclab_tasks.utils import parse_env_cfg

    import soarm_tasks  # noqa: F401
    from soarm_tasks.pick_place_env_cfg import cameras_enabled

    cfg = parse_env_cfg(task, num_envs=16)
    cfg.validate()
    assert cfg.scene.num_envs == 16
    assert cfg.scene.replicate_physics is False
    assert cfg.sim.dt == pytest.approx(C.PHYSICS_DT) and cfg.decimation == C.DECIMATION
    assert cfg.sim.physics.solve_articulation_contact_last is True
    assert cfg.language_instruction == "pick up the M6 hex bolt and place it in the bin"
    assert cfg.scene.fastener.spawn.usd_path.endswith("hex_bolt_M6x20.usda")
    assert "env_[^/]+" in cfg.scene.fastener.prim_path or "{ENV_REGEX_NS}" in cfg.scene.fastener.prim_path
    assert set(vars(cfg.observations.subtask_terms)) >= {"grasped", "lifted", "in_bin"}
    if "State" in task:
        assert not cameras_enabled(cfg) and cfg.observations.images is None
        assert cfg.observations.policy.concatenate_terms
    else:
        assert cameras_enabled(cfg)
        # image observation names are the contract's obs keys
        assert {C.WRIST_CAMERA.obs_key, C.OVERHEAD_CAMERA.obs_key} <= set(vars(cfg.observations.images))
        assert cfg.scene.overhead_cam.offset.pos == C.OVERHEAD_CAMERA.pos
        assert cfg.scene.wrist_cam.height == C.IMAGE_HEIGHT
    arm = cfg.actions.arm_action
    if "IK" in task:
        assert arm.body_name == C.EE_BODY_NAME and arm.controller.command_type == "pose"
    else:
        assert list(arm.joint_names) == list(C.ARM_JOINT_NAMES) and arm.use_default_offset is False
    assert list(cfg.actions.gripper_action.joint_names) == [C.GRIPPER_JOINT_NAME]


def test_fastener_mix_uses_multi_asset_spawner():
    import isaaclab.sim as sim_utils

    from soarm_tasks.pick_place_env_cfg import FastenerPickPlaceJointAbsEnvCfg

    cfg = FastenerPickPlaceJointAbsEnvCfg()
    cfg.fastener_names = ("hex_bolt_M5x16", "hex_bolt_M6x20", "hex_bolt_M8x20")
    cfg.__post_init__()
    assert isinstance(cfg.scene.fastener.spawn, sim_utils.MultiAssetSpawnerCfg)
    assert len(cfg.scene.fastener.spawn.assets_cfg) == 3


def test_expert_plan_aligns_jaws_across_bolt():
    torch = pytest.importorskip("torch")
    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    from soarm_tasks.expert import FastenerPickExpert, top_down_quat
    from soarm_tasks.mdp.events import LYING_QUAT

    yaws = torch.tensor([0.0, 0.7, -1.2, 2.5, -3.0])
    n = len(yaws)
    q = quat_mul(quat_from_angle_axis(yaws + math.pi / 2, torch.tensor([[0.0, 0.0, 1.0]]).expand(n, -1)),
                 torch.tensor([LYING_QUAT]).expand(n, -1))
    pose = torch.cat([torch.tensor([[0.2, -0.07, 0.005]]).expand(n, -1), q], -1)
    ex = FastenerPickExpert(n, "cpu", work_cell.BIN.center)
    ex.plan(pose, torch.arange(n))
    axis = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]]).expand(n, -1))
    assert torch.allclose(torch.atan2(axis[:, 1], axis[:, 0]), torch.atan2(torch.sin(yaws), torch.cos(yaws)),
                          atol=1e-5)
    gx = quat_apply(top_down_quat(ex.target, ex.dyaw), torch.tensor([[1.0, 0.0, 0.0]]).expand(n, -1))
    assert torch.allclose((gx * axis).sum(-1), torch.zeros(n), atol=1e-5)  # jaws close across the axis
    assert (ex.dyaw.abs() <= math.pi / 2 + 1e-6).all()
