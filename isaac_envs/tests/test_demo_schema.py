"""Tests for the demo HDF5 writer/reader (GitHub issue #33). No GPU and no Kit needed.

The writer is Isaac Lab's ``HDF5DatasetFileHandler`` subclass in ``soarm_tasks.recorders``, so
these tests need Isaac Lab, torch and h5py importable (inside the Isaac image:
``/isaac-sim/python.sh -m pytest /isaac_envs/tests``) and are skipped elsewhere. An episode is
built the way the recorder manager builds it (``EpisodeData.add`` per step), written, then read
back and validated with ``soarm_tasks.demo_schema``.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # isaac_envs/

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab", reason="needs Isaac Lab (run inside the Isaac image)")

from soarm_common import contract as C  # noqa: E402
from soarm_tasks import demo_schema as S  # noqa: E402

T = 7  # steps per test episode
H, W = C.IMAGE_HEIGHT, C.IMAGE_WIDTH


def make_episode(source="scripted", seed=3, success=True, images=True, steps=T):
    """An EpisodeData filled like the recorder manager fills it (one ``add`` per key and step)."""
    from isaaclab.utils.datasets import EpisodeData

    ep = EpisodeData()
    ep.add("initial_state/articulation/robot/joint_position", torch.zeros(C.NUM_JOINTS))
    ep.add("initial_state/rigid_object/fastener/root_pose", torch.tensor([0.2, -0.07, 0.0065, 0, 0, 0, 1.0]))
    for t in range(steps):
        pose = torch.tensor([0.2, -0.07, 0.08, 1.0, 0.0, 0.0, 0.0, 1.0])  # x-flip quat: gripper down
        ep.add(S.ACTIONS, pose)
        ep.add(S.EE_POSE_ACTIONS, pose)
        ep.add(S.JOINT_POS_TARGETS, torch.full((C.NUM_JOINTS,), 0.01 * t))
        ep.add("obs", {"joint_pos": torch.full((C.NUM_JOINTS,), 0.01 * t), "joint_vel": torch.zeros(C.NUM_JOINTS)})
        if images:
            ep.add("obs", {cam.obs_key: torch.full((H, W, 3), t, dtype=torch.uint8) for cam in C.CAMERAS})
        ep.add("states/articulation/robot/joint_position", torch.zeros(C.NUM_JOINTS))
    ep.seed = seed
    ep.success = success
    ep.soarm_meta = S.episode_attrs(fastener="hex_bolt_M6x20", prompt="pick up the M6 hex bolt and place it in the bin",
                                    source=source, env_id=1, reset_index=4, num_envs=2)
    ep.pre_export()
    return ep


def write_file(path, episodes, env_name="Soarm-PickPlace-Fastener-IK-Abs-v0"):
    from soarm_tasks.recorders import DemoDatasetFileHandler

    handler = DemoDatasetFileHandler()
    handler.create(str(path), env_name=env_name)
    for ep in episodes:
        handler.write_episode(ep)
    handler.close()
    return str(path)


def test_written_file_is_valid_isaac_lab_format(tmp_path):
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    path = write_file(tmp_path / "demos.hdf5", [make_episode(), make_episode(seed=4)])
    assert S.validate(path) == []
    # Isaac Lab's own reader (replay_demos.py, Mimic) loads it
    reader = HDF5DatasetFileHandler()
    reader.open(path)
    assert reader.get_env_name() == "Soarm-PickPlace-Fastener-IK-Abs-v0"
    assert reader.get_num_episodes() == 2
    ep = reader.load_episode("demo_1", "cpu")
    assert ep.seed == 4 and bool(ep.success)
    assert ep.get_next_action().shape == (S.EE_POSE_ACTION_DIM,)
    assert ep.get_initial_state()["rigid_object"]["fastener"]["root_pose"].shape == (1, 7)  # as recorded
    reader.close()


def test_episode_and_file_attributes_round_trip(tmp_path):
    path = write_file(tmp_path / "demos.hdf5", [make_episode(source="teleop")])
    arrays, attrs = S.load_episode(path, "demo_0")
    assert attrs["seed"] == 3 and attrs["success"]
    assert attrs["fastener"] == "hex_bolt_M6x20" and attrs["source"] == "teleop"
    assert attrs["prompt"] == "pick up the M6 hex bolt and place it in the bin"
    assert (attrs["env_id"], attrs["reset_index"], attrs["num_envs"]) == (1, 4, 2)
    assert attrs["num_samples"] == T
    assert arrays[S.JOINT_POS_TARGETS].shape == (T, C.NUM_JOINTS)
    assert arrays["obs/wrist_rgb"].dtype == "uint8" and arrays["obs/wrist_rgb"].shape == (T, H, W, 3)
    assert int(arrays["obs/overhead_rgb"][5, 0, 0, 0]) == 5  # step order kept
    with h5py.File(path) as f:
        data = f["data"]
        assert json.loads(data.attrs["soarm_joint_names"]) == list(C.JOINT_NAMES)
        assert int(data.attrs["soarm_contract_version"]) == C.CONTRACT_VERSION
        assert json.loads(data.attrs["soarm_image_keys"]) == ["obs/wrist_rgb", "obs/overhead_rgb"]
    info = S.file_info(path)
    assert (info["demos"], info["steps"], info["successes"], info["sources"]) == (1, T, 1, ["teleop"])


def test_validate_reports_missing_and_misshaped_data(tmp_path):
    path = write_file(tmp_path / "no_images.hdf5", [make_episode(images=False)])
    assert S.validate(path, require_images=False) == []
    problems = S.validate(path)
    assert any("obs/wrist_rgb" in p for p in problems) and any("obs/overhead_rgb" in p for p in problems)

    path = write_file(tmp_path / "bad.hdf5", [make_episode()])
    with h5py.File(path, "r+") as f:
        demo = f["data/demo_0"]
        del demo[S.JOINT_POS_TARGETS]
        demo.create_dataset(S.JOINT_POS_TARGETS, data=torch.zeros(T - 1, 5).numpy())
        del demo.attrs["prompt"]
    problems = S.validate(path)
    assert any("missing attr 'prompt'" in p for p in problems)
    assert any(S.JOINT_POS_TARGETS in p and "steps" in p for p in problems)
    assert any(S.JOINT_POS_TARGETS in p and "shape" in p for p in problems)


def test_episode_names_sort_numerically(tmp_path):
    path = write_file(tmp_path / "many.hdf5", [make_episode(images=False, steps=1) for _ in range(12)])
    assert S.episode_names(path)[:3] == ["demo_0", "demo_1", "demo_2"]
    assert S.episode_names(path)[-1] == "demo_11"


def test_swap_actions_for_joint_space_replay(tmp_path):
    src = write_file(tmp_path / "demos.hdf5", [make_episode(images=False)])
    dst = str(tmp_path / "demos_joint.hdf5")
    assert S.copy_with_actions(src, dst, S.JOINT_POS_TARGETS, "Soarm-PickPlace-Fastener-JointAbs-v0") == 1
    a, _ = S.load_episode(dst, "demo_0", keys=(S.ACTIONS, S.JOINT_POS_TARGETS))
    assert (a[S.ACTIONS] == a[S.JOINT_POS_TARGETS]).all() and a[S.ACTIONS].shape == (T, C.NUM_JOINTS)
    assert S.file_info(dst)["env_name"] == "Soarm-PickPlace-Fastener-JointAbs-v0"
    assert S.file_info(src)["env_name"] == "Soarm-PickPlace-Fastener-IK-Abs-v0"  # source untouched
    with pytest.raises(KeyError):
        S.copy_with_actions(src, str(tmp_path / "x.hdf5"), "no_such_key", "x")


def test_demo_recorder_cfg_registered_and_complete():
    import gymnasium as gym

    import soarm_tasks
    from soarm_tasks.recorders import DemoDatasetFileHandler, DemoRecorderManagerCfg, TeleopDemoRecorderManagerCfg

    for task, entry in soarm_tasks.DEMO_RECORDERS.items():
        assert gym.spec(task).kwargs["demo_recorder_cfg_entry_point"] == f"soarm_tasks.{entry}"
    cfg = DemoRecorderManagerCfg()
    assert cfg.dataset_file_handler_class_type is DemoDatasetFileHandler
    # Isaac Lab's default terms (initial state, states, actions, ...) are kept
    for term in ("record_initial_state", "record_post_step_states", "record_pre_step_actions",
                 "record_contract_actions", "record_camera_images", "record_episode_meta"):
        assert getattr(cfg, term) is not None, term
    assert cfg.record_episode_meta.source == "scripted"
    assert TeleopDemoRecorderManagerCfg().record_episode_meta.source == "teleop"


def test_teleop_task_takes_se3_device_actions():
    from isaaclab_tasks.utils import parse_env_cfg

    import soarm_tasks  # noqa: F401
    from soarm_tasks import teleop
    from soarm_tasks.pick_place_env_cfg import GRIPPER_CLOSED, GRIPPER_OPEN

    cfg = parse_env_cfg("Soarm-PickPlace-Fastener-IK-Rel-v0", num_envs=1)
    arm, grip = cfg.actions.arm_action, cfg.actions.gripper_action
    assert arm.controller.use_relative_mode is False and arm.top_down  # the term integrates the deltas
    assert arm.body_name == C.EE_BODY_NAME and arm.scale == 1.0
    assert grip.open_command_expr == {C.GRIPPER_JOINT_NAME: GRIPPER_OPEN}
    assert grip.close_command_expr == {C.GRIPPER_JOINT_NAME: GRIPPER_CLOSED}
    devices = cfg.teleop_devices.devices
    assert set(devices) == {"keyboard", "spacemouse"}
    assert devices["keyboard"].pos_sensitivity == teleop.KEYBOARD_POS_STEP
    assert devices["keyboard"].gripper_term  # 7th action element: +1 open / -1 close
