"""Manager-based fastener pick-and-place task for the SO-ARM101 (GitHub issue #32).

Modelled on Isaac Lab's ``isaaclab_tasks/contrib/stack/config/so101`` (cube stack): same PhysX
scene settings (``solve_articulation_contact_last``), the same full-pose IK action
(``SO101PoseIKActionCfg``) for the IK-Abs variant, and the same observation groups (``policy``,
``subtask_terms`` for Mimic). Robot, cameras, timing and keys come from ``soarm_common``.

Three variants are registered in ``soarm_tasks/__init__.py``:

=====================================  ==========================================  ========
task id                                action (dim)                                cameras
=====================================  ==========================================  ========
Soarm-PickPlace-Fastener-JointAbs-v0   absolute joint targets, contract order (6)  yes
Soarm-PickPlace-Fastener-IK-Abs-v0     EE pose in base frame pos+quat xyzw, gripper  yes
                                       joint target (7 + 1)
Soarm-PickPlace-Fastener-State-v0      absolute joint targets (6)                  no
=====================================  ==========================================  ========

The whole module is pure configuration: it builds without launching Kit (see the tests).
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import SO101PoseIKControllerCfg

from assets.fasteners import fastener_spawn_cfg
from soarm_common import contract as C
from soarm_common.cameras import camera_prim_path, make_camera_cfg
from soarm_common.robot import make_physx_cfg, make_robot_cfg

from . import mdp
from . import randomization as R
from .language import instruction_for
from .spawners import BinCfg, XformCfg
from .work_cell import (
    BIN,
    DEFAULT_FASTENER,
    FASTENER_SPAWN_X,
    FASTENER_SPAWN_Y,
    FASTENER_SPAWN_YAW,
    FASTENER_SPAWN_Z,
    TABLE_CENTER,
    TABLE_COLOR,
    TABLE_SIZE,
    BinSpec,
)

ROBOT = SceneEntityCfg("robot", body_names=[C.EE_BODY_NAME], joint_names=[C.GRIPPER_JOINT_NAME])
FASTENER = SceneEntityCfg("fastener")
GRIPPER_OPEN = 1.0  # rad; the scripted expert's open angle (the joint opens to 1.745)


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

# [start:scene_helpers]
def fastener_spawner(names: list[str] | tuple[str, ...]):
    """Spawner for one fastener per env: the USD itself for one name, else a
    ``MultiAssetSpawnerCfg`` mix (Part 2), one asset per env (needs ``replicate_physics=False``)."""
    if len(names) == 1:
        return fastener_spawn_cfg(names[0])
    return sim_utils.MultiAssetSpawnerCfg(assets_cfg=[fastener_spawn_cfg(n) for n in names])


def make_bin_cfg(spec: BinSpec) -> AssetBaseCfg:
    """Scene entity of one static bin. Part 2 adds one field per bin to a scene subclass."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Bins/{spec.name}",
        spawn=BinCfg(spec=spec),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(spec.center[0], spec.center[1], 0.0)),
    )
# [end:scene_helpers]


_WRIST_CAM_PATH = camera_prim_path(C.WRIST_CAMERA, "{ENV_REGEX_NS}/Robot", "{ENV_REGEX_NS}")
_OVERHEAD_CAM_PATH = camera_prim_path(C.OVERHEAD_CAMERA, "{ENV_REGEX_NS}/Robot", "{ENV_REGEX_NS}")


# [start:scene]
@configclass
class FastenerSceneCfg(InteractiveSceneCfg):
    """Robot on a table, one bin, one fastener and the contract's two cameras, per env.

    ``{ENV_REGEX_NS}`` expands to ``/World/envs/env_[^/]+``: a one-segment wildcard, so the leaf
    names below never match prims deeper in another env's tree.
    """

    robot: ArticulationCfg = make_robot_cfg("{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(), PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR, roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER),
    )

    bin = make_bin_cfg(BIN)

    fastener = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fastener",
        spawn=fastener_spawner([DEFAULT_FASTENER]),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(sum(FASTENER_SPAWN_X) / 2, sum(FASTENER_SPAWN_Y) / 2, FASTENER_SPAWN_Z), rot=mdp.LYING_QUAT
        ),
    )

    # The camera spawner needs the parent prim of a regex path to exist: an Xform per env.
    cameras_root = AssetBaseCfg(prim_path=_OVERHEAD_CAM_PATH.rsplit("/", 1)[0], spawn=XformCfg())
    wrist_cam: CameraCfg | None = make_camera_cfg(C.WRIST_CAMERA, _WRIST_CAM_PATH)
    overhead_cam: CameraCfg | None = make_camera_cfg(C.OVERHEAD_CAMERA, _OVERHEAD_CAM_PATH)

    light = AssetBaseCfg(  # same dome light as soarm_common.scene
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
# [end:scene]


# ═══════════════════════════════════════════════════════════════════════════
# Actions
# ═══════════════════════════════════════════════════════════════════════════

def gripper_joint_action() -> base_mdp.JointPositionActionCfg:
    """Gripper target, absolute joint angle in rad (contract: ~0 closed, 1.745 open)."""
    return base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[C.GRIPPER_JOINT_NAME], scale=1.0, use_default_offset=False
    )


@configclass
class JointAbsActionsCfg:
    """The contract action: absolute joint-position targets in JOINT_NAMES order (6)."""

    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(C.ARM_JOINT_NAMES), scale=1.0, use_default_offset=False,
        preserve_order=True,
    )
    gripper_action = gripper_joint_action()


# [start:ik_action]
@configclass
class IKAbsActionsCfg:
    """Absolute EE pose of gripper_frame_link in the robot base frame [pos(3), quat xyzw(4)],
    then the gripper joint target (rad): 8 dims. Field order = action order."""

    arm_action = SO101PoseIKActionCfg(
        asset_name="robot",
        joint_names=list(C.ARM_JOINT_NAMES),
        body_name=C.EE_BODY_NAME,
        scale=1.0,
        controller=SO101PoseIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="adaptive_dls",
            ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
            # Isaac Lab's SO-101 task uses 0.5. The fastener grasp needs the jaws square to the
            # bolt axis, so orientation is tracked at full weight; it is still served by the
            # wrist only, and position by all five joints.
            orientation_weight=1.0,
            orientation_joint_names=("wrist_flex", "wrist_roll"),
            joint_limit_avoidance_gain=0.5,
            joint_limit_avoidance_margin=0.3,
        ),
    )
    gripper_action = gripper_joint_action()
# [end:ik_action]


# ═══════════════════════════════════════════════════════════════════════════
# Observations
# ═══════════════════════════════════════════════════════════════════════════

_BIN_TOP = BIN.wall_height + 0.01
_IN_BIN = {"center": BIN.center, "inner_size": BIN.inner_size, "top_z": _BIN_TOP}


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Robot and object state. ``joint_pos`` is the contract state (absolute, rad)."""

        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_vel = ObsTerm(func=base_mdp.joint_vel)
        ee_pose = ObsTerm(func=mdp.ee_pose_b, params={"robot_cfg": ROBOT})
        gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"robot_cfg": ROBOT})
        fastener_pose = ObsTerm(func=mdp.object_pose_b, params={"object_cfg": FASTENER})
        bin_pos = ObsTerm(func=mdp.bin_pos_b, params={"center": BIN.center, "floor_z": BIN.floor_top})
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ImagesCfg(ObsGroup):
        """uint8 RGB (N, 224, 224, 3). Term names = contract CameraSpec.obs_key."""

        wrist_rgb = ObsTerm(func=base_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("wrist_cam"),
                                                             "normalize": False})
        overhead_rgb = ObsTerm(func=base_mdp.image_rgb, params={"sensor_cfg": SceneEntityCfg("overhead_cam"),
                                                                "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # [start:subtasks]
    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask signals (bool per env), for Mimic's subtask segmentation and for eval."""

        grasped = ObsTerm(func=mdp.fastener_grasped, params={"robot_cfg": ROBOT, "object_cfg": FASTENER})
        lifted = ObsTerm(func=mdp.fastener_lifted, params={"object_cfg": FASTENER, "min_height": 0.03})
        in_bin = ObsTerm(func=mdp.fastener_in_bin, params={**_IN_BIN, "object_cfg": FASTENER})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    # [end:subtasks]

    policy: PolicyCfg = PolicyCfg()
    images: ImagesCfg | None = ImagesCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ═══════════════════════════════════════════════════════════════════════════
# Events and terminations
# ═══════════════════════════════════════════════════════════════════════════

# [start:events]
@configclass
class FastenerEventCfg:
    """Asset fixes at start-up, then domain randomization (ranges: ``soarm_tasks.randomization``).

    Order within a mode is field order. Later DR terms (camera-mount jitter, actuator bias) go at
    the end of this class.
    """

    # -- start-up (once) ---------------------------------------------------------------------
    jaws_convex = EventTerm(func=mdp.jaws_convex_decomposition, mode="prestartup")
    fastener_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": FASTENER, "static_friction_range": R.FASTENER_STATIC_FRICTION,
                "dynamic_friction_range": R.FASTENER_DYNAMIC_FRICTION, "restitution_range": (0.0, 0.0),
                "num_buckets": R.FRICTION_BUCKETS, "make_consistent": True},
    )
    fastener_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": FASTENER, "mass_distribution_params": R.FASTENER_MASS_SCALE, "operation": "scale"},
    )
    # -- every reset ---------------------------------------------------------------------------
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(C.ARM_JOINT_NAMES)),
                "position_range": R.ROBOT_JOINT_OFFSET, "velocity_range": (0.0, 0.0)},
    )
    fastener_pose = EventTerm(
        func=mdp.reset_fastener_lying,
        mode="reset",
        params={"asset_cfg": FASTENER, "x_range": FASTENER_SPAWN_X, "y_range": FASTENER_SPAWN_Y,
                "yaw_range": FASTENER_SPAWN_YAW, "spin_range": R.FASTENER_SPIN, "z": FASTENER_SPAWN_Z},
    )
    lighting = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",
        params={"intensity_range": R.LIGHT_INTENSITY, "color_variation": R.LIGHT_COLOR_VARIATION},
    )
# [end:events]


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    fastener_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": FASTENER}
    )
    success = DoneTerm(func=mdp.fastener_placed, params={**_IN_BIN, "object_cfg": FASTENER, "robot_cfg": ROBOT})


# ═══════════════════════════════════════════════════════════════════════════
# Environments
# ═══════════════════════════════════════════════════════════════════════════

@configclass
class FastenerPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Base cfg: cameras on, actions set by the variants below."""

    scene: FastenerSceneCfg = FastenerSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions = MISSING
    events: FastenerEventCfg = FastenerEventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    curriculum = None

    # Fasteners spawned (one per env; several names = a mix) and the prompt for Part 1. Per-env
    # prompts of a mix come from soarm_tasks.mdp.language_instructions(env).
    fastener_names: tuple[str, ...] = (DEFAULT_FASTENER,)
    language_instruction: str = instruction_for(DEFAULT_FASTENER)

    # Used by Isaac Lab Mimic / the stack-style subtask terms.
    gripper_joint_names: list[str] = [C.GRIPPER_JOINT_NAME]
    gripper_open_val: float = GRIPPER_OPEN
    gripper_threshold: float = 0.2

    def __post_init__(self):
        # contract timing: 120 Hz physics, 30 Hz control, render every 2nd physics step
        self.sim.dt = C.PHYSICS_DT
        self.decimation = C.DECIMATION
        self.sim.render_interval = C.RENDER_INTERVAL
        self.sim.physics = make_physx_cfg()
        self.episode_length_s = 20.0
        self.scene.fastener.spawn = fastener_spawner(self.fastener_names)
        if len(self.fastener_names) == 1:
            self.language_instruction = instruction_for(self.fastener_names[0])
        self.num_rerenders_on_reset = 1 if cameras_enabled(self) else 0
        # viewport camera when a Kit window is open (--viz kit)
        self.sim.default_visualizer_cfg = KitVisualizerCfg(eye=(0.75, -0.55, 0.45), lookat=(0.2, 0.0, 0.05))

    def disable_cameras(self) -> None:
        """Drop both cameras and the image observations (state-only, no RTX rendering)."""
        self.scene.wrist_cam = None
        self.scene.overhead_cam = None
        self.scene.cameras_root = None
        self.observations.images = None
        self.num_rerenders_on_reset = 0


def cameras_enabled(cfg: FastenerPickPlaceEnvCfg) -> bool:
    """Whether the cfg still has a camera (see :meth:`FastenerPickPlaceEnvCfg.disable_cameras`)."""
    return cfg.scene.wrist_cam is not None or cfg.scene.overhead_cam is not None


@configclass
class FastenerPickPlaceJointAbsEnvCfg(FastenerPickPlaceEnvCfg):
    actions: JointAbsActionsCfg = JointAbsActionsCfg()


@configclass
class FastenerPickPlaceIKAbsEnvCfg(FastenerPickPlaceEnvCfg):
    actions: IKAbsActionsCfg = IKAbsActionsCfg()


@configclass
class FastenerPickPlaceStateEnvCfg(FastenerPickPlaceJointAbsEnvCfg):
    """Joint-abs actions, no cameras, policy observations concatenated into one vector (RL)."""

    def __post_init__(self):
        super().__post_init__()
        self.disable_cameras()
        self.observations.policy.concatenate_terms = True
