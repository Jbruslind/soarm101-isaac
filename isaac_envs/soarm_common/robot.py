"""Isaac Lab robot and physics configs for the SO-ARM101, built from :mod:`soarm_common.contract`.

Needs Isaac Lab (import inside a launched Isaac Sim app).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

from . import contract as C


def make_robot_cfg(prim_path: str = C.ENV_ROBOT_PRIM_PATH) -> ArticulationCfg:
    """SO-ARM101 articulation: the importer-3.0 USD, all joints at 0, implicit PD drives.

    The spawned prim is the articulation root (convert_urdf.py puts it there).
    """
    arm, grip = C.SIM_ARM_DRIVE, C.SIM_GRIPPER_DRIVE
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=C.ROBOT_USD_PATH,
            rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={name: 0.0 for name in C.JOINT_NAMES}),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(C.ARM_JOINT_NAMES),
                joint_velocity_limit=arm["velocity_limit"],
                joint_effort_limit=arm["effort_limit"],
                stiffness=arm["stiffness"],
                damping=arm["damping"],
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[C.GRIPPER_JOINT_NAME],
                joint_velocity_limit=grip["velocity_limit"],
                joint_effort_limit=grip["effort_limit"],
                stiffness=grip["stiffness"],
                damping=grip["damping"],
            ),
        },
    )


def make_physx_cfg() -> PhysxCfg:
    """PhysX scene settings from Isaac Lab's SO-101 cube-stack task (contrib/stack/stack_env_cfg.py).

    solve_articulation_contact_last resolves contacts after the joint drives, so object contacts
    stall the position-driven jaw instead of the jaw tunnelling into a grasped object (Isaac Lab
    3.0 CHANGELOG; GitHub issue #28). It doesn't change free-space motion.
    """
    return PhysxCfg(
        solve_articulation_contact_last=True,
        bounce_threshold_velocity=0.01,
        friction_correlation_distance=0.00625,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**21,
    )


def check_joint_limits(robot, tol: float = 1e-3) -> list[str]:
    """Compare the spawned articulation's joint names/limits with the contract.

    Returns a list of mismatch messages (empty if they agree) and prints them.
    """
    problems = []
    names = list(robot.joint_names)
    if names[: C.NUM_JOINTS] != list(C.JOINT_NAMES):
        problems.append(f"joint order {names} != contract {list(C.JOINT_NAMES)}")
    limits = robot.data.joint_pos_limits.torch[0].cpu().tolist()
    for name, (lo, hi) in C.JOINT_LIMITS.items():
        if name not in names:
            continue
        ulo, uhi = limits[names.index(name)]
        if abs(ulo - lo) > tol or abs(uhi - hi) > tol:
            problems.append(f"{name} limits USD ({ulo:.4f}, {uhi:.4f}) != contract ({lo:.4f}, {hi:.4f})")
    for p in problems:
        print(f"[WARN] soarm_common contract: {p}", flush=True)
    return problems


JAW_LINK_NAMES = ("gripper_link", "moving_jaw_so101_v1_link")


def use_convex_decomposition_for_jaws(robot_prim_path: str, stage=None) -> int:
    """Collide the gripper jaws of the robot at ``robot_prim_path`` as convex decompositions
    instead of convex hulls. Call it after spawning and before the simulation starts.

    The URDF import gives every mesh a single convex hull; the hulls of the curved jaws fill the
    space between them, so a closing gripper squeezes the object out or traps it in the hull
    instead of holding it. The collision meshes are instanced, so their instanceable ancestors
    are de-instanced (on this stage only) first. Links are found by name, so this doesn't depend
    on the USD layout. Returns the number of collision meshes changed.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = stage or omni.usd.get_context().get_stage()
    meshes, instances = [], set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path), Usd.TraverseInstanceProxies()):
        if prim.GetName() not in JAW_LINK_NAMES:
            continue
        for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if mesh.HasAPI(UsdPhysics.MeshCollisionAPI):
                meshes.append(mesh.GetPath())
                parent = mesh
                while parent.IsInstanceProxy() or parent.IsInstance():
                    if parent.IsInstance():
                        instances.add(parent.GetPath())
                    parent = parent.GetParent()
    for path in instances:
        stage.GetPrimAtPath(path).SetInstanceable(False)
    for path in meshes:
        UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path)).GetApproximationAttr().Set("convexDecomposition")
    return len(meshes)
