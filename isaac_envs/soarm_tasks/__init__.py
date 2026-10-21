"""Gym-registered, manager-based Isaac Lab tasks for the SO-ARM101 fastener tutorials.

Importing this package registers (configs are only built when an env is made):

* ``Soarm-PickPlace-Fastener-JointAbs-v0``  absolute joint targets (the contract action), cameras
* ``Soarm-PickPlace-Fastener-IK-Abs-v0``    absolute EE pose + gripper target, cameras
* ``Soarm-PickPlace-Fastener-State-v0``     absolute joint targets, no cameras, flat state obs

How it gets imported
--------------------
The package lives in ``isaac_envs/``, which docker-compose mounts read-only at ``/isaac_envs``;
it is not pip-installed. Put that directory on ``sys.path`` and import it before ``gym.make``::

    sys.path.insert(0, "/isaac_envs")
    import soarm_tasks  # registers the tasks
    env = gym.make("Soarm-PickPlace-Fastener-IK-Abs-v0", cfg=parse_env_cfg(...))

For Isaac Lab's own scripts (``list_envs.py``, ``record_demos.py``, ...), which only import
``isaaclab_tasks``, run them through ``soarm_tasks/scripts/with_soarm_tasks.py``; it imports this
package and then runs the script unchanged. ``soarm_tasks/scripts/list_tasks.py`` lists the
registered Soarm tasks.
"""

try:
    import gymnasium as gym
except ImportError:  # plain-Python use (language, work_cell) outside the Isaac image
    gym = None

from .language import instruction_for  # noqa: F401

TASK_IDS = {
    "Soarm-PickPlace-Fastener-JointAbs-v0": "pick_place_env_cfg:FastenerPickPlaceJointAbsEnvCfg",
    "Soarm-PickPlace-Fastener-IK-Abs-v0": "pick_place_env_cfg:FastenerPickPlaceIKAbsEnvCfg",
    "Soarm-PickPlace-Fastener-State-v0": "pick_place_env_cfg:FastenerPickPlaceStateEnvCfg",
}

# [start:register]
for _task_id, _cfg in TASK_IDS.items() if gym is not None else ():
    if _task_id not in gym.registry:
        gym.register(
            id=_task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}.{_cfg}"},
            disable_env_checker=True,
        )
# [end:register]
