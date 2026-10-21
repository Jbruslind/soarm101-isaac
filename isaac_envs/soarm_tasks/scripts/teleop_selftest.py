"""Headless check of the keyboard teleop path: record_demos.py with synthetic key presses.

Runs Isaac Lab's ``scripts/tools/record_demos.py`` unchanged on
``Soarm-PickPlace-Fastener-IK-Rel-v0 --teleop_device keyboard``, with one substitution: the
keyboard device is :class:`ScriptedKeyboard`, an ``Se3Keyboard`` (the task's own cfg and key
bindings) whose key presses come from the scripted expert instead of a person. Each step it
decides which keys a person would hold (W/S, A/D, Q/E, C/V) and when to tap K, and feeds those
press/release events to ``Se3Keyboard._on_keyboard_event``, the callback Kit calls for real keys.
Everything after that (device output, IK-Rel action, recorder, success check, HDF5) is the path
a human session takes. Usage, in the isaac-sim container::

    /isaac-sim/python.sh /isaac_envs/soarm_tasks/scripts/teleop_selftest.py \\
        --dataset_file /data/demos/fastener_m6/teleop_selftest.hdf5 --num_demos 1

Arguments are record_demos.py's (``--task`` and ``--teleop_device`` are filled in). Check the
file afterwards with ``soarm_tasks/scripts/demo_tools.py validate``.
"""

from __future__ import annotations

import os
import runpy
import sys
from types import SimpleNamespace

ISAAC_ENVS = os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs")
RECORD_DEMOS = "/workspace/isaaclab/scripts/tools/record_demos.py"
TASK = "Soarm-PickPlace-Fastener-IK-Rel-v0"

_ENV = []  # the env record_demos.py creates (captured from gym.make)
MAX_EPISODE_STEPS = 900  # record_demos.py has no timeout; give up on an attempt after this
READY_TOL = 0.01  # m, EE distance from the ready pose at which steering starts
MAX_WAIT_STEPS = 150  # start anyway after this many steps (the IK may settle just outside READY_TOL)


def make_scripted_keyboard(cfg):
    """An Se3Keyboard driven by the scripted expert (built after Kit has started)."""
    import carb
    import numpy as np
    import torch
    from isaaclab.devices import Se3Keyboard
    from isaaclab.utils.math import quat_apply

    from soarm_tasks.expert import ExpertParams, FastenerPickExpert
    from soarm_tasks.work_cell import BIN

    press, release = carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE

    class ScriptedKeyboard(Se3Keyboard):
        # [start:scripted_keyboard]
        def __init__(self, cfg):
            # Se3Keyboard.__init__ subscribes to the Kit window's keyboard, which a headless run
            # does not have; set up the same state and bindings without it.
            self.pos_sensitivity, self.rot_sensitivity = cfg.pos_sensitivity, cfg.rot_sensitivity
            self.gripper_term, self._sim_device = cfg.gripper_term, cfg.sim_device
            self._create_key_bindings()
            self._close_gripper = False
            self._delta_pos, self._delta_rot = np.zeros(3), np.zeros(3)
            self._additional_callbacks = {}
            self._held: set[str] = set()
            self._expert = None
            self.key_presses = 0
            self._steps = 0
            self._started = False
            self._wait_steps = 0

        def __del__(self):
            pass

        def _key(self, kind, name):
            self._on_keyboard_event(SimpleNamespace(type=kind, input=SimpleNamespace(name=name)))
            self.key_presses += kind == press

        def reset(self):
            super().reset()
            self._held.clear()
            self._steps = 0
            self._started = False
            self._wait_steps = 0
            if self._expert is not None:
                self._expert.reset()

        def advance(self) -> torch.Tensor:
            self._hold(self._keys_to_hold())
            return super().advance()

        def _hold(self, keys: set[str]):
            for k in self._held - keys:
                self._key(release, k)
            for k in keys - self._held:
                self._key(press, k)
            self._held = keys
        # [end:scripted_keyboard]

        def _keys_to_hold(self) -> set[str]:
            env = _ENV[0]
            term = env.action_manager.get_term("arm_action")
            if term.needs_sync[0]:  # first step after a reset: the EE target is not set yet
                return set()
            if self._expert is None or not self._started:
                # a person waits until the arm has (about) reached the ready pose before steering
                ee = env.obs_buf["policy"]["ee_pose"][0, :3]
                dist = float((ee - term.target_pose[0, :3]).norm())
                self._wait_steps += 1
                if self._wait_steps % 50 == 0:
                    print(f"[teleop-selftest] waiting for the ready pose: EE {dist * 1000:.1f} mm from target "
                          f"after {self._wait_steps} steps", flush=True)
                if dist > READY_TOL and self._wait_steps < MAX_WAIT_STEPS:
                    return set()
                print(f"[teleop-selftest] start steering after {self._wait_steps} steps "
                      f"(EE {dist * 1000:.1f} mm from the ready pose)", flush=True)
                self._started = True
            if self._expert is None:
                # a little slower than the expert's default (6 mm/step) so 3 mm key steps keep up
                params = ExpertParams(max_speed=0.004, carry_speed=0.004)
                self._expert = FastenerPickExpert(env.num_envs, env.device, BIN.center, params)
            pol = env.obs_buf["policy"]
            cmd = self._expert.act(pol["ee_pose"][:, :3], pol["fastener_pose"])[0]
            target = term.target_pose[0]
            self._steps += 1
            if self._steps % 50 == 0:
                print(f"[teleop-selftest] step {self._steps} phase={self._expert.phase_names()[0]} "
                      f"ee={[round(v, 3) for v in pol['ee_pose'][0, :3].tolist()]} "
                      f"target={[round(v, 3) for v in target[:3].tolist()]} "
                      f"fastener={[round(v, 3) for v in pol['fastener_pose'][0, :3].tolist()]} "
                      f"keys={sorted(self._held)} presses={self.key_presses}", flush=True)
            if self._steps > MAX_EPISODE_STEPS and "R" in self._additional_callbacks:
                # what a person does when an attempt fails: press R (record_demos.py resets)
                print(f"[teleop-selftest] no success after {MAX_EPISODE_STEPS} steps: pressing R", flush=True)
                self._key(press, "R")
                self._key(release, "R")
                return set()
            keys = set()
            # move the target towards the expert's command, one key per axis, half-step deadband
            err = (cmd[:3] - target[:3]).tolist()
            for e, (plus, minus) in zip(err, (("W", "S"), ("A", "D"), ("Q", "E"))):
                if abs(e) > self.pos_sensitivity / 2:
                    keys.add(plus if e > 0 else minus)
            ex = torch.tensor([[1.0, 0.0, 0.0]], device=cmd.device)
            yaw_cmd, yaw_tgt = (torch.atan2(v[0, 1], v[0, 0]) for v in
                                (quat_apply(cmd[None, 3:7], ex), quat_apply(target[None, 3:7], ex)))
            dyaw = float(torch.remainder(yaw_cmd - yaw_tgt + torch.pi, 2 * torch.pi) - torch.pi)
            if abs(dyaw) > self.rot_sensitivity / 2:
                keys.add("C" if dyaw > 0 else "V")
            # K toggles the gripper: tap it when the expert wants the other state
            if (float(cmd[7]) < 0.5) != self._close_gripper:
                self._key(press, "K")
                self._key(release, "K")
            return keys

    return ScriptedKeyboard(cfg)


def main():
    sys.path.insert(0, ISAAC_ENVS)
    import gymnasium as gym

    import isaaclab.devices.teleop_device_factory as factory
    import soarm_tasks  # noqa: F401  (registers the tasks)

    make = gym.make

    def capture_make(*a, **kw):
        env = make(*a, **kw)
        _ENV.append(env.unwrapped)
        return env

    def create_device(name, devices_cfg, callbacks=None):
        assert name == "keyboard", name
        device = make_scripted_keyboard(devices_cfg[name])
        for key, cb in (callbacks or {}).items():
            device.add_callback(key, cb)
        print(f"[teleop-selftest] scripted keyboard: pos {device.pos_sensitivity} m/step, "
              f"rot {device.rot_sensitivity} rad/step", flush=True)
        return device

    gym.make = capture_make
    factory.create_teleop_device = create_device  # record_demos.py imports it from here
    sys.argv = [RECORD_DEMOS, "--task", TASK, "--teleop_device", "keyboard", *sys.argv[1:]]
    sys.path.insert(0, os.path.dirname(RECORD_DEMOS))
    runpy.run_path(RECORD_DEMOS, run_name="__main__")


if __name__ == "__main__":
    main()
