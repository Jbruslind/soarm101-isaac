"""Inspect, validate and re-target fastener demo HDF5 files (no Kit, no GPU).

    python demo_tools.py info     FILE                 env, demos, steps, successes, size, attrs of demo_0
    python demo_tools.py validate FILE [--no-images]   schema check (soarm_tasks.demo_schema), exit 1 on problems
    python demo_tools.py swap     SRC DST --actions joint_pos_targets --task Soarm-PickPlace-Fastener-JointAbs-v0
                                                       copy with another recorded action as ``actions``,
                                                       for replay_demos.py on that task

In the isaac-sim container run it with ``/isaac-sim/python.sh`` (it has h5py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from soarm_tasks import demo_schema as S  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info").add_argument("file")
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.add_argument("--no-images", action="store_true")
    v.add_argument("--no-ee", action="store_true", help="joint-space task: no ee_pose_actions")
    s = sub.add_parser("swap")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--actions", default=S.JOINT_POS_TARGETS, choices=(S.JOINT_POS_TARGETS, S.EE_POSE_ACTIONS))
    s.add_argument("--task", default="Soarm-PickPlace-Fastener-JointAbs-v0")
    a = p.parse_args(argv)

    if a.cmd == "info":
        info = S.file_info(a.file)
        names = S.episode_names(a.file)
        if names:
            arrays, attrs = S.load_episode(a.file, names[0], keys=(S.ACTIONS, S.EE_POSE_ACTIONS, S.JOINT_POS_TARGETS))
            info["demo_0"] = {"attrs": attrs, **{k: list(v.shape) for k, v in arrays.items()}}
        print(json.dumps(info, indent=1, default=str))
        return 0
    if a.cmd == "validate":
        problems = S.validate(a.file, require_images=not a.no_images, require_ee=not a.no_ee)
        for line in problems:
            print("PROBLEM", line)
        info = S.file_info(a.file)
        print(f"{'OK' if not problems else 'INVALID'}: {a.file} env={info['env_name']} demos={info['demos']} "
              f"steps={info['steps']} successes={info['successes']} {info['bytes'] / 2**20:.1f} MB")
        return 1 if problems else 0
    n = S.copy_with_actions(a.src, a.dst, a.actions, a.task)
    print(f"wrote {a.dst}: {n} demos, actions = {a.actions}, env = {a.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
