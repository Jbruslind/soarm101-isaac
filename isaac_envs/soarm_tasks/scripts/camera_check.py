"""How big is the fastener in the 224 x 224 policy images? (GitHub issue #32)

Runs the scripted expert on the IK-Abs task and, at the pick pose (the step the jaws start to
close), projects the fastener's axis end points (head top, shank tip) and its head width into
the wrist and overhead cameras with their intrinsics and poses. Prints the lengths in pixels per
episode and saves a contact sheet (overhead | wrist, one row per episode) as a PNG.

    docker compose --profile collect run --rm isaac-sim /isaac-sim/python.sh \
        /isaac_envs/soarm_tasks/scripts/camera_check.py --episodes 4 --out /data/camera_check.png
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="Soarm-PickPlace-Fastener-IK-Abs-v0")
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="/data/camera_check.png")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.require_kit = True
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))


def project(spec, parent_pos_w, parent_quat_w, points_w):
    """Pixel (u, v) and depth of world points (K, 3) in the camera ``spec`` of the contract,
    mounted on a parent frame (the EE link or the env origin). Uses the contract's pinhole
    intrinsics and its "world" camera convention (looks along +x, +y left, +z up), so it checks
    the contract itself, not a renderer's view of it."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    dev = points_w.device
    cam_q = quat_mul(parent_quat_w, torch.tensor(spec.rot, device=dev))
    cam_p = parent_pos_w + quat_apply(parent_quat_w, torch.tensor(spec.pos, device=dev))
    p_c = quat_apply(quat_inv(cam_q).expand(len(points_w), -1), points_w - cam_p)
    f_px = spec.width * spec.focal_length / spec.horizontal_aperture
    depth = p_c[:, 0]
    u = spec.width / 2 - f_px * p_c[:, 1] / depth
    v = spec.height / 2 - f_px * p_c[:, 2] / depth
    return torch.stack([u, v], -1), depth


def main():
    with launch_simulation(None, args):
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.utils.math import quat_apply
        from isaaclab_tasks.utils import parse_env_cfg
        from PIL import Image, ImageDraw

        import soarm_tasks  # noqa: F401
        from assets.fasteners import spec_from_name
        from soarm_common import contract as C
        from soarm_tasks.expert import PHASE_NAMES, FastenerPickExpert
        from soarm_tasks.work_cell import BIN

        cfg = parse_env_cfg(args.task, num_envs=1)
        cfg.seed = args.seed
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        spec = spec_from_name(cfg.fastener_names[0])
        head_top, tip = spec.head_height * 1e-3, -spec.length * 1e-3
        robot = u.scene["robot"]
        ee_id = robot.find_bodies(C.EE_BODY_NAME)[0][0]
        obs, _ = env.reset(seed=args.seed)
        expert = FastenerPickExpert(1, u.device, BIN.center)
        rows, stats = [], {"overhead": [], "wrist": []}
        close = PHASE_NAMES.index("close")
        for ep in range(args.episodes):
            expert.reset()
            for _ in range(400):
                pol = obs["policy"]
                action = expert.act(pol["ee_pose"][:, :3], pol["fastener_pose"])
                obs, _, term, trunc, _ = env.step(action)
                if int(expert.phase[0]) == close:
                    break
            # the images in obs were rendered after this step, at the pick pose
            f = u.scene["fastener"].data.root_link_pose_w.torch[0]
            axis = quat_apply(f[3:7], torch.tensor([0.0, 0.0, 1.0], device=u.device))
            across = torch.linalg.cross(axis, torch.tensor([0.0, 0.0, 1.0], device=u.device))
            across = across / across.norm()
            pts = torch.stack([f[:3] + head_top * axis, f[:3] + tip * axis,
                               f[:3] + 0.5 * spec.head_width * 1e-3 * across,
                               f[:3] - 0.5 * spec.head_width * 1e-3 * across])
            tiles = []
            ee = robot.data.body_link_pose_w.torch[0, ee_id]
            origin = torch.cat([u.scene.env_origins[0], torch.tensor([0.0, 0.0, 0.0, 1.0], device=u.device)])
            for name, cspec, parent in (("overhead", C.OVERHEAD_CAMERA, origin), ("wrist", C.WRIST_CAMERA, ee)):
                uv, depth = project(cspec, parent[:3], parent[3:7], pts)
                length = float((uv[0] - uv[1]).norm())
                width = float((uv[2] - uv[3]).norm())
                inside = bool(((uv >= 0) & (uv < C.IMAGE_SIZE)).all() and (depth > 0).all())
                stats[name].append((length, width, inside, float(depth[:2].mean())))
                print(f"[cam] ep {ep}: {name:8s} bolt length {length:5.1f} px, head width {width:4.1f} px, "
                      f"depth {float(depth[:2].mean()):.3f} m, in view {inside}", flush=True)
                img = Image.fromarray(obs["images"][cspec.obs_key][0].cpu().numpy().astype(np.uint8))
                draw = ImageDraw.Draw(img)
                (u0, v0), (u1, v1) = uv[0].tolist(), uv[1].tolist()
                draw.line([(u0, v0), (u1, v1)], fill=(255, 0, 0), width=1)
                draw.text((3, 3), f"{name} {length:.0f}px", fill=(255, 255, 0))
                tiles.append(img)
            row = Image.new("RGB", (2 * C.IMAGE_SIZE + 4, C.IMAGE_SIZE), (255, 255, 255))
            row.paste(tiles[0], (0, 0))
            row.paste(tiles[1], (C.IMAGE_SIZE + 4, 0))
            rows.append(row)
            obs, _ = env.reset()
        sheet = Image.new("RGB", (rows[0].width, len(rows) * (C.IMAGE_SIZE + 4)), (255, 255, 255))
        for i, row in enumerate(rows):
            sheet.paste(row, (0, i * (C.IMAGE_SIZE + 4)))
        sheet.save(args.out)
        for name, s in stats.items():
            lengths = [x[0] for x in s]
            widths = [x[1] for x in s]
            print(f"[cam] {name:8s} bolt length px: min {min(lengths):.1f} mean {sum(lengths) / len(lengths):.1f} "
                  f"max {max(lengths):.1f}; head width px mean {sum(widths) / len(widths):.1f}; "
                  f"all in view: {all(x[2] for x in s)}", flush=True)
        print(f"[cam] contact sheet: {args.out}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
