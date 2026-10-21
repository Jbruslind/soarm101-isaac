"""Drop test for the procedural fasteners (issue #30).

Run inside the Isaac Sim container (from docker/):
    docker compose --profile collect run --rm isaac-sim         /isaac-sim/python.sh /isaac_envs/assets/fastener_drop_test.py --render
Results (JSON + PNGs) go to --out (default /data/fastener_drop, i.e. data/fastener_drop on the host).

Each part is dropped in three orientations onto a kinematic ground box (top at z = 0) at dt = 1/120
(lowest collision point 20 mm above the ground). After a settle phase (default 5 s) we simulate a
window (default 5 s) and report:
  drift     |p(end) - p(settle)| and rotation change over the window   (< 1 mm required)
  vmax      max linear speed over the window, and over its last second (jitter)
  pen       deepest collision support point below the ground over the window
Optionally renders the settled M6x20 hex bolt.
"""
import argparse
import json
import math
import sys

sys.path.insert(0, "/isaac_envs")
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--parts", nargs="+", default=[
    "hex_bolt_M3x10", "hex_bolt_M6x20", "socket_cap_M6x20", "hex_bolt_M8x40", "socket_cap_M3x10",
    "hex_bolt_M5x16", "socket_cap_M8x40", "hex_nut_M6", "hex_nut_M3", "washer_M6", "washer_M3", "washer_M4",
    "washer_M8"])
parser.add_argument("--variants", nargs="+", default=["/isaac_envs/assets/usd/fasteners:16"],
                    help="usd_dir:round_segments pairs (as generated with --round-segments), dropped side by side")
parser.add_argument("--poses", nargs="+", default=["flat", "upright", "tilted", "flipped"])
parser.add_argument("--settle", type=float, default=5.0)
parser.add_argument("--window", type=float, default=5.0)
parser.add_argument("--dt", type=float, default=1 / 120.0)
parser.add_argument("--render", action="store_true")
parser.add_argument("--trace", action="store_true")
parser.add_argument("--out", default="/data/fastener_drop")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.render

import isaaclab.sim as sim_utils  # noqa: E402

sim_cfg = sim_utils.SimulationCfg(dt=args.dt, render_interval=1)

with launch_simulation(sim_cfg, args):
    import os

    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import quat_apply
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

    from assets import fasteners as F

    os.makedirs(args.out, exist_ok=True)
    sim = SimulationContext(sim_cfg)

    ground = sim_utils.CuboidCfg(
        size=(4.0, 4.0, 0.02),
        rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=PhysxCollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.2, 0.24), roughness=0.7),
    )
    ground.func("/World/ground", ground, translation=(0.0, 0.0, -0.01))
    dome = sim_utils.DomeLightCfg(intensity=400.0, color=(0.9, 0.92, 1.0))
    dome.func("/World/dome", dome)
    sun = sim_utils.DistantLightCfg(intensity=1200.0, angle=2.0)
    sun.func("/World/sun", sun, orientation=(0.3, 0.2, 0.0, 0.933))

    def quat_axis_angle(axis, ang):  # xyzw
        n = math.sqrt(sum(a * a for a in axis))
        s = math.sin(ang / 2) / n
        return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2))

    POSES = {
        "flat": quat_axis_angle((1, 0, 0), math.pi / 2),   # screw axis horizontal / nut, washer on edge
        "upright": (0.0, 0.0, 0.0, 1.0),                   # head up (screw on its tip) / nut, washer flat
        "tilted": quat_axis_angle((0.3, 1.0, 0.2), 0.7),
        "flipped": quat_axis_angle((1, 0, 0), math.pi),
    }
    POSES = {k: v for k, v in POSES.items() if k in args.poses}

    sim_utils.create_prim("/World/drop", "Xform")
    objs = {}
    for v, variant in enumerate(args.variants):
        usd_dir, segs = variant.rsplit(":", 1)
        segs = int(segs)
        vname = f"seg{segs}"
        for i, name in enumerate(args.parts):
            spec = F.spec_from_name(name)
            pts = torch.tensor(F.collision_support_points(spec, round_segments=segs))
            for j, (pose, q) in enumerate(POSES.items()):
                rot = torch.tensor(q).expand(len(pts), 4)
                zmin = float(quat_apply(rot, pts)[:, 2].min())
                pos = (0.3 * v + 0.08 * j, 0.06 * i - 0.3, 0.02 - zmin)
                cfg = RigidObjectCfg(
                    prim_path=f"/World/drop/{vname}_{name}_{pose}",
                    spawn=sim_utils.UsdFileCfg(usd_path=F.fastener_usd_path(name, usd_dir)),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=q),
                )
                objs[f"{vname}/{name}/{pose}"] = (spec, segs, RigidObject(cfg))

    if args.render:
        from isaaclab.sensors import Camera, CameraCfg

        cam = Camera(CameraCfg(
            prim_path="/World/cam_closeup", update_period=0.0, height=600, width=800, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.005, 20.0)),
        ))
        cam_all = Camera(CameraCfg(
            prim_path="/World/cam_overview", update_period=0.0, height=720, width=960, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
        ))

    sim.reset()
    dt = sim.get_physics_dt()
    masses = {}
    for key, (spec, _, obj) in objs.items():
        obj.update(dt)
        m = obj.root_view.get_masses()
        masses[key] = float(torch.as_tensor(m.torch if hasattr(m, "torch") else m).flatten()[0])

    def step(n):
        for _ in range(n):
            sim.step()
            for _, _, obj in objs.values():
                obj.update(dt)

    def support_min_z(spec, segs, obj):
        pts = torch.tensor(F.collision_support_points(spec, round_segments=segs), device=obj.device,
                           dtype=torch.float32)
        p = obj.data.root_pos_w.torch[0]
        q = obj.data.root_quat_w.torch[0]
        return float((quat_apply(q.expand(len(pts), 4), pts) + p)[:, 2].min())

    n_settle = int(round(args.settle / dt))
    n_win = int(round(args.window / dt))
    n_last = int(round(1.0 / dt))
    settle_t = {k: 0.0 for k in objs}
    for i in range(n_settle):
        step(1)
        if args.trace and (i % 3 == 0 and i < 60 or i == n_settle - 1):
            for k, (_, _, obj) in objs.items():
                print("TRACE", round((i + 1) * dt, 3), k, [round(float(x) * 1e3, 3) for x in obj.data.root_pos_w.torch[0]],
                      [round(float(x), 4) for x in obj.data.root_quat_w.torch[0]],
                      round(float(obj.data.root_lin_vel_w.torch[0].norm()) * 1e3, 1), flush=True)
        for k, (_, _, obj) in objs.items():
            if float(obj.data.root_lin_vel_w.torch[0].norm()) > 0.005:
                settle_t[k] = (i + 1) * dt
    p0 = {k: obj.data.root_pos_w.torch[0].double().clone() for k, (_, _, obj) in objs.items()}
    q0 = {k: obj.data.root_quat_w.torch[0].double().clone() for k, (_, _, obj) in objs.items()}
    vmax = {k: 0.0 for k in objs}
    vlast = {k: 0.0 for k in objs}
    wlast = {k: 0.0 for k in objs}
    pen = {k: -1.0 for k in objs}
    dev = {k: 0.0 for k in objs}  # max |p(t) - p(settle)| over the window (jitter amplitude)
    for i in range(n_win):
        step(1)
        for k, (spec, segs, obj) in objs.items():
            v = float(obj.data.root_lin_vel_w.torch[0].norm())
            dev[k] = max(dev[k], float((obj.data.root_pos_w.torch[0].double() - p0[k]).norm()))
            vmax[k] = max(vmax[k], v)
            if i >= n_win - n_last:
                vlast[k] = max(vlast[k], v)
                wlast[k] = max(wlast[k], float(obj.data.root_ang_vel_w.torch[0].norm()))
            if i % 6 == 0:
                pen[k] = max(pen[k], -support_min_z(spec, segs, obj))

    rows = []
    summary = {}
    for k, (spec, segs, obj) in objs.items():
        p1 = obj.data.root_pos_w.torch[0].double()
        q1 = obj.data.root_quat_w.torch[0].double()
        drift = float((p1 - p0[k]).norm()) * 1e3
        # rotation angle between q0 and q1 (robust for small angles)
        rot_drift = math.degrees(2 * math.atan2(float((q1 - q0[k]).norm()), float((q1 + q0[k]).norm())))
        rot_drift = min(rot_drift, 360 - rot_drift)
        ok = drift < 1.0 and dev[k] < 1e-3 and pen[k] < 0.5e-3 and vlast[k] < 0.01
        variant = k.split("/")[0]
        summary.setdefault(variant, [0, 0])
        summary[variant][0] += ok
        summary[variant][1] += 1
        row = dict(part=k, mass_g=round(masses[k] * 1e3, 3), manifest_g=F.manifest_entry(spec)["mass_g"],
                   settle_s=round(settle_t[k], 2), drift_mm=round(drift, 4), maxdev_mm=round(dev[k] * 1e3, 4), rot_drift_deg=round(rot_drift, 3),
                   vmax_mm_s=round(vmax[k] * 1e3, 2), vlast_mm_s=round(vlast[k] * 1e3, 2),
                   wlast_rad_s=round(wlast[k], 3), pen_mm=round(pen[k] * 1e3, 3),
                   z_mm=round(float(p1[2]) * 1e3, 3), ok=ok)
        rows.append(row)
        print("DROP", json.dumps(row), flush=True)
    for variant, (n_ok, n) in summary.items():
        print(f"DROP SUMMARY {variant}: {n_ok}/{n} ok", flush=True)
    with open(os.path.join(args.out, "drop_results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    if args.render:
        from PIL import Image

        key = next(k for k in objs if k.endswith("hex_bolt_M6x20/flat"))
        hero = objs[key][2]
        target = hero.data.root_pos_w.torch[0].clone()
        target[2] = 0.004
        eye = target + torch.tensor([0.035, -0.045, 0.03], device=target.device)
        cam.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        c2 = torch.tensor([[0.08, 0.0, 0.0]], device=target.device)
        cam_all.set_world_poses_from_view(c2 + torch.tensor([[0.25, -0.45, 0.3]], device=target.device), c2)
        for _ in range(40):
            sim.render()
            cam.update(dt)
            cam_all.update(dt)
        for c, fname in ((cam, "m6x20_hex_closeup.png"), (cam_all, "drop_overview.png")):
            rgb = c.data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            Image.fromarray(rgb[0, ..., :3].cpu().numpy().astype("uint8")).save(os.path.join(args.out, fname))
            print("saved", fname, flush=True)
