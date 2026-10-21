#!/usr/bin/env python3
"""Convert a URDF file to USD for Isaac Sim (URDF importer 3.0).

Usage (inside the Isaac Sim container; scripts/setup_robot_usd.sh wraps this):
    /isaac-sim/python.sh /usr/local/bin/convert_urdf.py \
        --urdf /robot_description/urdf/soarm101_isaacsim.urdf \
        --out  /usd_out/soarm101.usda

Isaac Sim 6.x ships URDF importer 3.0 (a Python pipeline that writes structured USD);
the old ``isaacsim.asset.importer.urdf.import_urdf`` / ``ImportConfig`` API is gone.
This uses Isaac Lab's ``UrdfConverter``, which wraps that pipeline.

Output layout. The importer writes ``<robot>/<robot>.usda`` plus a ``payloads/`` folder
(base/instances/materials/robot ``.usda``, ``geometries.usd`` with the meshes, and
``Physics/{physics,physx,mujoco}.usda``) that the root file references by relative path.
This script converts into a temporary directory and moves that tree next to ``--out``:
the root layer becomes ``--out`` and ``payloads/`` sits beside it. Keep them together.

Prim layout (differs from the 2.x importer used up to Isaac Sim 5.x):
  - The default prim is the robot (``so101_new_calib``); there is no ``/World`` wrapper,
    lights or physics scene, and no duplicate ``so101_new_calib_01``.
  - Links are nested along the kinematic chain under ``Geometry/`` (e.g.
    ``Geometry/base_link/shoulder_link/.../gripper_link/gripper_frame_link``);
    joints live under ``Physics/``.
  - The ``Physics`` variant set selects ``physics`` (neutral UsdPhysics, the default),
    ``physx``, ``mujoco`` or ``none``. Drive gains are not authored (the URDF has none);
    the envs set them through their actuator cfgs.

Post-processing (``--fix-articulation-root``, on by default). The importer puts
``PhysicsArticulationRootAPI`` on ``base_link`` and fixes the base with ``Physics/root_joint``
(world -> base_link). PhysX then treats that joint as an external constraint on a
*floating-base* articulation (Isaac Lab reports ``is_fixed_base == False`` and the base
drifts a few mm). This script moves the articulation root to the robot prim, an ancestor
of the fixed joint, which makes PhysX build a fixed-base articulation (the same fix
Isaac Lab applies in ``modify_articulation_root_properties`` when it creates a root joint).
"""
import argparse
import os
import shutil
import sys
import tempfile

from isaaclab.app import add_launcher_args, launch_simulation

ROOT_SCHEMAS = ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI")


def fix_articulation_root(physics_layer_path: str) -> None:
    """Move the articulation-root schemas from the root link to the robot (default) prim."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(physics_layer_path)
    robot = layer.GetPrimAtPath("/" + layer.defaultPrim)
    paths = []
    layer.Traverse(Sdf.Path.absoluteRootPath, lambda p: paths.append(p) if p.IsPrimPath() else None)

    moved = []
    for path in paths:
        spec = layer.GetPrimAtPath(path)
        if path == robot.path:
            continue
        op = spec.GetInfo("apiSchemas")
        items = list(op.prependedItems)
        found = [s for s in ROOT_SCHEMAS if s in items]
        if not found:
            continue
        op.prependedItems = [s for s in items if s not in ROOT_SCHEMAS]
        spec.SetInfo("apiSchemas", op)
        moved.append(str(path))
        # The Newton articulation-root attribute travels with its schema.
        attr = spec.attributes.get("newton:selfCollisionEnabled")
        if attr is not None:
            new = Sdf.AttributeSpec(robot, attr.name, attr.typeName)
            new.default = attr.default
            spec.RemoveProperty(attr)

    if not moved:
        print(f"[convert_urdf] no articulation root to move in {physics_layer_path}")
        return
    op = robot.GetInfo("apiSchemas")
    op.prependedItems = list(op.prependedItems) + [s for s in ROOT_SCHEMAS if s not in op.prependedItems]
    robot.SetInfo("apiSchemas", op)
    layer.Save()
    print(f"[convert_urdf] moved articulation root {moved} -> {robot.path}")


def main():
    parser = argparse.ArgumentParser(description="URDF to USD converter (URDF importer 3.0)")
    parser.add_argument("--urdf", required=True, help="Path to input URDF file")
    parser.add_argument("--out", required=True, help="Path of the root USD layer to write (payloads/ goes next to it)")
    parser.add_argument("--fix-base", action=argparse.BooleanOptionalAction, default=True)
    # Keep fixed joints: gripper_frame_link (the EE frame) is attached by a fixed joint.
    parser.add_argument("--merge-fixed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fix-articulation-root", action=argparse.BooleanOptionalAction, default=True,
                        help="Move the articulation root to the robot prim so PhysX sees a fixed base")
    add_launcher_args(parser)
    args = parser.parse_args()
    args.require_kit = True
    args.physics = "isaacsim_physx"

    urdf_path = os.path.abspath(args.urdf)
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)

    try:
        from isaaclab.physics import PhysicsCfg

        with launch_simulation(cfg=PhysicsCfg(), launcher_args=args):
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

            os.makedirs(out_dir, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=out_dir, prefix=".convert_") as tmp:
                cfg = UrdfConverterCfg(
                    asset_path=urdf_path,
                    usd_dir=tmp,
                    fix_base=args.fix_base,
                    merge_fixed_joints=args.merge_fixed,
                    force_usd_conversion=True,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="position"),
                )
                usd_path = UrdfConverter(cfg).usd_path
                gen_dir = os.path.dirname(usd_path)
                if args.fix_articulation_root:
                    fix_articulation_root(os.path.join(gen_dir, "payloads", "Physics", "physics.usda"))

                # The asset transformer records its scratch path in each layer's doc string;
                # drop it so regenerated files only differ when the asset does.
                from pxr import Sdf

                for root, _, files in os.walk(gen_dir):
                    for n in files:
                        if n.endswith(".usda"):
                            layer = Sdf.Layer.FindOrOpen(os.path.join(root, n))
                            if layer.documentation.startswith("Generated from Composed Stage"):
                                layer.documentation = ""
                                layer.Save()

                dst_payloads = os.path.join(out_dir, "payloads")
                if os.path.exists(dst_payloads):
                    shutil.rmtree(dst_payloads)
                shutil.move(os.path.join(gen_dir, "payloads"), dst_payloads)
                shutil.move(usd_path, out_path)

            uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
            if uid and gid:
                for root, dirs, files in os.walk(out_dir):
                    for n in dirs + files:
                        os.chown(os.path.join(root, n), int(uid), int(gid))
                os.chown(out_dir, int(uid), int(gid))
            print(f"USD saved to {out_path} (+ {os.path.join(out_dir, 'payloads')}/)")
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        print("You can convert manually in Isaac Sim GUI:")
        print(f"  File > Import > {args.urdf}")
        print("  Enable: Static Base, Allow Self-Collision")
        print(f"  Save as: {args.out}")
        sys.exit(1)


if __name__ == "__main__":
    main()
