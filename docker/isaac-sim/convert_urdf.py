#!/usr/bin/env python3
"""Convert a URDF file to USD for Isaac Sim.

Usage (inside the Isaac Sim container):
    /isaac-sim/python.sh /usr/local/bin/convert_urdf.py \
        --urdf /robot_description/urdf/soarm101_isaacsim.urdf \
        --out  /robot_description/usd/soarm101.usd

Isaac Sim 6.x ships URDF importer 3.0 (a Python pipeline that writes structured USD);
the old ``isaacsim.asset.importer.urdf.import_urdf`` / ``ImportConfig`` API is gone.
This uses Isaac Lab's ``UrdfConverter``, which wraps that pipeline. The importer names
the output file after the robot, so the result is renamed to ``--out`` (same directory,
so the layers it references still resolve).

Note: importer 3.0 lays out the prims differently from the 2.x importer that produced
the committed robot_description/usd/soarm101.usd, which the envs expect at
``Robot/so101_new_calib``. Check the prim paths before replacing that file.
"""
import argparse
import os
import shutil
import sys

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description="URDF to USD converter")
    parser.add_argument("--urdf", required=True, help="Path to input URDF file")
    parser.add_argument("--out", required=True, help="Path to output USD file")
    parser.add_argument("--fix-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-fixed", action=argparse.BooleanOptionalAction, default=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.require_kit = True
    args.physics = "isaacsim_physx"

    urdf_path = os.path.abspath(args.urdf)
    out_path = os.path.abspath(args.out)

    try:
        from isaaclab.physics import PhysicsCfg

        with launch_simulation(cfg=PhysicsCfg(), launcher_args=args):
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

            cfg = UrdfConverterCfg(
                asset_path=urdf_path,
                usd_dir=os.path.dirname(out_path),
                fix_base=args.fix_base,
                merge_fixed_joints=args.merge_fixed,
                force_usd_conversion=True,
                joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="position"),  # position drive
            )
            usd_path = UrdfConverter(cfg).usd_path
            if os.path.abspath(usd_path) != out_path:
                shutil.move(usd_path, out_path)
            print(f"USD saved to {out_path}")
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        print("You can convert manually in Isaac Sim GUI:")
        print(f"  File > Import > {args.urdf}")
        print("  Enable: Static Base, Allow Self-Collision")
        print(f"  Save as: {args.out}")
        sys.exit(1)


if __name__ == "__main__":
    main()
