#!/usr/bin/env python3
"""Convert a URDF file to USD for Isaac Sim.

Usage (inside the Isaac Sim container):
    /isaac-sim/python.sh /usr/local/bin/convert_urdf.py \
        --urdf /robot_description/urdf/soarm101_isaacsim.urdf \
        --out  /robot_description/usd/soarm101.usd
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="URDF to USD converter")
    parser.add_argument("--urdf", required=True, help="Path to input URDF file")
    parser.add_argument("--out", required=True, help="Path to output USD file")
    parser.add_argument("--fix-base", action="store_true", default=True)
    parser.add_argument("--merge-fixed", action="store_true", default=True)
    args = parser.parse_args()

    try:
        from isaacsim import SimulationApp
        sim = SimulationApp({"headless": True})

        from isaacsim.asset.importer.urdf import import_urdf, ImportConfig

        config = ImportConfig()
        config.fix_base = args.fix_base
        config.merge_fixed_joints = args.merge_fixed
        config.make_instanceable = False
        config.default_drive_type = 1  # position drive

        import_urdf(args.urdf, config, args.out)
        print(f"USD saved to {args.out}")
        sim.close()
    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        print("You can convert manually in Isaac Sim GUI:")
        print(f"  File > Import > {args.urdf}")
        print("  Enable: Static Base, Allow Self-Collision")
        print(f"  Save as: {args.out}")
        sys.exit(1)


if __name__ == "__main__":
    main()
