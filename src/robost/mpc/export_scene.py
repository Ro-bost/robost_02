#!/usr/bin/env python3
"""Export the RS06 MuJoCo scene (robot + terrain + torque motors) for the C++ MPC.

Reuses robost.cli.scene.build_scene, so the C++ side sees exactly the same
robot, terrain, effort limits and "stand" keyframe as the Python tools.
"""

import argparse
from pathlib import Path

import mujoco

from robost.cli.scene import build_scene
from robost.paths import ROOT
from robost.simulation.terrain import Terrain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-cm", type=int, default=0, help="0 = flat ground")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stairs_cm:
        terrain, name = Terrain("stairs", args.stairs_cm), f"scene_stairs_{args.stairs_cm}cm"
    else:
        terrain, name = Terrain("flat"), "scene_flat"
    _, model, _ = build_scene(terrain=terrain)
    output = args.output or ROOT / "runs" / "mpc" / f"{name}.mjb"
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(output), None)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
