#!/usr/bin/env python3
"""Export the RS06 MuJoCo scene (robot + terrain + torque motors) for the C++ MPC.

Reuses robost.cli.scene.build_scene, so the C++ side sees exactly the same
robot, terrain, effort limits and "stand" keyframe as the Python tools.
"""

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import mujoco

from robost.cli.scene import build_scene
from robost.paths import ROOT
from robost.simulation.terrain import Terrain


def map_terrain(path):
    """Terrain made of the box geoms of a history/rs02/maps/*.xml course."""
    text = Path(path).read_text()
    boxes = []
    for name, pos, size in re.findall(
        r'name="(course_\d+)" type="box" pos="([^"]+)" size="([^"]+)"', text
    ):
        boxes.append(
            SimpleNamespace(
                name=name,
                pos=[float(v) for v in pos.split()],
                size=[float(v) for v in size.split()],
                friction=[1.0, 0.005, 0.0001],
                strip=False,
            )
        )
    return SimpleNamespace(iter_boxes=lambda: iter(boxes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-cm", type=int, default=0, help="0 = flat ground")
    parser.add_argument("--map", type=Path, help="RL course xml, e.g. history/rs02/maps/stairs_15cm.xml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.map:
        terrain, name = map_terrain(args.map), f"scene_rl_{args.map.stem}"
    elif args.stairs_cm:
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
