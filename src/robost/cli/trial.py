"""Internal single-trial worker for independent matrix evaluation processes."""

import argparse
import importlib
import math
from pathlib import Path

from robost.cli.common import ADAPTERS, HEIGHTS_CM


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=ADAPTERS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--stairs-cm", type=int, choices=HEIGHTS_CM, default=18)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new directory")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must fit uint32")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("Speed must be finite and positive")
    if not math.isfinite(args.duration) or args.duration <= 3:
        parser.error("Duration must be finite and exceed 3 seconds")
    import torch
    from robost.rl import base

    adapter = importlib.import_module("robost.rl." + args.adapter)
    args.num_envs = 1
    args.stop_go = False
    args.sample_actions = False
    args.noise_scale = 0.0
    args.policy_adapter = Path(adapter.__file__).resolve()
    args.evaluation_purpose = "RS06 fixed policy; nominal physics and first-failure evaluation"
    torch.set_num_threads(4)
    base.evaluate(args, cfg_factory=adapter.make_cfg)


if __name__ == "__main__":
    main()
