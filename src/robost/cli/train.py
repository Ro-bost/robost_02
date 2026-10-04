"""Train an RS06 policy stage with the fixed nominal hardware configuration."""

import argparse
import json
import math
from pathlib import Path

from robost.paths import CONFIG


def main(argv=None):
    settings = json.loads((CONFIG / "rs06_training.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Initial policy with the current observation layout",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=list(settings["stages"]), default="flat")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--std", type=float, default=settings["exploration_std"])
    parser.add_argument("--learning-rate", type=float, default=settings["learning_rate"])
    parser.add_argument(
        "--normalizer-pseudocount", type=int, default=settings["normalizer_pseudocount"]
    )
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new directory")
    if min(args.num_envs, args.iterations, args.normalizer_pseudocount) < 1:
        parser.error("Environment, iteration and normalizer counts must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must fit uint32")
    if not all(math.isfinite(value) for value in (args.std, args.learning_rate)):
        parser.error("Numeric arguments must be finite")
    if not 0 < args.std <= 1 or args.learning_rate <= 0:
        parser.error("Standard deviation must be in (0, 1] and learning rate must be positive")
    args.speed = settings["speed_m_s"]
    import torch
    from robost.rl.rs06 import train

    torch.set_num_threads(4)
    train(args)


if __name__ == "__main__":
    main()
