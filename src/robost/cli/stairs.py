#!/usr/bin/env python3
"""Run a fixed RS06 v5 policy on flat ground or the measured stair course."""

import argparse
import datetime
import hashlib
import json
import math
import os
import importlib
from pathlib import Path

from robost.paths import CONFIG, ROOT
from robost.cli.common import ADAPTERS, HEIGHTS_CM

DEFAULT_POLICY = json.loads((CONFIG / "rs06_policy.json").read_text())


def select_policy(height, checkpoint=None, adapter=None):
    if checkpoint is None:
        if adapter is not None:
            raise ValueError("Custom --adapter requires --checkpoint")
        if height not in HEIGHTS_CM:
            raise ValueError(f"Unsupported step height: {height}")
        return DEFAULT_POLICY["adapter"], ROOT / DEFAULT_POLICY["path"]
    if adapter is None:
        raise ValueError(
            "--checkpoint requires its matching --adapter (rs06/gait/support/route/rhythm)"
        )
    return adapter, checkpoint


def validate_checkpoint(checkpoint, *, default=False):
    """Reject missing or changed default weights before importing GPU libraries."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        guidance = (
            (
                " This source checkout may omit trained weights. Supply "
                "--checkpoint /path/to/policy.pt --adapter rs06 "
                "(or the checkpoint matching adapter)."
            )
            if default
            else ""
        )
        raise ValueError(f"Checkpoint missing: {checkpoint}.{guidance}")
    if default:
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual != DEFAULT_POLICY["sha256"]:
            raise ValueError(
                f"Default RS06 checkpoint SHA-256 mismatch: {checkpoint}. "
                "Restore the manifest-matching weights or explicitly select "
                "--checkpoint and --adapter for a different policy."
            )


def default_policy_description():
    status = DEFAULT_POLICY.get("status", "experimental")
    outcome = DEFAULT_POLICY.get("course_18cm_completed")
    note = (
        "18cm course not completed"
        if outcome is False
        else "18cm completion unverified"
        if outcome is None
        else "18cm completion is a simulation result"
    )
    return f"RS06 trained {status} policy; fixed checkpoint across heights; {note}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-cm", type=int, choices=HEIGHTS_CM, default=18)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="Forward command in m/s; evaluated default is 0.25",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run a scored test and save a 1x video instead of a live window",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--play-seconds", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional override; default is the fixed experimental RS06 checkpoint in config/rs06_policy.json",
    )
    parser.add_argument(
        "--adapter",
        choices=ADAPTERS,
        help="Must match the checkpoint training adapter; custom adapters require --checkpoint",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    using_default_policy = args.checkpoint is None
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed must be finite and positive")
    if not math.isfinite(args.duration) or args.duration <= 3:
        parser.error("duration must exceed 3 seconds")
    if not math.isfinite(args.play_seconds) or args.play_seconds < 0:
        parser.error("play-seconds must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must fit uint32")
    try:
        args.adapter, args.checkpoint = select_policy(args.stairs_cm, args.checkpoint, args.adapter)
        validate_checkpoint(args.checkpoint, default=using_default_policy)
    except ValueError as error:
        parser.error(str(error))
    os.environ.setdefault("MUJOCO_GL", "egl" if args.headless else "glfw")
    import torch
    import robost.rl.base as flat

    adapter = importlib.import_module("robost.rl." + args.adapter)
    torch.set_num_threads(4)
    args.num_envs = 1
    args.stop_go = False
    args.video = True
    args.sample_actions = False
    args.noise_scale = 0.0
    if args.output is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output = ROOT / f"runs/stairs/{args.stairs_cm}cm_{stamp}"
    terrain_label = (
        f"{args.stairs_cm}cm rise x 32cm tread | 10 up + landing + 10 down"
        if args.stairs_cm
        else "flat ground"
    )
    print(f"RS06 v5 | {terrain_label} | {args.speed}m/s")
    print("Policy:", args.checkpoint)
    if using_default_policy:
        print(default_policy_description())
    if args.stairs_cm:
        print("Raised edge strips: 5mm x 60mm.")
    print("No automatic resets. Hardware safety remains unverified.")
    if args.headless:
        args.policy_adapter = Path(adapter.__file__).resolve()
        args.evaluation_purpose = (
            default_policy_description()
            if using_default_policy
            else "RS06 deterministic policy evaluation with user checkpoint"
        )
        flat.evaluate(args, cfg_factory=adapter.make_cfg)
    else:
        flat.play(args, cfg_factory=adapter.make_cfg)


if __name__ == "__main__":
    main()
