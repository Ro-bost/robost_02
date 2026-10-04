"""Evaluate one fixed policy in independent processes; retain every trial and failure."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from robost.paths import ROOT
from robost.cli.common import ADAPTERS, HEIGHTS_CM


RESULT_METADATA = (
    "checkpoint_sha256",
    "robot",
    "mass_kg",
    "urdf_sha256",
    "model_input_sha256",
    "physics",
    "auto_reset",
    "source_snapshot_sha256",
    "adapter_sha256",
    "policy_adapter",
    "policy_adapter_sha256",
    "course_validation_sha256",
    "purpose",
    "controller",
    "training_provenance",
    "contact_backend_note",
    "quality_measurement_window",
    "hardware_pass",
    "numerical_gate_pass",
    "actual_duration_s",
    "speed_command",
    "duration",
    "joint_order",
)


def trial_record(folder, height, seed, repeat, returncode):
    """Keep reported physics and provenance; a flat trial has no stair completion."""
    record = dict(
        height_cm=height,
        terrain="flat" if height == 0 else "stairs",
        seed=seed,
        repeat=repeat,
        folder=str(folder.resolve()),
        process_exit_code=returncode,
        evaluation_available=False,
        course_completed=None,
        course_completed_at_s=None,
        course=None,
    )
    result_path = folder / "evaluation.json"
    if not result_path.is_file():
        return record
    try:
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            raise ValueError("evaluation.json must contain an object")
    except (OSError, ValueError) as error:
        record["evaluation_error"] = str(error)
        return record
    record["evaluation_available"] = True
    if result.get("trials"):
        record.update(result["trials"][0])
    for key in RESULT_METADATA:
        if key in result:
            record[key] = result[key]
    if height:
        for key in (
            "course",
            "course_completed",
            "course_completed_at_s",
            "left_course",
            "max_x_before_failure",
            "completion_definition",
            "max_foot_height_m",
        ):
            if key in result:
                record[key] = result[key]
    else:
        # Surviving flat ground or passing its quality gate does not complete
        # an absent staircase, even if an adapter emitted stale course fields.
        record.update(course=None, course_completed=None, course_completed_at_s=None)
    return record


def matrix_summary(args, checkpoint_sha256, records):
    stairs = [record for record in records if record["height_cm"] > 0]
    flat = [record for record in records if record["height_cm"] == 0]
    return dict(
        adapter=args.adapter,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha256,
        speed=args.speed,
        duration=args.duration,
        requested_heights_cm=args.heights,
        seeds=args.seeds,
        repeats=args.repeats,
        runs=records,
        total=len(records),
        stairs_total=len(stairs),
        flat_total=len(flat),
        completed=sum(record.get("course_completed") is True for record in stairs)
        if stairs
        else None,
        flat_survived=sum(record.get("survived") is True for record in flat),
        flat_numerical_gate_passed=sum(
            record.get("numerical_gate_pass") is True for record in flat
        ),
        failed_processes=sum(record["process_exit_code"] != 0 for record in records),
        missing_evaluations=sum(not record["evaluation_available"] for record in records),
        completion_note="completed counts staircase trials only; flat course_completed is null. "
        "Survival, numerical quality and hardware safety are separate outcomes.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=ADAPTERS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--heights",
        nargs="+",
        type=int,
        choices=HEIGHTS_CM,
        default=[18],
        help="Step heights in cm; 0 selects flat ground",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed must be finite and positive")
    if not math.isfinite(args.duration) or args.duration <= 3:
        parser.error("duration must be finite and exceed 3 seconds")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("seeds must be between 0 and 2**32 - 1")
    if len(set(args.heights)) != len(args.heights) or len(set(args.seeds)) != len(args.seeds):
        parser.error("heights and seeds must be unique; use repeats for repeated trials")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new directory")
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for height in args.heights:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                name = f"h{height}_seed{seed}_repeat{repeat}"
                folder = args.output / name
                command = [
                    sys.executable,
                    "-m",
                    "robost.cli.trial",
                    "--adapter",
                    args.adapter,
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--output",
                    str(folder.resolve()),
                    "--stairs-cm",
                    str(height),
                    "--speed",
                    str(args.speed),
                    "--duration",
                    str(args.duration),
                    "--seed",
                    str(seed),
                ]
                if args.video:
                    command.append("--video")
                with (args.output / f"{name}.log").open("x") as log:
                    run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                record = trial_record(folder, height, seed, repeat, run.returncode)
                records.append(record)
                summary = matrix_summary(args, checkpoint_sha256, records)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
                print(
                    name,
                    "completed=",
                    record.get("course_completed"),
                    "survived=",
                    record.get("survived"),
                    "time=",
                    record.get("course_completed_at_s"),
                    "failure=",
                    record.get("failure_reasons"),
                    flush=True,
                )
    # Finish collecting all trials and preserve the summary before reporting
    # infrastructure errors. A robot fall is a valid evaluation, not a process
    # error, and therefore does not change the command's exit status.
    if summary["failed_processes"] or summary["missing_evaluations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
