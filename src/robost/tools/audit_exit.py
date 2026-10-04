"""Read recorded trajectories; write a separate exit audit, never rewrite results."""

import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from robost.rl.validation import StairExitTracker


def audit(folder):
    result = json.loads((folder / "evaluation.json").read_text())
    if not result.get("stairs_cm"):
        raise ValueError(
            "Exit audit requires a staircase trial; flat ground has no course completion"
        )
    model = mujoco.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mujoco.MjData(model)
    ids = [model.site("robot/" + leg).id for leg in ("FR", "FL", "RR", "RL")]
    contacts = np.load(folder / "foot_contacts.npy")
    tracker = (
        StairExitTracker.from_metadata(result["course"], foot_radius=result["foot_radius_m"])
        if "course" in result
        else StairExitTracker(exit_x=4.30, foot_radius=0.02661)
    )
    for qpos, row in zip(np.load(folder / "qpos_env0.npy"), result["log_env0"]):
        if not row["alive"]:
            tracker.failed = True
            break
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        t = row["t"]
        start = result.get("measurement_start_s", 3.0)
        index = round((t - start) * result.get("sample_hz", 50))
        contact = contacts[index, 0] if 0 <= index < len(contacts) else np.zeros(4, bool)
        if tracker.update(t, data.site_xpos[ids], contact):
            break
    # Older 25 Hz pose logs can omit a failure between frames. The 50 Hz
    # evaluator's first-failure record remains authoritative in that case.
    first_failure = result.get("trials", [{}])[0].get("first_failure_s")
    if first_failure is not None and (
        tracker.completed_at is None or first_failure <= tracker.completed_at
    ):
        tracker.failed = True
        tracker.completed_at = None
    record = dict(
        original_course_completed=result["course_completed"],
        completion_definition=tracker.definition,
        audit_sample_hz=25,
        original_contact_hz=50,
        course_completed=tracker.completed_at is not None,
        course_completed_at_s=tracker.completed_at,
        landing_times_s=tracker.landing_times,
        original_first_failure_s=first_failure,
        left_course=tracker.left_course,
        failed_before_completion=tracker.failed,
        note="Recorded-state audit only; not a fresh physics rollout. Old completed videos may lack the new 2s exit observation.",
    )
    target = folder / ("exit_audit_v3.json" if "course" in result else "exit_audit_v2.json")
    with target.open("x") as handle:
        json.dump(record, handle, indent=2)
    print(folder, json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    for folder in parser.parse_args().folders:
        audit(folder)
