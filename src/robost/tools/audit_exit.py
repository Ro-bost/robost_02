"""Read recorded trajectories; write a separate v2 audit, never rewrite results."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from robost.rl.validation import StairExitTracker


def audit(folder):
    result = json.loads((folder / 'evaluation.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(folder / 'scene.mjb'))
    data = mujoco.MjData(model)
    ids = [model.site('robot/' + leg).id for leg in ('FR', 'FL', 'RR', 'RL')]
    contacts = np.load(folder / 'foot_contacts.npy')
    tracker = StairExitTracker()
    for qpos, row in zip(np.load(folder / 'qpos_env0.npy'), result['log_env0']):
        if not row['alive']:
            tracker.failed = True
            break
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        t = row['t']
        contact = contacts[round((t - 3.) / .02), 0] if t >= 3 else np.zeros(4, bool)
        if tracker.update(t, data.site_xpos[ids], contact):
            break
    record = dict(original_course_completed=result['course_completed'],
                  completion_definition=tracker.definition,
                  audit_sample_hz=25, original_contact_hz=50,
                  course_completed=tracker.completed_at is not None,
                  course_completed_at_s=tracker.completed_at,
                  landing_times_s=tracker.landing_times,
                  left_course=tracker.left_course, failed_before_completion=tracker.failed,
                  note='Recorded-state audit only; not a fresh physics rollout. Old completed videos may lack the new 2s exit observation.')
    target = folder / 'exit_audit_v2.json'
    with target.open('x') as handle:
        json.dump(record, handle, indent=2)
    print(folder, json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    for folder in parser.parse_args().folders:
        audit(folder)
