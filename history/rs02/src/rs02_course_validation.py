"""Dynamic stair exit validation, independent from the policy and simulator.

Every foot must land on the exit floor, but not at the same instant: trot has
swing feet. Lateral bounds apply on the staircase, not on the infinite exit
floor. Require two further seconds beyond the exit without a termination.
"""
import numpy as np


class StairExitTracker:
    definition = 'v2: each foot lands beyond x=4.30; all feet remain beyond exit for 2s; no reset or stair-side bypass'

    def __init__(self):
        self.landing_times = [None] * 4
        self.left_course = False
        self.failed = False
        self.clear_since = None
        self.completed_at = None

    def update(self, time, feet, contact, alive=True):
        if not alive:
            self.failed = True
            return False
        feet = np.asarray(feet)
        on_course = (feet[:, 0] > .75) & (feet[:, 0] < 4.30)
        self.left_course |= bool(np.any(on_course & (np.abs(feet[:, 1]) > .8)))
        landed = ((feet[:, 0] > 4.30) &
                  (np.abs(feet[:, 2] - .02661) < .02) & np.asarray(contact, dtype=bool))
        for k in range(4):
            if landed[k] and self.landing_times[k] is None:
                self.landing_times[k] = float(time)
        clear = all(t is not None for t in self.landing_times) and np.all(feet[:, 0] > 4.30)
        if clear:
            if self.clear_since is None:
                self.clear_since = float(time)
        else:
            self.clear_since = None
        valid = not self.failed and not self.left_course
        if valid and self.clear_since is not None and time - self.clear_since >= 2. - 1e-9:
            if self.completed_at is None:
                self.completed_at = float(time)
        return valid and self.completed_at is not None
