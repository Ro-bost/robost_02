"""Dynamic stair exit validation, independent from the policy and simulator.

Every foot must land on the exit floor, but not at the same instant: trot has
swing feet. Lateral bounds apply on the staircase, not on the infinite exit
floor. Require two further seconds beyond the exit without a termination.
"""

import numpy as np


class StairExitTracker:
    def __init__(
        self,
        *,
        exit_x,
        foot_radius,
        course_start=0.75,
        half_width=0.8,
        exit_floor_height=0.0,
        hold_seconds=2.0,
        landing_tolerance=0.02,
    ):
        """Require the course exit and robot radius explicitly.

        New evaluations should use ``from_metadata`` so extending a course
        cannot accidentally retain an earlier completion boundary.
        """
        values = (
            course_start,
            exit_x,
            half_width,
            foot_radius,
            exit_floor_height,
            hold_seconds,
            landing_tolerance,
        )
        if not np.isfinite(values).all():
            raise ValueError("Exit validation dimensions must be finite")
        if (
            exit_x <= course_start
            or min(half_width, foot_radius, hold_seconds, landing_tolerance) <= 0
        ):
            raise ValueError("Invalid exit validation dimensions")
        self.course_start = float(course_start)
        self.exit_x = float(exit_x)
        self.half_width = float(half_width)
        self.foot_radius = float(foot_radius)
        self.exit_floor_height = float(exit_floor_height)
        self.hold_seconds = float(hold_seconds)
        self.landing_tolerance = float(landing_tolerance)
        self.definition = (
            f"v3: each foot lands beyond x={self.exit_x:.2f}; "
            f"all feet remain beyond exit for {self.hold_seconds:g}s; "
            "no reset or stair-side bypass"
        )
        self.landing_times = [None] * 4
        self.left_course = False
        self.failed = False
        self.clear_since = None
        self.completed_at = None

    @classmethod
    def from_metadata(cls, course, *, foot_radius, **kwargs):
        """Use the geometry saved with the rollout, never infer from its height."""
        return cls(
            course_start=course["start_x_m"],
            exit_x=course["exit_x_m"],
            half_width=course["width_m"] / 2,
            foot_radius=foot_radius,
            **kwargs,
        )

    def update(self, time, feet, contact, alive=True):
        if not alive:
            self.failed = True
            return False
        feet = np.asarray(feet)
        on_course = (feet[:, 0] > self.course_start) & (feet[:, 0] < self.exit_x)
        self.left_course |= bool(np.any(on_course & (np.abs(feet[:, 1]) > self.half_width)))
        landed = (
            (feet[:, 0] > self.exit_x)
            & (
                np.abs(feet[:, 2] - self.exit_floor_height - self.foot_radius)
                < self.landing_tolerance
            )
            & np.asarray(contact, dtype=bool)
        )
        for k in range(4):
            if landed[k] and self.landing_times[k] is None:
                self.landing_times[k] = float(time)
        clear = all(t is not None for t in self.landing_times) and np.all(feet[:, 0] > self.exit_x)
        if clear:
            if self.clear_since is None:
                self.clear_since = float(time)
        else:
            self.clear_since = None
        valid = not self.failed and not self.left_course
        if (
            valid
            and self.clear_since is not None
            and time - self.clear_since >= self.hold_seconds - 1e-9
        ):
            if self.completed_at is None:
                self.completed_at = float(time)
        return valid and self.completed_at is not None
