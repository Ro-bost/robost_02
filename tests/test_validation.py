import unittest
import numpy as np
from robost.rl.validation import StairExitTracker


class ExitTests(unittest.TestCase):
    @staticmethod
    def legacy_tracker():
        return StairExitTracker(course_start=0.75, exit_x=4.30, half_width=0.8, foot_radius=0.02661)

    def test_sequential_trot_landings_and_two_second_exit(self):
        tracker = self.legacy_tracker()
        feet = np.array([[4.5, 0.0, 0.15]] * 4)
        for k in range(4):
            feet[:, 2] = 0.15
            feet[k, 2] = 0.02661
            self.assertFalse(tracker.update(k, feet, np.arange(4) == k))
        self.assertFalse(tracker.update(4.9, feet, [0, 0, 0, 1]))
        self.assertTrue(tracker.update(5.0, feet, [0, 0, 0, 1]))

    def test_side_bypass_fails_but_exit_floor_is_not_a_stair(self):
        tracker = self.legacy_tracker()
        feet = np.array([[4.6, 0.9, 0.02661]] * 4)
        tracker.update(0, feet, [1] * 4)
        self.assertTrue(tracker.update(2, feet, [1] * 4))
        tracker = self.legacy_tracker()
        feet[0, 0] = 3.9
        tracker.update(0, feet, [1] * 4)
        feet[:, 0] = 4.6
        tracker.update(1, feet, [1] * 4)
        self.assertFalse(tracker.update(3, feet, [1] * 4))
        self.assertTrue(tracker.left_course)

    def test_fall_or_restart_never_passes(self):
        tracker = self.legacy_tracker()
        feet = np.array([[4.6, 0.0, 0.02661]] * 4)
        tracker.update(0, feet, [1] * 4)
        tracker.update(1, feet, [1] * 4, alive=False)
        self.assertFalse(tracker.update(3, feet, [1] * 4))

    def test_long_course_cannot_complete_at_historical_exit(self):
        with self.assertRaises(TypeError):
            StairExitTracker()
        tracker = StairExitTracker.from_metadata(
            {"start_x_m": 0.75, "exit_x_m": 7.66, "width_m": 1.6}, foot_radius=0.02606
        )
        feet = np.array([[4.6, 0.0, 0.02606]] * 4)
        self.assertFalse(tracker.update(0.0, feet, [1] * 4))
        self.assertFalse(tracker.update(3.0, feet, [1] * 4))
        self.assertEqual(tracker.landing_times, [None] * 4)
        feet[:, 0] = 7.8
        self.assertFalse(tracker.update(4.0, feet, [1] * 4))
        self.assertTrue(tracker.update(6.0, feet, [1] * 4))

    def test_side_bypass_after_old_exit_is_still_failure(self):
        tracker = StairExitTracker(exit_x=7.66, foot_radius=0.02606)
        feet = np.array([[6.0, 0.81, 0.02606]] * 4)
        tracker.update(0.0, feet, [1] * 4)
        feet[:, 0] = 7.8
        tracker.update(1.0, feet, [1] * 4)
        self.assertFalse(tracker.update(3.0, feet, [1] * 4))
        self.assertTrue(tracker.left_course)

    def test_step_back_restarts_exit_hold(self):
        tracker = StairExitTracker(exit_x=7.66, foot_radius=0.02606)
        feet = np.array([[7.8, 0.0, 0.02606]] * 4)
        tracker.update(0.0, feet, [1] * 4)
        feet[0, 0] = 7.6
        tracker.update(1.0, feet, [1] * 4)
        feet[0, 0] = 7.8
        tracker.update(1.5, feet, [1] * 4)
        self.assertFalse(tracker.update(3.0, feet, [1] * 4))
        self.assertTrue(tracker.update(3.5, feet, [1] * 4))


if __name__ == "__main__":
    unittest.main()
