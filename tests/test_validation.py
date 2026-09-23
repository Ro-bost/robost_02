import unittest
import numpy as np
from robost.rl.validation import StairExitTracker


class ExitTests(unittest.TestCase):
    def test_sequential_trot_landings_and_two_second_exit(self):
        tracker = StairExitTracker()
        feet = np.array([[4.5, 0., .15]] * 4)
        for k in range(4):
            feet[:, 2] = .15
            feet[k, 2] = .02661
            self.assertFalse(tracker.update(k, feet, np.arange(4) == k))
        self.assertFalse(tracker.update(4.9, feet, [0, 0, 0, 1]))
        self.assertTrue(tracker.update(5., feet, [0, 0, 0, 1]))

    def test_side_bypass_fails_but_exit_floor_is_not_a_stair(self):
        tracker = StairExitTracker()
        feet = np.array([[4.6, .9, .02661]] * 4)
        tracker.update(0, feet, [1] * 4)
        self.assertTrue(tracker.update(2, feet, [1] * 4))
        tracker = StairExitTracker()
        feet[0, 0] = 3.9
        tracker.update(0, feet, [1] * 4)
        feet[:, 0] = 4.6
        tracker.update(1, feet, [1] * 4)
        self.assertFalse(tracker.update(3, feet, [1] * 4))
        self.assertTrue(tracker.left_course)

    def test_fall_or_restart_never_passes(self):
        tracker = StairExitTracker()
        feet = np.array([[4.6, 0., .02661]] * 4)
        tracker.update(0, feet, [1] * 4)
        tracker.update(1, feet, [1] * 4, alive=False)
        self.assertFalse(tracker.update(3, feet, [1] * 4))


if __name__ == '__main__':
    unittest.main()
