"""Recorded replay keeps flat labels, timing and the terminal failure pose."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from robost.tools.render_trial import load_recording, replay_indices, trial_label
from robost.tools.plot_trial import course_profile, motor_measurement


class ReplayTests(unittest.TestCase):
    def test_flat_measurements_use_rs06_motor_ratings_and_zero_height(self):
        result = {
            "robot": "rs06",
            "duration": 4.0,
            "speed_command": 0.25,
            "joint_order": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            "trials": [
                {
                    "joint_torque_rms": [1.0, 2.0, 6.0],
                    "ideal_motor_torque_rms_nm": [1.0, 2.0, 4.0],
                    "motor_rated_torque_nm": [6.0, 11.0, 11.0],
                }
            ],
        }
        self.assertEqual(course_profile(result), ([0.0, 2.0], [0.0, 0.0]))
        rms, ratings, label = motor_measurement(result)
        np.testing.assert_array_equal(rms, [1.0, 2.0, 4.0])
        np.testing.assert_array_equal(ratings, [6.0, 11.0, 11.0])
        self.assertNotIn("RS02", label)

    def test_flat_trial_without_stair_field_has_truthful_label(self):
        label = trial_label({"robot": "rs06", "video_policy_label": "RS06 adapted policy"})
        self.assertIn("flat", label)
        self.assertIn("RS06 adapted policy", label)
        self.assertNotIn("stairs", label)
        self.assertIn("18cm stairs", trial_label({"robot": "rs06", "stairs_cm": 18}))

    def test_half_interval_terminal_pose_is_preserved_at_correct_speed(self):
        times = np.array([0.02, 0.06, 0.10, 0.12])
        indices = replay_indices(times, 50)
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 2, 3])
        self.assertEqual(int(indices[-1]), 3)
        self.assertAlmostEqual(len(indices) / 50, 0.12)

    def test_explicit_telemetry_times_match_poses_and_failure_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            times = np.array([0.02, 0.06, 0.08])
            np.save(folder / "qpos_env0.npy", np.ones((3, 19)))
            np.savez(folder / "telemetry.npz", qpos_time=times)
            result = {"log_env0": [{"t": value, "alive": value < 0.08} for value in times]}
            poses, rows, loaded = load_recording(folder, result)
            self.assertEqual(len(poses), 3)
            self.assertFalse(rows[-1]["alive"])
            np.testing.assert_array_equal(loaded, times)
            np.save(folder / "qpos_time.npy", np.array([0.02, 0.06]))
            with self.assertRaisesRegex(ValueError, "matching lengths"):
                load_recording(folder, result)

    def test_recording_mismatch_is_rejected_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.save(folder / "qpos_env0.npy", np.ones((2, 19)))
            with self.assertRaisesRegex(ValueError, "matching lengths"):
                load_recording(folder, {"log_env0": [{"t": 0.02}]})


if __name__ == "__main__":
    unittest.main()
