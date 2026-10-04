"""No-reset playback termination and cleanup without a display or GPU."""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from robost.rl import base
from robost.rl.viewer import NoResetMujocoViewer


class FakeEnvironment:
    def __init__(self, fail_after=2):
        self.unwrapped = self
        self.cfg = SimpleNamespace(viewer=SimpleNamespace(env_idx=0))
        self.step_dt = 0.02
        self.reset_buf = torch.tensor([False])
        self.termination_manager = SimpleNamespace(
            active_terms=("stalled",), get_term=lambda _: self.reset_buf
        )
        self.steps = 0
        self.fail_after = fail_after
        self.manual_reset_pending = False

    def get_observations(self):
        return {}

    def step(self, actions):
        del actions
        if self.manual_reset_pending:
            raise AssertionError("A terminated environment must never be stepped")
        self.steps += 1
        self.reset_buf = torch.tensor([self.steps >= self.fail_after])
        self.manual_reset_pending = bool(self.reset_buf.any())

    def reset(self):
        # Match mjlab: explicit reset clears the guard but leaves reset_buf.
        self.steps = 0
        self.manual_reset_pending = False


class ViewerTests(unittest.TestCase):
    def viewer(self, finite):
        env = FakeEnvironment()
        viewer = NoResetMujocoViewer(
            env, lambda _: None, close_on_failure=finite, enable_perturbations=False
        )
        viewer._interrupted = False
        return env, viewer

    def test_bounded_playback_exits_immediately_after_terminal_step(self):
        env, viewer = self.viewer(True)
        self.assertTrue(viewer._execute_step())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(viewer._execute_step())
            self.assertFalse(viewer._execute_step())
        self.assertEqual(env.steps, 2)
        self.assertTrue(viewer._interrupted)
        self.assertTrue(viewer._is_paused)
        self.assertEqual(viewer.first_failure["reasons"], {"0": ["stalled"]})
        self.assertAlmostEqual(viewer.first_failure["time_s"], 0.04)
        self.assertEqual(output.getvalue().count("Playback stopped"), 1)

    def test_unlimited_playback_pauses_and_retains_failed_pose(self):
        env, viewer = self.viewer(False)
        with contextlib.redirect_stdout(io.StringIO()):
            viewer._execute_step()
            viewer._execute_step()
            viewer.resume()  # Resuming without an explicit reset is also blocked.
            self.assertFalse(viewer._execute_step())
        self.assertEqual(env.steps, 2)
        self.assertTrue(viewer._is_paused)
        self.assertFalse(viewer._interrupted)

    def test_preexisting_failure_never_steps(self):
        env, viewer = self.viewer(True)
        env.reset_buf[:] = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(viewer._execute_step())
        self.assertEqual(env.steps, 0)
        self.assertTrue(viewer._interrupted)

    def test_first_step_before_mjlab_creates_reset_buffer(self):
        env, viewer = self.viewer(True)
        del env.reset_buf
        self.assertTrue(viewer._execute_step())
        self.assertEqual(env.steps, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(viewer._execute_step())
        self.assertEqual(env.steps, 2)
        self.assertTrue(viewer._interrupted)

    def test_manual_reset_ignores_stale_buffer_and_retains_first_failure(self):
        env, viewer = self.viewer(False)
        with contextlib.redirect_stdout(io.StringIO()):
            viewer._execute_step()
            viewer._execute_step()
            first_failure = viewer.first_failure
            viewer.reset_environment()
            self.assertTrue(bool(env.reset_buf.any()))
            self.assertTrue(viewer._execute_step())
            self.assertEqual(env.steps, 1)
            self.assertFalse(viewer._execute_step())
        self.assertIs(viewer.first_failure, first_failure)
        self.assertTrue(viewer._episode_failed)

    def test_play_closes_environment_when_policy_loading_fails(self):
        args = SimpleNamespace(
            seed=42, stairs_cm=0, speed=0.25, checkpoint=Path("unused.pt"), play_seconds=15.0
        )
        cfg = MagicMock()
        raw_env = MagicMock()
        runner = MagicMock()
        runner.load.side_effect = RuntimeError("bad checkpoint")
        with (
            patch.object(base, "ManagerBasedRlEnv", return_value=raw_env),
            patch.object(base, "RslRlVecEnvWrapper"),
            patch.object(base, "MjlabOnPolicyRunner", return_value=runner),
        ):
            with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                base.play(args, cfg_factory=lambda *a, **k: cfg)
        raw_env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
