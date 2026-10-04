import unittest
from types import SimpleNamespace
import torch
import robost.rl.gait as gait


class GaitTests(unittest.TestCase):
    def test_trot_diagonals(self):
        env = SimpleNamespace(episode_length_buf=torch.arange(100), step_dt=0.02, device="cpu")
        phases = gait.leg_phases(env)
        torch.testing.assert_close(phases[:, 0], phases[:, 3])
        torch.testing.assert_close(phases[:, 1], phases[:, 2])
        self.assertTrue(torch.all((phases < 0.65).sum(-1) >= 2))
        self.assertTrue(torch.isfinite(gait.clock_observation(env)).all())

    def test_scan_sees_ahead_from_above(self):
        offsets, directions = gait.AheadPattern(size=(0.3, 0.05), resolution=0.05).generate_rays(
            None, "cpu"
        )
        self.assertAlmostEqual(float(offsets[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(offsets[:, 0].max()), 0.3)
        self.assertTrue(torch.all(offsets[:, 2] == 0.5))
        self.assertTrue(torch.all(directions[:, 2] == -1.0))

    def test_physics_unchanged(self):
        cfg = gait.make_cfg(1, evaluate=True)
        self.assertEqual(cfg.sim.mujoco.timestep, 0.002)
        self.assertEqual(cfg.decimation, 10)
        self.assertEqual(list(cfg.observations["actor"].terms)[-1], "gait_clock")
        self.assertEqual(len(cfg.actions), 1)
        self.assertFalse(cfg.curriculum)
        self.assertEqual(cfg.rewards["shank_collision"].weight, -1.0)
        self.assertEqual(
            next(s for s in cfg.scene.sensors if s.name == "swing_scan").ray_alignment, "world"
        )


if __name__ == "__main__":
    unittest.main()
