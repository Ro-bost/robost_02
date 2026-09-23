import unittest
from types import SimpleNamespace
import torch
import mujoco
import numpy as np
import robost.rl.rhythm as rhythm


class RhythmTests(unittest.TestCase):
    def test_impact_cost_only_above_threshold(self):
        force = torch.tensor([[[0., 0., 10.]] * 4, [[0., 0., 120.]] * 4])
        env = SimpleNamespace(scene={'shank_ground': SimpleNamespace(data=SimpleNamespace(force=force))})
        torch.testing.assert_close(rhythm.shank_force_cost(env), torch.tensor([0., 1.]))

    def test_knee_reserve_and_clock(self):
        cfg = rhythm.make_cfg(1, evaluate=True)
        self.assertAlmostEqual(cfg.actions['joint_pos'].clip['FR_calf_joint'][1], -.87856-.15)
        self.assertEqual(cfg.rewards['gait_contact'].weight, 3.)
        self.assertIn('joint_range', cfg.terminations)

    def test_training_height_range_preserves_final_course(self):
        for difficulty, rise in ((0., .12), (1., .20)):
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name='terrain')
            cfg = rhythm.HighCourseCfg(size=(7., 3.))
            output = cfg.function(difficulty, spec, np.random.default_rng(42))
            tops = [g.geom.pos[2] + g.geom.size[2] for g in output.geometries]
            self.assertAlmostEqual(tops[1], rise)
            self.assertAlmostEqual(max(tops), 5*rise)
            self.assertEqual(len(tops), 10)


if __name__ == '__main__':
    unittest.main()
