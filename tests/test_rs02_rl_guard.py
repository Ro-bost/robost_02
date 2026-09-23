import unittest
import torch
import rs02_rl_guard as guard


class GuardTests(unittest.TestCase):
    def test_only_brakes_outward_near_limits(self):
        position = torch.tensor([[-1.5, -2.3, -1.01, -2.3]])
        velocity = torch.tensor([[0., -5., 5., 2.]])
        target = torch.tensor([[-1.4, -2.317, -1.02856, -2.]])
        limits = torch.tensor([[[-2.397, -.87856]] * 4])
        clip = torch.tensor([[[-2.317, -1.02856]] * 4])
        out = guard.guarded_target(target, position, velocity, limits, torch.tensor(80.), clip)
        torch.testing.assert_close(out, torch.tensor([[-1.4, -2.05, -1.26, -2.]]))
        self.assertTrue(torch.all(out >= clip[..., 0]))
        self.assertTrue(torch.all(out <= clip[..., 1]))

    def test_failure_checks_preserved(self):
        cfg = guard.make_cfg(1, evaluate=True)
        self.assertIsInstance(cfg.actions['joint_pos'], guard.GuardedJointActionCfg)
        self.assertIn('joint_range', cfg.terminations)
        self.assertIn('illegal_contact', cfg.terminations)


if __name__ == '__main__':
    unittest.main()
