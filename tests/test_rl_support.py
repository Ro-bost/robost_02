import unittest
from types import SimpleNamespace
import torch
import robost.rl.support as support
from robost.simulation.hardware import get_hardware


class Scene(dict):
    pass


class SupportTests(unittest.TestCase):
    def test_hard_limit_failure(self):
        data = SimpleNamespace(
            joint_pos=torch.tensor([[0.0, 1.004], [0.0, 1.006]]),
            joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        self.assertEqual(support.range_exceeded(env).tolist(), [False, True])

    def test_stall_and_episode_reset(self):
        scene = Scene(
            robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(1, 3)))
        )
        scene.env_origins = torch.zeros(1, 3)
        env = SimpleNamespace(scene=scene, episode_length_buf=torch.tensor([1]), step_dt=0.02)
        self.assertFalse(support.stalled(env).item())
        env.episode_length_buf[:] = 152
        self.assertTrue(support.stalled(env).item())
        scene["robot"].data.root_link_pos_w[:, 0] = 0.05
        self.assertFalse(support.stalled(env).item())
        env.episode_length_buf[:] = 1
        scene["robot"].data.root_link_pos_w.zero_()
        self.assertFalse(support.stalled(env).item())

    def test_safety_and_observation_compatibility(self):
        cfg = support.make_cfg(1, evaluate=True)
        self.assertEqual(cfg.rewards["upright"].params["terrain_sensor_names"], ("terrain_scan",))
        self.assertIn("joint_range", cfg.terminations)
        self.assertIn("stalled", cfg.terminations)
        self.assertAlmostEqual(
            cfg.actions["joint_pos"].clip["FR_calf_joint"][0],
            get_hardware().soft_joint_ranges[2, 0] + 0.08,
        )
        self.assertEqual(list(cfg.observations["actor"].terms)[-1], "gait_clock")

    def test_side_bypass_during_second_half_of_long_course(self):
        scene = Scene(
            robot=SimpleNamespace(
                site_names=["FR", "FL", "RR", "RL"],
                data=SimpleNamespace(
                    site_pos_w=torch.tensor([[[6.0, 0.81, 0.2]] * 4, [[7.8, 0.81, 0.03]] * 4])
                ),
            )
        )
        scene.env_origins = torch.zeros(2, 3)
        env = SimpleNamespace(scene=scene)
        self.assertEqual(support.foot_outside(env).tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
