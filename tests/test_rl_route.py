import unittest
from types import SimpleNamespace
import numpy as np
import torch
import robost.rl.route as route


class Scene(dict):
    pass


class RouteTests(unittest.TestCase):
    def test_heading_and_lateral_are_observable(self):
        robot = SimpleNamespace(
            data=SimpleNamespace(
                heading_w=torch.tensor([0.0, np.pi / 2]),
                root_link_pos_w=torch.tensor([[0.0, 0.2, 0.0], [0.0, 3.4, 0.0]]),
            )
        )
        scene = Scene(robot=robot)
        scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        observation = route.course_observation(SimpleNamespace(scene=scene))
        np.testing.assert_allclose(observation, [[0.0, 1.0, 0.2], [1.0, 0.0, 0.4]], atol=1e-6)

    def test_upstream_heading_controller_and_safety(self):
        cfg = route.make_cfg(1, evaluate=True)
        self.assertTrue(cfg.commands["twist"].heading_command)
        self.assertEqual(cfg.commands["twist"].ranges.ang_vel_z, (-0.5, 0.5))
        self.assertEqual(cfg.commands["twist"].rel_world_envs, 1.0)
        self.assertIn("joint_range", cfg.terminations)
        self.assertEqual(list(cfg.observations["actor"].terms)[-1], "course_direction")


if __name__ == "__main__":
    unittest.main()
