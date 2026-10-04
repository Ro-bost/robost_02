"""Regression tests: course dimensions and unchanged policy configuration."""

import unittest
import torch
import mujoco
import numpy as np
from types import SimpleNamespace
from robost.rl.stairs import CourseCfg, make_cfg, planar_tracking, course_bounds, course_end
from robost.rl.base import get_hardware
from robost.simulation.hardware import JOINT_NAMES
from robost.simulation.terrain import Terrain
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg


class StairTests(unittest.TestCase):
    def test_multirow_curriculum_compiles_unique_physical_tiles(self):
        cfg = TerrainGeneratorCfg(
            seed=42,
            size=(11.0, 3.0),
            num_rows=3,
            curriculum=True,
            sub_terrains={"course": CourseCfg()},
        )
        generator = TerrainGenerator(cfg)
        spec = mujoco.MjSpec()
        generator.compile(spec)
        model = spec.compile()
        self.assertEqual(model.ngeom, 3 * 40)
        self.assertEqual(len({model.geom(i).name for i in range(model.ngeom)}), model.ngeom)
        np.testing.assert_allclose(np.diff(generator.terrain_origins[:, 0, 0]), 11.0)
        for row, rise in enumerate((0.02, 0.10, 0.18)):
            first = model.geom(f"course_tile_{row * 40}_up_1")
            self.assertAlmostEqual(first.pos[2] + first.size[2], rise)
            strip = model.geom(f"course_tile_{row * 40}_up_1_strip_up")
            self.assertAlmostEqual(strip.pos[2] + strip.size[2], rise + 0.005)

    def test_course_geometry(self):
        for difficulty, rise in ((0.0, 0.02), (0.5, 0.10), (1.0, 0.18)):
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            out = CourseCfg(size=(11.0, 3.0)).function(difficulty, spec, None)
            terrain = Terrain("stairs", rise * 100)
            self.assertEqual(len(out.geometries), 40)
            np.testing.assert_allclose(out.origin, [1.0, 1.5, 0.0])
            for output, box in zip(out.geometries[1:], terrain.iter_boxes()):
                np.testing.assert_allclose(output.geom.pos, np.array(box.pos) + [1.0, 1.5, 0.0])
                np.testing.assert_allclose(output.geom.size, box.size)
                np.testing.assert_allclose(output.geom.friction, [1.0, 0.005, 0.0001])
            self.assertTrue(all(g.geom.mass == 0 for g in out.geometries))
            self.assertAlmostEqual(out.geometries[10].geom.size[0] * 2, 1.0)
            self.assertAlmostEqual(out.geometries[1].geom.size[0] * 2, 0.32)
            self.assertAlmostEqual(out.geometries[1].geom.size[1] * 2, 1.6)
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        with self.assertRaises(ValueError):
            CourseCfg(size=(7.0, 3.0)).function(1.0, spec, None)

    def test_eval_no_curriculum(self):
        cfg = make_cfg(1, evaluate=True)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertFalse(cfg.curriculum)
        self.assertEqual(list(cfg.observations["actor"].terms)[-1], "height_scan")
        self.assertEqual(cfg.sim.mujoco.timestep, 0.002)
        self.assertEqual(len(cfg.actions["joint_pos"].clip), 12)
        bounds = get_hardware().soft_joint_ranges[JOINT_NAMES.index("FR_calf_joint")]
        self.assertAlmostEqual(bounds[0], -2.44346)
        self.assertAlmostEqual(cfg.actions["joint_pos"].clip["FR_calf_joint"][0], bounds[0] + 0.03)
        self.assertAlmostEqual(cfg.actions["joint_pos"].clip["FR_calf_joint"][1], bounds[1] - 0.03)

    def test_vertical_motion_not_horizontal_error(self):
        data = SimpleNamespace(
            root_link_lin_vel_b=torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.2]])
        )
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(
                get_command=lambda _: torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]])
            ),
        )
        torch.testing.assert_close(planar_tracking(env), torch.ones(2))

    def test_exit_is_failure_not_success(self):
        class Scene(dict):
            pass

        scene = Scene(
            robot=SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor(
                        [[1.0, 0.7, 0.0], [8.7, 0.0, 0.0], [5.4, 0.0, 0.0], [1.0, 0.0, 0.0]]
                    )
                )
            )
        )
        scene.env_origins = torch.zeros(4, 3)
        env = SimpleNamespace(scene=scene)
        self.assertEqual(course_bounds(env).tolist(), [True, False, False, False])
        self.assertEqual(course_end(env).tolist(), [False, True, False, False])


if __name__ == "__main__":
    unittest.main()
