"""CPU-only guards for RS06 policy adaptation; never instantiate a GPU env."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from mjlab.entity import Entity
from mjlab.terrains.terrain_generator import TerrainGenerator
import mujoco
import numpy as np
import torch

from robost.paths import CONFIG
from robost.rl import rhythm, rs06
from robost.simulation.hardware import get_hardware
from robost.simulation.terrain import FRICTION, Terrain, add_to_spec


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = rhythm.make_cfg(1, seed=42, evaluate=True)
        cls.reference_model = Entity(cls.reference.scene.entities["robot"]).spec.compile()
        cls.settings = json.loads((CONFIG / "rs06_training.json").read_text())

    def assert_same_terrain(self, actual, expected):
        # The upstream default creates a fresh, capture-free lambda returning
        # an empty MjSpec on every config construction. Compare its code and
        # captures as well as all other terrain fields, not function identity.
        self.assertIs(actual.spec_fn.__code__, expected.spec_fn.__code__)
        self.assertEqual(actual.spec_fn.__closure__, expected.spec_fn.__closure__)
        self.assertEqual(actual.spec_fn.__defaults__, expected.spec_fn.__defaults__)
        self.assertEqual(replace(actual, spec_fn=expected.spec_fn), expected)

    def test_evaluation_preserves_rhythm_policy_interface_and_failure_gates(self):
        for stage in self.settings["stages"]:
            with self.subTest(stage=stage):
                cfg = rs06.make_cfg(1, seed=42, evaluate=True, stage=stage)
                self.assertFalse(cfg.auto_reset)
                for name in (
                    "actions",
                    "observations",
                    "terminations",
                    "events",
                    "sim",
                    "commands",
                    "rewards",
                    "curriculum",
                ):
                    self.assertEqual(getattr(cfg, name), getattr(self.reference, name), name)
                self.assertEqual(cfg.scene.entities, self.reference.scene.entities)
                self.assertEqual(cfg.scene.sensors, self.reference.scene.sensors)
                self.assert_same_terrain(cfg.scene.terrain, self.reference.scene.terrain)
                self.assertEqual(cfg.decimation, self.reference.decimation)
                self.assertEqual(cfg.episode_length_s, self.reference.episode_length_s)

    def test_training_retains_hardware_actions_observations_and_existing_terminations(self):
        fields = (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "body_pos",
            "body_quat",
            "jnt_range",
            "jnt_axis",
            "jnt_pos",
            "geom_type",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_contype",
            "geom_conaffinity",
            "geom_friction",
            "dof_armature",
            "dof_damping",
            "dof_frictionloss",
            "actuator_forcerange",
            "actuator_ctrlrange",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_trnid",
        )
        for stage in self.settings["stages"]:
            with self.subTest(stage=stage):
                cfg = rs06.make_cfg(1, seed=42, stage=stage)
                for name in ("actions", "observations"):
                    self.assertEqual(getattr(cfg, name), getattr(self.reference, name), name)
                # Extra constraint storage prevents buffer overflow in batched
                # stair training; it must not alter solver or physics options.
                self.assertGreaterEqual(cfg.sim.njmax, self.reference.sim.njmax)
                self.assertEqual(
                    replace(cfg.sim, njmax=self.reference.sim.njmax), self.reference.sim
                )
                self.assertEqual(cfg.scene.entities, self.reference.scene.entities)
                self.assertEqual(cfg.scene.sensors, self.reference.scene.sensors)
                self.assertEqual(cfg.decimation, 10)
                for name, term in self.reference.terminations.items():
                    self.assertEqual(cfg.terminations[name], term, name)
                added = set(cfg.terminations) - set(self.reference.terminations)
                self.assertEqual(
                    added, set() if stage == "flat" else {"course_bounds", "course_end"}
                )
                model = Entity(cfg.scene.entities["robot"]).spec.compile()
                self.assertAlmostEqual(
                    float(model.body_mass.sum()), get_hardware().mass_kg, places=8
                )
                for name in fields:
                    np.testing.assert_array_equal(
                        getattr(model, name), getattr(self.reference_model, name), err_msg=name
                    )
                np.testing.assert_array_equal(model.opt.gravity, [0.0, 0.0, -9.81])
                self.assertEqual(cfg.sim.mujoco.timestep, 0.002)

    def test_training_resets_are_explicit_and_do_not_randomize_physics(self):
        for stage, settings in self.settings["stages"].items():
            with self.subTest(stage=stage):
                cfg = rs06.make_cfg(1, seed=42, stage=stage)
                self.assertTrue(cfg.auto_reset)
                self.assertEqual(set(cfg.events), {"reset_base", "reset_robot_joints"})
                self.assertEqual(cfg.events, self.reference.events)
                self.assertTrue(all(event.mode == "reset" for event in cfg.events.values()))
                self.assertFalse(cfg.observations["actor"].enable_corruption)
                self.assertEqual(cfg.episode_length_s, settings["episode_seconds"])
                command = cfg.commands["twist"]
                self.assertEqual(command.ranges.lin_vel_x, (self.settings["speed_m_s"],) * 2)
                self.assertEqual(command.ranges.lin_vel_y, (0.0, 0.0))
                self.assertEqual(command.rel_standing_envs, 0.0)
                self.assertGreater(command.resampling_time_range[0], cfg.episode_length_s)
                self.assertEqual(command.ranges.heading, (0.0, 0.0))

    def test_stage_terrain_rows_and_limits_follow_saved_configuration(self):
        for stage, settings in self.settings["stages"].items():
            with self.subTest(stage=stage):
                cfg = rs06.make_cfg(1, seed=42, stage=stage)
                if stage == "flat":
                    self.assertEqual(settings["terrain_rows"], 0)
                    self.assert_same_terrain(cfg.scene.terrain, self.reference.scene.terrain)
                    self.assertEqual(cfg.curriculum, {})
                    continue
                generator = cfg.scene.terrain.terrain_generator
                self.assertIs(
                    cfg.curriculum["terrain"].func,
                    rs06.mixed_levels if stage == "mixed" else rs06.adaptation_levels,
                )
                self.assertEqual(generator.num_rows, settings["terrain_rows"])
                self.assertEqual(generator.num_cols, 1)
                self.assertEqual(
                    cfg.scene.terrain.max_init_terrain_level, settings.get("max_initial_level", 0)
                )
                self.assertTrue(generator.curriculum)
                course = replace(generator.sub_terrains["course"], size=generator.size)
                self.assertEqual(course.min_rise, settings["min_rise_m"])
                self.assertEqual(course.max_rise, settings["max_rise_m"])
                for difficulty in np.linspace(0.0, 1.0, settings["terrain_rows"]):
                    spec = mujoco.MjSpec()
                    spec.worldbody.add_body(name="terrain")
                    output = course.function(difficulty, spec, np.random.default_rng(42))
                    rise = settings["min_rise_m"] + difficulty * (
                        settings["max_rise_m"] - settings["min_rise_m"]
                    )
                    # Tile prefixes guarantee global name uniqueness when
                    # the generator appends several courses to one MjSpec.
                    geoms = [item.geom for item in output.geometries]
                    np.testing.assert_array_equal(output.origin, [1.0, 1.5, 0.0])
                    if rise == 0.0:
                        self.assertEqual(len(geoms), 1)
                        np.testing.assert_array_equal(geoms[0].pos, [5.5, 1.5, -0.05])
                        np.testing.assert_array_equal(geoms[0].size, [5.5, 1.5, 0.05])
                        np.testing.assert_array_equal(geoms[0].friction, FRICTION)
                        continue
                    first = next(g for g in geoms if g.name.endswith("_up_1"))
                    plateau = next(g for g in geoms if g.name.endswith("_plateau"))
                    strip = next(g for g in geoms if g.name.endswith("_up_1_strip_up"))
                    self.assertAlmostEqual(first.pos[2] + first.size[2], rise)
                    self.assertAlmostEqual(first.size[0] * 2, 0.32)
                    self.assertAlmostEqual(first.size[1] * 2, 1.6)
                    self.assertAlmostEqual(plateau.pos[2] + plateau.size[2], 10 * rise)
                    np.testing.assert_allclose(strip.size, [0.03, 0.8, 0.0025])
                    np.testing.assert_array_equal(first.friction, [1.0, 0.005, 0.0001])
                    self.assertEqual(len(output.geometries), 40)

    def test_mixed_compiled_rows_match_flat_and_direct_stair_geometry(self):
        cfg = rs06.make_cfg(256, seed=42, stage="mixed")
        self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 6)
        self.assertEqual(cfg.episode_length_s, 70.0)
        self.assertEqual(cfg.rewards["gait_contact"].weight, 1.5)
        self.assertEqual(cfg.rewards["swing_obstacle"].weight, -1.0)
        self.assertEqual(cfg.rewards["pose"].weight, 0.5)
        generator = TerrainGenerator(cfg.scene.terrain.terrain_generator, device="cpu")
        spec = mujoco.MjSpec()
        generator.compile(spec)
        model = spec.compile()
        self.assertEqual(model.ngeom, 1 + 9 * 40)
        names = [model.geom(i).name for i in range(model.ngeom)]
        self.assertEqual(len(set(names)), model.ngeom)
        self.assertEqual(generator.terrain_origins.shape, (10, 1, 3))
        np.testing.assert_allclose(np.diff(generator.terrain_origins[:, 0, 0]), 11.0)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        fields = ("quat", "friction", "contype", "conaffinity", "solref", "solimp", "margin", "gap")
        for row, cm in ((0, 0), (1, 2), (9, 18)):
            with self.subTest(row=row, rise_cm=cm):
                origin = generator.terrain_origins[row, 0]
                if cm:
                    terrain = Terrain("stairs", cm)
                    direct_spec = mujoco.MjSpec()
                    direct_spec.worldbody.add_body(name="terrain")
                    add_to_spec(direct_spec, terrain)
                    direct = direct_spec.compile()
                    prefix = f"course_tile_{1 + (row - 1) * 40}_"
                    for box in terrain.iter_boxes():
                        actual = model.geom(prefix + box.name)
                        expected = direct.geom("course_" + box.name)
                        np.testing.assert_allclose(
                            actual.pos - origin, expected.pos, rtol=0.0, atol=1e-12
                        )
                        np.testing.assert_allclose(actual.size, expected.size, rtol=0.0, atol=1e-12)
                        for field in fields:
                            np.testing.assert_array_equal(
                                getattr(actual, field), getattr(expected, field)
                            )
                else:
                    terrain = Terrain("flat")
                    # Only its floor occupies the flat tile, including the
                    # spawn, stair location and full exit distance.
                    ids = np.flatnonzero(np.abs(model.geom_pos[:, 0] - (origin[0] + 4.5)) < 5.5)
                    self.assertEqual(len(ids), 1)
                    floor = model.geom(int(ids[0]))
                    np.testing.assert_allclose(floor.pos - origin, [4.5, 0.0, -0.05])
                    np.testing.assert_array_equal(floor.size, [5.5, 1.5, 0.05])
                    np.testing.assert_array_equal(floor.friction, FRICTION)
                    self.assertEqual(floor.contype, 1)
                    self.assertEqual(floor.conaffinity, 1)
                # Avoid exact riser edges, where floating translations can
                # make equality choose either adjacent surface.
                for x in (-0.5, 0.0, 0.7501, 1.1, 3.7, 4.8, 7.499, 8.8):
                    for y in (0.0, 0.7):
                        ray_origin = origin + [x, y, 3.0]
                        distance = mujoco.mj_ray(
                            model,
                            data,
                            ray_origin,
                            np.array([0.0, 0.0, -1.0]),
                            None,
                            1,
                            -1,
                            np.array([-1], dtype=np.int32),
                        )
                        self.assertAlmostEqual(3.0 - distance, terrain.height(x, y), places=10)

    def curriculum_env(self, positions, failed, common_step_counter=100):
        class Scene(dict):
            pass

        terrain = SimpleNamespace(terrain_levels=torch.ones(len(positions), dtype=torch.long))

        def update(env_ids, up, down):
            terrain.last_update = (env_ids.clone(), up.clone(), down.clone())
            terrain.terrain_levels[env_ids] += up.long() - down.long()

        terrain.update_env_origins = update
        scene = Scene(
            robot=SimpleNamespace(
                data=SimpleNamespace(root_link_pos_w=torch.tensor(positions, dtype=torch.float32))
            )
        )
        scene.env_origins = torch.zeros(len(positions), 3)
        scene.terrain = terrain
        return SimpleNamespace(
            scene=scene,
            common_step_counter=common_step_counter,
            termination_manager=SimpleNamespace(terminated=torch.tensor(failed, dtype=torch.bool)),
        )

    def test_training_curriculum_demotes_midcourse_failures_and_blocks_failed_promotion(self):
        env = self.curriculum_env(
            [[2.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [True, True, False, False, False],
        )
        rs06.adaptation_levels(env, torch.arange(5))
        _, up, down = env.scene.terrain.last_update
        self.assertEqual(up.tolist(), [False, False, True, False, False])
        self.assertEqual(down.tolist(), [True, True, False, False, True])

    def test_training_curriculum_handles_subset_origins_and_initial_reset(self):
        env = self.curriculum_env(
            [[12.0, 0.0, 0.0], [28.0, 0.1, 0.0], [38.0, 0.8, 0.0]], [True, False, False]
        )
        env.scene.env_origins[:, 0] = torch.tensor([10.0, 20.0, 30.0])
        subset = torch.tensor([1, 2])
        rs06.adaptation_levels(env, subset)
        ids, up, down = env.scene.terrain.last_update
        self.assertEqual(ids.tolist(), [1, 2])
        self.assertEqual(up.tolist(), [True, False])
        self.assertEqual(down.tolist(), [False, False])
        self.assertEqual(env.scene.terrain.terrain_levels[0], 1)
        env.common_step_counter = 0
        rs06.adaptation_levels(env, torch.arange(3))
        _, up, down = env.scene.terrain.last_update
        self.assertFalse(bool(up.any()))
        self.assertFalse(bool(down.any()))

    def test_mixed_replay_preserves_fixed_rows_and_origins_after_promotions(self):
        class Scene(dict):
            @property
            def env_origins(self):
                return self.terrain.env_origins

        n = 12
        terrain = SimpleNamespace(
            terrain_levels=torch.full((n,), 6, dtype=torch.long),
            terrain_types=torch.zeros(n, dtype=torch.long),
            terrain_origins=torch.zeros(10, 1, 3),
        )
        terrain.terrain_origins[:, 0, 0] = torch.arange(10) * 11.0
        terrain.env_origins = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ].clone()

        def update(ids, up, down):
            terrain.terrain_levels[ids] = (
                terrain.terrain_levels[ids] + up.long() - down.long()
            ).clamp(0, 9)
            terrain.env_origins[ids] = terrain.terrain_origins[
                terrain.terrain_levels[ids], terrain.terrain_types[ids]
            ]

        terrain.update_env_origins = update
        scene = Scene(
            robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=terrain.env_origins.clone()))
        )
        scene.terrain = terrain
        env = SimpleNamespace(
            scene=scene,
            common_step_counter=0,
            termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
        )
        ids = torch.arange(n)
        rs06.mixed_levels(env, ids)
        self.assertEqual(terrain.terrain_levels.tolist(), [0, 1, 6, 6, 0, 2, 6, 6, 0, 3, 6, 6])
        torch.testing.assert_close(
            terrain.env_origins,
            terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
        )
        env.common_step_counter = 100
        for _ in range(3):
            scene["robot"].data.root_link_pos_w = terrain.env_origins + torch.tensor(
                [8.0, 0.0, 0.0]
            )
            state = rs06.mixed_levels(env, ids)
        self.assertEqual(terrain.terrain_levels.tolist(), [0, 1, 9, 9, 0, 2, 9, 9, 0, 3, 9, 9])
        torch.testing.assert_close(
            terrain.env_origins,
            terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
        )
        self.assertEqual(state["max"], 9)
        self.assertAlmostEqual(float(state["mean"]), float(terrain.terrain_levels.float().mean()))
        # Failing reset subsets restore fixed low rows, demote adaptive rows,
        # and leave every non-reset environment's origin untouched.
        before_levels = terrain.terrain_levels.clone()
        before_origins = terrain.env_origins.clone()
        subset = torch.tensor([1, 2, 5, 9])
        env.termination_manager.terminated[:] = True
        scene["robot"].data.root_link_pos_w = terrain.env_origins + torch.tensor([2.0, 0.0, 0.0])
        rs06.mixed_levels(env, subset)
        self.assertEqual(terrain.terrain_levels[subset].tolist(), [1, 8, 2, 3])
        unchanged = torch.tensor([0, 3, 4, 6, 7, 8, 10, 11])
        torch.testing.assert_close(terrain.terrain_levels[unchanged], before_levels[unchanged])
        torch.testing.assert_close(terrain.env_origins[unchanged], before_origins[unchanged])
        torch.testing.assert_close(
            terrain.env_origins,
            terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
        )

    def test_initialization_failure_preserves_source_and_inputs_before_gpu_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "dummy.pt"
            checkpoint.write_bytes(b"dummy checkpoint; must never reach torch.load")
            output = folder / "failed_training"
            args = SimpleNamespace(
                output=output,
                checkpoint=checkpoint,
                num_envs=1,
                seed=42,
                stage="low",
                std=0.15,
                learning_rate=0.0003,
                iterations=1,
                normalizer_pseudocount=1000000,
            )

            def fail_initialization(cfg, *, device):
                self.assertTrue(cfg.auto_reset)
                self.assertEqual(device, "cuda:0")
                # This callback replaces environment creation itself. Evidence
                # must already exist before it can fail; no CUDA work occurs.
                early = json.loads((output / "run.json").read_text())
                for name in ("rl/rs06.py", "simulation/model.py"):
                    content = (output / "source_snapshot" / name).read_bytes()
                    self.assertEqual(
                        hashlib.sha256(content).hexdigest(), early["source_snapshot_sha256"][name]
                    )
                self.assertIn("rs06_quadruped.urdf", early["model_input_sha256"])
                self.assertIn("rs06_training.json", early["model_input_sha256"])
                for name, expected in early["model_input_sha256"].items():
                    content = (output / "model_inputs" / name).read_bytes()
                    self.assertEqual(hashlib.sha256(content).hexdigest(), expected)
                raise RuntimeError("simulated initialization failure")

            with (
                patch.object(
                    rs06.base, "ManagerBasedRlEnv", side_effect=fail_initialization
                ) as create_env,
                patch.object(rs06.torch, "load") as load_checkpoint,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated initialization failure"):
                    rs06.train(args)
            create_env.assert_called_once()
            load_checkpoint.assert_not_called()
            result = json.loads((output / "run.json").read_text())
            self.assertIs(result["completed"], False)
            self.assertIn("simulated initialization failure", result["error"])
            self.assertGreaterEqual(result["elapsed_seconds"], 0.0)
            self.assertIs(result["hardware_pass"], None)
            self.assertEqual(
                checkpoint.read_bytes(), b"dummy checkpoint; must never reach torch.load"
            )


if __name__ == "__main__":
    unittest.main()
