"""Staged RS06 policy adaptation with frozen nominal hardware and strict evaluation.

Training may reset failed episodes to collect PPO samples. Evaluation always
uses the existing one-episode, first-failure gate and never auto-resets.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg, TerrainGeometry, TerrainOutput

from robost.paths import CONFIG, RS06_PACKAGE
from robost.rl import base, rhythm, stairs
from robost.simulation.terrain import FRICTION, Terrain

SETTINGS = json.loads((CONFIG / "rs06_training.json").read_text())


@dataclass
class AdaptationCourseCfg(stairs.CourseCfg):
    min_rise: float = 0.02
    max_rise: float = 0.18

    def function(self, difficulty, spec, rng):
        rise = self.min_rise + (self.max_rise - self.min_rise) * difficulty
        if rise == 0.0:
            # A zero-rise curriculum row is truly flat: retaining the stair
            # strips would create twenty 5 mm obstacles and change its purpose.
            if self.size[0] < Terrain().exit_x + 2.0:
                raise ValueError("terrain tile is too short for the complete stair course and exit")
            floor = spec.body("terrain").add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[self.size[0] / 2, self.size[1] / 2, -0.05],
                size=[self.size[0] / 2, self.size[1] / 2, 0.05],
                mass=0.0,
                group=0,
                contype=1,
                conaffinity=1,
                friction=FRICTION,
            )
            return TerrainOutput(
                origin=np.array([1.0, self.size[1] / 2, 0.0]),
                geometries=[TerrainGeometry(geom=floor)],
            )
        return super().function((rise - 0.02) / 0.16, spec, rng)


def adaptation_levels(env, env_ids):
    """Adjust training difficulty using the episode state before its reset.

    mjlab computes terminations before _reset_idx, and _reset_idx invokes the
    curriculum before resetting the scene or termination manager. ``terminated``
    therefore identifies hard failures, independently from training timeouts.
    Crossing this training distance is a curriculum signal, not an evaluation
    pass: only StairExitTracker checks the individual exit landings and hold.
    """
    terrain = env.scene.terrain
    relative = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    failed = env.termination_manager.terminated[env_ids]
    reached_distance = (relative[env_ids, 0] > Terrain().exit_x + 0.2) & (
        relative[env_ids, 1].abs() < 0.65
    )
    up = reached_distance & ~failed
    down = (failed | (relative[env_ids, 0] < 1.0)) & ~up
    if env.common_step_counter == 0:
        up[:] = False
        down[:] = False
    terrain.update_env_origins(env_ids, up, down)
    return {"mean": terrain.terrain_levels.float().mean(), "max": terrain.terrain_levels.max()}


def mixed_levels(env, env_ids):
    """Retain 25% flat and 25% low-course replay while 50% adapts.

    A fixed initial mix alone disappears as successful episodes are promoted.
    Pin the replay assignments after each curriculum update, including the
    initial reset. Only the reset subset changes; each new origin must match
    its assigned row before mjlab resets the robot there.
    """
    adaptation_levels(env, env_ids)
    terrain = env.scene.terrain
    flat_ids = env_ids[env_ids % 4 == 0]
    low_ids = env_ids[env_ids % 4 == 1]
    terrain.terrain_levels[flat_ids] = 0
    terrain.terrain_levels[low_ids] = (low_ids // 4) % 3 + 1
    replay_ids = env_ids[(env_ids % 4) < 2]
    terrain.env_origins[replay_ids] = terrain.terrain_origins[
        terrain.terrain_levels[replay_ids], terrain.terrain_types[replay_ids]
    ]
    return {"mean": terrain.terrain_levels.float().mean(), "max": terrain.terrain_levels.max()}


def make_cfg(num_envs=256, seed=42, evaluate=False, *, stage="flat"):
    # Start from evaluation physics: no mass/COM/friction/actuator randomization.
    # This also keeps the old policy's observation order and actuator mapping.
    cfg = rhythm.make_cfg(num_envs, seed, evaluate=True)
    if evaluate:
        return cfg
    settings = SETTINGS["stages"][stage]
    cfg.auto_reset = True  # training samples only; never the completion evaluator
    # Multiple terrain tiles create more initialization constraints. This is
    # array capacity only; no contact pair or solver setting is removed.
    cfg.sim.njmax = 1024
    cfg.episode_length_s = settings["episode_seconds"]
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (SETTINGS["speed_m_s"],) * 2
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.resampling_time_range = (1000.0, 1000.0)
    cfg.rewards["gait_contact"].weight = settings["gait_contact_weight"]
    cfg.rewards["swing_obstacle"].weight = settings["swing_obstacle_weight"]
    cfg.rewards["pose"].weight = settings["pose_weight"]
    cfg.rewards["course_alignment"] = RewardTermCfg(func=stairs.course_alignment, weight=-3.0)
    if stage != "flat":
        course = AdaptationCourseCfg(
            min_rise=settings["min_rise_m"], max_rise=settings["max_rise_m"]
        )
        cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            max_init_terrain_level=settings.get("max_initial_level", 0),
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(11.0, 3.0),
                num_rows=settings["terrain_rows"],
                num_cols=1,
                curriculum=True,
                sub_terrains={"course": course},
            ),
        )
        cfg.curriculum = {
            "terrain": CurriculumTermCfg(
                func=mixed_levels if stage == "mixed" else adaptation_levels
            )
        }
        cfg.terminations["course_bounds"] = TerminationTermCfg(func=stairs.course_bounds)
        cfg.terminations["course_end"] = TerminationTermCfg(func=stairs.course_end, time_out=True)
    return cfg


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(
        vars(args),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        settings=SETTINGS,
        training_auto_reset=True,
        evaluation_auto_reset=False,
        initialization="actor+critic including normalization; fresh optimizer and exploration std",
        hardware_pass=None,
    )
    raw_env = None
    try:
        with (
            (args.output / "training.log").open("x", buffering=1) as log,
            contextlib.redirect_stdout(log),
        ):
            cfg = make_cfg(args.num_envs, args.seed, stage=args.stage)
            agent = base.runner_cfg()
            agent.seed = args.seed
            agent.num_steps_per_env = SETTINGS["num_steps_per_env"]
            agent.save_interval = SETTINGS["save_interval"]
            agent.actor.distribution_cfg.update(init_std=args.std, learn_std=False)
            agent.algorithm.entropy_coef = 0.0
            agent.algorithm.learning_rate = args.learning_rate
            agent.algorithm.schedule = SETTINGS["learning_rate_schedule"]
            # Import the lazy model builder before capturing source, so even an
            # initialization failure retains its exact inputs and code.
            from robost.simulation import model as _model_builder  # noqa: F401

            hardware = base.get_hardware()
            snapshot = base.local_source_snapshot()
            # Running with `python -m` registers this adapter as __main__.
            snapshot[Path(__file__).resolve().relative_to(base.SOURCE)] = Path(
                __file__
            ).read_bytes()
            record["source_snapshot_sha256"] = {
                str(name): hashlib.sha256(value).hexdigest() for name, value in snapshot.items()
            }
            for name, value in snapshot.items():
                target = args.output / "source_snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
            inputs = [
                hardware.urdf,
                RS06_PACKAGE / "actuator_params.yaml",
                RS06_PACKAGE / "knee_linkage.csv",
                RS06_PACKAGE / "urdf_summary.json",
                CONFIG / "courses.json",
                CONFIG / "rs06_training.json",
            ]
            (args.output / "model_inputs").mkdir()
            record["model_input_sha256"] = {}
            for source in inputs:
                value = source.read_bytes()
                (args.output / "model_inputs" / source.name).write_bytes(value)
                record["model_input_sha256"][source.name] = hashlib.sha256(value).hexdigest()
            (args.output / "run.json").write_text(json.dumps(record, default=str, indent=2))
            raw_env = base.ManagerBasedRlEnv(cfg, device="cuda:0")
            model = raw_env.sim.mj_model
            robot = raw_env.scene["robot"]
            if abs(float(model.body_mass.sum()) - hardware.mass_kg) > 1e-5:
                raise RuntimeError("Unexpected training hardware mass")
            record.update(
                mass_kg=float(model.body_mass.sum()),
                joint_order=list(robot.joint_names),
                gravity_m_s2=model.opt.gravity.tolist(),
                physics_dt_s=float(model.opt.timestep),
                control_dt_s=raw_env.step_dt,
                urdf_sha256=hashlib.sha256(hardware.urdf.read_bytes()).hexdigest(),
                terrain_stage=args.stage,
                controller="unchanged bounded PD60/60/80, damping2",
                constraint_capacity=cfg.sim.njmax,
                reward_weights={key: value.weight for key, value in cfg.rewards.items()},
            )
            env = base.RslRlVecEnvWrapper(raw_env, clip_actions=3.0)
            runner = base.MjlabOnPolicyRunner(env, asdict(agent), str(args.output), device="cuda:0")
            checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
            for name in ("actor", "critic"):
                module = getattr(runner.alg, name)
                module.load_state_dict(checkpoint[name + "_state_dict"], strict=True)
            record["normalizer_counts_before"] = {
                name: float(getattr(runner.alg, name).obs_normalizer.count)
                for name in ("actor", "critic")
            }
            with torch.no_grad():
                runner.alg.actor.distribution.std_param.fill_(args.std)
                # Keep the loaded mean/variance and initial action mapping, but
                # let the new robot's observations influence running statistics.
                for name in ("actor", "critic"):
                    getattr(runner.alg, name).obs_normalizer.count.clamp_(
                        max=args.normalizer_pseudocount
                    )
            record["normalizer_counts_start"] = {
                name: float(getattr(runner.alg, name).obs_normalizer.count)
                for name in ("actor", "critic")
            }
            record["learning_rate_schedule"] = agent.algorithm.schedule
            record["runner_config"] = asdict(agent)
            (args.output / "run.json").write_text(json.dumps(record, default=str, indent=2))
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            record["output_checkpoint_sha256"] = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(args.output.glob("model_*.pt"))
            }
            record["completed"] = True
            record["elapsed_seconds"] = time.monotonic() - started
    except BaseException as error:
        record.update(
            completed=False, error=repr(error), elapsed_seconds=time.monotonic() - started
        )
        raise
    finally:
        try:
            if raw_env is not None:
                raw_env.close()
        except Exception as error:
            record["close_error"] = repr(error)
            raise
        finally:
            (args.output / "run.json").write_text(json.dumps(record, default=str, indent=2))
    print("Training complete:", args.output)
