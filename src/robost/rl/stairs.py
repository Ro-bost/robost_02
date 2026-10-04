"""Terrain-aware PPO fine tuning; ideal simulated height sensing, not hardware validation."""

from dataclasses import dataclass

import robost.rl.base as flat
from robost.simulation.terrain import Terrain, add_to_spec, FRICTION
from robost.simulation.hardware import JOINT_NAMES
import mujoco
import numpy as np
import torch
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainOutput,
    TerrainGeometry,
)


@dataclass
class CourseCfg(SubTerrainCfg):
    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        # A generator appends every tile to this one body and does not rename
        # named geoms. Reserve a prefix before adding this tile's forty boxes.
        name_prefix = f"course_tile_{len(body.geoms)}_"
        # Training varies only rise up to the measured 18 cm. Tread dimensions,
        # ten-riser flights, strips and friction match the evaluation scene.
        terrain = Terrain("stairs", (0.02 + 0.16 * difficulty) * 100)
        if self.size[0] < terrain.exit_x + 2.0:
            raise ValueError("terrain tile is too short for the complete stair course and exit")
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[self.size[0] / 2, self.size[1] / 2, -0.05],
            size=[self.size[0] / 2, self.size[1] / 2, 0.05],
            mass=0.0,
            group=0,
            contype=1,
            conaffinity=1,
            friction=FRICTION,
        )
        geoms = [floor] + add_to_spec(
            spec, terrain, offset=(1.0, self.size[1] / 2, 0.0), name_prefix=name_prefix
        )
        return TerrainOutput(
            origin=np.array([1.0, self.size[1] / 2, 0.0]),
            geometries=[TerrainGeometry(geom=geom) for geom in geoms],
        )


def terrain_height_reward(env):
    # Median under-body scan, NOT absolute world height on raised terrain.
    heights = flat.env_mdp.height_scan(env, "body_height")
    return torch.exp(
        -((heights.median(dim=1).values - flat.get_hardware().standing_height) / 0.08).square()
    )


def course_bounds(env):
    relative = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    return (relative[:, 1].abs() > 0.65) | (relative[:, 0] < -0.5)


def course_end(env):
    return (
        env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
        > Terrain().exit_x + 1.0
    )


def planar_tracking(env):
    # Ascending/descending necessarily has nonzero vertical velocity. Do not
    # punish that as a failure to obey a horizontal speed command.
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    command = env.command_manager.get_command("twist")[:, :2]
    return torch.exp(-(velocity - command).square().sum(-1) / 0.25**2)


def course_alignment(env):
    robot = env.scene["robot"]
    lateral = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    quat = robot.data.root_link_quat_w
    yaw = torch.atan2(
        2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    return lateral.square() + yaw.square()


def levels(env, env_ids):
    terrain = env.scene.terrain
    relative = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    up = (relative[env_ids, 0] > Terrain().exit_x + 0.2) & (relative[env_ids, 1].abs() < 0.65)
    down = relative[env_ids, 0] < 1.0
    if env.common_step_counter == 0:
        up[:] = False
        down[:] = False
    terrain.update_env_origins(env_ids, up, down & ~up)
    return {"mean": terrain.terrain_levels.float().mean(), "max": terrain.terrain_levels.max()}


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = flat.make_cfg(num_envs, seed, evaluate)
    # Restrict requested angles, not just motor torque. MuJoCo soft constraints
    # still allow dynamic overshoot, which remains a measured evaluation failure.
    cfg.actions["joint_pos"].clip = {
        name: (float(low) + 0.03, float(high) - 0.03)
        for name, (low, high) in zip(JOINT_NAMES, flat.get_hardware().soft_joint_ranges)
    }
    cfg.rewards["dof_pos_limits"].weight = -5.0
    scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    body_scan = RayCastSensorCfg(
        name="body_height",
        frame=ObjRef(type="body", name="base", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    cfg.scene.sensors += (scan, body_scan)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["height_scan"] = ObservationTermCfg(
            func=flat.env_mdp.height_scan, params={"sensor_name": "terrain_scan"}, scale=0.2
        )
    cfg.rewards["base_height"] = RewardTermCfg(func=terrain_height_reward, weight=0.5)
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(func=planar_tracking, weight=4.0)
    cfg.rewards["foot_clearance"].params["target_height"] = 0.10
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.10
    if not evaluate:
        cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            max_init_terrain_level=1,
            terrain_generator=TerrainGeneratorCfg(
                seed=seed,
                size=(11.0, 3.0),
                num_rows=10,
                curriculum=True,
                sub_terrains={"course": CourseCfg()},
            ),
        )
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.03, 0.03),
            "yaw": (-0.03, 0.03),
        }
        command = cfg.commands["twist"]
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.15, 0.35)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.resampling_time_range = (90.0, 90.0)
        cfg.episode_length_s = 70.0
        cfg.curriculum = {"terrain": CurriculumTermCfg(func=levels)}
        cfg.rewards["course_alignment"] = RewardTermCfg(func=course_alignment, weight=-1.0)
        cfg.terminations["course_bounds"] = TerminationTermCfg(func=course_bounds)
        cfg.terminations["course_end"] = TerminationTermCfg(func=course_end, time_out=True)
    return cfg
