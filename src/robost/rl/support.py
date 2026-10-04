"""Stair policy refinement with terrain-relative body pose and explicit stall/limit failures."""

import torch
import robost.rl.gait as gait
from robost.simulation.hardware import get_hardware
from robost.simulation.terrain import Terrain
from mjlab.managers import RewardTermCfg, TerminationTermCfg


def support_height_reward(env):
    robot = env.scene["robot"]
    sites = [robot.site_names.index(leg) for leg in gait.flat.LEGS]
    foot_z = robot.data.site_pos_w[:, sites, 2]
    heights = env.scene["foot_height_scan"].data.heights
    ground = (foot_z - heights).mean(-1)
    relative = robot.data.root_link_pos_w[:, 2] - ground
    return torch.exp(-((relative - get_hardware().standing_height) / 0.08).square())


def range_exceeded(env):
    data = env.scene["robot"].data
    return (
        (data.joint_pos < data.joint_pos_limits[..., 0] - 0.005)
        | (data.joint_pos > data.joint_pos_limits[..., 1] + 0.005)
    ).any(-1)


def stalled(env):
    x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    step = env.episode_length_buf
    if not hasattr(env, "_robost_progress_x"):
        env._robost_progress_x = x.clone()
        env._robost_progress_step = step.clone()
    advanced = (x > env._robost_progress_x + 0.04) | (step <= 1)
    env._robost_progress_x = torch.where(advanced, x, env._robost_progress_x)
    env._robost_progress_step = torch.where(advanced, step, env._robost_progress_step)
    return (step - env._robost_progress_step) * env.step_dt > 3.0


def foot_outside(env, course_start=None, exit_x=None, half_width=None):
    if course_start is None or exit_x is None or half_width is None:
        course = Terrain()
        course_start = course.start if course_start is None else course_start
        exit_x = course.exit_x if exit_x is None else exit_x
        half_width = course.width / 2 if half_width is None else half_width
    robot = env.scene["robot"]
    sites = [robot.site_names.index(leg) for leg in gait.flat.LEGS]
    feet = robot.data.site_pos_w[:, sites] - env.scene.env_origins[:, None, :]
    in_course = (feet[:, :, 0] > course_start) & (feet[:, :, 0] < exit_x)
    return (in_course & (feet[:, :, 1].abs() > half_width)).any(-1)


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = gait.make_cfg(num_envs, seed, evaluate)
    # Restore upstream Go1 rough-terrain attitude reference instead of requiring
    # a horizontal body while front/rear feet stand at different stair levels.
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    cfg.rewards["base_height"] = RewardTermCfg(func=support_height_reward, weight=0.8)
    cfg.rewards["shank_collision"].weight = -2.0
    cfg.actions["joint_pos"].clip = {
        k: (lo + 0.03, hi - 0.03) if "calf" in k else (lo, hi)
        for k, (lo, hi) in cfg.actions["joint_pos"].clip.items()
    }
    cfg.terminations["joint_range"] = TerminationTermCfg(func=range_exceeded)
    cfg.terminations["stalled"] = TerminationTermCfg(func=stalled)
    course = Terrain("stairs")
    cfg.terminations["foot_outside"] = TerminationTermCfg(
        func=foot_outside,
        params={
            "course_start": course.start,
            "exit_x": course.exit_x,
            "half_width": course.width / 2,
        },
    )
    return cfg
