"""Support-relative stair PPO with observable course direction.

Uses mjlab's standard heading/world velocity command. Global route state is
ideal simulated localization, not a claim that the real robot has an estimator.
All commands still go through the learned joint policy and bounded actuators.
"""
import torch
import robost.rl.gait as gait
import robost.rl.support as support
from mjlab.managers.observation_manager import ObservationTermCfg


def course_observation(env):
    robot = env.scene['robot']
    heading = robot.data.heading_w
    lateral = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.stack((heading.sin(), heading.cos(), lateral.clamp(-1., 1.)), dim=-1)


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = support.make_cfg(num_envs, seed, evaluate)
    for group in ('actor', 'critic'):
        cfg.observations[group].terms['course_direction'] = ObservationTermCfg(func=course_observation)
    command = cfg.commands['twist']
    command.heading_command = True
    command.ranges.heading = (0., 0.)
    command.rel_heading_envs = 1.
    command.rel_world_envs = 1.
    command.heading_control_stiffness = 1.5
    command.ranges.ang_vel_z = (-.5, .5)
    if not evaluate:
        command.ranges.lin_vel_x = (.20, .30)
        cfg.rewards['course_alignment'].weight = -3.
    return cfg


if __name__ == '__main__':
    gait.main(make_cfg, __file__)
