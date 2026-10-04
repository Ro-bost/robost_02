"""Phase-conditioned terrain PPO refinement (no prescribed joint trajectory).

Clock/contact-conditioning design reference: Improbable-AI/walk-these-ways.
PPO/physics/MDP implementation remains mjlab + RSL-RL; this is a local adapter,
not a claim to run the complete Walk These Ways code or its pretrained policy.
"""

from dataclasses import dataclass
import torch
import robost.rl.base as flat
import robost.rl.stairs as stairs
from mjlab.managers import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg, ObjRef


def leg_phases(env):
    phase = env.episode_length_buf.float() * env.step_dt * 1.5
    return torch.remainder(
        phase[:, None] + torch.tensor([0.0, 0.5, 0.5, 0.0], device=env.device), 1.0
    )


def clock_observation(env):
    phase = leg_phases(env)[:, 0] * 2 * torch.pi
    return torch.stack((phase.sin(), phase.cos()), dim=-1)


def phase_contact_reward(env):
    desired = leg_phases(env) < 0.65
    contact = env.scene["feet_ground_contact"].data.found > 0
    return (desired == contact).float().mean(-1)


@dataclass
class AheadPattern(GridPatternCfg):
    def generate_rays(self, mj_model, device):
        offsets, directions = super().generate_rays(mj_model, device)
        offsets[:, 0] += 0.15
        offsets[:, 2] += 0.5
        return offsets, directions


def swing_obstacle_cost(env):
    robot = env.scene["robot"]
    sites = [robot.site_names.index(leg) for leg in flat.LEGS]
    foot_z = robot.data.site_pos_w[:, sites, 2]
    scan = env.scene["swing_scan"].data
    heights = scan.hit_pos_w[:, :, 2].reshape(env.num_envs, 4, -1)
    valid = (scan.distances >= 0).reshape(env.num_envs, 4, -1)
    ahead = torch.where(valid, heights, torch.full_like(heights, -10.0)).max(-1).values
    # Reward clearance above the upcoming tread, only during mid-swing.
    # Raycasts affect rewards, not robot position or externally applied force.
    phase = leg_phases(env)
    swing = ((phase - 0.65) / 0.35).clamp(0.0, 1.0)
    weight = torch.sin(torch.pi * swing).square()
    error = ((ahead + 0.065 - foot_z).clamp(min=0, max=0.3) / 0.1).square()
    return (error * weight).mean(-1)


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = stairs.make_cfg(num_envs, seed, evaluate)
    cfg.scene.sensors += (
        RayCastSensorCfg(
            name="swing_scan",
            frame=tuple(ObjRef(type="site", name=l, entity="robot") for l in flat.LEGS),
            # The calf/site frame can pitch beyond 90 degrees during high swing;
            # extracting its yaw would flip the lookahead backward. This straight
            # course has fixed world +X travel, independent of the foot orientation.
            pattern=AheadPattern(size=(0.3, 0.05), resolution=0.05),
            ray_alignment="world",
            max_distance=2.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
        ),
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["gait_clock"] = ObservationTermCfg(func=clock_observation)
    cfg.rewards["gait_contact"] = RewardTermCfg(func=phase_contact_reward, weight=0.8)
    cfg.rewards["swing_obstacle"] = RewardTermCfg(func=swing_obstacle_cost, weight=-1.0)
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0
    cfg.rewards["shank_collision"].weight = -1.0
    cfg.rewards["dof_pos_limits"].weight = -10.0
    cfg.actions["joint_pos"].clip = {
        k: (lo + 0.02, hi - 0.02) for k, (lo, hi) in cfg.actions["joint_pos"].clip.items()
    }
    return cfg
