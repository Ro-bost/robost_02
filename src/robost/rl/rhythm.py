"""Route PPO refinement: stronger swing/contact timing and knee-limit reserve."""

from dataclasses import dataclass
import robost.rl.route as route
from robost.rl.stairs import CourseCfg
from mjlab.managers import RewardTermCfg


def shank_force_cost(env):
    # One sensor slot per calf geom; penalize measured impacts, not only touch.
    force = env.scene["shank_ground"].data.force.norm(dim=-1)
    return ((force - 20.0).clamp(min=0.0, max=500.0) / 100.0).square().mean(-1)


@dataclass
class HighCourseCfg(CourseCfg):
    def function(self, difficulty, spec, rng):
        # Reuse exactly the same geometry generator, now focused on 12--18cm.
        return super().function((0.12 + 0.06 * difficulty - 0.02) / 0.16, spec, rng)


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = route.make_cfg(num_envs, seed, evaluate)
    cfg.rewards["gait_contact"].weight = 3.0
    cfg.rewards["shank_impact"] = RewardTermCfg(func=shank_force_cost, weight=-2.0)
    # Failure replay showed knee extension overshoot, even on the exit floor.
    # Reserve another 0.07 rad at the extension end; do not change hard limits.
    cfg.actions["joint_pos"].clip = {
        name: (low, high - 0.07) if "calf" in name else (low, high)
        for name, (low, high) in cfg.actions["joint_pos"].clip.items()
    }
    if not evaluate:
        cfg.scene.terrain.terrain_generator.sub_terrains = {"course": HighCourseCfg()}
    return cfg
