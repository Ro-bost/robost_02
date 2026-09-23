"""Predictive joint-limit braking through the existing torque-bounded PD motors.

This is a controller, NOT a hard state clamp or a guaranteed safety barrier.
External contact can still overcome available torque; original failures remain.
"""
from dataclasses import asdict, dataclass
import torch
import robost.rl.gait as gait
import robost.rl.rhythm as rhythm
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


def guarded_target(target, position, velocity, limits, kp, clip):
    predicted = position + velocity * .04
    brake = position - (4. / kp) * velocity
    lower = (predicted < limits[..., 0] + .12) & (velocity < 0.)
    upper = (predicted > limits[..., 1] - .12) & (velocity > 0.)
    target = torch.where(lower, torch.maximum(target, brake), target)
    target = torch.where(upper, torch.minimum(target, brake), target)
    return torch.clamp(target, min=clip[..., 0], max=clip[..., 1])


class GuardedJointAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.kp = torch.tensor([80. if 'calf' in name else 60. for name in self.target_names],
                               device=self.device)

    def apply_actions(self):
        data = self._entity.data
        ids = self._target_ids
        target = guarded_target(self._processed_actions - data.encoder_bias[:, ids],
                                data.joint_pos[:, ids], data.joint_vel[:, ids],
                                data.joint_pos_limits[:, ids], self.kp, self._clip)
        self._entity.set_joint_position_target(target, joint_ids=ids)


@dataclass(kw_only=True)
class GuardedJointActionCfg(JointPositionActionCfg):
    def build(self, env):
        return GuardedJointAction(self, env)


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = rhythm.make_cfg(num_envs, seed, evaluate)
    cfg.actions['joint_pos'] = GuardedJointActionCfg(**asdict(cfg.actions['joint_pos']))
    return cfg


if __name__ == '__main__':
    gait.main(make_cfg, __file__)
