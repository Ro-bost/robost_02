"""PD damping probe, retaining original RS02 mass, geometry and torque bounds."""
import robost.rl.gait as gait


def make_cfg(num_envs=256,seed=42,evaluate=False):
    cfg=gait.make_cfg(num_envs,seed,evaluate)
    for actuator in cfg.scene.entities['robot'].articulation.actuators:
        actuator.damping=3.
    return cfg


if __name__=='__main__':gait.main(make_cfg,__file__)
