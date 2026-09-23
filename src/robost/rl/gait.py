"""Phase-conditioned terrain PPO refinement (no prescribed joint trajectory).

Clock/contact-conditioning design reference: Improbable-AI/walk-these-ways.
PPO/physics/MDP implementation remains mjlab + RSL-RL; this is an RS02 adapter,
not a claim to run the complete Walk These Ways code or its pretrained policy.
"""
import argparse
import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
import robost.rl.base as flat
import robost.rl.stairs as stairs
from mjlab.managers import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg, ObjRef


def leg_phases(env):
    phase=env.episode_length_buf.float()*env.step_dt*1.5
    return torch.remainder(phase[:,None]+torch.tensor([0.,.5,.5,0.],device=env.device),1.)


def clock_observation(env):
    phase=leg_phases(env)[:,0]*2*torch.pi
    return torch.stack((phase.sin(),phase.cos()),dim=-1)


def phase_contact_reward(env):
    desired=leg_phases(env)<.65
    contact=env.scene['feet_ground_contact'].data.found>0
    return (desired==contact).float().mean(-1)


@dataclass
class AheadPattern(GridPatternCfg):
    def generate_rays(self,mj_model,device):
        offsets,directions=super().generate_rays(mj_model,device)
        offsets[:,0]+=.15;offsets[:,2]+=.5
        return offsets,directions


def swing_obstacle_cost(env):
    robot=env.scene['robot']
    sites=[robot.site_names.index(leg) for leg in flat.LEGS]
    foot_z=robot.data.site_pos_w[:,sites,2]
    scan=env.scene['swing_scan'].data
    heights=scan.hit_pos_w[:,:,2].reshape(env.num_envs,4,-1)
    valid=(scan.distances>=0).reshape(env.num_envs,4,-1)
    ahead=torch.where(valid,heights,torch.full_like(heights,-10.)).max(-1).values
    # Reward clearance above the upcoming tread, only during mid-swing.
    # Raycasts affect rewards, not robot position or externally applied force.
    phase=leg_phases(env)
    swing=((phase-.65)/.35).clamp(0.,1.)
    weight=torch.sin(torch.pi*swing).square()
    error=((ahead+.065-foot_z).clamp(min=0,max=.3)/.1).square()
    return (error*weight).mean(-1)


def make_cfg(num_envs=256,seed=42,evaluate=False):
    cfg=stairs.make_cfg(num_envs,seed,evaluate)
    cfg.scene.sensors+=(RayCastSensorCfg(name='swing_scan',
        frame=tuple(ObjRef(type='site',name=l,entity='robot') for l in flat.LEGS),
        # The calf/site frame can pitch beyond 90 degrees during high swing;
        # extracting its yaw would flip the lookahead backward. This straight
        # course has fixed world +X travel, independent of the foot orientation.
        pattern=AheadPattern(size=(.3,.05),resolution=.05),ray_alignment='world',
        max_distance=2.,exclude_parent_body=True,include_geom_groups=(0,)),)
    for group in ('actor','critic'):
        cfg.observations[group].terms['gait_clock']=ObservationTermCfg(func=clock_observation)
    cfg.rewards['gait_contact']=RewardTermCfg(func=phase_contact_reward,weight=.8)
    cfg.rewards['swing_obstacle']=RewardTermCfg(func=swing_obstacle_cost,weight=-1.)
    cfg.rewards['foot_clearance'].weight=0.
    cfg.rewards['foot_swing_height'].weight=0.
    cfg.rewards['shank_collision'].weight=-1.
    cfg.rewards['dof_pos_limits'].weight=-10.
    cfg.actions['joint_pos'].clip={k:(lo+.02,hi-.02) for k,(lo,hi) in cfg.actions['joint_pos'].clip.items()}
    return cfg


def train(args,cfg_factory=make_cfg):
    args.output.mkdir(parents=True,exist_ok=False)
    for source in (Path(__file__),Path(stairs.__file__),Path(flat.__file__),Path(args.policy_adapter)):
        (args.output/source.name).write_text(source.read_text())
    (args.output/'run.json').write_text(json.dumps(vars(args),default=str,indent=2))
    with (args.output/'training.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log):
        cfg=cfg_factory(args.num_envs,args.seed)
        cfg.scene.terrain.max_init_terrain_level=args.initial_level
        if args.clean_training:
            cfg.observations['actor'].enable_corruption=False
            for key in tuple(cfg.events):
                if key not in ('reset_base','reset_robot_joints'):cfg.events.pop(key)
        agent=flat.runner_cfg()
        agent.actor.distribution_cfg.update(init_std=args.std,learn_std=False)
        agent.algorithm.entropy_coef=0.
        agent.algorithm.learning_rate=3e-4
        env=flat.RslRlVecEnvWrapper(flat.ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=3.)
        runner=flat.MjlabOnPolicyRunner(env,asdict(agent),str(args.output),device='cuda:0')
        checkpoint=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
        for name in ('actor','critic'):
            module=getattr(runner.alg,name)
            module.load_state_dict(stairs.expand_state(checkpoint[name+'_state_dict'],module.state_dict()))
        with torch.no_grad():runner.alg.actor.distribution.std_param.fill_(args.std)
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        env.close()
    print('Training complete:',args.output)


def main(cfg_factory=make_cfg,policy_adapter=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','evaluate','play'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=512)
    parser.add_argument('--iterations',type=int,default=1500)
    parser.add_argument('--initial-level',type=int,choices=range(10),default=4)
    parser.add_argument('--std',type=float,default=.15)
    parser.add_argument('--clean-training',action='store_true',help='Stage 1: nominal physics and exact state, robustness not claimed')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--speed',type=float,default=.25)
    parser.add_argument('--duration',type=float,default=30.)
    parser.add_argument('--play-seconds',type=float,default=0.)
    parser.add_argument('--stairs-cm',type=int,choices=[0,2,4,6,8,10,12,14,15,16,18,20],default=15)
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args();args.stop_go=False
    args.policy_adapter=Path(policy_adapter or __file__).resolve()
    if args.num_envs<1 or args.iterations<1:parser.error('Positive counts required')
    if not 0.<args.std<=1. or not np.isfinite(args.speed) or not np.isfinite(args.duration) or args.duration<=3:parser.error('Invalid numeric argument')
    if not np.isfinite(args.play_seconds) or args.play_seconds<0:parser.error('Invalid play-seconds')
    if args.mode=='evaluate' and args.stairs_cm and args.num_envs!=1:parser.error('Stairs evaluation requires --num-envs 1')
    torch.set_num_threads(4)
    if args.mode=='train':train(args,cfg_factory)
    elif args.mode=='play':flat.play(args,cfg_factory=cfg_factory)
    else:
        args.evaluation_purpose='phase-conditioned terrain policy, deterministic; no external support'
        flat.evaluate(args,cfg_factory=cfg_factory)


if __name__=='__main__':main()
