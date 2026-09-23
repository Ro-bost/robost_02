"""Terrain-aware PPO fine tuning; ideal simulated height sensing, not hardware validation."""
import argparse
import contextlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import robost.rl.base as flat
import mujoco
import numpy as np
import torch
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainOutput, TerrainGeometry


@dataclass
class CourseCfg(SubTerrainCfg):
    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body('terrain')
        rise = .02 + .18 * difficulty
        geometries = []
        sections = [(0., self.size[0], 0.)]
        sections += [(1.75 + (i-1)*.3, .3, i*rise) for i in range(1,5)]
        sections += [(2.95, 1., 5*rise)]
        sections += [(3.95 + (i-1)*.3, .3, (5-i)*rise) for i in range(1,5)]
        for i, (left, length, top) in enumerate(sections):
            halfz = .05 if i == 0 else top/2
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[left+length/2,self.size[1]/2,-.05 if i==0 else top/2],
                size=[length/2,self.size[1]/2 if i==0 else .8,halfz],
                mass=0,group=0,contype=1,conaffinity=1,friction=[1,.005,.0001])
            geometries.append(TerrainGeometry(geom=geom))
        return TerrainOutput(origin=np.array([1.,self.size[1]/2,0.]),geometries=geometries)


def terrain_height_reward(env):
    # Median under-body scan, NOT absolute world height on raised terrain.
    heights=flat.env_mdp.height_scan(env,'body_height')
    return torch.exp(-((heights.median(dim=1).values-.35)/.08).square())


def course_bounds(env):
    relative=env.scene['robot'].data.root_link_pos_w-env.scene.env_origins
    return (relative[:,1].abs()>.65)|(relative[:,0]<-.5)


def course_end(env):
    return env.scene['robot'].data.root_link_pos_w[:,0]-env.scene.env_origins[:,0]>5.3


def planar_tracking(env):
    # Ascending/descending necessarily has nonzero vertical velocity. Do not
    # punish that as a failure to obey a horizontal speed command.
    velocity=env.scene['robot'].data.root_link_lin_vel_b[:,:2]
    command=env.command_manager.get_command('twist')[:,:2]
    return torch.exp(-(velocity-command).square().sum(-1)/.25**2)


def course_alignment(env):
    robot=env.scene['robot']
    lateral=robot.data.root_link_pos_w[:,1]-env.scene.env_origins[:,1]
    quat=robot.data.root_link_quat_w
    yaw=torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
    return lateral.square()+yaw.square()


def levels(env,env_ids):
    terrain=env.scene.terrain
    relative=env.scene['robot'].data.root_link_pos_w-env.scene.env_origins
    up=(relative[env_ids,0]>4.5)&(relative[env_ids,1].abs()<.65)
    down=relative[env_ids,0]<1.
    if env.common_step_counter==0:up[:]=False;down[:]=False
    terrain.update_env_origins(env_ids,up,down&~up)
    return {'mean':terrain.terrain_levels.float().mean(),'max':terrain.terrain_levels.max()}


def make_cfg(num_envs=256,seed=42,evaluate=False):
    cfg=flat.make_cfg(num_envs,seed,evaluate)
    # Restrict requested angles, not just motor torque. MuJoCo soft constraints
    # still allow dynamic overshoot, which remains a measured evaluation failure.
    joints=flat.ET.parse(flat.PACKAGE/'rs02.xml').getroot().iter('joint')
    cfg.actions['joint_pos'].clip={j.get('name'):(float(j.get('range').split()[0])+.03,
        float(j.get('range').split()[1])-.03) for j in joints if j.get('range')}
    cfg.rewards['dof_pos_limits'].weight=-5.
    scan=RayCastSensorCfg(name='terrain_scan',frame=ObjRef(type='body',name='base',entity='robot'),
        ray_alignment='yaw',pattern=GridPatternCfg(size=(1.6,1.),resolution=.1),
        max_distance=5.,exclude_parent_body=True,include_geom_groups=(0,))
    body_scan=RayCastSensorCfg(name='body_height',frame=ObjRef(type='body',name='base',entity='robot'),
        ray_alignment='yaw',pattern=GridPatternCfg(size=(.2,.2),resolution=.1),
        max_distance=5.,exclude_parent_body=True,include_geom_groups=(0,))
    cfg.scene.sensors += (scan,body_scan)
    for group in ('actor','critic'):
        cfg.observations[group].terms['height_scan']=ObservationTermCfg(
            func=flat.env_mdp.height_scan,params={'sensor_name':'terrain_scan'},scale=.2)
    cfg.rewards['base_height']=RewardTermCfg(func=terrain_height_reward,weight=.5)
    cfg.rewards['track_linear_velocity']=RewardTermCfg(func=planar_tracking,weight=4.)
    cfg.rewards['foot_clearance'].params['target_height']=.10
    cfg.rewards['foot_swing_height'].params['target_height']=.10
    if not evaluate:
        cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',max_init_terrain_level=1,
            terrain_generator=TerrainGeneratorCfg(seed=seed,size=(7.,3.),num_rows=10,
                curriculum=True,sub_terrains={'course':CourseCfg()}))
        cfg.events['reset_base'].params['pose_range']={'x':(-.05,.05),'y':(-.03,.03),'yaw':(-.03,.03)}
        command=cfg.commands['twist'];command.rel_standing_envs=0.
        command.ranges.lin_vel_x=(.15,.35);command.ranges.lin_vel_y=(0.,0.)
        command.ranges.ang_vel_z=(0.,0.);command.resampling_time_range=(50.,50.)
        cfg.episode_length_s=40.
        cfg.curriculum={'terrain':CurriculumTermCfg(func=levels)}
        cfg.rewards['course_alignment']=RewardTermCfg(func=course_alignment,weight=-1.)
        cfg.terminations['course_bounds']=TerminationTermCfg(func=course_bounds)
        cfg.terminations['course_end']=TerminationTermCfg(func=course_end,time_out=True)
    return cfg


def expand_state(old,new):
    """Preserve learned proprioception weights; initialize new sensing weights to zero."""
    result={k:v.clone() for k,v in new.items()}
    for key,value in old.items():
        if value.shape==result[key].shape:result[key]=value.clone()
        elif key=='mlp.0.weight':
            result[key].zero_();result[key][:,:value.shape[1]]=value
        elif key in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
            result[key].fill_(0. if key.endswith('_mean') else 1.)
            result[key][:,:value.shape[1]]=value
        else:raise ValueError(f'Unexpected checkpoint mismatch: {key}')
    return result


def train(args):
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'adapter_source.py').write_text(Path(__file__).read_text())
    (args.output/'flat_adapter_source.py').write_text(Path(flat.__file__).read_text())
    (args.output/'run.json').write_text(json.dumps(vars(args),default=str,indent=2))
    # A full training log is retained locally, without uploading telemetry.
    with (args.output/'training.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log):
        cfg=make_cfg(args.num_envs,args.seed)
        cfg.scene.terrain.max_init_terrain_level=args.initial_level
        env=flat.RslRlVecEnvWrapper(flat.ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=3.)
        runner=flat.MjlabOnPolicyRunner(env,asdict(flat.runner_cfg()),str(args.output),device='cuda:0')
        checkpoint=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
        for name in ('actor','critic'):
            module=getattr(runner.alg,name)
            module.load_state_dict(expand_state(checkpoint[name+'_state_dict'],module.state_dict()))
        if args.exploration_std is not None:
            with torch.no_grad():runner.alg.actor.distribution.std_param.fill_(args.exploration_std)
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        env.close()
    print('Training complete:',args.output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','evaluate','play'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=512)
    parser.add_argument('--iterations',type=int,default=500)
    parser.add_argument('--initial-level',type=int,choices=range(10),default=1,
        help='Maximum initial training difficulty row, 0=2cm to 9=20cm')
    parser.add_argument('--exploration-std',type=float,help='Optional fresh exploration std when fine tuning')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--speed',type=float,default=.25)
    parser.add_argument('--duration',type=float,default=40.)
    parser.add_argument('--play-seconds',type=float,default=0.)
    parser.add_argument('--stairs-cm',type=int,choices=[0,2,4,6,8,10,12,14,15,16,18,20],default=15,
        help='Final user courses: 15/20; lower heights are curriculum diagnostics only')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--sample-actions',action='store_true',help='Diagnostic only: retain PPO action sampling during evaluation')
    parser.add_argument('--noise-scale',type=float,default=1.)
    args=parser.parse_args();args.stop_go=False
    if args.num_envs<1 or args.iterations<1:parser.error('Positive counts required')
    if not np.isfinite(args.noise_scale) or args.noise_scale<0:parser.error('Invalid noise-scale')
    if not np.isfinite(args.play_seconds) or args.play_seconds<0:parser.error('Invalid play-seconds')
    if args.exploration_std is not None and (not np.isfinite(args.exploration_std) or args.exploration_std<=0):parser.error('Exploration std must be finite and positive')
    if not np.isfinite(args.speed) or not np.isfinite(args.duration) or args.duration<=3:parser.error('Invalid speed/duration')
    if args.mode=='evaluate' and args.stairs_cm and args.num_envs!=1:parser.error('Stairs evaluation requires --num-envs 1')
    torch.set_num_threads(4)
    if args.mode=='train':train(args)
    elif args.mode=='play':flat.play(args,cfg_factory=make_cfg)
    else:
        args.policy_adapter=Path(__file__).resolve()
        args.evaluation_purpose='terrain-aware PPO evaluation on fixed full staircase; not hardware certification'
        flat.evaluate(args,cfg_factory=make_cfg)
