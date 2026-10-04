"""RS02 adapter for upstream mjlab velocity MDP and RSL-RL PPO.

No handwritten footstep state machine, position teleport, or support force.
Use the separate rs02-rl conda environment. Upstream remains unmodified.
"""
from pathlib import Path
import argparse
import json
import hashlib
import math
import os
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import asdict

os.environ.setdefault('MUJOCO_GL', 'glfw' if len(sys.argv)>1 and sys.argv[1]=='play' else 'egl')
os.environ.setdefault('WANDB_MODE', 'disabled')
import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as env_mdp
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch, TerrainHeightSensorCfg, ObjRef
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity import mdp
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'RS02_4족로봇_최종정리_2026-09-15/04_URDF_IsaacSim/rs02_quadruped'
LEGS=('FR','FL','RR','RL')


def robot_spec():
    root=ET.parse(PACKAGE/'docs/mujoco_check_model.xml').getroot()
    root.find('compiler').set('meshdir',str(PACKAGE/'meshes'))
    # Only remove old scene and actuation; inertias, links and collision sizes
    # are preserved. mjlab owns terrain and creates bounded PD actuators.
    for tag in ('actuator','sensor','keyframe','option'):
        old=root.find(tag)
        if old is not None:root.remove(old)
    world=root.find('worldbody')
    for child in list(world):
        if child.tag!='body':world.remove(child)
    base=world.find("body[@name='base']")
    base.set('pos','0 0 .36661')
    for body in root.iter('body'):
        for i,geom in enumerate(body.findall('geom')):
            if geom.get('type')=='mesh':
                geom.set('group','1');geom.set('contype','0');geom.set('conaffinity','0')
            else:
                geom.set('name',body.get('name')+'_collision')
                geom.set('group','3');geom.set('contype','1');geom.set('conaffinity','1')
        if body.get('name','').endswith('_foot'):
            for site in list(body.findall('site')):body.remove(site)
            ET.SubElement(body,'site',name=body.get('name').split('_')[0],size='.005',group='5')
            ET.SubElement(body,'geom',type='sphere',size='.02661',rgba='.08 .08 .08 1',
                group='1',contype='0',conaffinity='0',mass='0')
    ET.SubElement(base,'site',name='imu',size='.005',group='5')
    sensors=ET.SubElement(root,'sensor')
    ET.SubElement(sensors,'gyro',name='imu_ang_vel',site='imu')
    ET.SubElement(sensors,'velocimeter',name='imu_lin_vel',site='imu')
    ET.SubElement(sensors,'subtreeangmom',name='root_angmom',body='base')
    return mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'))


def robot_cfg():
    return EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,.36661),joint_pos={
            '.*_hip_joint':0.,'.*_thigh_joint':.68022,'.*_calf_joint':-1.3664},joint_vel={'.*':0.}),
        articulation=EntityArticulationInfoCfg(actuators=(
            BuiltinPositionActuatorCfg(target_names_expr=('.*_hip_joint','.*_thigh_joint'),
                stiffness=60.,damping=2.,effort_limit=17.,armature=.012),
            BuiltinPositionActuatorCfg(target_names_expr=('.*_calf_joint',),
                stiffness=80.,damping=2.,effort_limit=25.2,armature=.012),
        ),soft_joint_pos_limit_factor=.95),
    )


def motor_cost(env):
    # Effort regularizer; measured output, not PD target. Actuator order is
    # resolved from joint names, not assumed to match FR/FL/RR/RL.
    robot=env.scene['robot']
    return robot.data.qfrc_actuator.square().sum(dim=-1)


def flat_height_reward(env):
    z=env.scene['robot'].data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
    return torch.exp(-((z-.35)/.06).square())


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg=unitree_go1_flat_env_cfg()
    cfg.scene.entities={'robot':robot_cfg()}
    cfg.scene.num_envs=num_envs;cfg.seed=seed
    for sensor in cfg.scene.sensors:
        if isinstance(sensor,TerrainHeightSensorCfg):
            sensor.frame=tuple(ObjRef(type='site',name=l,entity='robot') for l in LEGS)
    bad=ContactSensorCfg(name='body_ground',
        primary=ContactMatch(mode='geom',pattern=('base_collision',)+tuple(l+'_thigh_collision' for l in LEGS),entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='none',num_slots=1,history_length=4)
    shank=ContactSensorCfg(name='shank_ground',
        primary=ContactMatch(mode='geom',pattern=tuple(l+'_calf_collision' for l in LEGS),entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='none',num_slots=1)
    cfg.scene.sensors=cfg.scene.sensors+(bad,shank)
    cfg.viewer.body_name='base';cfg.viewer.distance=2.
    cfg.events['base_com'].params['asset_cfg'].body_names=('base',)
    for name in ('upright','body_ang_vel'):
        cfg.rewards[name].params['asset_cfg'].body_names=('base',)
    cfg.actions['joint_pos'].scale={'.*_hip_joint':.20,'.*_thigh_joint':.35,'.*_calf_joint':.40}
    cfg.sim.mujoco.timestep=.002;cfg.decimation=10  # 500Hz physics / 50Hz policy
    cfg.sim.mujoco.iterations=20;cfg.sim.njmax=400
    cfg.rewards['track_linear_velocity'].params['std']=.25
    cfg.rewards['pose'].weight=.5
    # Early v1 rewarded all four airborne during a collapse. Keep upstream's
    # Go1 setting (zero air-time reward), and explicitly penalize termination.
    cfg.rewards['air_time'].weight=0.
    cfg.rewards['air_time'].params['command_threshold']=.1
    cfg.rewards['foot_clearance'].params['target_height']=.06
    cfg.rewards['foot_swing_height'].params['target_height']=.06
    cfg.rewards['motor_effort']=RewardTermCfg(func=motor_cost,weight=-.0002)
    cfg.rewards['termination']=RewardTermCfg(func=env_mdp.is_terminated,weight=-100.)
    cfg.rewards['base_height']=RewardTermCfg(func=flat_height_reward,weight=.5)
    cfg.rewards['shank_collision']=RewardTermCfg(func=mdp.self_collision_cost,weight=-.2,params={'sensor_name':'shank_ground'})
    cfg.terminations['illegal_contact']=TerminationTermCfg(func=mdp.illegal_contact,params={'sensor_name':'body_ground'})
    cfg.terminations['fell_over'].params['limit_angle']=math.radians(45)
    cmd=cfg.commands['twist'];cmd.heading_command=False;cmd.ranges.heading=None
    cmd.rel_heading_envs=0.;cmd.rel_forward_envs=0.;cmd.rel_standing_envs=.1
    cmd.ranges.lin_vel_x=(0.,.6);cmd.ranges.lin_vel_y=(-.1,.1);cmd.ranges.ang_vel_z=(-.3,.3)
    cfg.curriculum={}
    # Baseline training: no artificial pushes. A later robustness suite must
    # add disturbances explicitly instead of disguising resets as recovery.
    cfg.events.pop('push_robot',None)
    if evaluate:
        cfg.observations['actor'].enable_corruption=False
        for key in tuple(cfg.events):
            if key not in ('reset_base','reset_robot_joints'):cfg.events.pop(key)
        cfg.events['reset_base'].params['pose_range']={'x':(-.02,.02),'y':(-.02,.02),'yaw':(-.03,.03)}
        cfg.episode_length_s=35.
        cfg.auto_reset=False
    return cfg


def runner_cfg():
    cfg=unitree_go1_ppo_runner_cfg()
    cfg.experiment_name='rs02_dynamic_velocity';cfg.logger='tensorboard';cfg.upload_model=False
    cfg.actor.hidden_dims=(256,128,128);cfg.critic.hidden_dims=(256,128,128)
    cfg.actor.distribution_cfg['init_std']=.4
    cfg.save_interval=100;cfg.clip_actions=3.
    return cfg


def add_stair_course(spec, rise):
    """The user's fixed 5-up / 1m landing / 5-down course, not training terrain."""
    body=spec.body('terrain')
    sections=[(.75+(i-1)*.30,.30,i*rise) for i in range(1,5)]
    sections += [(1.95,1.,5*rise)]
    sections += [(2.95+(i-1)*.30,.30,(5-i)*rise) for i in range(1,5)]
    for i,(left,length,top) in enumerate(sections):
        body.add_geom(name=f'course_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[left+length/2,0,top/2],size=[length/2,.8,top/2],
            rgba=[.35,.45,.55,1.],contype=1,conaffinity=1,group=0,
            friction=[1.,.005,.0001])


def smoke(args):
    cfg=make_cfg(args.num_envs,args.seed,evaluate=True)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    robot=env.scene['robot'];model=env.sim.mj_model
    print('RS02 mass:',model.body_mass.sum(),'joints:',robot.joint_names)
    assert abs(model.body_mass.sum()-16.940606)<1e-4
    assert len(robot.joint_names)==12
    obs,_=env.reset();maxforce=0.
    for _ in range(100):
        obs,reward,terminated,truncated,info=env.step(torch.zeros(args.num_envs,12,device='cuda:0'))
        assert all(torch.isfinite(v).all() for v in obs.values())
        maxforce=max(maxforce,float(robot.data.actuator_force.abs().max()))
    result={'mass_kg':float(model.body_mass.sum()),'joint_names':list(robot.joint_names),
        'observation_shapes':{k:list(v.shape) for k,v in obs.items()},
        'finite':True,'standing_z_mean':float(robot.data.root_link_pos_w[:,2].mean()),
        'peak_actuator_force':maxforce,'terminated_count':int(terminated.sum()),
        'self_collision_enabled':True,'hardware_pass':None}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'smoke.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));env.close()


def train(args):
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=make_cfg(args.num_envs,args.seed)
    agent=runner_cfg();agent.seed=args.seed
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=agent.clip_actions)
    runner=MjlabOnPolicyRunner(env,asdict(agent),str(args.output),device='cuda:0')
    if args.checkpoint:runner.load(str(args.checkpoint))
    (args.output/'adapter_source.py').write_text(Path(__file__).read_text())
    (args.output/'run.json').write_text(json.dumps({'num_envs':args.num_envs,'seed':args.seed,'iterations':args.iterations,'framework_commit':'27577db821fe321c819072a851bdda234b89f32d','hardware_pass':None},indent=2))
    runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
    env.close()


def play(args, cfg_factory=make_cfg):
    from mjlab.viewer import NativeMujocoViewer
    cfg=cfg_factory(1,args.seed,evaluate=True)
    if args.stairs_cm:
        cfg.scene.spec_fn=lambda spec:add_stair_course(spec,args.stairs_cm/100)
    cfg.auto_reset=True;cfg.episode_length_s=1e6
    command=cfg.commands['twist'];command.rel_standing_envs=0.
    command.resampling_time_range=(1e6,1e6)
    command.ranges.lin_vel_x=(args.speed,args.speed)
    command.ranges.lin_vel_y=(0.,0.)
    if not command.heading_command:command.ranges.ang_vel_z=(0.,0.)
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device='cuda:0'),clip_actions=3.)
    runner=MjlabOnPolicyRunner(env,asdict(runner_cfg()),device='cuda:0')
    runner.load(str(args.checkpoint),load_cfg={'actor':True},strict=True,map_location='cuda:0')
    policy=runner.get_inference_policy(device='cuda:0')
    print('Live learned policy. Speed:',args.speed,'m/s. GUI auto-resets on failure; use evaluate for pass/fail.')
    play_seconds=getattr(args,'play_seconds',0.)
    previous_threads={t.ident for t in threading.enumerate()}
    try:
        NativeMujocoViewer(env,policy,enable_perturbations=False).run(
            num_steps=round(play_seconds/.02) if play_seconds else None)
    finally:
        # MuJoCo 3.11 passive close signals the render thread but does not join
        # it. Let our viewer finish before Python's glfw.terminate atexit hook.
        for thread in threading.enumerate():
            if thread.ident not in previous_threads and '_launch_internal' in thread.name:
                thread.join(timeout=5.)
    env.close()


def evaluate(args, cfg_factory=make_cfg):
    """Every reset/termination is a failure; never count a later restart as pass."""
    from scipy.spatial.transform import Rotation
    from rs02_course_validation import StairExitTracker
    evaluator_source=Path(__file__).read_bytes()
    validation_source=Path(__file__).with_name('rs02_course_validation.py').read_bytes()
    policy_source=Path(args.policy_adapter).read_bytes() if hasattr(args,'policy_adapter') else None
    # Capture imported local adapter dependencies too (support/route reuse gait).
    source_snapshot={}
    for module_name,module in list(sys.modules.items()):
        source_path=getattr(module,'__file__',None)
        if module_name.startswith('rs02_') and source_path and Path(source_path).parent==ROOT:
            source_snapshot[Path(source_path).name]=Path(source_path).read_bytes()
    cfg=cfg_factory(args.num_envs,args.seed,evaluate=True)
    if args.stairs_cm:
        cfg.scene.spec_fn=lambda spec:add_stair_course(spec,args.stairs_cm/100)
    # Preserve terminal state; failed trials must never restart during evaluation.
    cfg.auto_reset=False;cfg.episode_length_s=args.duration+10
    cmd=cfg.commands['twist'];cmd.rel_standing_envs=0.
    cmd.resampling_time_range=(1e6,1e6)
    cmd.ranges.lin_vel_x=(args.speed,args.speed)
    cmd.ranges.lin_vel_y=(0.,0.)
    if not cmd.heading_command:cmd.ranges.ang_vel_z=(0.,0.)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    agent=runner_cfg();wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),device='cuda:0')
    runner.load(str(args.checkpoint),load_cfg={'actor':True},strict=True,map_location='cuda:0')
    policy=runner.get_inference_policy(device='cuda:0')
    obs=wrapped.get_observations();robot=env.scene['robot']
    if getattr(args,'sample_actions',False):
        with torch.no_grad():policy.distribution.std_param.mul_(getattr(args,'noise_scale',1.))
    alive=np.ones(args.num_envs,dtype=bool);first_failure=np.full(args.num_envs,np.nan)
    failure_reasons=[[] for _ in range(args.num_envs)]
    completed=False;left_course=False;max_x=0.;completed_at=None;max_foot_z=0.
    exit_tracker=StairExitTracker()
    rows=[];poses=[];vels=[];worldvels=[];positions=[];yaws=[];tilts=[];masks=[];forces=[];contacts=[];slips=[];shanks=[];shankforces=[];jointvel=[];margins=[];references=[];times=[]
    sites=[robot.site_names.index(l) for l in LEGS]
    with torch.inference_mode():
        for i in range(round(args.duration/env.step_dt)):
            now=(i+1)*env.step_dt
            reference=0. if args.stop_go and (now<3 or 13<=now<18) else args.speed
            if args.stop_go:
                env.command_manager.get_term('twist').vel_command_b[:,0]=reference
                obs=wrapped.get_observations()
            obs,rew,done,extra=wrapped.step(policy(obs,stochastic_output=getattr(args,'sample_actions',False)))
            ended=done.cpu().numpy().astype(bool)
            for k in np.flatnonzero(alive & ended):
                failure_reasons[k]=[name for name in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(name)[k])]
            first_failure[alive & ended]=now;alive &= ~ended
            v=robot.data.root_link_lin_vel_b.cpu().numpy().copy()
            quat=robot.data.root_link_quat_w.cpu().numpy()
            rpy=Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')
            contact=env.scene['feet_ground_contact'].data.found.cpu().numpy()>0
            if args.stairs_cm and alive[0]:
                feet=robot.data.site_pos_w[:,sites].cpu().numpy()[0]
                max_x=max(max_x,float(robot.data.root_link_pos_w[0,0]))
                max_foot_z=max(max_foot_z,float(np.max(feet[:,2])))
                completed=exit_tracker.update(now,feet,contact[0])
                left_course=exit_tracker.left_course
                completed_at=exit_tracker.completed_at
            elif args.stairs_cm:
                exit_tracker.failed=True
            footv=robot.data.site_vel_w[:,sites,:2].cpu().numpy()
            if now>=3:
                references.append(reference);times.append(now)
                vels.append(v);tilts.append(rpy[:,:2]);masks.append(alive.copy())
                worldvels.append(robot.data.root_link_lin_vel_w.cpu().numpy().copy())
                positions.append(robot.data.root_link_pos_w.cpu().numpy().copy())
                yaws.append(rpy[:,2].copy())
                forces.append(robot.data.qfrc_actuator.cpu().numpy().copy())
                contacts.append(contact.copy());slips.append(np.linalg.norm(footv,axis=-1)*contact)
                shanks.append(env.scene['shank_ground'].data.found.cpu().numpy()>0)
                shankforces.append(env.scene['shank_ground'].data.force.cpu().numpy().copy())
                jointvel.append(robot.data.joint_vel.cpu().numpy().copy())
                jointpos=robot.data.joint_pos.cpu().numpy()
                bounds=np.array([env.sim.mj_model.joint('robot/'+name).range for name in robot.joint_names])
                margins.append(np.minimum(jointpos-bounds[:,0],bounds[:,1]-jointpos))
            if i%2==0 or not alive[0]:
                poses.append(env.sim.data.qpos[0].cpu().numpy().copy())
                rows.append({'t':now,'alive':bool(alive[0]),'command':reference,'vx':float(v[0,0]),'rpy':rpy[0].tolist()})
            if args.stairs_cm and completed:break
            if args.stairs_cm and not alive[0]:break
    vels=np.array(vels);tilts=np.array(tilts);masks=np.array(masks,dtype=bool).reshape(-1,args.num_envs);forces=np.array(forces)
    worldvels=np.array(worldvels);positions=np.array(positions);yaws=np.array(yaws)
    contact=np.array(contacts,dtype=bool).reshape(-1,args.num_envs,4);slips=np.array(slips);shanks=np.array(shanks)
    shankforces=np.linalg.norm(np.array(shankforces),axis=-1)
    jointvel=np.array(jointvel);margins=np.array(margins)
    references=np.array(references);times=np.array(times)
    trials=[]
    for k in range(args.num_envs):
        valid=masks[:,k]
        if valid.any():
            vx=vels[valid,k,0];rp=tilts[valid,k]
            target=references[valid]
            rmse=float(np.sqrt(np.mean((vx-target)**2)))
            avg=float(vx.mean());rms=float(np.rad2deg(np.sqrt(np.mean(rp**2))));peak=float(np.rad2deg(np.max(np.abs(rp))))
            cycles=np.abs(np.diff(contact[valid,k].astype(int),axis=0)).sum(axis=0).tolist()
            slip=float(slips[valid,k].sum()/max(1,contact[valid,k].sum()))
            trms=np.sqrt(np.mean(forces[valid,k]**2,axis=0)).tolist()
            shank_fraction=float(shanks[valid,k].any(axis=-1).mean())
            joint_peak=np.max(np.abs(jointvel[valid,k]),axis=0).tolist()
            margin=float(margins[valid,k].min())
            torque_peak=np.max(np.abs(forces[valid,k]),axis=0).tolist()
            speed_limits=np.array([42.9/1.48 if '_calf_' in n else 42.9 for n in robot.joint_names])
            passed=bool(alive[k] and rmse<=.1 and abs(avg-target.mean())<=.2*args.speed and rms<=8 and peak<=20 and min(cycles)>=4 and slip<=.08 and shank_fraction==0 and margin>=-.005 and np.all(joint_peak<=speed_limits))
        else:rmse=avg=rms=peak=slip=shank_fraction=margin=None;cycles=[];trms=[];joint_peak=[];torque_peak=[];passed=False
        trials.append({'trial':k,'survived':bool(alive[k]),'first_failure_s':None if np.isnan(first_failure[k]) else float(first_failure[k]),'mean_vx':avg,'vx_rmse':rmse,'roll_pitch_rms_deg':rms,'roll_pitch_peak_deg':peak,'foot_contact_transitions':cycles,'stance_foot_slip_m_s':slip,'joint_torque_rms':trms,'shank_contact_fraction':shank_fraction,'joint_limit_min_margin_rad':margin,'joint_speed_peak':joint_peak,'joint_torque_peak':torque_peak,'numerical_gate_pass':passed})
        trials[-1]['shank_force_peak_n']=float(shankforces[valid,k].max()) if valid.any() else None
        trials[-1]['failure_reasons']=failure_reasons[k]
        trials[-1]['shank_force_sum_mean_n']=float(shankforces[valid,k].sum(axis=-1).mean()) if valid.any() else None
        trials[-1]['shank_over_10n_fraction']=float((shankforces[valid,k].max(axis=-1)>10).mean()) if valid.any() else None
        if valid.any():
            foot=contact[valid,k]
            trials[-1]['mean_world_vx']=float(worldvels[valid,k,0].mean())
            trials[-1]['max_abs_yaw_deg']=float(np.rad2deg(np.abs(yaws[valid,k]).max()))
            trials[-1]['max_abs_lateral_m']=float(np.abs(positions[valid,k,1]-float(env.scene.env_origins[k,1])).max())
            trials[-1]['support_count_fraction']=[float((foot.sum(-1)==n).mean()) for n in range(5)]
            trials[-1]['diagonal_pair_only_fraction']=float(((foot[:,0]&foot[:,3]&~foot[:,1]&~foot[:,2])|
                (foot[:,1]&foot[:,2]&~foot[:,0]&~foot[:,3])).mean())
            ideal_ratio=np.array([1.48 if '_calf_' in name else 1. for name in robot.joint_names])
            motor_rms=np.array(trms)/ideal_ratio
            trials[-1]['ideal_motor_torque_rms_nm']=motor_rms.tolist()
            trials[-1]['ideal_motor_rms_over_6nm_rating']=(motor_rms/6.).tolist()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'evaluator_source.py').write_bytes(evaluator_source)
    (args.output/'source_snapshot').mkdir()
    for name,content in source_snapshot.items():(args.output/'source_snapshot'/name).write_bytes(content)
    result={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256(evaluator_source).hexdigest(),'joint_order':list(robot.joint_names),'speed_command':args.speed,'duration':args.duration,'seed':args.seed,'survivors':int(alive.sum()),'trials':trials,'numerical_gate_pass':all(t['numerical_gate_pass'] for t in trials),'visual_review_required':True,'hardware_pass':None,'sample_hz':50,'log_env0':rows}
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2))
    np.save(args.output/'qpos_env0.npy',np.array(poses))
    np.save(args.output/'foot_contacts.npy',contact)
    result['command_profile']='0-3 stop, 3-13 walk, 13-18 stop, 18-end walk' if args.stop_go else 'constant'
    result['actual_duration_s']=now
    result['auto_reset']=False
    result['stochastic_policy']=getattr(args,'sample_actions',False)
    result['noise_scale']=getattr(args,'noise_scale',1.)
    result['source_snapshot_sha256']={name:hashlib.sha256(content).hexdigest() for name,content in source_snapshot.items()}
    result['contact_metric_note']='50Hz samples; shank sensor has one slot per geom. Not a bound on all 500Hz impact peaks.'
    result['motor_metric_note']='6Nm rated torque from supplied readme; calf uses ideal fixed 1.48 ratio. No efficiency, variable linkage ratio, torque-speed or thermal model; not hardware certification.'
    if args.stairs_cm:
        result['stairs_cm']=args.stairs_cm
        result['course_completed']=completed
        result['course_completed_at_s']=completed_at
        result['completion_definition']=exit_tracker.definition
        result['exit_foot_landing_times_s']=exit_tracker.landing_times
        (args.output/'course_validation_source.py').write_bytes(validation_source)
        result['course_validation_sha256']=hashlib.sha256(validation_source).hexdigest()
        result['max_foot_height_m']=max_foot_z
        result['left_course']=left_course
        result['max_x_before_failure']=max_x
        result['numerical_gate_pass']=False  # Flat quality criteria do not certify stairs.
        result['purpose']=getattr(args,'evaluation_purpose','flat-trained policy stair transfer diagnostic, not stair-trained policy')
    if hasattr(args,'policy_adapter'):
        result['policy_adapter']=str(args.policy_adapter)
        result['policy_adapter_sha256']=hashlib.sha256(policy_source).hexdigest()
        (args.output/'policy_adapter_source.py').write_bytes(policy_source)
    if args.stop_go:
        stopped=(times>=15)&(times<18)
        result['settled_stop_max_abs_vx']=float(np.max(np.abs(vels[stopped,:,0])))
        result['numerical_gate_pass']=result['numerical_gate_pass'] and result['settled_stop_max_abs_vx']<=.08
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2))
    mujoco.mj_saveModel(env.sim.mj_model,str(args.output/'scene.mjb'))
    # Render at actual speed. This replays recorded states, never drives physics.
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        model=env.sim.mj_model;data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,480,800)
        camera=mujoco.MjvCamera();camera.distance=1.8;camera.azimuth=100;camera.elevation=-18
        opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0]
        with imageio.get_writer(str(args.output/'evaluation_1x.mp4'),fps=25) as writer:
            for q,row in zip(poses,rows):
                data.qpos[:]=q;mujoco.mj_forward(model,data);camera.lookat[:]=q[:3]
                renderer.update_scene(data,camera=camera,scene_option=opt)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,800,35),fill='black')
                draw.text((8,8),f"RS02 learned policy | 1x replay | t={row['t']:.2f}s | target {row['command']} m/s | trial alive={row['alive']}",fill='white')
                writer.append_data(np.asarray(frame))
        renderer.close()
    print(json.dumps({k:v for k,v in result.items() if k!='log_env0'},indent=2));env.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['smoke','train','evaluate','play'])
    ap.add_argument('--num-envs',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--iterations',type=int,default=500)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--speed',type=float,default=.25)
    ap.add_argument('--duration',type=float,default=30.)
    ap.add_argument('--play-seconds',type=float,default=0.,help='GUI: zero runs until closed; positive limits simulation seconds')
    ap.add_argument('--video',action='store_true')
    ap.add_argument('--stop-go',action='store_true',help='evaluation: stop, walk, stop, restart')
    ap.add_argument('--stairs-cm',type=int,choices=[0,15,20],default=0,help='single-env evaluation of fixed stair course; policy is flat-trained')
    ap.add_argument('--output',type=Path,default=ROOT/'output/rl/smoke')
    args=ap.parse_args()
    if args.num_envs<1 or args.iterations<1:ap.error('num-envs and iterations must be positive')
    if not math.isfinite(args.play_seconds) or args.play_seconds<0:ap.error('play-seconds must be finite and nonnegative')
    if not math.isfinite(args.speed) or not math.isfinite(args.duration) or args.duration<=3:
        ap.error('speed must be finite and duration must exceed 3 seconds')
    torch.set_num_threads(4)
    if args.mode in ('evaluate','play') and not args.checkpoint:ap.error('evaluate/play requires --checkpoint')
    if args.stop_go and args.duration<25:ap.error('stop-go evaluation requires at least 25 seconds')
    if args.stairs_cm and (args.mode!='evaluate' or args.num_envs!=1):ap.error('stair transfer diagnostic requires evaluate --num-envs 1')
    {'smoke':smoke,'train':train,'evaluate':evaluate,'play':play}[args.mode](args)
