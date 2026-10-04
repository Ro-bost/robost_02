#!/usr/bin/env python3
"""Run the preserved RS02 policies without importing current RS06 code."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / 'src'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('WANDB_MODE', 'disabled')

POLICIES = {
    15: 'd878602c32f4ff41ae9bc601143c6ed2635a64cede417bba6e062031a3ebd52f',
    20: '0c3201204a2086961bfc0803387c3d25e98f0199ba30b003356cfc898bf2e264',
}


def load_adapter():
    # Relocate the historical model; assets/docs is a local relative symlink.
    import rs02_rl as base
    base.PACKAGE = HERE / 'assets'
    import rs02_rl_rhythm as rhythm
    return base, rhythm


def check_inputs():
    import mujoco
    import numpy as np
    from mjlab.entity import Entity

    base, rhythm = load_adapter()
    model = Entity(base.robot_cfg()).spec.compile()
    if not np.isclose(model.body_mass.sum(), 16.940606, atol=1e-8):
        raise RuntimeError('Unexpected RS02 robot mass')
    for height, expected in POLICIES.items():
        checkpoint = HERE / 'policies' / f'{height}cm.pt'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Checkpoint hash mismatch: {checkpoint}')
        # Compare the original source generator to the map extracted from a
        # successful evaluation MJB. No current RS06 geometry is referenced.
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name='terrain')
        base.add_stair_course(spec, height / 100)
        generated = spec.compile()
        preserved = mujoco.MjModel.from_xml_path(str(HERE / 'maps' / f'stairs_{height}cm.xml'))
        for index in range(9):
            a, b = generated.geom(f'course_{index}'), preserved.geom(f'course_{index}')
            for field in ('pos', 'size', 'quat', 'friction', 'contype', 'conaffinity'):
                np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=0, atol=1e-14)
        cfg = rhythm.make_cfg(1, evaluate=True)
        if cfg.auto_reset or cfg.decimation != 10 or cfg.sim.mujoco.timestep != .002:
            raise RuntimeError('Historical evaluation configuration changed')
    print(json.dumps({'robot_mass_kg': float(model.body_mass.sum()),
                      'joints': model.nv - 6, 'actuators': model.nu,
                      'verified_maps_cm': list(POLICIES),
                      'policy_sha256': POLICIES, 'auto_reset': False}, indent=2))


def smoke(args, base, rhythm):
    import numpy as np
    import torch
    cfg = rhythm.make_cfg(1, args.seed, evaluate=True)
    cfg.scene.spec_fn = lambda spec: base.add_stair_course(spec, args.height / 100)
    cfg.auto_reset = False
    cfg.episode_length_s = args.duration + 10
    cmd = cfg.commands['twist']
    cmd.rel_standing_envs = 0.
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.ranges.lin_vel_x = (args.speed, args.speed)
    cmd.ranges.lin_vel_y = (0., 0.)
    if not cmd.heading_command:
        cmd.ranges.ang_vel_z = (0., 0.)
    raw = base.ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env = base.RslRlVecEnvWrapper(raw, clip_actions=3.)
        runner = base.MjlabOnPolicyRunner(env, asdict(base.runner_cfg()), device='cuda:0')
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True, map_location='cuda:0')
        policy = runner.get_inference_policy(device='cuda:0')
        obs = env.get_observations()
        first_failure = None
        peak = np.zeros(12)
        for index in range(math.ceil(args.duration / raw.step_dt)):
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs, stochastic_output=False))
            if not all(bool(torch.isfinite(value).all()) for value in obs.values()):
                raise RuntimeError('Non-finite observations')
            peak = np.maximum(peak, raw.scene['robot'].data.qfrc_actuator[0].abs().cpu().numpy())
            if bool(done.any()):
                first_failure = {'time_s': (index + 1) * raw.step_dt,
                    'terms': [name for name in raw.termination_manager.active_terms
                              if bool(raw.termination_manager.get_term(name)[0])]}
                break
        result = {'height_cm': args.height, 'policy_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  'steps': index + 1, 'duration_s': (index + 1) * raw.step_dt,
                  'finite': True, 'first_failure': first_failure, 'auto_reset': False,
                  'peak_joint_torque_nm': peak.tolist(),
                  'final_base_xyz_m': raw.scene['robot'].data.root_link_pos_w[0].cpu().tolist(),
                  'course_pass': None, 'hardware_pass': None,
                  'purpose': 'policy-load and short rollout smoke only, not a complete-course result'}
        print(json.dumps(result, indent=2))
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / 'smoke.json').write_text(json.dumps(result, indent=2))
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'smoke', 'evaluate', 'train'))
    parser.add_argument('--height', type=int, choices=(15, 20), default=15)
    parser.add_argument('--checkpoint', type=Path, help='Custom checkpoint for a new experiment')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--initial-level', type=int, choices=range(10), default=4)
    parser.add_argument('--std', type=float, default=.15)
    parser.add_argument('--clean-training', action='store_true')
    args = parser.parse_args()
    args.duration = args.duration if args.duration is not None else (2. if args.mode == 'smoke' else 40.)
    if not all(math.isfinite(value) and value > 0 for value in (args.duration, args.speed, args.std)):
        parser.error('duration, speed and std must be finite and positive')
    if args.num_envs < 1 or args.iterations < 1 or args.std > 1:
        parser.error('Counts must be positive and std must be at most 1')
    if args.mode == 'evaluate' and args.duration < 3.02:
        parser.error('Historical evaluation metrics require duration >= 3.02 seconds')
    if args.mode in ('evaluate', 'train') and args.output is None:
        parser.error('--output is required for evaluate/train')
    if args.output is not None:
        args.output = args.output.resolve()
        if args.output.is_relative_to(HERE):
            parser.error('Write experiment outputs outside the preserved RS02 package')
        if args.output.exists():
            parser.error('Output already exists; choose a new directory')
    if args.checkpoint is None:
        args.checkpoint = HERE / 'policies' / f'{args.height}cm.pt'
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != POLICIES[args.height]:
            parser.error('Preserved checkpoint hash mismatch')
    else:
        args.checkpoint = args.checkpoint.resolve()
    if args.mode == 'check':
        check_inputs()
        return
    base, rhythm = load_adapter()
    base.torch.set_num_threads(4)
    args.stairs_cm = args.height
    args.stop_go = False
    args.sample_actions = False
    args.noise_scale = 1.
    args.play_seconds = 0.
    args.policy_adapter = HERE / 'src' / 'rs02_rl_rhythm.py'
    if args.mode == 'train':
        rhythm.gait.train(args, cfg_factory=rhythm.make_cfg)
    elif args.mode == 'evaluate':
        args.num_envs = 1
        args.evaluation_purpose = 'preserved RS02 rhythm policy, deterministic; no external support'
        base.evaluate(args, cfg_factory=rhythm.make_cfg)
    else:
        smoke(args, base, rhythm)


if __name__ == '__main__':
    main()
