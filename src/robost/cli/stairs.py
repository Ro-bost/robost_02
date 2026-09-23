#!/usr/bin/env python3
"""Run an evaluated RS02 stair policy in MuJoCo."""
import argparse
import datetime
import math
import os
import importlib
from pathlib import Path

from robost.paths import ROOT
DEFAULT_POLICIES={
    15:('rhythm',ROOT/'checkpoints/rs02_stairs/15cm.pt'),
    20:('rhythm',ROOT/'checkpoints/rs02_stairs/20cm.pt'),
}


def select_policy(height,checkpoint=None,adapter=None):
    if checkpoint is None:
        if adapter is not None:raise ValueError('Custom --adapter requires --checkpoint')
        return DEFAULT_POLICIES[height]
    if adapter is None:raise ValueError('--checkpoint requires its matching --adapter (gait/support/route/rhythm)')
    return adapter,checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stairs-cm',type=int,choices=[15,20],default=15)
    parser.add_argument('--speed',type=float,default=.25,help='Forward command in m/s; evaluated default is 0.25')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--headless',action='store_true',help='Run a scored test and save a 1x video instead of a live window')
    parser.add_argument('--duration',type=float,default=40.)
    parser.add_argument('--play-seconds',type=float,default=0.)
    parser.add_argument('--checkpoint',type=Path,help='Optional override; default selects the evaluated policy for the chosen height')
    parser.add_argument('--adapter',choices=['gait','support','route','rhythm'],
        help='Must match the checkpoint training adapter; custom adapters require --checkpoint')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.speed) or args.speed<=0:parser.error('speed must be finite and positive')
    if not math.isfinite(args.duration) or args.duration<=3:parser.error('duration must exceed 3 seconds')
    if not math.isfinite(args.play_seconds) or args.play_seconds<0:parser.error('play-seconds must be finite and nonnegative')
    try:args.adapter,args.checkpoint=select_policy(args.stairs_cm,args.checkpoint,args.adapter)
    except ValueError as error:parser.error(str(error))
    os.environ.setdefault('MUJOCO_GL','egl' if args.headless else 'glfw')
    import torch
    import robost.rl.base as flat
    adapter=importlib.import_module('robost.rl.'+args.adapter)
    torch.set_num_threads(4)
    if not args.checkpoint.is_file():parser.error(f'Checkpoint missing: {args.checkpoint}')
    args.num_envs=1;args.stop_go=False;args.video=True
    args.sample_actions=False;args.noise_scale=0.
    if args.output is None:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.output=ROOT/f'runs/stairs/{args.stairs_cm}cm_{stamp}'
    print(f'RS02 | {args.stairs_cm}cm | 5 up + landing + 5 down | {args.speed}m/s')
    print('Policy:',args.checkpoint)
    print('Course traversal demo; not certified hardware-safe. GUI resets are not test passes.')
    if args.headless:
        args.policy_adapter=Path(adapter.__file__).resolve()
        args.evaluation_purpose='User repeatable deterministic staircase test'
        flat.evaluate(args,cfg_factory=adapter.make_cfg)
    else:flat.play(args,cfg_factory=adapter.make_cfg)


if __name__=='__main__':main()
