"""Render recorded MuJoCo states, with an explicit replay-speed label."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def render(report, playback_speed=4., stride=1):
    report=Path(report);r=json.loads(report.read_text())
    scene=report.with_name(report.name.replace('_report.json','_scene.xml'))
    stem=report.name.removesuffix('_report.json')
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    camera=mujoco.MjvCamera();camera.azimuth=100;camera.elevation=-17;camera.distance=1.8
    renderer=mujoco.Renderer(m,480,800)
    rows=[row for row in r['log'] if 'qpos' in row][::stride]
    if not rows: raise ValueError('report has no recorded qpos frames')
    video=report.with_name(stem+'_replay.mp4')
    dt=float(np.median(np.diff([x['t'] for x in rows]))) if len(rows)>1 else .1
    writer=imageio.get_writer(str(video),fps=playback_speed/dt,codec='libx264',quality=7)
    snapshots=set([0,len(rows)-1])
    for x in (1.95,2.45,3.4):
        snapshots.add(min(range(len(rows)),key=lambda i:abs(rows[i]['xyz'][0]-x)))
    try:
        for i,row in enumerate(rows):
            d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d)
            camera.lookat[:]=d.qpos[:3]
            renderer.update_scene(d,camera=camera,scene_option=option)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,800,42),fill=(15,20,25))
            draw.text((12,6),f"RS02 | {r['terrain']} {r['step_height_cm'] if r['terrain']=='stairs' else ''} cm | recorded physics | {playback_speed:g}x replay",fill='white')
            draw.text((12,23),f"Simulation {row['t']:.1f} s | command {r['command_speed_m_s']:.2f} m/s | {row['phase']} | x={row['xyz'][0]:.2f} m",fill='white')
            writer.append_data(np.asarray(frame))
            if i in snapshots:frame.save(report.with_name(f'{stem}_t{row["t"]:.1f}.png'))
    finally:
        writer.close();renderer.close()
    print(video.resolve())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('report',type=Path)
    ap.add_argument('--playback-speed',type=float,default=4.);ap.add_argument('--stride',type=int,default=1)
    args=ap.parse_args()
    if args.playback_speed<=0 or args.stride<1:ap.error('playback-speed and stride must be positive')
    render(args.report,args.playback_speed,args.stride)
