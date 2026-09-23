"""Render saved physics states at 1x; this does not rerun or drive simulation."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def render(folder):
    target = folder / 'evaluation_1x.mp4'
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((folder / 'evaluation.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(folder / 'scene.mjb'))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 1.8
    camera.azimuth = 100
    camera.elevation = -18
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    with mujoco.Renderer(model, 480, 800) as renderer, imageio.get_writer(str(target), fps=25) as writer:
        for qpos, row in zip(np.load(folder / 'qpos_env0.npy'), result['log_env0']):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            camera.lookat[:] = qpos[:3]
            renderer.update_scene(data, camera=camera, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 800, 35), fill='black')
            draw.text((8, 8), f"RS02 {result['stairs_cm']}cm | recorded physics 1x | t={row['t']:.2f}s | "
                      f"target {row['command']}m/s | trial alive={row['alive']}", fill='white')
            writer.append_data(np.asarray(frame))
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    render(parser.parse_args().folder)
