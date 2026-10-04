"""Render saved physics states at 1x; this does not rerun or drive simulation."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def load_recording(folder, result):
    """Read matched poses and timestamps; reject silent zip truncation."""
    poses = np.load(folder / "qpos_env0.npy")
    rows = result["log_env0"]
    if (folder / "qpos_time.npy").is_file():
        times = np.load(folder / "qpos_time.npy")
    elif (folder / "telemetry.npz").is_file():
        with np.load(folder / "telemetry.npz") as telemetry:
            times = (
                telemetry["qpos_time"].copy()
                if "qpos_time" in telemetry
                else np.array([row["t"] for row in rows])
            )
    else:
        times = np.array([row["t"] for row in rows])
    if poses.ndim != 2 or times.ndim != 1 or len(poses) != len(rows) or len(poses) != len(times):
        raise ValueError("Recorded poses, log rows and pose timestamps must have matching lengths")
    if not len(times) or not np.isfinite(times).all() or not np.isfinite(poses).all():
        raise ValueError("Recorded poses and timestamps must be nonempty and finite")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Recorded timestamps must be nonnegative and strictly increasing")
    if not np.allclose(times, [row["t"] for row in rows], atol=1e-8, rtol=0):
        raise ValueError("Pose timestamps disagree with the recorded log")
    return poses, rows, times


def replay_indices(times, sample_hz):
    """Hold the latest recorded pose on its original policy-time grid.

    A usual 40 ms pose interval occupies two 50 Hz video frames. The final
    20 ms failure sample occupies one; no interpolated pose is invented.
    """
    if not np.isfinite(sample_hz) or sample_hz <= 0:
        raise ValueError("Replay sample rate must be finite and positive")
    ticks = (np.asarray(times) - times[0]) * sample_hz
    if not np.allclose(ticks, np.rint(ticks), atol=1e-6, rtol=0):
        raise ValueError("Recorded pose times do not fit the reported sample rate")
    frame_ticks = np.arange(round(ticks[-1]) + 1)
    return np.searchsorted(np.rint(ticks), frame_ticks, side="right") - 1


def trial_label(result):
    robot = str(result.get("robot", "RS02")).upper()
    height = result.get("stairs_cm", 0)
    terrain = f"{height}cm stairs" if height else "flat"
    policy = result.get("video_policy_label")
    return f"{robot} | {terrain}" + (f" | {policy}" if policy else "")


def render(folder):
    folder = Path(folder)
    target = folder / "evaluation_1x.mp4"
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((folder / "evaluation.json").read_text())
    poses, rows, times = load_recording(folder, result)
    sample_hz = result.get("sample_hz", 50)
    indices = replay_indices(times, sample_hz)
    label = trial_label(result)
    model = mujoco.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 1.8
    camera.azimuth = 100
    camera.elevation = -18
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    with (
        mujoco.Renderer(model, 480, 800) as renderer,
        imageio.get_writer(str(target), fps=sample_hz) as writer,
    ):
        previous_index = None
        previous_frame = None
        for index in indices:
            if index == previous_index:
                writer.append_data(previous_frame)
                continue
            qpos, row = poses[index], rows[index]
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            camera.lookat[:] = qpos[:3]
            renderer.update_scene(data, camera=camera, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 800, 50), fill="black")
            draw.text((8, 6), label, fill="white")
            draw.text(
                (8, 26),
                f"recorded physics 1x | t={times[index]:.2f}s | "
                f"target {row['command']}m/s | trial alive={row['alive']}",
                fill="white",
            )
            previous_frame = np.asarray(frame)
            previous_index = index
            writer.append_data(previous_frame)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    render(parser.parse_args().folder)
