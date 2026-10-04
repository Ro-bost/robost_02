"""CPU MuJoCo RS06 scene preview and bounded standing-hold smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl" if not os.environ.get("DISPLAY") else "glfw")
import mujoco
import numpy as np

from robost.paths import ROOT
from robost.simulation.hardware import get_hardware, JOINT_NAMES
from robost.simulation.model import robot_spec
from robost.simulation.terrain import add_to_spec, Terrain, FRICTION


def build_scene(payload="nominal", terrain=None):
    """Create a floating robot, physical stairs and twelve bounded motors."""
    hardware = get_hardware(payload)
    terrain = Terrain() if terrain is None else terrain
    spec = robot_spec(payload)
    ground = spec.worldbody.add_body(name="terrain")
    ground.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(20.0, 4.0, 0.1),
        friction=FRICTION,
        rgba=(0.7, 0.7, 0.7, 1.0),
    )
    add_to_spec(spec, terrain)
    spec.worldbody.add_light(name="scene_light", pos=(2.0, -2.0, 6.0), dir=(0.0, 0.0, -1.0))
    for name, limit in zip(JOINT_NAMES, hardware.effort_limits):
        spec.add_actuator(
            name=name + "_motor",
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[1.0] + [0.0] * 9,
            ctrllimited=True,
            ctrlrange=(-limit, limit),
            forcelimited=True,
            forcerange=(-limit, limit),
        )
    spec.visual.headlight.ambient[:] = (0.5, 0.5, 0.5)
    spec.visual.headlight.diffuse[:] = (0.6, 0.6, 0.6)
    spec.visual.headlight.specular[:] = (0.1, 0.1, 0.1)
    model = spec.compile()
    data = mujoco.MjData(model)
    for name, angle in hardware.standing_joint_positions.items():
        data.qpos[model.joint(name).qposadr] = angle
    # qpos0 is the URDF reference configuration, not a valid standing pose.
    # Export an explicit initialization key without changing joint references
    # or model physics; external controllers can reset to this same state.
    spec.add_key(name="stand", qpos=data.qpos.copy())
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    return spec, model, data


def hold_step(model, data, hardware):
    qadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    vadr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
    q = data.qpos[qadr]
    desired = np.array([hardware.standing_joint_positions[name] for name in JOINT_NAMES])
    torque = np.tile([60.0, 60.0, 80.0], 4) * (desired - q) - 2.0 * data.qvel[vadr]
    # Fixed URDF effort cap, matching the initial RL transfer diagnostic.
    # The supplied linear speed envelope is exposed separately in hardware.py.
    data.ctrl[:] = np.clip(torque, -hardware.effort_limits, hardware.effort_limits)
    mujoco.mj_step(model, data)


def configure_camera(camera, terrain):
    camera.lookat[:] = ((terrain.start + terrain.end) / 2 - 0.4, 0.0, 0.65)
    camera.distance = 10.0
    camera.azimuth = 110.0
    camera.elevation = -23.0


def snapshot(model, data, terrain, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {output}")
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1440, 900
    camera = mujoco.MjvCamera()
    configure_camera(camera, terrain)
    option = mujoco.MjvOption()
    option.geomgroup[:] = (1, 1, 1, 0, 0, 0)
    with mujoco.Renderer(model, height=900, width=1440) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render()).save(output)


def run_hold(model, data, hardware, duration, viewer=None):
    peak = np.zeros(12)
    square = np.zeros(12)
    first_failure = None
    initial = data.qpos[:3].copy()
    minimum_z = float(data.qpos[2])
    steps = int(np.ceil(duration / model.opt.timestep))
    completed = 0
    for step in range(steps):
        if viewer is not None and not viewer.is_running():
            break
        started = time.monotonic()
        hold_step(model, data, hardware)
        completed += 1
        effort = data.actuator_force.copy()
        peak = np.maximum(peak, np.abs(effort))
        square += effort**2
        minimum_z = min(minimum_z, float(data.qpos[2]))
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        reason = (
            "nonfinite_state" if not finite else "base_below_0.20m" if data.qpos[2] < 0.20 else None
        )
        if reason and first_failure is None:
            first_failure = {"time_s": float(data.time), "reason": reason}
        if not finite:
            break
        if viewer is not None:
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    return {
        "duration_s": float(data.time),
        "steps": completed,
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "initial_base_xyz_m": initial.tolist(),
        "final_base_xyz_m": data.qpos[:3].tolist(),
        "minimum_base_z_m": minimum_z,
        "peak_joint_torque_nm": peak.tolist(),
        "rms_joint_torque_nm": np.sqrt(square / max(completed, 1)).tolist(),
        "first_failure": first_failure,
        "auto_reset": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("nominal", "5kg"), default="nominal")
    parser.add_argument("--check", action="store_true", help="Compile without stepping")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    hardware = get_hardware(args.payload)
    terrain = Terrain()
    spec, model, data = build_scene(args.payload, terrain)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or ROOT / "runs" / hardware.robot / ("scene_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    (output / "scene.xml").write_text(spec.to_xml())
    mujoco.mj_saveModel(model, str(output / "scene.mjb"), None)
    result = {
        "robot": hardware.robot,
        "payload": args.payload,
        "urdf": str(hardware.urdf),
        "urdf_sha256": hashlib.sha256(hardware.urdf.read_bytes()).hexdigest(),
        "mass_kg": float(model.body_mass.sum()),
        "joint_count": 12,
        "mujoco_version": mujoco.__version__,
        "terrain": terrain.metadata(),
        "control": "standing_hold_pd" if not args.check else "compile_only",
        "course_pass": None,
        "walking_pass": None,
        "hardware_pass": None,
        "limitations": [
            "Standing hold does not test locomotion or course completion.",
            f"Nominal fixed knee armature {hardware.armatures[2]:g} kg m²; four-bar dynamics not fully simulated.",
            "Fixed joint effort caps; no measured torque-speed, latency, thermal or compliance model.",
        ],
        "joint_names": list(JOINT_NAMES),
        "effort_limits_nm": hardware.effort_limits.tolist(),
    }
    if not args.check:
        if args.headless:
            result["smoke"] = run_hold(model, data, hardware, args.duration)
        else:
            from mujoco import viewer as mjviewer

            with mjviewer.launch_passive(model, data) as viewer:
                configure_camera(viewer.cam, terrain)
                viewer.opt.geomgroup[:] = (1, 1, 1, 0, 0, 0)
                result["smoke"] = run_hold(model, data, hardware, args.duration, viewer)
    np.save(output / "final_qpos.npy", data.qpos)
    (output / "scene_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if args.snapshot:
        snapshot(model, data, terrain, args.snapshot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
