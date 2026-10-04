#!/usr/bin/env python3
"""Source-backed RS06 v5 URDF MuJoCo preview."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco


from robost.simulation.hardware import JOINT_NAMES, LEGS, get_hardware


def robot_spec(payload: str = "nominal", *, floating: bool = True) -> mujoco.MjSpec:
    """Import the supplied URDF without changing inertias, ranges or shapes.

    URDF visual/collision names are qualified by their link to avoid duplicate
    names in v5. Collision groups only control display; masks remain enabled.
    Motor armature uses the supplied nominal value. CPU simulations can apply
    hardware.update_reflected_inertia for the angle-dependent knee inertia.
    """
    hardware = get_hardware(payload)
    root = ET.parse(hardware.urdf).getroot()
    options = root.find("mujoco")
    if options is None:
        options = ET.SubElement(root, "mujoco")
    compiler = options.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(options, "compiler")
    compiler.attrib.update(
        meshdir=str(hardware.urdf.parent / "meshes"),
        strippath="true",
        discardvisual="false",
        fusestatic="false",
        balanceinertia="false",
    )
    for link in root.findall("link"):
        for role in ("collision", "visual"):
            for index, geom in enumerate(link.findall(role)):
                suffix = geom.get("name", str(index))
                geom.set("name", f"{link.get('name')}_{role}_{suffix}")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    base = spec.body("base")
    if floating:
        base.pos = (0.0, 0.0, hardware.standing_height)
        base.add_joint(name="root", type=mujoco.mjtJoint.mjJNT_FREE)
    for name, armature in zip(JOINT_NAMES, hardware.armatures):
        spec.joint(name).armature = armature
    for geom in spec.geoms:
        if "_collision_" in geom.name:
            geom.group = 3
            geom.contype = 1
            geom.conaffinity = 1
            # Nominal friction; source proposal 0.6–1.0.
            geom.friction = (1.0, 0.005, 0.0001)
        else:
            geom.group = 1
            geom.contype = 0
            geom.conaffinity = 0
    for leg in LEGS:
        spec.body(leg + "_foot").add_site(name=leg, size=[0.005, 0, 0], group=5)
    imu = spec.body("imu_link")
    imu.add_site(name="imu", size=[0.005, 0, 0], group=5)
    for name, kind in (
        ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="base",
    )
    spec.option.timestep = 0.002
    spec.option.gravity = (0.0, 0.0, -9.81)
    return spec


def load_model(
    pose_name: str = "stand", payload: str = "nominal"
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a fixed-base preview with all supplied visual and fixed links."""
    hardware = get_hardware(payload)
    model = robot_spec(payload, floating=False).compile()
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.65, 0.65, 0.65)
    model.vis.headlight.specular[:] = (0.1, 0.1, 0.1)
    data = mujoco.MjData(model)
    if pose_name == "stand":
        pose = hardware.standing_joint_positions
    elif pose_name == "cad":
        angle = -95.083
        pose = {
            f"{leg}_{joint}_joint": value
            for leg in LEGS
            for joint, value in (("hip", 0.0), ("thigh", 0.0), ("calf", math.radians(angle)))
        }
    else:
        raise ValueError(f"지원하지 않는 자세입니다: {pose_name}")
    for joint_name, value in pose.items():
        data.qpos[model.joint(joint_name).qposadr] = value
    mujoco.mj_forward(model, data)
    return model, data


def print_summary(model: mujoco.MjModel, pose_name: str, payload: str = "nominal") -> None:
    hardware = get_hardware(payload)
    print(f"MuJoCo {mujoco.__version__}")
    print(f"URDF: {hardware.urdf}")
    print(
        f"로드 결과: body={model.nbody}, joint={model.njnt}, "
        f"geom={model.ngeom}, mesh={model.nmesh}, mass={model.body_mass.sum():.6f} kg"
    )
    print(f"표시 자세: {pose_name}; payload={payload}")


def configure_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (0.0, 0.0, -0.18)
    camera.distance = 1.25
    camera.azimuth = 135.0
    camera.elevation = -18.0


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, output: Path) -> None:
    from PIL import Image

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    configure_camera(camera)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = (0, 1, 1, 0, 0, 0)
    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    Image.fromarray(renderer.render()).save(output)
    renderer.close()
    print(f"스냅샷 저장: {output}")


def show_viewer(model: mujoco.MjModel, data: mujoco.MjData, duration: float = 0) -> None:
    import mujoco.viewer

    print("PNG와 동일한 외형·카메라로 표시합니다. 이 모드는 정적 미리보기입니다.")
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    start = time.monotonic()
    try:
        with viewer.lock():
            configure_camera(viewer.cam)
            viewer.opt.geomgroup[:] = (0, 1, 1, 0, 0, 0)
        while viewer.is_running():
            viewer.sync()
            if duration > 0 and time.monotonic() - start >= duration:
                break
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        time.sleep(0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("nominal", "5kg"), default="nominal")
    parser.add_argument(
        "--check",
        action="store_true",
        help="URDF 로드와 수치만 확인하고 GUI를 열지 않음",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="GUI 대신 지정 경로에 960x720 PNG를 렌더링",
    )
    parser.add_argument(
        "--pose",
        choices=("cad", "stand"),
        default="stand",
        help="stand: PNG와 같은 서기 자세(기본값), cad: F3D/STEP 저장 자세",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="미리보기 자동 종료 시간(초); 0이면 창을 닫을 때까지 유지",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data = load_model(args.pose, args.payload)
    print_summary(model, args.pose, args.payload)
    if args.check:
        return
    if args.snapshot is not None:
        save_snapshot(model, data, args.snapshot.resolve())
        return
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("GUI 표시용 DISPLAY 또는 WAYLAND_DISPLAY가 없습니다.")
    show_viewer(model, data, args.duration)


if __name__ == "__main__":
    main()
