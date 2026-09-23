#!/usr/bin/env python3
"""RS02 URDF를 현재 폴더 기준으로 MuJoCo에 로드하고 표시한다.

원본 생성 스크립트의 Windows 절대경로를 사용하지 않는다. URDF에 MuJoCo
compiler 옵션을 임시로 삽입해 visual STL과 fixed link를 보존한 뒤, 문서에
기록된 기본 서기 자세를 적용한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import mujoco


from robost_paths import ROOT, PACKAGE
URDF = PACKAGE / "rs02_quadruped.urdf"
POSE = PACKAGE / "docs" / "standing_pose.json"
SUMMARY = PACKAGE / "urdf_summary.json"


def load_model(pose_name: str = "stand") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Visual mesh와 fixed body를 보존하며 URDF를 읽는다."""
    source = URDF.read_text(encoding="utf-8")
    compiler = (
        '<mujoco><compiler meshdir="meshes" strippath="true" '
        'discardvisual="false" fusestatic="false" '
        'balanceinertia="false"/></mujoco>'
    )
    marker = '<robot name="rs02_quadruped">'
    if marker not in source:
        raise RuntimeError(f"예상한 robot 태그를 찾지 못했습니다: {URDF}")
    patched = source.replace(marker, marker + compiler, 1)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".urdf",
            prefix=".rs02_mujoco_",
            dir=PACKAGE,
            delete=False,
        ) as stream:
            stream.write(patched)
            temp_path = Path(stream.name)
        model = mujoco.MjModel.from_xml_path(str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    palette = {
        "base": (0.62, 0.66, 0.67, 1.0),
        "hip": (0.11, 0.42, 0.52, 1.0),
        "thigh": (0.45, 0.50, 0.53, 1.0),
        "calf": (0.76, 0.35, 0.15, 1.0),
        "foot": (0.12, 0.12, 0.12, 1.0),
    }
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            part = "base" if body_name == "base" else body_name.split("_", 1)[1]
            model.geom_rgba[geom_id] = palette[part]
        elif body_name.endswith("_foot"):
            model.geom_group[geom_id] = 1
            model.geom_rgba[geom_id] = palette["foot"]
        else:
            model.geom_group[geom_id] = 3

    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.65, 0.65, 0.65)
    model.vis.headlight.specular[:] = (0.1, 0.1, 0.1)

    data = mujoco.MjData(model)
    if pose_name == "stand":
        pose = json.loads(POSE.read_text(encoding="utf-8"))["joints"]
    elif pose_name == "cad":
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        pose = {}
        for leg, values in summary["legs"].items():
            pose[f"{leg}_hip_joint"] = 0.0
            pose[f"{leg}_thigh_joint"] = 0.0
            pose[f"{leg}_calf_joint"] = math.radians(values["knee_cad_deg"])
    else:
        raise ValueError(f"지원하지 않는 자세입니다: {pose_name}")
    for joint_name, value in pose.items():
        joint_id = model.joint(joint_name).id
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)
    return model, data


def print_summary(model: mujoco.MjModel, pose_name: str) -> None:
    total_mass = float(model.body_mass.sum())
    print(f"MuJoCo {mujoco.__version__}")
    print(f"URDF: {URDF}")
    print(
        "로드 결과: "
        f"body={model.nbody}, joint={model.njnt}, "
        f"geom={model.ngeom}, mesh={model.nmesh}, mass={total_mass:.4f} kg"
    )
    if pose_name == "cad":
        print("표시 자세: CAD 저장 자세 (hip=0, thigh=0, calf=-95.338 deg)")
    else:
        print("표시 자세: 서기 자세 (hip=0, thigh=0.68022, calf=-1.3664 rad)")


def configure_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (0.0, 0.0, -0.18)
    camera.distance = 1.25
    camera.azimuth = 135.0
    camera.elevation = -18.0


def save_snapshot(
    model: mujoco.MjModel, data: mujoco.MjData, output: Path
) -> None:
    from PIL import Image

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
    parser.add_argument('--duration', type=float, default=0,
                        help='미리보기 자동 종료 시간(초); 0이면 창을 닫을 때까지 유지')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not URDF.is_file() or not POSE.is_file():
        raise FileNotFoundError("RS02 URDF 패키지를 현재 폴더에서 찾지 못했습니다.")
    model, data = load_model(args.pose)
    print_summary(model, args.pose)
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
