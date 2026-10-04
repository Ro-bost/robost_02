"""Source-backed hardware values and RS06 four-bar transmission calculations.

Joint order is FR, FL, RR, RL, each hip/thigh/calf.  The supplied URDF is the
authority for mass, limits and damping; the CSV is the authority for knee ratio.
The linear torque-speed envelope is an estimate from actuator_params.yaml, not
a measured motor curve.  No randomization is applied by this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robost.paths import RS06_PACKAGE

LEGS = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf"))


@dataclass(frozen=True)
class Hardware:
    robot: str
    payload: str
    urdf: Path
    mass_kg: float
    standing_height: float
    foot_radius: float
    standing_joint_positions: dict[str, float]
    joint_ranges: np.ndarray
    soft_joint_ranges: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray
    armatures: np.ndarray
    rated_motor_torques: np.ndarray

    def knee_ratio(self, q):
        """Signed d(motor output angle)/d(knee angle), from CAD lookup table."""
        angles, ratios = _knee_table()
        return np.interp(q, angles, ratios)

    def joint_velocity_limits(self, q):
        q = np.asarray(q, dtype=float)
        limits = np.broadcast_to(self.velocity_limits, q.shape).copy()
        limits[..., 2::3] = (480 * 2 * np.pi / 60) / np.abs(self.knee_ratio(q[..., 2::3]))
        return limits

    def reflected_armatures(self, q):
        q = np.asarray(q, dtype=float)
        values = np.broadcast_to(self.armatures, q.shape).copy()
        values[..., 2::3] = 0.012 * self.knee_ratio(q[..., 2::3]) ** 2
        return values

    def torque_envelope(self, q, qvel):
        """Conservative estimated motoring envelope, capped at URDF effort.

        This also bounds braking by the motoring envelope; regenerative braking
        has not been measured. The 30 Nm knee structural cap always applies.
        """
        q = np.asarray(q, dtype=float)
        speed_fraction = np.maximum(0.0, 1.0 - np.abs(qvel) / self.joint_velocity_limits(q))
        peak = np.broadcast_to(self.effort_limits, q.shape).copy()
        peak[..., 2::3] = 36.0 * np.abs(self.knee_ratio(q[..., 2::3]))
        return np.minimum(self.effort_limits, peak * speed_fraction)


@lru_cache(maxsize=1)
def _knee_table():
    with (RS06_PACKAGE / "knee_linkage.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    angles = np.array([float(row["q_calf_rad"]) for row in rows])[::-1]
    ratios = np.array([float(row["ratio"]) for row in rows])[::-1]
    return angles, ratios


@lru_cache(maxsize=2)
def get_hardware(payload: str = "nominal") -> Hardware:
    """Read the current RS06 nominal or supplied 5 kg comparison model."""
    if payload not in ("nominal", "5kg"):
        raise ValueError(f"Unsupported RS06 payload: {payload!r}")
    urdf = RS06_PACKAGE / (
        "rs06_quadruped.urdf" if payload == "nominal" else "rs06_quadruped_payload5kg.urdf"
    )
    summary = json.loads((RS06_PACKAGE / "urdf_summary.json").read_text())
    pose = {
        f"{leg}_{part}_joint": angle
        for leg in LEGS
        for part, angle in summary["legs"][leg]["stand_rad"].items()
    }
    height, radius = summary["base_height_stand_m"], summary["legs"]["FR"]["foot_radius"]
    armatures, rated = [0.0042, 0.012, 0.02628], [6.0, 11.0, 11.0]
    root = ET.parse(urdf).getroot()
    mass = sum(float(item.attrib["value"]) for item in root.findall("link/inertial/mass"))
    joints = {item.attrib["name"]: item for item in root.findall("joint")}
    ranges, soft_ranges, effort, velocity = [], [], [], []
    for name in JOINT_NAMES:
        joint = joints[name]
        limit = joint.find("limit").attrib
        hard = [float(limit["lower"]), float(limit["upper"])]
        ranges.append(hard)
        soft = joint.find("safety_controller")
        soft_ranges.append(
            hard
            if soft is None
            else [float(soft.attrib["soft_lower_limit"]), float(soft.attrib["soft_upper_limit"])]
        )
        effort.append(float(limit["effort"]))
        velocity.append(float(limit["velocity"]))
    return Hardware(
        "rs06",
        payload,
        urdf,
        mass,
        height,
        radius,
        pose,
        np.array(ranges),
        np.array(soft_ranges),
        np.array(effort),
        np.array(velocity),
        np.tile(armatures, 4),
        np.tile(rated, 4),
    )


def update_reflected_inertia(model, data, hardware: Hardware | None = None):
    """Update CPU MuJoCo armatures; return the corresponding inertial bias.

    The caller must subtract this 12-joint bias from qfrc_applied *before* its
    next mj_step/mj_forward. For I(q), Lagrange's equation adds .5 I'(q) qdot².
    This helper does not modify existing applied forces or actuator torques.
    GPU batches need their own per-environment implementation.
    """
    hardware = hardware or get_hardware()
    qadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    dadr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
    q, v = data.qpos[qadr], data.qvel[dadr]
    model.dof_armature[dadr] = hardware.reflected_armatures(q)
    epsilon = 1e-6
    derivative = (
        hardware.reflected_armatures(q + epsilon) - hardware.reflected_armatures(q - epsilon)
    ) / (2 * epsilon)
    return 0.5 * derivative * v**2
