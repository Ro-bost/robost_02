"""One collision geometry and height model for CPU and GPU stair scenes.

The course repeats the approved side section: 32 cm from the rear riser to
the nosing tip, and 18 cm from the lower tread to the upper nosing top.
The 6 cm nosings are 3 cm thick, raised 5 mm and overhang the riser by 3 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
from robost.paths import CONFIG

import numpy as np


FRICTION = (1.0, 0.005, 0.0001)
COURSE_PROFILES = json.loads((CONFIG / "courses.json").read_text())
STRIP_FRICTION = (
    FRICTION[0] * COURSE_PROFILES["real"]["strip_friction_multiplier"],
    *FRICTION[1:],
)


@dataclass(frozen=True)
class TerrainBox:
    name: str
    left: float
    length: float
    width: float
    bottom: float
    top: float
    strip: bool = False

    @property
    def pos(self):
        return (self.left + self.length / 2, 0.0, (self.bottom + self.top) / 2)

    @property
    def size(self):
        return (self.length / 2, self.width / 2, (self.top - self.bottom) / 2)

    @property
    def friction(self):
        return STRIP_FRICTION if self.strip else FRICTION


class Terrain:
    """Measured stair dimensions in metres.

    Rise is measured from the lower plain tread to the upper nosing top;
    tread depth includes the overhang. Structural pitches subtract the raised
    height and overhang respectively. Height includes the solid nosings.
    """

    def __init__(self, kind="stairs", step_height_cm=18):
        if kind not in ("flat", "stairs"):
            raise ValueError("terrain kind must be flat or stairs")
        if not math.isfinite(step_height_cm) or step_height_cm <= 0:
            raise ValueError("step height must be finite and positive")
        self.kind = kind
        profile = COURSE_PROFILES["real"]
        self.start, self.rise, self.width = (
            profile["start_m"],
            step_height_cm / 100,
            profile["width_m"],
        )
        self.tread, self.count = profile["tread_m"], profile["count"]
        self.strip_height, self.strip_depth = profile["strip_height_m"], profile["strip_depth_m"]
        self.strip_thickness = profile["strip_thickness_m"]
        self.strip_overhang = profile["strip_overhang_m"]
        self.body_rise = self.rise - self.strip_height
        self.riser_pitch = self.tread - self.strip_overhang
        if kind == "stairs" and self.body_rise <= 0:
            raise ValueError("step height must exceed the raised nosing height")
        # Keep the first physical nosing tip at the configured course start.
        self.riser_start = self.start + self.strip_overhang
        self.landing_length = profile["landing_m"]
        self.plateau_start = self.riser_start + (self.count - 1) * self.riser_pitch
        self.descent_start = self.plateau_start + self.landing_length
        self.end = self.descent_start + (self.count - 1) * self.riser_pitch + self.strip_overhang
        self.exit_x = self.end + profile["exit_clearance_m"]
        self.transitions = (
            (
                [
                    (self.riser_start + i * self.riser_pitch, (i + 1) * self.body_rise)
                    for i in range(self.count)
                ]
                + [
                    (
                        self.descent_start + i * self.riser_pitch,
                        (self.count - i - 1) * self.body_rise,
                    )
                    for i in range(self.count)
                ]
            )
            if kind == "stairs"
            else []
        )
        self.sections = (
            (
                [
                    (
                        f"up_{i + 1}",
                        self.riser_start + i * self.riser_pitch,
                        self.riser_pitch,
                        (i + 1) * self.body_rise,
                    )
                    for i in range(self.count - 1)
                ]
                + [
                    (
                        "plateau",
                        self.plateau_start,
                        self.landing_length,
                        self.count * self.body_rise,
                    )
                ]
                + [
                    (
                        f"down_{i + 1}",
                        self.descent_start + i * self.riser_pitch,
                        self.riser_pitch,
                        (self.count - i - 1) * self.body_rise,
                    )
                    for i in range(self.count - 1)
                ]
            )
            if kind == "stairs"
            else []
        )
        self.strips = []
        if kind == "stairs" and self.strip_height:
            for name, left, length, top in self.sections:
                if name.startswith("up_") or name == "plateau":
                    self.strips.append(
                        (
                            name + "_strip_up",
                            left - self.strip_overhang,
                            self.strip_depth,
                            top + self.strip_height,
                        )
                    )
                if name.startswith("down_") or name == "plateau":
                    self.strips.append(
                        (
                            name + "_strip_down",
                            left + length + self.strip_overhang - self.strip_depth,
                            self.strip_depth,
                            top + self.strip_height,
                        )
                    )

    def iter_boxes(self):
        for name, left, length, top in self.sections:
            yield TerrainBox(name, left, length, self.width, 0.0, top)
        for name, left, length, top in self.strips:
            yield TerrainBox(name, left, length, self.width, top - self.strip_thickness, top, True)

    def height(self, x, y=None):
        x = np.asarray(x, dtype=float)
        if y is not None:
            x, y = np.broadcast_arrays(x, np.asarray(y, dtype=float))
        values = np.zeros_like(x)
        # Use the same box footprints as the collision geometry, including the
        # strips. The half-open interval convention resolves shared boundaries.
        for box in self.iter_boxes():
            inside = (x >= box.left) & (x < box.left + box.length)
            if y is not None:
                inside &= np.abs(y) <= box.width / 2
            values = np.where(inside, np.maximum(values, box.top), values)
        return float(values) if values.ndim == 0 else values

    def metadata(self):
        edges = sorted(
            {
                round(edge, 12)
                for box in self.iter_boxes()
                for edge in (box.left, box.left + box.length)
            }
        )
        surface = [
            [left, right, self.height((left + right) / 2)] for left, right in zip(edges, edges[1:])
        ]
        return {
            "preset": "real",
            "rise_m": self.rise,
            "rise_reference": "lower plain tread to upper nosing top",
            "body_rise_m": self.body_rise,
            "tread_depth_m": self.tread,
            "tread_reference": "rear structural riser to front nosing tip",
            "riser_pitch_m": self.riser_pitch,
            "risers_up": self.count,
            "risers_down": self.count,
            "width_m": self.width,
            "start_x_m": self.start,
            "first_riser_x_m": self.riser_start,
            "plateau_start_x_m": self.plateau_start,
            "landing_length_m": self.landing_length,
            "descent_start_x_m": self.descent_start,
            "end_x_m": self.end,
            "exit_x_m": self.exit_x,
            "strip_height_m": self.strip_height,
            "strip_depth_m": self.strip_depth,
            "strip_thickness_m": self.strip_thickness,
            "strip_overhang_m": self.strip_overhang,
            "strip_friction_multiplier": COURSE_PROFILES["real"]["strip_friction_multiplier"],
            "strip_friction": list(STRIP_FRICTION),
            "strip_placement": "upper tread at each riser; both edges of top landing",
            "friction": list(FRICTION),
            "surface_segments_m": surface,
        }


def add_to_spec(
    spec,
    terrain=None,
    *,
    offset=(0.0, 0.0, 0.0),
    terrain_body="terrain",
    contype=1,
    conaffinity=1,
    name_prefix="course_",
):
    """Add solid stair and strip boxes to an existing MjSpec terrain body.

    The caller supplies the floor. Return the geoms for mjlab TerrainGeometry.
    """
    import mujoco

    terrain = Terrain() if terrain is None else terrain
    body = spec.body(terrain_body)
    geoms = []
    for box in terrain.iter_boxes():
        pos = [p + shift for p, shift in zip(box.pos, offset)]
        geoms.append(
            body.add_geom(
                name=name_prefix + box.name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=box.size,
                mass=0.0,
                group=0,
                contype=contype,
                conaffinity=conaffinity,
                friction=box.friction,
                rgba=[0.25, 0.25, 0.25, 1.0] if box.strip else [0.35, 0.45, 0.55, 1.0],
            )
        )
    return geoms
