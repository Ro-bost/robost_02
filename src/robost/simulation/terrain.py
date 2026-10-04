"""One collision geometry and height model for CPU and GPU stair scenes.

The measured course has ten rising and ten falling risers. Anti-slip strips
occupy the upper tread next to each riser: the left edge on ascent and right
edge on descent. Their measured shape is modelled without changing friction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
from robost.paths import CONFIG

import numpy as np


FRICTION = (1.0, 0.005, 0.0001)
COURSE_PROFILES = json.loads((CONFIG / "courses.json").read_text())


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


class Terrain:
    """Measured stair dimensions in metres.

    ``transitions`` describes the structural risers, while ``height`` includes
    the 5 mm raised strips. Passing y to height also accounts for course width.
    The final riser reaches the floor; the top riser reaches the 1 m landing.
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
        self.landing_length = profile["landing_m"]
        self.plateau_start = self.start + (self.count - 1) * self.tread
        self.descent_start = self.plateau_start + self.landing_length
        self.end = self.descent_start + (self.count - 1) * self.tread
        self.exit_x = self.end + profile["exit_clearance_m"]
        self.transitions = (
            (
                [(self.start + i * self.tread, (i + 1) * self.rise) for i in range(self.count)]
                + [
                    (self.descent_start + i * self.tread, (self.count - i - 1) * self.rise)
                    for i in range(self.count)
                ]
            )
            if kind == "stairs"
            else []
        )
        self.sections = (
            (
                [
                    (f"up_{i + 1}", self.start + i * self.tread, self.tread, (i + 1) * self.rise)
                    for i in range(self.count - 1)
                ]
                + [("plateau", self.plateau_start, self.landing_length, self.count * self.rise)]
                + [
                    (
                        f"down_{i + 1}",
                        self.descent_start + i * self.tread,
                        self.tread,
                        (self.count - i - 1) * self.rise,
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
                        (name + "_strip_up", left, self.strip_depth, top + self.strip_height)
                    )
                if name.startswith("down_") or name == "plateau":
                    self.strips.append(
                        (
                            name + "_strip_down",
                            left + length - self.strip_depth,
                            self.strip_depth,
                            top + self.strip_height,
                        )
                    )

    def iter_boxes(self):
        for name, left, length, top in self.sections:
            yield TerrainBox(name, left, length, self.width, 0.0, top)
        for name, left, length, top in self.strips:
            yield TerrainBox(name, left, length, self.width, top - self.strip_height, top, True)

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
            "tread_depth_m": self.tread,
            "risers_up": self.count,
            "risers_down": self.count,
            "width_m": self.width,
            "start_x_m": self.start,
            "plateau_start_x_m": self.plateau_start,
            "landing_length_m": self.landing_length,
            "descent_start_x_m": self.descent_start,
            "end_x_m": self.end,
            "exit_x_m": self.exit_x,
            "strip_height_m": self.strip_height,
            "strip_depth_m": self.strip_depth,
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
                friction=FRICTION,
                rgba=[0.25, 0.25, 0.25, 1.0] if box.strip else [0.35, 0.45, 0.55, 1.0],
            )
        )
    return geoms
