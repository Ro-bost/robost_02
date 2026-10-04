"""TPU foot material and explicit foot–terrain contact parameters."""

from robost.simulation.hardware import LEGS
from robost.simulation.terrain import STRIP_FRICTION

FOOT_CONDIM = 4
FOOT_FRICTION = (0.8, 0.003, 0.0001)
FOOT_GEOM_NAMES = tuple(leg + "_foot_collision_foot" for leg in LEGS)


def configure_foot_contacts(spec):
    """Apply the same effective TPU contact material in CPU and RL scenes.

    Same-priority geom contacts take the maximum friction component, which
    would hide the lower TPU coefficients beneath terrain defaults. Explicit
    pairs retain terrain sliding friction on nosings and foot torsion on all
    surfaces. Call after terrain and robot attachment, before compilation.
    """
    feet = [g for g in spec.geoms if g.name.rsplit("/", 1)[-1] in FOOT_GEOM_NAMES]
    if len(feet) != len(FOOT_GEOM_NAMES):
        raise ValueError("Expected the four RS06 foot collision geoms before configuring contacts")
    terrain = spec.body("terrain")
    if terrain is None:
        raise ValueError("Expected the terrain body before configuring foot contacts")
    existing = {frozenset((p.geomname1, p.geomname2)): p for p in spec.pairs}
    for foot in feet:
        for surface in terrain.geoms:
            if not ((foot.contype & surface.conaffinity) or (surface.contype & foot.conaffinity)):
                continue
            if not surface.name:
                raise ValueError("Foot contact surfaces must have names before configuring pairs")
            key = frozenset((foot.name, surface.name))
            pair = existing.get(key)
            if pair is None:
                pair = spec.add_pair(
                    name=f"foot_terrain_{foot.name}_{surface.name}",
                    geomname1=foot.name,
                    geomname2=surface.name,
                )
                existing[key] = pair
            sliding = (
                STRIP_FRICTION[0]
                if surface.name.endswith(("_strip_up", "_strip_down"))
                else FOOT_FRICTION[0]
            )
            pair.condim = FOOT_CONDIM
            pair.friction = (
                sliding,
                sliding,
                FOOT_FRICTION[1],
                FOOT_FRICTION[2],
                FOOT_FRICTION[2],
            )
