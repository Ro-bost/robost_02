"""Measured terrain geometry, CPU scene and height-query consistency."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robost.simulation.terrain import Terrain, add_to_spec, FRICTION, STRIP_FRICTION
from robost.cli.scene import build_scene


class TerrainTests(unittest.TestCase):
    def test_measured_course_dimensions_and_strip_collision(self):
        terrain = Terrain()
        self.assertEqual(len(terrain.transitions), 20)
        self.assertEqual(len(terrain.sections), 19)
        self.assertEqual(len(terrain.strips), 20)
        self.assertAlmostEqual(terrain.plateau_start, 3.606)
        self.assertAlmostEqual(terrain.end, 7.462)
        self.assertAlmostEqual(terrain.exit_x, 7.612)
        self.assertAlmostEqual(terrain.height(4.0), 1.75)
        for box in terrain.iter_boxes():
            if box.strip:
                self.assertAlmostEqual(box.length, 0.06)
                self.assertAlmostEqual(box.top - box.bottom, 0.03)
            self.assertAlmostEqual(box.width, 1.6)
        # The landing has a strip at both top risers; each strip raises actual
        # collision height. Ground before/after and outside the width stays zero.
        self.assertAlmostEqual(terrain.height(0.76), 0.18)
        self.assertAlmostEqual(terrain.height(0.82), 0.175)
        self.assertAlmostEqual(terrain.height(3.61), 1.755)
        self.assertAlmostEqual(terrain.height(4.60), 1.755)
        self.assertAlmostEqual(terrain.height(4.62), 1.575)
        self.assertAlmostEqual(terrain.height(4.92), 1.58)
        np.testing.assert_allclose(terrain.height([0.7, 7.47, 4.0], [0.0, 0.0, 0.81]), 0.0)

    def test_approved_section_repeats_and_mirrors_at_every_riser(self):
        terrain = Terrain()
        bodies = {b.name: b for b in terrain.iter_boxes() if not b.strip}
        for strip in (b for b in terrain.iter_boxes() if b.strip):
            name, direction = strip.name.rsplit("_strip_", 1)
            body = bodies[name]
            self.assertAlmostEqual(strip.top - body.top, 0.005)
            self.assertAlmostEqual(strip.top - strip.bottom, 0.03)
            self.assertAlmostEqual(strip.length, 0.06)
            if direction == "up":
                self.assertAlmostEqual(body.left - strip.left, 0.003)
                lower_x = body.left - terrain.riser_pitch / 2
            else:
                self.assertAlmostEqual(strip.left + strip.length - body.left - body.length, 0.003)
                lower_x = body.left + body.length + terrain.riser_pitch / 2
            lower_top = terrain.height(lower_x)
            self.assertAlmostEqual(strip.top - lower_top, 0.18)
            if name != "plateau":
                self.assertAlmostEqual(body.length + 0.003, 0.32)
        self.assertEqual(Terrain("flat").sections, [])
        self.assertEqual(Terrain("flat").strips, [])
        with self.assertRaises(ValueError):
            Terrain("stairs", 0.5)

    def test_nosing_contact_has_configured_sliding_friction_and_real_undercut(self):
        terrain = Terrain()
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        add_to_spec(spec, terrain)
        probe = spec.worldbody.add_body(name="probe", pos=[0.751, 0, 0.189])
        probe.add_freejoint()
        probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], friction=FRICTION)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contacts = [
            c
            for c in data.contact
            if any(model.geom(int(i)).name == "course_up_1_strip_up" for i in c.geom)
        ]
        self.assertTrue(contacts)
        for contact in contacts:
            self.assertEqual(contact.dim, 3)
            np.testing.assert_allclose(contact.friction[:2], [1.25, 1.25])
        # Upward rays distinguish the overhang underside from a solid riser.
        geomid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([0.751, 0.4, 0.14]),
            np.array([0.0, 0.0, 1.0]),
            None,
            True,
            model.body("probe").id,
            geomid,
        )
        self.assertAlmostEqual(distance, 0.01)
        self.assertEqual(model.geom(int(geomid[0])).name, "course_up_1_strip_up")

    def test_height_sampler_matches_mujoco_rays_cpu_and_spec(self):
        terrain = Terrain()
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[15.0, 3.0, 0.1])
        spec.worldbody.add_body(name="terrain")
        geoms = add_to_spec(spec, terrain)
        self.assertEqual(len(geoms), 39)
        for geom in geoms:
            expected = STRIP_FRICTION if "_strip_" in geom.name else FRICTION
            np.testing.assert_allclose(geom.friction, expected)
            self.assertEqual(geom.contype, 1)
            self.assertEqual(geom.conaffinity, 1)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Probe both sides of every riser and every 6 cm strip edge, at centre,
        # inside the side edge and just outside; no robot shadows these rays.
        xs = [-0.1, terrain.end + 0.1]
        for box in terrain.iter_boxes():
            xs.extend(
                [
                    box.left - 1e-5,
                    box.left + 1e-5,
                    box.left + box.length - 1e-5,
                    box.left + box.length + 1e-5,
                ]
            )
        geomid = np.array([-1], dtype=np.int32)
        for x in xs:
            for y in (0.0, 0.79, 0.81):
                distance = mujoco.mj_ray(
                    model,
                    data,
                    np.array([x, y, 3.0]),
                    np.array([0.0, 0.0, -1.0]),
                    None,
                    True,
                    -1,
                    geomid,
                )
                self.assertAlmostEqual(3.0 - distance, terrain.height(x, y), places=9)
        scene, _, _ = build_scene(terrain=terrain)
        root = ET.fromstring(scene.to_xml())
        for box in terrain.iter_boxes():
            geom = root.find(f"worldbody/body[@name='terrain']/geom[@name='course_{box.name}']")
            np.testing.assert_allclose(np.fromstring(geom.get("pos"), sep=" "), box.pos)
            np.testing.assert_allclose(np.fromstring(geom.get("size"), sep=" "), box.size)

    def test_serialized_surface_includes_strips_and_full_course(self):
        terrain = Terrain()
        metadata = terrain.metadata()
        surface = metadata["surface_segments_m"]
        self.assertEqual(surface[0][0], terrain.start)
        self.assertAlmostEqual(surface[-1][1], terrain.end)
        for left, right, top in surface:
            self.assertGreater(right, left)
            self.assertAlmostEqual(terrain.height((left + right) / 2), top)
        self.assertEqual(metadata["risers_up"], 10)
        self.assertEqual(metadata["risers_down"], 10)
        self.assertEqual(metadata["strip_height_m"], 0.005)
        self.assertEqual(metadata["strip_thickness_m"], 0.03)
        self.assertEqual(metadata["strip_overhang_m"], 0.003)
        self.assertEqual(metadata["strip_friction"], [1.25, 0.005, 0.0001])


if __name__ == "__main__":
    unittest.main()
