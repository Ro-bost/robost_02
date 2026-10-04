"""Measured terrain geometry, CPU scene and height-query consistency."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robost.simulation.terrain import Terrain, add_to_spec
from robost.cli.scene import build_scene


class TerrainTests(unittest.TestCase):
    def test_measured_course_dimensions_and_strip_collision(self):
        terrain = Terrain()
        self.assertEqual(len(terrain.transitions), 20)
        self.assertEqual(len(terrain.sections), 19)
        self.assertEqual(len(terrain.strips), 20)
        self.assertAlmostEqual(terrain.plateau_start, 3.63)
        self.assertAlmostEqual(terrain.end, 7.51)
        self.assertAlmostEqual(terrain.exit_x, 7.66)
        self.assertAlmostEqual(terrain.height(4.0), 1.8)
        for box in terrain.iter_boxes():
            if box.strip:
                self.assertAlmostEqual(box.length, 0.06)
                self.assertAlmostEqual(box.top - box.bottom, 0.005)
            self.assertAlmostEqual(box.width, 1.6)
        # The landing has a strip at both top risers; each strip raises actual
        # collision height. Ground before/after and outside the width stays zero.
        self.assertAlmostEqual(terrain.height(0.76), 0.185)
        self.assertAlmostEqual(terrain.height(0.82), 0.18)
        self.assertAlmostEqual(terrain.height(3.64), 1.805)
        self.assertAlmostEqual(terrain.height(4.62), 1.805)
        self.assertAlmostEqual(terrain.height(4.64), 1.62)
        self.assertAlmostEqual(terrain.height(4.94), 1.625)
        np.testing.assert_allclose(terrain.height([0.7, 7.52, 4.0], [0.0, 0.0, 0.81]), 0.0)

    def test_height_sampler_matches_mujoco_rays_cpu_and_spec(self):
        terrain = Terrain()
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[15.0, 3.0, 0.1])
        spec.worldbody.add_body(name="terrain")
        geoms = add_to_spec(spec, terrain)
        self.assertEqual(len(geoms), 39)
        for geom in geoms:
            np.testing.assert_allclose(geom.friction, [1.0, 0.005, 0.0001])
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


if __name__ == "__main__":
    unittest.main()
