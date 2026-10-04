"""URDF preservation and linkage envelope checks for RS06 v5."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robost.simulation.hardware import get_hardware, JOINT_NAMES
from robost.simulation.model import load_model, robot_spec


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = get_hardware()
        cls.model, cls.data = load_model()
        cls.source = ET.parse(cls.hardware.urdf).getroot()

    def test_supplied_mass_and_full_inertia_are_preserved(self):
        model = self.model
        self.assertAlmostEqual(model.body_mass.sum(), 18.082756, places=8)
        self.assertEqual(model.nbody, len(self.source.findall("link")) + 1)
        for link in self.source.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            body = model.body(link.get("name"))
            self.assertAlmostEqual(float(body.mass[0]), float(inertial.find("mass").get("value")))
            inertia = inertial.find("inertia").attrib
            expected = np.array(
                [
                    [float(inertia["ixx"]), float(inertia["ixy"]), float(inertia["ixz"])],
                    [float(inertia["ixy"]), float(inertia["iyy"]), float(inertia["iyz"])],
                    [float(inertia["ixz"]), float(inertia["iyz"]), float(inertia["izz"])],
                ]
            )
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, body.iquat)
            rotation = rotation.reshape(3, 3)
            # MuJoCo's eigensolver drops sub-1e-7 products of inertia when
            # diagonalizing the supplied full tensor (base ixy is 6.6e-8).
            np.testing.assert_allclose(
                rotation @ np.diag(body.inertia) @ rotation.T, expected, atol=1e-7
            )

    def test_every_collision_primitive_and_hard_range_is_preserved(self):
        count = 0
        for link in self.source.findall("link"):
            for index, collision in enumerate(link.findall("collision")):
                name = f"{link.get('name')}_collision_{collision.get('name', str(index))}"
                geom = self.model.geom(name)
                self.assertEqual(int(geom.contype[0]), 1)
                self.assertEqual(int(geom.conaffinity[0]), 1)
                shape = collision.find("geometry")[0]
                if shape.tag == "box":
                    np.testing.assert_allclose(
                        geom.size, np.fromstring(shape.get("size"), sep=" ") / 2
                    )
                elif shape.tag == "cylinder":
                    np.testing.assert_allclose(
                        geom.size[:2], [float(shape.get("radius")), float(shape.get("length")) / 2]
                    )
                elif shape.tag == "sphere":
                    self.assertAlmostEqual(geom.size[0], float(shape.get("radius")))
                else:
                    self.fail(f"Unexpected source collision shape: {shape.tag}")
                count += 1
        self.assertEqual(count, 35)
        for index, name in enumerate(JOINT_NAMES):
            np.testing.assert_allclose(
                self.model.joint(name).range, self.hardware.joint_ranges[index]
            )
        self.assertEqual(
            len(set(self.model.geom(i).name for i in range(self.model.ngeom))), self.model.ngeom
        )

    def test_floating_model_sites_payload_and_visual_materials(self):
        model = robot_spec().compile()
        self.assertEqual((model.nq, model.nv), (19, 18))
        self.assertEqual(model.site("imu").bodyid[0], model.body("imu_link").id)
        for leg in ("FR", "FL", "RR", "RL"):
            self.assertEqual(model.site(leg).bodyid[0], model.body(leg + "_foot").id)
        comparison = robot_spec(payload="5kg").compile()
        self.assertAlmostEqual(comparison.body_mass.sum(), 18.720756, places=8)
        self.assertGreater(len(np.unique(model.geom_rgba, axis=0)), 4)

    def test_cad_knee_table_and_structural_effort_cap(self):
        hw = self.hardware
        q = np.array([hw.standing_joint_positions[name] for name in JOINT_NAMES])
        q[2::3] = -2.53073  # Supplied hard limit and CSV row, not rounded README.
        np.testing.assert_allclose(hw.knee_ratio(q[2::3]), -3.7841)
        np.testing.assert_allclose(hw.joint_velocity_limits(q)[2::3], 480 * 2 * np.pi / 60 / 3.7841)
        np.testing.assert_allclose(hw.reflected_armatures(q)[2::3], 0.012 * 3.7841**2)
        np.testing.assert_allclose(hw.torque_envelope(q, np.zeros(12)), hw.effort_limits)
        np.testing.assert_allclose(hw.torque_envelope(q, hw.joint_velocity_limits(q)), 0.0)
        self.assertTrue(np.all(hw.torque_envelope(q, np.ones(12) * 10) <= hw.effort_limits))


if __name__ == "__main__":
    unittest.main()
