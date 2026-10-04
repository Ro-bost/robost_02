"""CPU scene smoke checks; a standing hold is not a walking result."""

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from robost.cli.scene import build_scene, run_hold
from robost.simulation.hardware import get_hardware, JOINT_NAMES
from robost.simulation.model import robot_spec
from robost.simulation.contact import configure_foot_contacts


class SceneTests(unittest.TestCase):
    def test_actual_foot_contacts_use_tpu_friction_and_nosing_override(self):
        _, model, data = build_scene()
        foot = model.geom("FR_foot_collision_foot").id
        radius = model.geom_size[foot, 0]
        self.assertEqual(model.npair, 4 * 40)
        for surface, position, sliding in (
            ("floor", [0.0, 0.0, radius - 1e-4], 0.8),
            ("course_up_1", [0.90, 0.0, 0.175 + radius - 1e-4], 0.8),
            ("course_up_1_strip_up", [0.78, 0.0, 0.18 + radius - 1e-4], 1.25),
        ):
            with self.subTest(surface=surface):
                mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
                mujoco.mj_forward(model, data)
                data.qpos[:3] += np.asarray(position) - data.geom_xpos[foot]
                mujoco.mj_forward(model, data)
                expected_geoms = {foot, model.geom(surface).id}
                contacts = [c for c in data.contact if set(c.geom) == expected_geoms]
                self.assertTrue(contacts)
                for contact in contacts:
                    self.assertEqual(contact.dim, 4)
                    np.testing.assert_array_equal(
                        contact.friction, [sliding, sliding, 0.003, 0.0001, 0.0001]
                    )

    def test_contact_configuration_is_idempotent_and_respects_collision_masks(self):
        spec, _, _ = build_scene()
        terrain = spec.body("terrain")
        for name, mask in (("terrain_visual_only", 0), ("terrain_disjoint_mask", 2)):
            terrain.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.1, 0.1, 0.1],
                mass=0.0,
                contype=mask,
                conaffinity=mask,
            )
        configure_foot_contacts(spec)
        configure_foot_contacts(spec)
        self.assertEqual(len(spec.pairs), 160)
        paired = {name for pair in spec.pairs for name in (pair.geomname1, pair.geomname2)}
        self.assertNotIn("terrain_visual_only", paired)
        self.assertNotIn("terrain_disjoint_mask", paired)
        self.assertFalse(any("_calf_collision_" in name for name in paired))

    def test_exported_scene_restores_standing_pose_without_changing_references(self):
        spec, model, initialized = build_scene()
        hardware = get_hardware()
        # A keyframe must not shift the URDF coordinate system or hard limits.
        original = robot_spec().compile()
        np.testing.assert_array_equal(model.qpos0, original.qpos0)
        np.testing.assert_array_equal(model.jnt_range, original.jnt_range)
        np.testing.assert_array_equal(model.dof_armature, original.dof_armature)
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "scene.mjb"
            mujoco.mj_saveModel(model, str(binary), None)
            roundtrips = (
                mujoco.MjModel.from_xml_string(spec.to_xml()),
                mujoco.MjModel.from_binary_path(str(binary)),
            )
            for encoding, loaded in zip(("xml", "mjb"), roundtrips):
                data = mujoco.MjData(loaded)
                mujoco.mj_resetDataKeyframe(loaded, data, loaded.key("stand").id)
                mujoco.mj_forward(loaded, data)
                np.testing.assert_array_equal(data.qpos, initialized.qpos)
                np.testing.assert_allclose(data.site_xpos, initialized.site_xpos)
                indices = [loaded.joint(name).qposadr[0] for name in JOINT_NAMES]
                angles = data.qpos[indices]
                self.assertTrue(np.all(angles >= hardware.joint_ranges[:, 0]))
                self.assertTrue(np.all(angles <= hardware.joint_ranges[:, 1]))
                for field in (
                    "qpos0",
                    "body_mass",
                    "body_inertia",
                    "jnt_range",
                    "dof_armature",
                    "geom_size",
                    "actuator_forcerange",
                ):
                    if encoding == "mjb":
                        np.testing.assert_array_equal(getattr(loaded, field), getattr(model, field))
                    else:
                        # MuJoCo's XML writer prints six significant digits;
                        # the MJB comparison above retains full precision.
                        np.testing.assert_allclose(
                            getattr(loaded, field), getattr(model, field), rtol=5e-6, atol=1e-12
                        )

    def test_physical_course_and_bounded_hold(self):
        _, model, data = build_scene()
        hardware = get_hardware()
        self.assertEqual(model.nu, 12)
        self.assertAlmostEqual(model.geom("course_up_1").size[0], 0.1585)
        self.assertAlmostEqual(model.geom("course_up_1_strip_up").size[2], 0.015)
        self.assertAlmostEqual(model.geom("course_plateau").pos[2] * 2, 1.75)
        np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -9.81])
        result = run_hold(model, data, hardware, 0.2)
        self.assertTrue(result["finite"])
        self.assertFalse(result["auto_reset"])
        self.assertIsNone(result["first_failure"])
        self.assertTrue(np.all(np.array(result["peak_joint_torque_nm"]) <= hardware.effort_limits))
        self.assertNotIn("walking_pass", result)

    def test_failure_is_retained_without_reset(self):
        _, model, data = build_scene()
        data.qpos[2] = 0.10
        result = run_hold(model, data, get_hardware(), 0.006)
        self.assertIsNotNone(result["first_failure"])
        self.assertFalse(result["auto_reset"])
        self.assertLess(result["first_failure"]["time_s"], result["duration_s"])


if __name__ == "__main__":
    unittest.main()
