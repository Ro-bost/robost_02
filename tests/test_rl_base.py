"""Model-adapter checks in the robost environment, before expensive training."""

import unittest
import hashlib
import json
import tempfile
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from mjlab.entity import Entity
from robost.rl.base import (
    robot_cfg,
    LEGS,
    make_cfg,
    evaluate,
    local_source_snapshot,
    settled_stop_max_abs_vx,
    checkpoint_training_provenance,
    evaluation_description,
    including_failure_metrics,
)
from robost.simulation.model import robot_spec
from robost.simulation.hardware import get_hardware


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = Entity(robot_cfg()).spec.compile()
        cls.original = robot_spec().compile()

    def test_inertias_and_joint_limits_preserved(self):
        m = self.m
        o = self.original
        self.assertAlmostEqual(m.body_mass.sum(), get_hardware().mass_kg, places=5)
        self.assertEqual(m.nv, 18)
        self.assertEqual(m.nu, 12)
        for name in [l + "_" + j + "_joint" for l in LEGS for j in ("hip", "thigh", "calf")]:
            np.testing.assert_allclose(m.joint(name).range, o.joint(name).range)
        for i in range(1, m.nbody):
            body = m.body(i)
            orig = o.body(body.name)
            np.testing.assert_allclose(body.mass, orig.mass)
            np.testing.assert_allclose(body.inertia, orig.inertia)

    def test_torque_limits_and_collision_masks(self):
        m = self.m
        for i in range(m.nu):
            name = m.joint(m.actuator_trnid[i, 0]).name
            limit = 30.0 if "_calf_" in name else 23.0 if "_thigh_" in name else 17.0
            np.testing.assert_allclose(m.actuator_forcerange[i], [-limit, limit])
        for i in range(m.ngeom):
            geom = m.geom(i)
            if "_collision_" in geom.name:
                self.assertEqual(int(geom.contype[0]), 1)
                self.assertEqual(int(geom.conaffinity[0]), 1)

    def test_no_crawl_or_artificial_support(self):
        cfg = make_cfg(16, evaluate=True)
        self.assertEqual(cfg.sim.mujoco.timestep, 0.002)
        self.assertEqual(cfg.decimation, 10)
        self.assertNotIn("push_robot", cfg.events)
        self.assertFalse(cfg.auto_reset)
        self.assertEqual(set(cfg.actions), {"joint_pos"})

    def test_snapshot_includes_loaded_hardware_factory(self):
        snapshot = local_source_snapshot()
        self.assertIn(Path("simulation/model.py"), snapshot)
        self.assertIn(Path("simulation/hardware.py"), snapshot)
        self.assertIn(b"def robot_spec(", snapshot[Path("simulation/model.py")])

    def test_no_reset_evaluation_rejects_parallel_batch(self):
        with self.assertRaisesRegex(ValueError, "num_envs=1"):
            evaluate(SimpleNamespace(num_envs=2))

    def test_stop_metric_handles_failure_before_stop_window(self):
        velocity = np.array([[[0.3, 0.0, 0.0]], [[0.04, 0.0, 0.0]], [[0.06, 0.0, 0.0]]])
        self.assertIsNone(settled_stop_max_abs_vx([0.02, 1.0, 6.8], velocity))
        self.assertAlmostEqual(settled_stop_max_abs_vx([14.98, 15.0, 17.98], velocity), 0.06)

    def test_including_failure_window_preserves_terminal_impact_and_limit_excursion(self):
        torque = np.array([[1.0, -2.0, 3.0], [2.0, -1.0, 2.0], [17.0, -23.0, 30.0]])
        margins = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3], [0.1, -0.012, 0.2]])
        shanks = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 102.37]])
        pre_failure = including_failure_metrics(torque[:-1], margins[:-1], shanks[:-1])
        all_samples = including_failure_metrics(torque, margins, shanks)
        self.assertEqual(pre_failure["shank_force_peak_n"], 0.0)
        self.assertEqual(pre_failure["joint_limit_min_margin_rad"], 0.1)
        self.assertEqual(all_samples["sample_count"], 3)
        self.assertEqual(all_samples["joint_torque_peak_nm"], [17.0, 23.0, 30.0])
        self.assertAlmostEqual(all_samples["joint_limit_min_margin_rad"], -0.012)
        self.assertAlmostEqual(all_samples["shank_force_peak_n"], 102.37)
        self.assertIn("including terminal failure", all_samples["measurement_window"])
        self.assertIn("does not bound 500Hz", all_samples["measurement_window"])

    def test_training_provenance_preserves_exact_source_and_reports_absence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "model_300.pt"
            checkpoint.write_bytes(b"checkpoint")
            absent, content = checkpoint_training_provenance(checkpoint)
            self.assertFalse(absent["available"])
            self.assertEqual(absent["status"], "missing")
            self.assertIsNone(content)
            raw = (
                json.dumps(
                    {"stage": "flat", "completed": True, "checkpoint_sha256": "initial hash"},
                    indent=2,
                ).encode()
                + b"\n"
            )
            source = folder / "run.json"
            source.write_bytes(raw)
            available, content = checkpoint_training_provenance(checkpoint)
            self.assertTrue(available["available"])
            self.assertEqual(content, raw)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(available["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(available["initial_checkpoint_sha256"], "initial hash")
            self.assertEqual(available["stage"], "flat")
            self.assertTrue(available["training_completed"])
            self.assertIn("does not verify", available["association"])

    def test_malformed_training_metadata_is_retained_without_claiming_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = b"{unfinished training record"
            (folder / "run.json").write_bytes(raw)
            provenance, content = checkpoint_training_provenance(folder / "policy.pt")
            self.assertFalse(provenance["available"])
            self.assertEqual(provenance["status"], "invalid_json")
            self.assertEqual(content, raw)
            self.assertIn("error", provenance)

    def test_flat_adapted_policy_purpose_is_not_labeled_transfer(self):
        expected = "RS06 adapted policy; fixed nominal physics and first-failure evaluation"
        purpose, label = evaluation_description(
            SimpleNamespace(stairs_cm=0, evaluation_purpose=expected)
        )
        self.assertEqual(purpose, expected)
        self.assertEqual(label, "RS06 adapted policy")
        self.assertNotIn("transfer", label)
        default, _ = evaluation_description(SimpleNamespace(stairs_cm=0))
        self.assertIn("flat ground", default)


if __name__ == "__main__":
    unittest.main()
