"""CPU-only regression checks for the standalone RS02 reproducer."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "history/rs02/run.py"
        spec = importlib.util.spec_from_file_location("history_launcher", path)
        cls.launcher = importlib.util.module_from_spec(spec)
        with (
            patch.object(sys, "path", sys.path.copy()),
            patch.object(sys, "dont_write_bytecode", True),
        ):
            spec.loader.exec_module(cls.launcher)
            cls.base, cls.rhythm = cls.launcher.load_adapter()

    def test_failure_before_quality_window_saves_first_terminal_state(self):
        # Failure on the second step also covers a frame that the normal
        # 25 Hz recording cadence would skip. No GPU or dynamics are mocked
        # into a successful result: this exercises only failure bookkeeping.
        for failure_step in (1, 2):
            with self.subTest(failure_step=failure_step), tempfile.TemporaryDirectory() as tmp:
                env = EarlyFailureEnv(failure_step)
                cfg = self.rhythm.make_cfg(1, seed=42, evaluate=True)
                cfg.auto_reset = True  # The evaluator must explicitly disable it.
                args = SimpleNamespace(
                    num_envs=1,
                    seed=42,
                    stairs_cm=15,
                    duration=40.0,
                    speed=0.25,
                    stop_go=False,
                    sample_actions=False,
                    checkpoint=self.launcher.HERE / "policies/15cm.pt",
                    output=Path(tmp) / "evaluation",
                    video=False,
                    policy_adapter=self.launcher.HERE / "src/rs02_rl_rhythm.py",
                )
                runner = SimpleNamespace(
                    load=lambda *a, **k: None,
                    get_inference_policy=lambda **k: lambda obs, **kw: torch.zeros(1, 12),
                )
                with (
                    patch.object(sys, "path", [str(self.launcher.HERE / "src"), *sys.path]),
                    patch.object(sys, "dont_write_bytecode", True),
                    patch.object(self.base, "ManagerBasedRlEnv", return_value=env),
                    patch.object(self.base, "RslRlVecEnvWrapper", side_effect=lambda e, **k: e),
                    patch.object(self.base, "MjlabOnPolicyRunner", return_value=runner),
                    redirect_stdout(io.StringIO()),
                ):
                    self.base.evaluate(args, cfg_factory=lambda *a, **k: cfg)
                self.assertEqual(env.steps, failure_step)
                self.assertTrue(env.closed)
                self.assertFalse(cfg.auto_reset)
                result = json.loads((args.output / "evaluation.json").read_text())
                trial = result["trials"][0]
                self.assertEqual(trial["first_failure_s"], failure_step * env.step_dt)
                self.assertEqual(result["actual_duration_s"], failure_step * env.step_dt)
                self.assertEqual(trial["failure_reasons"], ["forced_failure"])
                self.assertFalse(trial["survived"])
                self.assertFalse(trial["numerical_gate_pass"])
                self.assertIsNone(trial["mean_vx"])
                self.assertFalse(result["auto_reset"])
                self.assertFalse(result["course_completed"])
                self.assertFalse(result["log_env0"][-1]["alive"])
                np.testing.assert_array_equal(
                    np.load(args.output / "qpos_env0.npy")[-1], env.sim.data.qpos[0].numpy()
                )
                contacts = np.load(args.output / "foot_contacts.npy")
                self.assertEqual(contacts.shape, (0, 1, 4))
                self.assertEqual(contacts.dtype, np.bool_)
                self.assertTrue((args.output / "scene.mjb").is_file())

    def test_standalone_sibling_output_allowed_but_package_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "rs02"
            package.mkdir()
            existing = Path(tmp) / "existing"
            existing.mkdir()
            for output, accepted in (
                (Path(tmp) / "result", True),
                (package / "result", False),
                (existing, False),
            ):
                with (
                    self.subTest(output=output),
                    patch.object(self.launcher, "HERE", package),
                    patch.object(self.launcher, "check_inputs") as check,
                    patch.object(
                        sys,
                        "argv",
                        ["run.py", "check", "--checkpoint", "unused.pt", "--output", str(output)],
                    ),
                    patch("sys.stderr", new_callable=io.StringIO),
                ):
                    if accepted:
                        self.launcher.main()
                        check.assert_called_once()
                    else:
                        with self.assertRaises(SystemExit) as caught:
                            self.launcher.main()
                        self.assertEqual(caught.exception.code, 2)
                        check.assert_not_called()


class EarlyFailureEnv:
    """Minimal CPU environment that raises if stepped after termination."""

    def __init__(self, failure_step):
        self.failure_step = failure_step
        self.steps = 0
        self.closed = False
        self.step_dt = 0.02
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
            "</body></worldbody></mujoco>"
        )
        qpos = torch.tensor(model.qpos0).unsqueeze(0)
        self.sim = SimpleNamespace(mj_model=model, data=SimpleNamespace(qpos=qpos))
        self.scene = {
            "robot": SimpleNamespace(
                joint_names=[f"joint_{index}" for index in range(12)],
                site_names=["FR", "FL", "RR", "RL"],
                data=SimpleNamespace(
                    root_link_lin_vel_b=torch.zeros(1, 3),
                    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    root_link_pos_w=torch.zeros(1, 3),
                    site_pos_w=torch.zeros(1, 4, 3),
                    site_vel_w=torch.zeros(1, 4, 3),
                ),
            ),
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.ones(1, 4, dtype=torch.bool))
            ),
        }
        self.termination_manager = SimpleNamespace(
            active_terms=("forced_failure",),
            get_term=lambda name: torch.tensor([True]),
        )

    def get_observations(self):
        return {"actor": torch.zeros(1, 3)}

    def step(self, action):
        if self.steps >= self.failure_step:
            raise AssertionError("Evaluator stepped beyond first terminal state")
        self.steps += 1
        self.sim.data.qpos[0, 0] = self.steps
        return (
            self.get_observations(),
            torch.zeros(1),
            torch.tensor([self.steps == self.failure_step]),
            {},
        )

    def close(self):
        self.closed = True
