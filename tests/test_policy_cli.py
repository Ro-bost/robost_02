"""Validate training and trial commands before importing GPU dependencies."""

import builtins
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from robost.cli import train, trial


class PolicyCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.checkpoint = self.folder / "policy.pt"
        self.checkpoint.write_bytes(b"weights")
        self.output = self.folder / "new_run"
        self.arguments = ["--checkpoint", str(self.checkpoint), "--output", str(self.output)]

    def test_invalid_inputs_fail_before_gpu_import_and_preserve_outputs(self):
        original_import = builtins.__import__

        def cpu_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("robost.rl"):
                self.fail("Invalid arguments imported GPU dependencies")
            return original_import(name, *args, **kwargs)

        cases = (
            (train, ["--std", "nan"]),
            (train, ["--learning-rate", "0"]),
            (train, ["--num-envs", "0"]),
            (train, ["--seed", "-1"]),
            (trial, ["--adapter", "rs06", "--duration", "inf"]),
            (trial, ["--adapter", "rs06", "--speed", "0"]),
        )
        with mock.patch("builtins.__import__", side_effect=cpu_import):
            for module, extra in cases:
                with self.subTest(command=module.__name__, extra=extra):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        module.main(self.arguments + extra)
            self.assertFalse(self.output.exists())
            self.output.mkdir()
            marker = self.output / "failure.json"
            marker.write_text("retain")
            for module, extra in ((train, []), (trial, ["--adapter", "rs06"])):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.main(self.arguments + extra)
            self.assertEqual(marker.read_text(), "retain")

    def test_train_delegates_fixed_speed_and_stage_without_starting_gpu(self):
        trainer = mock.Mock()
        torch = SimpleNamespace(set_num_threads=mock.Mock())
        adapter = SimpleNamespace(train=trainer)
        with mock.patch.dict(sys.modules, {"torch": torch, "robost.rl.rs06": adapter}):
            train.main(self.arguments + ["--stage", "mixed", "--iterations", "2"])
        arguments = trainer.call_args.args[0]
        self.assertEqual(
            (arguments.stage, arguments.iterations, arguments.speed), ("mixed", 2, 0.25)
        )
        self.assertEqual(arguments.checkpoint, self.checkpoint)
        self.assertFalse(self.output.exists())

    def test_worker_uses_single_deterministic_trial_and_matching_adapter(self):
        evaluator = mock.Mock()
        base = SimpleNamespace(evaluate=evaluator)
        torch = SimpleNamespace(set_num_threads=mock.Mock())
        adapter = SimpleNamespace(__file__=str(self.folder / "adapter.py"), make_cfg=mock.Mock())
        with (
            mock.patch.dict(sys.modules, {"torch": torch, "robost.rl": SimpleNamespace(base=base)}),
            mock.patch.object(trial.importlib, "import_module", return_value=adapter),
        ):
            trial.main(self.arguments + ["--adapter", "rs06", "--stairs-cm", "0"])
        arguments = evaluator.call_args.args[0]
        self.assertEqual(arguments.num_envs, 1)
        self.assertEqual(arguments.stairs_cm, 0)
        self.assertFalse(arguments.sample_actions)
        self.assertFalse(arguments.stop_go)
        self.assertEqual(arguments.noise_scale, 0.0)
        self.assertIs(evaluator.call_args.kwargs["cfg_factory"], adapter.make_cfg)


if __name__ == "__main__":
    unittest.main()
