"""The matrix launcher must separate flat results and preserve failed subprocesses."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from robost.cli import evaluate


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "policy.pt"
        self.checkpoint.write_bytes(b"test checkpoint")
        self.output = self.root / "matrix"
        self.args = [
            "--adapter",
            "rs06",
            "--checkpoint",
            str(self.checkpoint),
            "--output",
            str(self.output),
            "--seeds",
            "42",
            "--repeats",
            "1",
        ]

    def test_flat_and_lower_stairs_keep_distinct_outcomes_and_provenance(self):
        def child(command, **kwargs):
            self.assertEqual(command[2], "robost.cli.trial")
            self.assertEqual(command[command.index("--adapter") + 1], "rs06")
            height = int(command[command.index("--stairs-cm") + 1])
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            result = dict(
                robot="rs06",
                mass_kg=18.082756,
                urdf_sha256="urdf hash",
                model_input_sha256={"input.urdf": "input hash"},
                physics={"timestep_s": 0.002, "gravity_m_s2": [0, 0, -9.81]},
                source_snapshot_sha256={"rl/rs06.py": "source hash"},
                checkpoint_sha256="checkpoint hash",
                auto_reset=False,
                purpose="fixed policy trial",
                hardware_pass=None,
                numerical_gate_pass=height == 0,
                course={"rise_m": height / 100},
                course_completed=True,
                course_completed_at_s=12.0,
                trials=[{"survived": True, "first_failure_s": None}],
            )
            (folder / "evaluation.json").write_text(json.dumps(result))
            return SimpleNamespace(returncode=0)

        with mock.patch.object(evaluate.subprocess, "run", side_effect=child) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate.main(self.args + ["--heights", "0", "2", "18"])
        self.assertEqual(run.call_count, 3)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(
            (summary["total"], summary["stairs_total"], summary["flat_total"]), (3, 2, 1)
        )
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["flat_survived"], 1)
        self.assertEqual(summary["flat_numerical_gate_passed"], 1)
        flat = summary["runs"][0]
        self.assertEqual(flat["terrain"], "flat")
        self.assertIsNone(flat["course_completed"])
        self.assertIsNone(flat["course"])
        self.assertIsNone(flat["course_completed_at_s"])
        self.assertEqual(flat["urdf_sha256"], "urdf hash")
        self.assertEqual(flat["source_snapshot_sha256"], {"rl/rs06.py": "source hash"})
        self.assertEqual(flat["physics"]["timestep_s"], 0.002)
        self.assertFalse(flat["auto_reset"])
        self.assertIsNone(flat["hardware_pass"])

    def test_failed_flat_process_has_no_invented_stair_failure_or_success(self):
        with mock.patch.object(
            evaluate.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate.main(self.args + ["--heights", "0"])
        self.assertEqual(error.exception.code, 1)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertIsNone(summary["completed"])
        self.assertEqual(summary["failed_processes"], 1)
        self.assertEqual(summary["missing_evaluations"], 1)
        self.assertIsNone(summary["runs"][0]["course_completed"])
        self.assertTrue((self.output / "h0_seed42_repeat0.log").is_file())

    def test_operational_errors_finish_collection_but_robot_falls_are_valid_results(self):
        def child(command, **kwargs):
            height = int(command[command.index("--stairs-cm") + 1])
            if height == 0:
                # Even an exit-zero child is an operational failure without its
                # evaluation file; the following stair trial must still run.
                return SimpleNamespace(returncode=0)
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            (folder / "evaluation.json").write_text(
                json.dumps(
                    {
                        "course_completed": False,
                        "trials": [
                            {
                                "survived": False,
                                "first_failure_s": 8.82,
                                "failure_reasons": ["fell_over"],
                            }
                        ],
                    }
                )
            )
            return SimpleNamespace(returncode=0)

        with mock.patch.object(evaluate.subprocess, "run", side_effect=child) as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate.main(self.args + ["--heights", "0", "18"])
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(run.call_count, 2)
            summary = json.loads((self.output / "summary.json").read_text())
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["missing_evaluations"], 1)
            self.assertEqual(summary["runs"][1]["failure_reasons"], ["fell_over"])
            # A fully recorded fall alone is a normal, unsuccessful rollout.
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate.main(
                    self.args + ["--heights", "18", "--output", str(self.root / "recorded_fall")]
                )
            self.assertIsNone(result)

    def test_invalid_arguments_cannot_launch_or_create_results(self):
        for arguments in (
            ["--speed", "nan"],
            ["--speed", "inf"],
            ["--speed", "0"],
            ["--duration", "nan"],
            ["--duration", "inf"],
            ["--duration", "3"],
            ["--repeats", "0"],
            ["--heights", "18", "18"],
            ["--seeds", "42", "42"],
            ["--seeds", "-1"],
        ):
            with self.subTest(arguments=arguments):
                with mock.patch.object(evaluate.subprocess, "run") as run:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        evaluate.main(self.args + arguments)
                run.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_existing_output_and_missing_checkpoint_are_preserved(self):
        self.output.mkdir()
        marker = self.output / "old_result.json"
        marker.write_text("preserved failure")
        with mock.patch.object(evaluate.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.main(self.args)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), "preserved failure")
        self.checkpoint.unlink()
        with mock.patch.object(evaluate.subprocess, "run") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.main(self.args + ["--output", str(self.root / "new")])
        run.assert_not_called()
        self.assertFalse((self.root / "new").exists())


if __name__ == "__main__":
    unittest.main()
