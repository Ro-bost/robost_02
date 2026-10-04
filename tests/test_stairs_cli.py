import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from robost.cli import stairs
from robost.paths import ROOT


class LauncherTests(unittest.TestCase):
    def test_every_height_uses_one_rs06_checkpoint_from_manifest(self):
        # Source-only installs intentionally omit checkpoint binaries.
        manifest = json.loads((ROOT / "config/rs06_policy.json").read_text())
        for height in stairs.HEIGHTS_CM:
            with self.subTest(height=height):
                self.assertEqual(stairs.select_policy(height), ("rs06", ROOT / manifest["path"]))
        self.assertTrue(manifest["status"])

    def test_custom_checkpoint_requires_explicit_adapter(self):
        with self.assertRaises(ValueError):
            stairs.select_policy(20, Path("custom.pt"))
        with self.assertRaises(ValueError):
            stairs.select_policy(20, adapter="route")
        self.assertEqual(
            stairs.select_policy(20, Path("custom.pt"), "route"), ("route", Path("custom.pt"))
        )
        self.assertEqual(
            stairs.select_policy(0, Path("adapted.pt"), "rs06"), ("rs06", Path("adapted.pt"))
        )

    def test_default_missing_or_mismatched_weights_fail_before_gpu_import(self):
        original_import = builtins.__import__
        gpu_imports = []

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("robost.rl"):
                gpu_imports.append(name)
                raise AssertionError("Invalid weights must fail before GPU library imports")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            manifest = dict(
                stairs.DEFAULT_POLICY,
                path=str(checkpoint),
                sha256=hashlib.sha256(b"expected").hexdigest(),
            )
            with (
                mock.patch.object(stairs, "DEFAULT_POLICY", manifest),
                mock.patch("builtins.__import__", side_effect=guarded_import),
            ):
                with (
                    contextlib.redirect_stderr(io.StringIO()) as error,
                    self.assertRaises(SystemExit),
                ):
                    stairs.main(["--headless"])
                self.assertIn("--checkpoint /path/to/policy.pt --adapter rs06", error.getvalue())
                checkpoint.write_bytes(b"different")
                with (
                    contextlib.redirect_stderr(io.StringIO()) as error,
                    self.assertRaises(SystemExit),
                ):
                    stairs.main(["--headless"])
                self.assertIn("SHA-256 mismatch", error.getvalue())
            self.assertEqual(gpu_imports, [])
            self.assertEqual(checkpoint.read_bytes(), b"different")

    def test_verified_default_and_explicit_custom_use_different_hash_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"expected")
            with mock.patch.object(
                stairs,
                "DEFAULT_POLICY",
                dict(stairs.DEFAULT_POLICY, sha256=hashlib.sha256(b"expected").hexdigest()),
            ):
                stairs.validate_checkpoint(checkpoint, default=True)
                checkpoint.write_bytes(b"custom policy")
                stairs.validate_checkpoint(checkpoint, default=False)
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    stairs.validate_checkpoint(checkpoint, default=True)


if __name__ == "__main__":
    unittest.main()
