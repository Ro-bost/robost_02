"""Paths for the source checkout (run with PYTHONPATH=src or an editable install)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
PACKAGE = ROOT / "models" / "rs02_quadruped"
BUILD = ROOT / "build"
