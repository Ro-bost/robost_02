"""Project paths shared by the simulation and RL entry points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "robost"
PACKAGE = ROOT / "assets" / "rs02"
BUILD = ROOT / "build"
