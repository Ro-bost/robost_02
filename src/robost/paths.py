"""Project paths shared by the simulation and RL entry points."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "robost"
RS06_PACKAGE = ROOT / "assets" / "rs06"
CONFIG = ROOT / "config"
