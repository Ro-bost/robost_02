"""Install the pinned CPU/GPU stack into the single ``robost`` conda environment.

Run with any Python 3.11+ interpreter from this source checkout. Existing conda
environments are retained; only ``robost`` is created or updated. The NVIDIA
driver is supplied by the host, not this environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = "robost"


def run(*command):
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conda",
        default=os.environ.get("CONDA_EXE") or shutil.which("conda"),
        help="conda executable; defaults to CONDA_EXE or PATH",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="install the CPU scene/preview dependencies; skip the CUDA/RL stack",
    )
    args = parser.parse_args()
    if not args.conda:
        parser.error("Conda is required; activate its shell or pass --conda /path/to/conda")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("This lock was validated on Linux x86_64 with an NVIDIA GPU")
    if not args.cpu_only and not shutil.which("git"):
        parser.error("Git is required to install the pinned mjlab source as a wheel")
    environments = json.loads(
        subprocess.check_output([args.conda, "env", "list", "--json"], text=True)
    )["envs"]
    if not any(Path(path).name == ENVIRONMENT for path in environments):
        run(args.conda, "env", "create", "--file", ROOT / "config/environment.yml", "--yes")
    prefix = [
        args.conda,
        "run",
        "--no-capture-output",
        "--name",
        ENVIRONMENT,
        "python",
        "-m",
        "pip",
    ]
    # The project stays editable: runtime assets belong to this checkout.
    if args.cpu_only:
        run(*prefix, "install", "--editable", f"{ROOT}[simulation,dev]")
    else:
        run(
            *prefix,
            "install",
            "--requirement",
            ROOT / "config/requirements.txt",
            "--extra-index-url",
            "https://pypi.nvidia.com/",
        )
        run(*prefix, "install", "--no-deps", "--editable", ROOT)
    run(*prefix, "check")
    print("Ready: conda activate robost")


if __name__ == "__main__":
    main()
