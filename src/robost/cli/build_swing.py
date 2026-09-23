"""Build the local MIT swing bridge without ROS or system package installation."""
from pathlib import Path
import shutil
import subprocess

from robost.paths import ROOT, BUILD


def build():
    compiler=shutil.which('g++')
    if compiler is None:
        raise RuntimeError('C++ compiler g++ is required to build libcheetah_swing.so')
    source=ROOT/'third_party/Cheetah-Software'
    if not (source/'common/src/Controllers/FootSwingTrajectory.cpp').is_file():
        raise RuntimeError('Missing dependencies; run python scripts/bootstrap_dependencies.py --only Cheetah-Software eigen3')
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler,'-O2','-shared','-fPIC',f'-I{ROOT/"third_party"}',
                    f'-I{source/"common/include"}',str(ROOT/'native/cheetah_swing_bridge.cpp'),
                    str(source/'common/src/Controllers/FootSwingTrajectory.cpp'),
                    '-o',str(BUILD/'libcheetah_swing.so')],check=True)


if __name__=='__main__':
    build()
