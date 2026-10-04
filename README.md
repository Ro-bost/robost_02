# URDF and Stair Map Updates

MuJoCo simulation for the **RS06 v5 quadruped**, with an updated URDF and a stair map based on measured dimensions. Robot and terrain construction are independent of the controller, so the same scene can be used for reinforcement learning or MPC development.

## Robot and terrain

| Parameter | Default |
|---|---|
| Robot mass, including electronics | 18.082756 kg |
| Actuated joints | 12 |
| Hip / thigh / calf torque limits | 17 / 23 / 30 Nm |
| Stair rise / tread depth | 18 / 32 cm |
| Stair count | 10 ascending + 10 descending |
| Landing length / stair width | 1.0 / 1.6 m |
| Raised edge strip | 5 mm high, 6 cm deep |

Robot assets are in `assets/rs06/`. Terrain dimensions are in `config/courses.json`. The URDF mass, inertia, joint limits and collision geometry are retained.

The default RS06 policy is experimental. Previous independent trials recorded 3/3 completions at 6, 10 and 12 cm, and 0/3 at 14 and 18 cm. Flat-ground survival was 3/3 over 60 seconds. Course completion does not establish gait quality or hardware safety. See [PROGRESS.md](PROGRESS.md) for results and remaining work.

## Installation

Use Linux x86_64 and Python 3.11. CPU simulation and GPU training share the **`robost` Conda environment**. GPU training and policy evaluation require an NVIDIA GPU and a CUDA 13-compatible driver. Run from a source checkout.

```bash
git clone https://github.com/Ro-bost/robost_02.git
cd robost_02
python scripts/setup_environment.py
conda activate robost
```

For robot and terrain development without the RL stack:

```bash
python scripts/setup_environment.py --cpu-only
conda activate robost
```

## Simulation

Preview the robot or run the stair scene with bounded standing control:

```bash
robost-preview
robost-scene --duration 30
```

The scene command exports `scene.xml` and `scene.mjb` to a new directory under `runs/`. Standing control is a scene check, not a walking test. To restore the exported standing pose:

```python
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
```

Controllers can construct the model directly through `robost.simulation.model.robot_spec()` or obtain the complete robot and terrain through `robost.cli.scene.build_scene()`.

View policy walking in a desktop window:

```bash
MUJOCO_GL=glfw robost-stairs --stairs-cm 12 --play-seconds 60
MUJOCO_GL=glfw robost-stairs --stairs-cm 18 --play-seconds 60
```

Playback ends at the first failure. Omit `--play-seconds` to keep the failed pose visible. Use `--stairs-cm 0` for flat ground.

Evaluate without a window and save a video:

```bash
MUJOCO_GL=egl robost-stairs --stairs-cm 18 --headless --duration 60
```

The default checkpoint is `assets/policies/rs06.pt`, verified against `config/rs06_policy.json`. Every height uses the same checkpoint. Results include `evaluation.json`, `evaluation_1x.mp4`, `telemetry.npz`, `qpos_env0.npy` and `scene.mjb`. Existing output directories are never overwritten.

Evaluate multiple heights and seeds in independent processes:

```bash
MUJOCO_GL=egl robost-evaluate \
  --adapter rs06 --checkpoint assets/policies/rs06.pt \
  --heights 0 6 10 12 14 18 --seeds 42 43 44 \
  --repeats 1 --speed .25 --duration 60 --output runs/evaluation
```

Evaluation uses one environment with automatic resets disabled. It stops at the first failure. Completion requires each foot to land on the exit floor, followed by two seconds with all feet beyond x = 7.66 m. Bypassing the stairs fails the trial.

## Training

PPO settings are in `config/rs06_training.json`. Adapt the current checkpoint on mixed terrain:

```bash
robost-train --checkpoint assets/policies/rs06.pt \
  --stage mixed --num-envs 256 --iterations 800 --output runs/training
```

Available stages are `flat`, `low`, `stairs` and `mixed`. The commanded speed is 0.25 m/s. Mixed training retains 25% flat terrain and 25% low stairs, with an adaptive curriculum for the remaining 50%. Actor, critic and observation normalization are loaded; the optimizer starts fresh. The final checkpoint after 800 iterations is `model_799.pt`.

Training resets collect PPO data. Rewards and curriculum promotion are not evidence of course completion. Evaluate each new policy independently, including flat ground.

## Repository and development

```text
robost/
├── src/robost/       Simulation, RL, CLI and analysis tools
├── tests/           Model, terrain, policy and evaluation checks
├── assets/          Current URDF, meshes and policy
├── config/          Terrain, policy, training and environment settings
├── scripts/         Environment installation
├── history/         Minimal RS02 reproduction package
├── README.md        Installation and usage
└── PROGRESS.md      Results and remaining work
```

```bash
make check
make lint
make test-core
make test-rl
```

CPU-only installations support `make check lint test-core`. RL tests require the full environment. Use `make format` to apply the shared Python format.

Add code under `src/`, tests under `tests/` and settings under `config/`. Record any changes to physical parameters or completion criteria when comparing results. Report completion, gait quality and hardware safety separately. Generated runs are excluded from Git.

The historical RS02 URDF, 15/20 cm maps and policies are in `history/`. See [history/README.md](history/README.md) for execution and further training. Third-party and supplied-asset notices are in [NOTICE](NOTICE).
