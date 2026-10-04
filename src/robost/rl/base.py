"""RS06 v5 adapter for upstream mjlab velocity MDP and RSL-RL PPO.

No handwritten footstep state machine, position teleport, or support force.
Use the robost environment. Upstream remains unmodified.
"""

from pathlib import Path
import json
import hashlib
import math
import os
import sys
import threading
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "glfw" if len(sys.argv) > 1 and sys.argv[1] == "play" else "egl")
os.environ.setdefault("WANDB_MODE", "disabled")
import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as env_mdp
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch, TerrainHeightSensorCfg, ObjRef
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity import mdp
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from robost.paths import RS06_PACKAGE, SOURCE, CONFIG
from robost.simulation.hardware import get_hardware
from robost.simulation.contact import configure_foot_contacts
from robost.simulation.terrain import Terrain, add_to_spec

LEGS = ("FR", "FL", "RR", "RL")


def robot_spec():
    """RS06 v5 URDF with unchanged inertias, hard ranges and collision shapes."""
    from robost.simulation.model import robot_spec as hardware_spec

    return hardware_spec()


def robot_cfg():
    hardware = get_hardware()
    return EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, hardware.standing_height),
            joint_pos=hardware.standing_joint_positions,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(f".*_{part}_joint",),
                    stiffness=80.0 if part == "calf" else 60.0,
                    damping=2.0,
                    effort_limit=limit,
                    armature=armature,
                )
                for part, limit, armature in zip(
                    ("hip", "thigh", "calf"), hardware.effort_limits[:3], hardware.armatures[:3]
                )
            ),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


def motor_cost(env):
    # Effort regularizer; measured output, not PD target. Actuator order is
    # resolved from joint names, not assumed to match FR/FL/RR/RL.
    robot = env.scene["robot"]
    return robot.data.qfrc_actuator.square().sum(dim=-1)


def flat_height_reward(env):
    z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((z - get_hardware().standing_height) / 0.06).square())


def make_cfg(num_envs=256, seed=42, evaluate=False):
    cfg = unitree_go1_flat_env_cfg()
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.spec_fn = configure_foot_contacts
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    for sensor in cfg.scene.sensors:
        if isinstance(sensor, TerrainHeightSensorCfg):
            sensor.frame = tuple(ObjRef(type="site", name=l, entity="robot") for l in LEGS)
    for sensor in cfg.scene.sensors:
        if isinstance(sensor, ContactSensorCfg) and sensor.name == "feet_ground_contact":
            sensor.primary = ContactMatch(
                mode="geom",
                pattern=tuple(leg + "_foot_collision_foot" for leg in LEGS),
                entity="robot",
            )
    bad = ContactSensorCfg(
        name="body_ground",
        primary=ContactMatch(
            mode="geom",
            pattern=(
                "base_collision_.*",
                ".*_hip_collision_.*",
                ".*_thigh_collision_.*",
                "realsense_d435i_collision_.*",
                "lidar_2d_collision_.*",
            ),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    shank = ContactSensorCfg(
        name="shank_ground",
        primary=ContactMatch(
            mode="geom", pattern=tuple(l + "_calf_collision_.*" for l in LEGS), entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = cfg.scene.sensors + (bad, shank)
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = 2.0
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
    for name in ("upright", "body_ang_vel"):
        cfg.rewards[name].params["asset_cfg"].body_names = ("base",)
    cfg.actions["joint_pos"].scale = {
        ".*_hip_joint": 0.20,
        ".*_thigh_joint": 0.35,
        ".*_calf_joint": 0.40,
    }
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10  # 500Hz physics / 50Hz policy
    cfg.sim.mujoco.iterations = 20
    cfg.sim.njmax = 400
    cfg.rewards["track_linear_velocity"].params["std"] = 0.25
    cfg.rewards["pose"].weight = 0.5
    # Early v1 rewarded all four airborne during a collapse. Keep upstream's
    # Go1 setting (zero air-time reward), and explicitly penalize termination.
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.1
    cfg.rewards["foot_clearance"].params["target_height"] = 0.06
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
    cfg.rewards["motor_effort"] = RewardTermCfg(func=motor_cost, weight=-0.0002)
    cfg.rewards["termination"] = RewardTermCfg(func=env_mdp.is_terminated, weight=-100.0)
    cfg.rewards["base_height"] = RewardTermCfg(func=flat_height_reward, weight=0.5)
    cfg.rewards["shank_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-0.2, params={"sensor_name": "shank_ground"}
    )
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact, params={"sensor_name": "body_ground"}
    )
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45)
    cmd = cfg.commands["twist"]
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_heading_envs = 0.0
    cmd.rel_forward_envs = 0.0
    cmd.rel_standing_envs = 0.1
    cmd.ranges.lin_vel_x = (0.0, 0.6)
    cmd.ranges.lin_vel_y = (-0.1, 0.1)
    cmd.ranges.ang_vel_z = (-0.3, 0.3)
    cfg.curriculum = {}
    # Baseline training: no artificial pushes. A later robustness suite must
    # add disturbances explicitly instead of disguising resets as recovery.
    cfg.events.pop("push_robot", None)
    if evaluate:
        cfg.observations["actor"].enable_corruption = False
        for key in tuple(cfg.events):
            if key not in ("reset_base", "reset_robot_joints"):
                cfg.events.pop(key)
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "yaw": (-0.03, 0.03),
        }
        cfg.episode_length_s = 35.0
        cfg.auto_reset = False
    return cfg


def runner_cfg():
    cfg = unitree_go1_ppo_runner_cfg()
    cfg.experiment_name = "rs06_dynamic_velocity"
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    cfg.actor.hidden_dims = (256, 128, 128)
    cfg.critic.hidden_dims = (256, 128, 128)
    cfg.actor.distribution_cfg["init_std"] = 0.4
    cfg.save_interval = 100
    cfg.clip_actions = 3.0
    return cfg


def add_stair_course(spec, rise=0.18):
    """Real-size 10-up / 1m landing / 10-down course including edge strips."""
    add_to_spec(spec, Terrain("stairs", rise * 100))
    configure_foot_contacts(spec)


def play(args, cfg_factory=make_cfg):
    from robost.rl.viewer import NoResetMujocoViewer

    cfg = cfg_factory(1, args.seed, evaluate=True)
    if args.stairs_cm:
        cfg.scene.spec_fn = lambda spec: add_stair_course(spec, args.stairs_cm / 100)
    cfg.auto_reset = False
    cfg.episode_length_s = 1e6
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1e6, 1e6)
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if not command.heading_command:
        command.ranges.ang_vel_z = (0.0, 0.0)
    raw_env = ManagerBasedRlEnv(cfg, device="cuda:0")
    previous_threads = {t.ident for t in threading.enumerate()}
    viewer = None
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=3.0)
        runner = MjlabOnPolicyRunner(env, asdict(runner_cfg()), device="cuda:0")
        runner.load(
            str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
        )
        policy = runner.get_inference_policy(device="cuda:0")
        print(
            "Live learned policy. Speed:",
            args.speed,
            "m/s. No automatic reset; use headless evaluation for scored results.",
        )
        play_seconds = getattr(args, "play_seconds", 0.0)
        viewer = NoResetMujocoViewer(
            env, policy, enable_perturbations=False, close_on_failure=bool(play_seconds)
        )
        viewer.run(
            num_steps=max(1, round(play_seconds / raw_env.step_dt)) if play_seconds else None
        )
    finally:
        try:
            # Also covers setup errors before the upstream run() cleanup starts.
            if viewer is not None:
                viewer.close()
            # MuJoCo 3.11 passive close signals the render thread but does not join
            # it. Finish before Python's glfw.terminate atexit hook.
            for thread in threading.enumerate():
                if thread.ident not in previous_threads and "_launch_internal" in thread.name:
                    thread.join(timeout=5.0)
        finally:
            raw_env.close()


def local_source_snapshot():
    """Capture local modules after model and environment factories import them."""
    snapshot = {}
    for module_name, module in list(sys.modules.items()):
        source_path = getattr(module, "__file__", None)
        if (
            module_name.startswith("robost.")
            and source_path
            and Path(source_path).is_relative_to(SOURCE)
        ):
            path = Path(source_path)
            snapshot[path.relative_to(SOURCE)] = path.read_bytes()
    return snapshot


def settled_stop_max_abs_vx(times, velocities):
    """Return no measurement if a first failure prevented the stop window."""
    stopped = (np.asarray(times) >= 15) & (np.asarray(times) < 18)
    return float(np.max(np.abs(np.asarray(velocities)[stopped, :, 0]))) if stopped.any() else None


def including_failure_metrics(joint_torque, joint_limit_margin, shank_force):
    """Retain terminal impacts separately from the pre-failure quality window."""
    torque = np.asarray(joint_torque)
    margins = np.asarray(joint_limit_margin)
    shanks = np.asarray(shank_force)
    count = len(torque)
    if len(margins) != count or len(shanks) != count:
        raise ValueError("Including-failure metrics require matching sample counts")
    return {
        "measurement_window": "all recorded 50Hz samples including terminal failure; "
        "no post-reset samples; does not bound 500Hz substep peaks",
        "sample_count": count,
        "joint_torque_peak_nm": np.max(np.abs(torque), axis=0).tolist() if count else [],
        "joint_limit_min_margin_rad": float(margins.min()) if count else None,
        "shank_force_peak_n": float(shanks.max()) if count else None,
    }


def checkpoint_training_provenance(checkpoint):
    """Snapshot adjacent training metadata without modifying its original file.

    Directory adjacency is evidence of association, not proof that a training
    run produced a particular checkpoint; retain that distinction in results.
    """
    source = Path(checkpoint).resolve().parent / "run.json"
    provenance = {
        "available": False,
        "status": "missing",
        "source_path": str(source),
        "association": "Adjacent run.json; checkpoint identity is recorded separately, "
        "and directory adjacency does not verify its training lineage.",
    }
    try:
        content = source.read_bytes()
    except FileNotFoundError:
        return provenance, None
    except OSError as error:
        provenance.update(status="unreadable", error=str(error))
        return provenance, None
    provenance.update(
        sha256=hashlib.sha256(content).hexdigest(), saved_path="checkpoint_training_run.json"
    )
    try:
        metadata = json.loads(content)
        if not isinstance(metadata, dict):
            raise ValueError("Training run metadata must be a JSON object")
    except (ValueError, UnicodeError) as error:
        provenance.update(status="invalid_json", error=str(error))
        return provenance, content
    provenance.update(
        available=True,
        status="available",
        training_completed=metadata.get("completed"),
        stage=metadata.get("stage", metadata.get("terrain_stage")),
        initial_checkpoint_sha256=metadata.get("checkpoint_sha256"),
    )
    return provenance, content


def evaluation_description(args):
    """Keep the full purpose in JSON and its concise first clause in video."""
    default = (
        "Policy evaluation on fixed stair course"
        if args.stairs_cm
        else "Policy evaluation on flat ground"
    )
    purpose = getattr(args, "evaluation_purpose", default)
    label = " ".join(purpose.split(";", 1)[0].split(",", 1)[0].split())
    return purpose, label if len(label) <= 48 else label[:45] + "..."


def evaluate(args, cfg_factory=make_cfg):
    """Every reset/termination is a failure; never count a later restart as pass."""
    if args.num_envs != 1:
        raise ValueError(
            "No-reset evaluation requires num_envs=1; use independent matrix trials for repeats"
        )
    from scipy.spatial.transform import Rotation
    from robost.rl.validation import StairExitTracker

    evaluator_source = Path(__file__).read_bytes()
    validation_source = Path(sys.modules[StairExitTracker.__module__].__file__).read_bytes()
    policy_source = (
        Path(args.policy_adapter).read_bytes() if hasattr(args, "policy_adapter") else None
    )
    training_provenance, training_run_source = checkpoint_training_provenance(args.checkpoint)
    purpose, video_policy_label = evaluation_description(args)
    cfg = cfg_factory(args.num_envs, args.seed, evaluate=True)
    if args.stairs_cm:
        cfg.scene.spec_fn = lambda spec: add_stair_course(spec, args.stairs_cm / 100)
    # A failed run ends at its first failure; no automatic restart.
    cfg.auto_reset = False
    cfg.episode_length_s = args.duration + 10
    cmd = cfg.commands["twist"]
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.ranges.lin_vel_x = (args.speed, args.speed)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    if not cmd.heading_command:
        cmd.ranges.ang_vel_z = (0.0, 0.0)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    agent = runner_cfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    # robot_spec imports simulation.model lazily while constructing the env.
    # Snapshot after that import, before any rollout, to retain its exact code.
    source_snapshot = local_source_snapshot()
    if getattr(args, "sample_actions", False):
        with torch.no_grad():
            policy.distribution.std_param.mul_(getattr(args, "noise_scale", 1.0))
    alive = np.ones(args.num_envs, dtype=bool)
    first_failure = np.full(args.num_envs, np.nan)
    failure_reasons = [[] for _ in range(args.num_envs)]
    failure_states = [None for _ in range(args.num_envs)]
    completed = False
    left_course = False
    max_x = 0.0
    completed_at = None
    max_foot_z = 0.0
    course = Terrain("stairs", args.stairs_cm) if args.stairs_cm else None
    hardware = get_hardware()
    exit_tracker = (
        StairExitTracker.from_metadata(course.metadata(), foot_radius=hardware.foot_radius)
        if course
        else None
    )
    rows = []
    poses = []
    vels = []
    worldvels = []
    positions = []
    yaws = []
    tilts = []
    masks = []
    forces = []
    contacts = []
    slips = []
    shanks = []
    shankforces = []
    jointvel = []
    jointpositions = []
    margins = []
    references = []
    times = []
    sites = [robot.site_names.index(l) for l in LEGS]
    with torch.inference_mode():
        for i in range(round(args.duration / env.step_dt)):
            now = (i + 1) * env.step_dt
            reference = 0.0 if args.stop_go and (now < 3 or 13 <= now < 18) else args.speed
            if args.stop_go:
                env.command_manager.get_term("twist").vel_command_b[:, 0] = reference
                obs = wrapped.get_observations()
            obs, rew, done, extra = wrapped.step(
                policy(obs, stochastic_output=getattr(args, "sample_actions", False))
            )
            ended = done.cpu().numpy().astype(bool)
            for k in np.flatnonzero(alive & ended):
                failure_reasons[k] = [
                    name
                    for name in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(name)[k])
                ]
            first_failure[alive & ended] = now
            alive &= ~ended
            v = robot.data.root_link_lin_vel_b.cpu().numpy().copy()
            quat = robot.data.root_link_quat_w.cpu().numpy()
            rpy = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler("xyz")
            contact = env.scene["feet_ground_contact"].data.found.cpu().numpy() > 0
            if args.stairs_cm and alive[0]:
                feet = robot.data.site_pos_w[:, sites].cpu().numpy()[0]
                max_x = max(max_x, float(robot.data.root_link_pos_w[0, 0]))
                max_foot_z = max(max_foot_z, float(np.max(feet[:, 2])))
                completed = exit_tracker.update(now, feet, contact[0])
                left_course = exit_tracker.left_course
                completed_at = exit_tracker.completed_at
            elif args.stairs_cm:
                exit_tracker.failed = True
            footv = robot.data.site_vel_w[:, sites, :2].cpu().numpy()
            references.append(reference)
            times.append(now)
            vels.append(v)
            tilts.append(rpy[:, :2])
            masks.append(alive.copy())
            worldvels.append(robot.data.root_link_lin_vel_w.cpu().numpy().copy())
            positions.append(robot.data.root_link_pos_w.cpu().numpy().copy())
            yaws.append(rpy[:, 2].copy())
            forces.append(robot.data.qfrc_actuator.cpu().numpy().copy())
            contacts.append(contact.copy())
            slips.append(np.linalg.norm(footv, axis=-1) * contact)
            shanks.append(env.scene["shank_ground"].data.found.cpu().numpy() > 0)
            shankforces.append(env.scene["shank_ground"].data.force.cpu().numpy().copy())
            jointvel.append(robot.data.joint_vel.cpu().numpy().copy())
            jointpos = robot.data.joint_pos.cpu().numpy().copy()
            jointpositions.append(jointpos)
            bounds = np.array(
                [env.sim.mj_model.joint("robot/" + name).range for name in robot.joint_names]
            )
            margins.append(np.minimum(jointpos - bounds[:, 0], bounds[:, 1] - jointpos))
            for k in np.flatnonzero(ended):
                if failure_states[k] is None:
                    failure_states[k] = {
                        "time_s": now,
                        "joint_order": list(robot.joint_names),
                        "joint_position_rad": jointpos[k].tolist(),
                        "joint_velocity_rad_s": jointvel[-1][k].tolist(),
                        "joint_torque_nm": forces[-1][k].tolist(),
                        "hard_limit_margin_rad": margins[-1][k].tolist(),
                    }
            if (
                i % 2 == 0
                or bool(ended.any())
                or completed
                or i == round(args.duration / env.step_dt) - 1
            ):
                poses.append(env.sim.data.qpos[0].cpu().numpy().copy())
                rows.append(
                    {
                        "t": now,
                        "alive": bool(alive[0]),
                        "command": reference,
                        "vx": float(v[0, 0]),
                        "rpy": rpy[0].tolist(),
                    }
                )
            if args.stairs_cm and completed:
                break
            if not alive.any():
                break
    vels = np.array(vels)
    tilts = np.array(tilts)
    masks = np.array(masks)
    forces = np.array(forces)
    worldvels = np.array(worldvels)
    positions = np.array(positions)
    yaws = np.array(yaws)
    contact = np.array(contacts)
    slips = np.array(slips)
    shanks = np.array(shanks)
    shankforces = np.linalg.norm(np.array(shankforces), axis=-1)
    jointvel = np.array(jointvel)
    jointpositions = np.array(jointpositions)
    margins = np.array(margins)
    references = np.array(references)
    times = np.array(times)
    trials = []
    for k in range(args.num_envs):
        valid = masks[:, k]
        if valid.any():
            vx = vels[valid, k, 0]
            rp = tilts[valid, k]
            target = references[valid]
            rmse = float(np.sqrt(np.mean((vx - target) ** 2)))
            avg = float(vx.mean())
            rms = float(np.rad2deg(np.sqrt(np.mean(rp**2))))
            peak = float(np.rad2deg(np.max(np.abs(rp))))
            cycles = np.abs(np.diff(contact[valid, k].astype(int), axis=0)).sum(axis=0).tolist()
            slip = float(slips[valid, k].sum() / max(1, contact[valid, k].sum()))
            trms = np.sqrt(np.mean(forces[valid, k] ** 2, axis=0)).tolist()
            shank_fraction = float(shanks[valid, k].any(axis=-1).mean())
            joint_peak = np.max(np.abs(jointvel[valid, k]), axis=0).tolist()
            margin = float(margins[valid, k].min())
            torque_peak = np.max(np.abs(forces[valid, k]), axis=0).tolist()
            speed_limits = np.array(
                [
                    33.96 if "_calf_" in n else 42.94 if "_hip_" in n else 50.27
                    for n in robot.joint_names
                ]
            )
            velocity_ratios = np.abs(jointvel[valid, k]) / speed_limits
            for j, name in enumerate(robot.joint_names):
                if "_calf_" in name:
                    velocity_ratios[:, j] = (
                        np.abs(jointvel[valid, k, j])
                        * np.abs(hardware.knee_ratio(jointpositions[valid, k, j]))
                        / 50.27
                    )
            passed = bool(
                alive[k]
                and rmse <= 0.1
                and abs(avg - target.mean()) <= 0.2 * args.speed
                and rms <= 8
                and peak <= 20
                and min(cycles) >= 4
                and slip <= 0.08
                and shank_fraction == 0
                and margin >= -0.005
                and np.all(velocity_ratios <= 1.0)
            )
        else:
            rmse = avg = rms = peak = slip = shank_fraction = margin = None
            cycles = []
            trms = []
            joint_peak = []
            torque_peak = []
            passed = False
        trials.append(
            {
                "trial": k,
                "survived": bool(alive[k]),
                "first_failure_s": None if np.isnan(first_failure[k]) else float(first_failure[k]),
                "mean_vx": avg,
                "vx_rmse": rmse,
                "roll_pitch_rms_deg": rms,
                "roll_pitch_peak_deg": peak,
                "foot_contact_transitions": cycles,
                "stance_foot_slip_m_s": slip,
                "joint_torque_rms": trms,
                "shank_contact_fraction": shank_fraction,
                "joint_limit_min_margin_rad": margin,
                "joint_speed_peak": joint_peak,
                "joint_torque_peak": torque_peak,
                "numerical_gate_pass": passed,
            }
        )
        trials[-1]["including_failure"] = including_failure_metrics(
            forces[:, k], margins[:, k], shankforces[:, k]
        )
        trials[-1]["shank_force_peak_n"] = (
            float(shankforces[valid, k].max()) if valid.any() else None
        )
        trials[-1]["failure_reasons"] = failure_reasons[k]
        trials[-1]["first_failure_state"] = failure_states[k]
        trials[-1]["shank_force_sum_mean_n"] = (
            float(shankforces[valid, k].sum(axis=-1).mean()) if valid.any() else None
        )
        trials[-1]["shank_over_10n_fraction"] = (
            float((shankforces[valid, k].max(axis=-1) > 10).mean()) if valid.any() else None
        )
        if valid.any():
            foot = contact[valid, k]
            trials[-1]["mean_world_vx"] = float(worldvels[valid, k, 0].mean())
            trials[-1]["max_abs_yaw_deg"] = float(np.rad2deg(np.abs(yaws[valid, k]).max()))
            trials[-1]["max_abs_lateral_m"] = float(
                np.abs(positions[valid, k, 1] - float(env.scene.env_origins[k, 1])).max()
            )
            trials[-1]["support_count_fraction"] = [
                float((foot.sum(-1) == n).mean()) for n in range(5)
            ]
            trials[-1]["diagonal_pair_only_fraction"] = float(
                (
                    (foot[:, 0] & foot[:, 3] & ~foot[:, 1] & ~foot[:, 2])
                    | (foot[:, 1] & foot[:, 2] & ~foot[:, 0] & ~foot[:, 3])
                ).mean()
            )
            ideal_ratio = np.ones_like(forces[valid, k])
            for j, name in enumerate(robot.joint_names):
                if "_calf_" in name:
                    ideal_ratio[:, j] = np.abs(hardware.knee_ratio(jointpositions[valid, k, j]))
            motor_rms = np.sqrt(np.mean((forces[valid, k] / ideal_ratio) ** 2, axis=0))
            rated = np.array([6.0 if "_hip_" in name else 11.0 for name in robot.joint_names])
            trials[-1]["joint_speed_limit_max_ratio"] = float(velocity_ratios.max())
            trials[-1]["ideal_motor_torque_rms_nm"] = motor_rms.tolist()
            trials[-1]["motor_rated_torque_nm"] = rated.tolist()
            trials[-1]["ideal_motor_rms_over_rating"] = (motor_rms / rated).tolist()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evaluator_source.py").write_bytes(evaluator_source)
    if training_run_source is not None:
        (args.output / training_provenance["saved_path"]).write_bytes(training_run_source)
    (args.output / "source_snapshot").mkdir()
    for name, content in source_snapshot.items():
        target = args.output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(evaluator_source).hexdigest(),
        "joint_order": list(robot.joint_names),
        "speed_command": args.speed,
        "duration": args.duration,
        "seed": args.seed,
        "survivors": int(alive.sum()),
        "trials": trials,
        "numerical_gate_pass": all(t["numerical_gate_pass"] for t in trials),
        "visual_review_required": True,
        "hardware_pass": None,
        "sample_hz": 50,
        "log_env0": rows,
    }
    result.update(
        purpose=purpose,
        video_policy_label=video_policy_label,
        training_provenance=training_provenance,
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    np.save(args.output / "qpos_env0.npy", np.array(poses))
    np.save(args.output / "foot_contacts.npy", contact)
    np.savez_compressed(
        args.output / "telemetry.npz",
        time=times,
        qpos_time=np.array([row["t"] for row in rows]),
        qpos=np.array(poses),
        joint_position=jointpositions,
        joint_velocity=jointvel,
        joint_torque=forces,
        alive=masks,
        foot_contacts=contact,
        shank_force=shankforces,
    )
    result["command_profile"] = (
        "0-3 stop, 3-13 walk, 13-18 stop, 18-end walk" if args.stop_go else "constant"
    )
    result["actual_duration_s"] = now
    result["measurement_start_s"] = float(times[0])
    result["stochastic_policy"] = getattr(args, "sample_actions", False)
    result["noise_scale"] = getattr(args, "noise_scale", 1.0)
    result["source_snapshot_sha256"] = {
        str(name): hashlib.sha256(content).hexdigest() for name, content in source_snapshot.items()
    }
    result["contact_metric_note"] = (
        "50Hz samples; shank sensor has one slot per geom. Not a bound on all 500Hz impact peaks."
    )
    result["robot"] = "rs06"
    result["mass_kg"] = float(env.sim.mj_model.body_mass.sum())
    result["urdf_sha256"] = hashlib.sha256(hardware.urdf.read_bytes()).hexdigest()
    result["foot_radius_m"] = hardware.foot_radius
    result["motor_rated_torques_nm"] = [
        6.0 if "_hip_" in name else 11.0 for name in robot.joint_names
    ]
    result["auto_reset"] = False
    result["quality_measurement_window"] = (
        "all pre-failure samples including startup; failures are retained"
    )
    result["motor_metric_note"] = (
        "RS06 hip rated 6Nm, thigh/calf rated 11Nm. Motor-equivalent torque and speed checks use the CAD knee ratio curve. Simulation uses supplied nominal constant armature and bounded ideal PD, without measured torque-speed, thermal, latency or flexible-link dynamics. No hardware certification."
    )
    if args.stairs_cm:
        result["stairs_cm"] = args.stairs_cm
        result["course"] = course.metadata()
        result["course_completed"] = completed
        result["course_completed_at_s"] = completed_at
        result["completion_definition"] = exit_tracker.definition
        result["exit_foot_landing_times_s"] = exit_tracker.landing_times
        (args.output / "course_validation_source.py").write_bytes(validation_source)
        result["course_validation_sha256"] = hashlib.sha256(validation_source).hexdigest()
        result["max_foot_height_m"] = max_foot_z
        result["left_course"] = left_course
        result["max_x_before_failure"] = max_x
        result["numerical_gate_pass"] = False  # Flat quality criteria do not certify stairs.
    if hasattr(args, "policy_adapter"):
        result["policy_adapter"] = str(args.policy_adapter)
        result["policy_adapter_sha256"] = hashlib.sha256(policy_source).hexdigest()
        (args.output / "policy_adapter_source.py").write_bytes(policy_source)
    if args.stop_go:
        stop_max = settled_stop_max_abs_vx(times, vels)
        result["settled_stop_max_abs_vx"] = stop_max
        result["numerical_gate_pass"] = (
            result["numerical_gate_pass"] and stop_max is not None and stop_max <= 0.08
        )
    # Preserve small model inputs as well as the compiled mesh-bearing MJB.
    inputs = [
        hardware.urdf,
        RS06_PACKAGE / "actuator_params.yaml",
        RS06_PACKAGE / "knee_linkage.csv",
        RS06_PACKAGE / "urdf_summary.json",
        CONFIG / "courses.json",
    ]
    input_dir = args.output / "model_inputs"
    input_dir.mkdir()
    result["model_input_sha256"] = {}
    for source in inputs:
        content = source.read_bytes()
        (input_dir / source.name).write_bytes(content)
        result["model_input_sha256"][source.name] = hashlib.sha256(content).hexdigest()
    result["physics"] = {
        "timestep_s": float(env.sim.mj_model.opt.timestep),
        "gravity_m_s2": env.sim.mj_model.opt.gravity.tolist(),
        "control_hz": 1.0 / env.step_dt,
        "self_collision_enabled": True,
    }
    result["contact_backend_note"] = (
        "MuJoCo Warp warns that cylinder-cylinder and cylinder-box pairs have at most one contact per pair; CPU/GPU contact equivalence is not established."
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    mujoco.mj_saveModel(env.sim.mj_model, str(args.output / "scene.mjb"))
    # Use the same timestamp-aware saved-state renderer as offline review.
    if args.video:
        from robost.tools.render_trial import render

        render(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "log_env0"}, indent=2))
    env.close()
