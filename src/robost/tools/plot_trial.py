"""Scientific figure from recorded trial data, not generated illustrative motion."""

import argparse
import json
from pathlib import Path
import numpy as np


def course_profile(result):
    """Read the recorded terrain rather than imposing today's course on old data."""
    if not result.get("stairs_cm"):
        length = result.get("duration", 60.0) * abs(result.get("speed_command", 0.25)) + 1.0
        return [0.0, length], [0.0, 0.0]
    course = result.get("course")
    if course is not None:
        segments = course["surface_segments_m"]
        x = [0.0]
        z = [0.0]
        for left, right, top in segments:
            x.extend((left, left, right))
            z.extend((z[-1], top, top))
        x.extend((course["end_x_m"], course["exit_x_m"] + 1.0))
        z.extend((0.0, 0.0))
        return x, z
    height = result["stairs_cm"] / 100.0
    return (
        [0.0, 0.75, 1.05, 1.35, 1.65, 1.95, 2.95, 3.25, 3.55, 3.85, 4.15, 5.5],
        np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0]) * height,
    )


def motor_measurement(result):
    """Use recorded RS06 motor conversion on flat and stair trials alike."""
    trial = result["trials"][0]
    ratings = trial.get("motor_rated_torque_nm", result.get("motor_rated_torques_nm"))
    if trial.get("ideal_motor_torque_rms_nm") is not None and ratings:
        return (
            np.array(trial["ideal_motor_torque_rms_nm"]),
            np.array(ratings),
            "Ideal motor RMS from recorded linkage ratio; no loss/thermal model",
        )
    if str(result.get("robot", "rs02")).lower() == "rs02":
        ratio = np.array([1.48 if "_calf_" in name else 1.0 for name in result["joint_order"]])
        rms = (
            np.array(trial["joint_torque_rms"]) / ratio
            if trial["joint_torque_rms"]
            else np.array([])
        )
        return rms, np.full(len(ratio), 6.0), "Historical RS02 ideal motor RMS (knee / 1.48)"
    return (
        np.array(trial["joint_torque_rms"]),
        None,
        "Joint RMS torque; motor conversion unavailable",
    )


def plot(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from robost.tools.render_trial import load_recording

    folder = Path(folder)
    target = folder / "trial_measurements.png"
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((folder / "evaluation.json").read_text())
    qpos, rows, _ = load_recording(folder, result)
    alive = np.array([row["alive"] for row in rows])
    contact = np.load(folder / "foot_contacts.npy")[:, 0]
    time = result.get("measurement_start_s", 3.0) + np.arange(len(contact)) / result["sample_hz"]
    if (folder / "telemetry.npz").is_file():
        with np.load(folder / "telemetry.npz") as telemetry:
            if "time" in telemetry:
                time = telemetry["time"].copy()
    if len(time) != len(contact):
        raise ValueError("Contact samples and timestamps must have matching lengths")
    failure = result["trials"][0]["first_failure_s"]
    if failure is not None:
        contact = contact[time < failure]
        time = time[time < failure]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout="constrained")
    x, z = course_profile(result)
    axes[0].step(x, z, where="post", color="gray", label="Terrain top")
    axes[0].plot(qpos[alive, 0], qpos[alive, 2], color="#187b95", label="Body position")
    axes[0].set(
        xlabel="World X [m]",
        ylabel="World Z [m]",
        title="Measured body trajectory (before first failure)",
    )
    axes[0].legend(loc="upper right")
    if len(time):
        axes[1].imshow(
            contact.T,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            extent=(time[0], time[-1] + 1 / result["sample_hz"], 3.5, -0.5),
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Failure occurred before the contact measurement window",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set(
        yticks=range(4),
        yticklabels=["FR", "FL", "RR", "RL"],
        xlabel="Time [s]",
        title="Measured foot contact: blue = contact, white = swing/no contact",
    )
    labels = [
        name.replace("_joint", "")
        .replace("_thigh", "-T")
        .replace("_hip", "-H")
        .replace("_calf", "-K")
        for name in result["joint_order"]
    ]
    rms, ratings, torque_title = motor_measurement(result)
    if len(rms):
        colors = np.where(rms > ratings, "#c15b38", "#187b95") if ratings is not None else "#187b95"
        axes[2].bar(labels, rms, color=colors)
        if ratings is not None:
            axes[2].plot(
                labels, ratings, "_", color="#aa3322", markersize=18, label="Motor rated torque"
            )
            axes[2].legend(loc="upper right")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No valid pre-failure torque samples",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    axes[2].set(ylabel="Torque [Nm]", title=torque_title)
    for ax in (axes[0], axes[2]):
        ax.grid(axis="y", alpha=0.2)
    robot_name = str(result.get("robot", "RS02")).upper()
    terrain = f"{result['stairs_cm']} cm stairs" if result.get("stairs_cm") else "flat ground"
    outcome = (
        f"course completed = {result['course_completed']}"
        if result.get("stairs_cm")
        else f"survived = {result['trials'][0]['survived']}"
    )
    fig.suptitle(
        f"{robot_name} {terrain} | command {result['speed_command']} m/s | "
        f"seed {result['seed']} | {outcome}\n"
        "Recorded simulation measurements; NOT hardware certification",
        fontsize=13,
    )
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    plot(parser.parse_args().folder)
