"""Scientific figure from recorded trial data, not generated illustrative motion."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    target = folder / 'trial_measurements.png'
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((folder / 'evaluation.json').read_text())
    qpos = np.load(folder / 'qpos_env0.npy')
    rows = result['log_env0']
    alive = np.array([row['alive'] for row in rows])
    contact = np.load(folder / 'foot_contacts.npy')[:, 0]
    time = 3. + np.arange(len(contact)) / result['sample_hz']
    failure = result['trials'][0]['first_failure_s']
    if failure is not None:
        contact = contact[time < failure]
        time = time[time < failure]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout='constrained')
    height = result['stairs_cm'] / 100.
    x = [0., .75, 1.05, 1.35, 1.65, 1.95, 2.95, 3.25, 3.55, 3.85, 4.15, 5.5]
    z = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0]) * height
    axes[0].step(x, z, where='post', color='gray', label='Stair top')
    axes[0].plot(qpos[alive, 0], qpos[alive, 2], color='#187b95', label='Body position')
    axes[0].set(xlabel='World X [m]', ylabel='World Z [m]', title='Measured body trajectory (before first failure)')
    axes[0].legend(loc='upper right')
    axes[1].imshow(contact.T, origin='upper', interpolation='nearest', aspect='auto',
                   extent=(time[0], time[-1] + .02, 3.5, -.5), cmap='Blues', vmin=0, vmax=1)
    axes[1].set(yticks=range(4), yticklabels=['FR', 'FL', 'RR', 'RL'], xlabel='Time [s]',
                title='Measured foot contact: blue = contact, white = swing/no contact')
    trial = result['trials'][0]
    ratio = np.array([1.48 if '_calf_' in name else 1. for name in result['joint_order']])
    rms = np.array(trial['joint_torque_rms']) / ratio
    labels = [name.replace('_joint', '').replace('_thigh', '-T').replace('_hip', '-H').replace('_calf', '-K')
              for name in result['joint_order']]
    axes[2].bar(labels, rms, color=np.where(rms > 6., '#c15b38', '#187b95'))
    axes[2].axhline(6., color='#aa3322', linestyle='--', label='Supplied motor rated torque: 6 Nm')
    axes[2].set(ylabel='Motor RMS torque [Nm]', title='Ideal motor-equivalent RMS (knee / 1.48, no loss/thermal model)')
    axes[2].legend(loc='upper right')
    for ax in (axes[0], axes[2]):
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f"RS02 {result['stairs_cm']} cm | command {result['speed_command']} m/s | "
                 f"seed {result['seed']} | course completed = {result['course_completed']}\n"
                 'Recorded simulation measurements; NOT hardware certification', fontsize=13)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    plot(parser.parse_args().folder)
