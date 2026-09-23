"""Independent-process repeated stair evaluations; retain failures and all logs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from robost_paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', choices=['gait', 'support', 'route', 'rhythm', 'guard'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heights', nargs='+', type=int, choices=[15, 20], default=[15, 20])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--duration', type=float, default=40.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for height in args.heights:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                name = f'h{height}_seed{seed}_repeat{repeat}'
                folder = args.output / name
                command = [sys.executable, '-m', f'rs02_rl_{args.adapter}', 'evaluate',
                           '--checkpoint', str(args.checkpoint.resolve()), '--output', str(folder.resolve()),
                           '--num-envs', '1', '--stairs-cm', str(height), '--speed', str(args.speed),
                           '--duration', str(args.duration), '--seed', str(seed)]
                if args.video:
                    command.append('--video')
                with (args.output / f'{name}.log').open('x') as log:
                    run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                record = dict(height_cm=height, seed=seed, repeat=repeat,
                              folder=str(folder.resolve()), process_exit_code=run.returncode)
                if (folder / 'evaluation.json').exists():
                    result = json.loads((folder / 'evaluation.json').read_text())
                    record.update({key: result[key] for key in ('course_completed', 'course_completed_at_s',
                        'left_course', 'max_x_before_failure', 'completion_definition', 'checkpoint_sha256')})
                    record.update(result['trials'][0])
                records.append(record)
                summary = dict(adapter=args.adapter, checkpoint=str(args.checkpoint.resolve()),
                               speed=args.speed, runs=records,
                               completed=sum(r.get('course_completed', False) for r in records),
                               total=len(records))
                (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
                print(name, 'completed=', record.get('course_completed'),
                      'time=', record.get('course_completed_at_s'),
                      'failure=', record.get('failure_reasons'), flush=True)


if __name__ == '__main__':
    main()
