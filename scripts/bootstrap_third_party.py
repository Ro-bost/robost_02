"""Restore pinned upstream sources without replacing local work."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def main():
    pins = json.loads((ROOT / 'config/third-party.lock.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=[p['name'] for p in pins])
    args = parser.parse_args()
    for pin in pins:
        if args.only and pin['name'] not in args.only:
            continue
        target = ROOT / 'third_party' / pin['name']
        if target.exists():
            if not (target / '.git').exists():
                raise SystemExit(f'Existing non-git directory: {target}')
            if git('-C', target, 'status', '--porcelain'):
                raise SystemExit(f'Local changes present: {target}')
            if git('-C', target, 'rev-parse', 'HEAD') != pin['commit']:
                raise SystemExit(f'Existing checkout differs from pin: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            git('init', target)
            git('-C', target, 'remote', 'add', 'origin', pin['url'])
            git('-C', target, 'fetch', '--depth=1', 'origin', pin['commit'])
            git('-C', target, 'checkout', '--detach', 'FETCH_HEAD')
        print(pin['name'], git('-C', target, 'rev-parse', 'HEAD'))


if __name__ == '__main__':
    main()
