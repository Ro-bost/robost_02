"""Build an explicit source archive and optional selected policy archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('README.md', 'AGENTS.md', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml',
         'Makefile', '.gitignore', '.gitattributes', '.ignore')
TREES = ('src', 'tests', 'native', 'scripts', 'config', 'docs', 'models', '.github')
SKIP = {'__pycache__', '.git'}
EXTENSIONS = {'.py', '.cpp', '.md', '.json', '.toml', '.yml', '.yaml', '.txt',
              '.lock', '.png', '.stl', '.xml', '.urdf'}


def source_files():
    paths = [ROOT / p for p in FILES] + [ROOT / 'third_party/README.md']
    for tree in TREES:
        for p in (ROOT / tree).rglob('*'):
            if p.is_file() and not p.is_symlink() and not SKIP.intersection(p.parts) and p.suffix in EXTENSIONS:
                paths.append(p)
    return sorted(set(paths))


def archive(target, paths, manifest_name):
    # Exclusive creation protects any previously delivered release.
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        manifest = {}
        for path in paths:
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            manifest[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            info = tarfile.TarInfo('robost/' + name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('robost/' + manifest_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(f'{target}: {len(paths)} files, {target.stat().st_size / 1024**2:.2f} MiB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/robost-source.tar.gz')
    parser.add_argument('--with-policies', action='store_true')
    args = parser.parse_args()
    paths = source_files()
    policies = []
    policy_target = args.output.with_name('robost-policies.tar.gz')
    if args.with_policies:
        for record in json.loads((ROOT / 'config/policies.json').read_text()):
            p = ROOT / record['path']
            if hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
                raise SystemExit(f'Policy checksum mismatch: {p}')
            policies.append(p)
    for target in [args.output] + ([policy_target] if policies else []):
        if target.exists():
            raise SystemExit(f'Release already exists: {target}; choose a new output directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive(args.output, paths, 'SOURCE_MANIFEST.json')
    if policies:
        archive(policy_target, policies + [ROOT / 'config/policies.json'], 'POLICY_MANIFEST.json')


if __name__ == '__main__':
    main()
