"""Check portable documentation links, runtime model integrity and release selection."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

from export_release import ROOT, source_files


def main():
    errors = []
    for folder in ('src', 'scripts', 'tests'):
        for p in (ROOT / folder).glob('*.py'):
            try:
                ast.parse(p.read_text(), filename=str(p))
            except SyntaxError as e:
                errors.append(str(e))
    for name, expected in json.loads((ROOT / 'models/manifest.json').read_text()).items():
        p = ROOT / 'models' / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append(f'Model checksum mismatch: {p}')
    paths = source_files()
    selected = set(paths)
    for p in paths:
        if p.stat().st_size > 50 * 1024**2:
            errors.append(f'Oversized source file: {p}')
        if p.suffix != '.md':
            continue
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            link = unquote(link.split('#')[0].strip('<>'))
            if not link or '://' in link or link.startswith('mailto:'):
                continue
            target = (p.parent / link).resolve()
            if not target.exists():
                errors.append(f'Broken link: {p.relative_to(ROOT)} -> {link}')
            elif target.is_file() and target not in selected:
                errors.append(f'Link target absent from source release: {p.relative_to(ROOT)} -> {link}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'OK: syntax, model hashes, Markdown links, {len(paths)} release files '
          f'({sum(p.stat().st_size for p in paths)/1024**2:.2f} MiB uncompressed)')


if __name__ == '__main__':
    main()
