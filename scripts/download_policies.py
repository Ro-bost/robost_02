"""Download the two evaluated RS02 stair policies from the GitHub release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "policies.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(record: dict, force: bool = False) -> Path:
    target = ROOT / record["path"]
    if target.exists() and sha256(target) == record["sha256"]:
        print(f"ready: {target.relative_to(ROOT)}")
        return target
    if target.exists() and not force:
        raise SystemExit(f"Checksum mismatch: {target}. Use --force to replace it.")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        request = Request(record["url"], headers={"User-Agent": "robost-rs02/0.2"})
        with urlopen(request, timeout=60) as response:
            shutil.copyfileobj(response, temporary)
    try:
        if temp_path.stat().st_size != record["bytes"] or sha256(temp_path) != record["sha256"]:
            raise SystemExit(f"Downloaded policy failed verification: {record['url']}")
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"downloaded: {target.relative_to(ROOT)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an invalid local file")
    args = parser.parse_args()
    for record in json.loads(MANIFEST.read_text()):
        download(record, force=args.force)


if __name__ == "__main__":
    main()
