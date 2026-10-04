from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"


def main() -> None:
    manifest = DATA / "MANIFEST_SHA256.txt"
    failures: list[str] = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name, size = line.split("  ")
        path = DATA / name
        checked += 1
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        payload = path.read_bytes()
        if len(payload) != int(size):
            failures.append(f"size mismatch: {name}")
        if hashlib.sha256(payload).hexdigest() != digest:
            failures.append(f"hash mismatch: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {checked} source-data files.")


if __name__ == "__main__":
    main()
