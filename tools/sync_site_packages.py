#!/usr/bin/env python3
"""Sync the adapters integration into an installed vllm wheel.

The cluster venv carries a prebuilt vllm 0.11.0 wheel; the adaptation
layer is pure Python, so serving the fork means overlaying exactly the
files this branch changes relative to the upstream base. Hand-copying
drifts (a stale protocol.py broke the shared-site contract once) —
run this instead:

    python tools/sync_site_packages.py /path/to/venv

It copies every file in `git diff --name-only BASE..HEAD`, deletes
overlay files the branch no longer has, and prints a manifest.
"""

import subprocess
import sys
from pathlib import Path

BASE = "b8b302cde434df8c9289a2b465406b47ebab1c2d"   # upstream v0.11.0


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    venv = Path(sys.argv[1])
    sp = next(venv.glob("lib/python3.*/site-packages"))
    repo = Path(__file__).resolve().parents[1]
    files = subprocess.run(
        ["git", "diff", "--name-only", BASE, "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    synced = 0
    for rel in files:
        if not rel.startswith("vllm/"):
            continue                     # tests/ etc. are not installed
        src, dst = repo / rel, sp / rel
        if not src.exists():
            if dst.exists():
                dst.unlink()
                print(f"removed {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        synced += 1
    print(f"synced {synced} files from {repo} @ "
          f"{subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo, capture_output=True, text=True).stdout.strip()} "
          f"-> {sp}")


if __name__ == "__main__":
    main()
