#!/usr/bin/env python3
"""
Watch docs/code roots and keep the `coordexp-codebase` skill indexes in sync.

Implementation:
- Polling (stdlib only). Runs in the foreground.
- On any tracked-file change under watch roots, re-run update_docs_index.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


WATCH_EXTS = {".md", ".py", ".sh", ".yaml", ".yml"}


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError(f"Could not find repo root from: {start}")


def _snapshot(roots: list[Path]) -> dict[str, tuple[int, int]]:
    """
    Fingerprint tracked files: path -> (mtime_ns, size).
    Deletions/creations also change the dict.
    """
    state: dict[str, tuple[int, int]] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts:
                continue
            if p.suffix.lower() not in WATCH_EXTS:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            state[str(p)] = (st.st_mtime_ns, st.st_size)
    return state


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch-roots",
        default="docs,src,scripts,configs",
        help="Comma-separated watch roots relative to repo root. Default: docs,src,scripts,configs",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds. Default: 2.0",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    repo_root = _find_repo_root(script_path)
    update_script = script_path.parent / "update_docs_index.py"

    roots: list[Path] = []
    for token in args.watch_roots.split(","):
        token = token.strip()
        if not token:
            continue
        roots.append((repo_root / token).resolve())

    update_cmd = [sys.executable, str(update_script)]

    last_state = _snapshot(roots)
    subprocess.run(update_cmd, cwd=str(repo_root), check=False)

    while True:
        time.sleep(args.interval)
        new_state = _snapshot(roots)
        if new_state == last_state:
            continue
        subprocess.run(update_cmd, cwd=str(repo_root), check=False)
        last_state = new_state


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
