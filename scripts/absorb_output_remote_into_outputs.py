#!/usr/bin/env python3
"""Absorb output_remote files into outputs without disturbing active writers.

The migration contract is intentionally conservative:

- compare `output_remote/` against `outputs/` by relative path
- stop if same-path files disagree in size
- copy only files that do not yet exist in `outputs/`
- leave `output_remote/` untouched so active training can continue writing there
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _scan(source: Path, dest: Path, *, sample_limit: int) -> dict[str, Any]:
    missing: list[str] = []
    conflicts: list[dict[str, Any]] = []
    copied_roots: set[str] = set()

    for dirpath, _, filenames in os.walk(source):
        base = Path(dirpath)
        for filename in filenames:
            src = base / filename
            rel = src.relative_to(source)
            dst = dest / rel
            rel_text = rel.as_posix()
            if dst.exists():
                try:
                    src_size = src.stat().st_size
                    dst_size = dst.stat().st_size
                except FileNotFoundError:
                    continue
                if src_size != dst_size:
                    conflicts.append(
                        {
                            "relative_path": rel_text,
                            "source_size": src_size,
                            "dest_size": dst_size,
                        }
                    )
                continue
            missing.append(rel_text)
            if rel.parts:
                copied_roots.add(rel.parts[0])

    return {
        "source": str(source.resolve()),
        "destination": str(dest.resolve()),
        "missing_count": len(missing),
        "conflict_count": len(conflicts),
        "top_level_roots": sorted(copied_roots),
        "missing_samples": missing[:sample_limit],
        "conflict_samples": conflicts[:sample_limit],
    }


def _run_rsync(source: Path, dest: Path) -> None:
    if shutil.which("rsync") is None:
        raise SystemExit("rsync is required for --apply but was not found on PATH")
    subprocess.run(
        ["rsync", "-a", "--ignore-existing", f"{source}/", f"{dest}/"],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="output_remote", help="source root")
    parser.add_argument("--dest", default="outputs", help="destination root")
    parser.add_argument(
        "--report",
        default="temp/output_remote_absorb/report.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="number of sample entries to keep in the report",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the no-overwrite copy after a clean precheck",
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    report_path = Path(args.report)

    if not source.exists():
        raise SystemExit(f"source root does not exist: {source}")
    if not dest.exists():
        raise SystemExit(f"destination root does not exist: {dest}")

    report = _scan(source, dest, sample_limit=args.sample_limit)
    report["apply_requested"] = bool(args.apply)
    report["apply_completed"] = False

    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report["conflict_count"] > 0:
        report["status"] = "blocked_on_conflicts"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    if args.apply and report["missing_count"] > 0:
        _run_rsync(source, dest)
        report["apply_completed"] = True

    report["status"] = "ok"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
