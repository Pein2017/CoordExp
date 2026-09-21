"""Read-only output/source boundary check; never execute or delete findings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path


FORBIDDEN_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".md", ".sh"})
FORBIDDEN_DIRECTORIES = frozenset({".venv", "venv", ".git", "__pycache__"})


def scan_output_root(root: Path) -> dict:
    root = Path(root).absolute()
    if not root.is_dir():
        raise FileNotFoundError(f"output root is missing or not a directory: {root}")
    if root.is_symlink():
        raise ValueError(f"select the resolved output root explicitly: {root.resolve()}")
    findings = []
    files = 0
    links = 0

    def onerror(error):
        raise error

    for directory, directories, filenames in os.walk(root, followlinks=False, onerror=onerror):
        for name in sorted(directories + filenames):
            path = Path(directory) / name
            if path.is_symlink():
                links += 1
                if not path.exists():
                    findings.append({"path": str(path), "reason": "broken_symlink"})
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                findings.append({"path": str(path), "reason": "source_or_prose_in_outputs"})
            elif name in FORBIDDEN_DIRECTORIES:
                findings.append({"path": str(path), "reason": "runtime_or_checkout_in_outputs"})
            if name in filenames and not path.is_symlink():
                files += 1
    return {"root": str(root), "files": files, "symlinks_not_followed": links,
            "findings": findings, "passed": not findings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    reports = [scan_output_root(root) for root in args.root]
    for report in reports:
        report["finding_count"] = len(report["findings"])
        report["reasons"] = dict(Counter(item["reason"] for item in report["findings"]))
        if not args.details:
            report.pop("findings")
    print(json.dumps({"passed": all(item["passed"] for item in reports), "roots": reports}, indent=2))
    raise SystemExit(0 if all(item["passed"] for item in reports) else 1)


if __name__ == "__main__":
    main()
