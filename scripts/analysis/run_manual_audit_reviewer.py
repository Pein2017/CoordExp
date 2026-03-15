#!/usr/bin/env python
"""Launch a lightweight browser reviewer for unmatched-proposal manual audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.manual_audit_reviewer import (  # noqa: E402
    serve_manual_audit_reviewer,
)


def _resolve_audit_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Audit path does not exist: {path}")
    candidates = [
        path / "manual_audit_recommended96.csv",
        path / "manual_audit_priority48.csv",
        path / "manual_audit_full96.csv",
        path / "manual_audit_recommended96.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix == ".csv":
            return candidate
    csv_candidates = sorted(path.glob("manual_audit*.csv"))
    if csv_candidates:
        return csv_candidates[0]
    raise FileNotFoundError(
        f"Could not find a manual audit CSV under directory: {path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        required=True,
        help=(
            "Path to a manual-audit CSV, or a directory containing "
            "manual_audit_recommended96.csv / manual_audit_priority48.csv."
        ),
    )
    parser.add_argument(
        "--labels-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional writable labels JSONL. Defaults to "
            "<audit-csv-dir>/manual_audit_labels.jsonl."
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Best-effort attempt to open the reviewer in a local browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_csv = _resolve_audit_csv(args.audit_csv)
    serve_manual_audit_reviewer(
        audit_csv=audit_csv,
        labels_jsonl=args.labels_jsonl,
        host=str(args.host),
        port=int(args.port),
        open_browser=bool(args.open_browser),
    )


if __name__ == "__main__":
    main()
