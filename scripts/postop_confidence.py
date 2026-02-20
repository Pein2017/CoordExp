#!/usr/bin/env python
"""YAML-first entrypoint for offline confidence post-operation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.eval.confidence_postop import run_confidence_postop_from_config


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "YAML config requires PyYAML (import yaml). Install it in the ms env."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("confidence post-op config must be a YAML mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoordExp confidence post-op (offline, CPU-only, YAML-first)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to confidence post-op YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    summary = run_confidence_postop_from_config(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
