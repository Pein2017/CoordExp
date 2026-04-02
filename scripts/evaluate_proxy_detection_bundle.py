#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.eval.proxy_eval_bundle import (
    _resolve_artifacts,
    options_from_config,
    run_proxy_eval_bundle,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "YAML config requires PyYAML (import yaml). Install it in the ms env."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("proxy eval bundle config must be a YAML mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark-aligned COCO and proxy-expanded eval views from one scored run."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    artifacts = _resolve_artifacts(cfg)
    options = options_from_config(cfg)
    summary = run_proxy_eval_bundle(artifacts, options=options)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

