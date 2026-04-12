#!/usr/bin/env python3
"""Run the duplication-collapse analysis study from a YAML manifest."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _preconfigure_cuda_visible_devices(config_path: Path) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    execution = raw.get("execution") or {}
    value = str(execution.get("cuda_visible_devices") or "").strip()
    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the study YAML config.")
    args = parser.parse_args()
    _preconfigure_cuda_visible_devices(args.config)
    from src.analysis.duplication_collapse_analysis import (  # noqa: E402
        run_duplication_collapse_analysis_study,
    )

    result = run_duplication_collapse_analysis_study(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
