from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.analyze_human13_row_contrast_successor import (
    analyze_successor_outputs,
)


ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-12-human13-k-union-to-greedy-overfit-screen"
)


def test_successor_analysis_requires_exact_declared_output_arms() -> None:
    with pytest.raises(ValueError, match="differ"):
        analyze_successor_outputs(
            manifest_path=ROOT / "manifest/human13-k-union-manifest.json",
            source_discovery_path=ROOT / "discovery/source/trajectories.jsonl",
            output_paths=(),
            successor_arm_ids=("R1",),
            resolved_plan_sha256="a" * 64,
            resolved_config_sha256="b" * 64,
        )
