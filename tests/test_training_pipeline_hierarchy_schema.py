from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import ConfigLoader
from test_detection_training_config_contract import (
    _detection_payload,
    _stage2_rollout_correction_payload,
)
from src.config.schema import DetectionTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "pipeline_id"),
    [
        (
            "configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml",
            "stage1_research_teacher_forcing",
        ),
        (
            "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml",
            "stage2_rollout_correction",
        ),
    ],
)
def test_pipeline_id_is_the_materialized_target_hierarchy_selector(
    relative_path: str,
    pipeline_id: str,
) -> None:
    cfg = ConfigLoader.load_materialized_training_config(str(REPO_ROOT / relative_path))

    assert isinstance(cfg, DetectionTrainingConfig)
    assert cfg.pipeline.id == pipeline_id


def test_target_hierarchy_rejects_custom_trainer_variant_selector() -> None:
    payload = _stage2_rollout_correction_payload()
    payload["custom"] = {"trainer_variant": "stage2_rollout_correction"}

    with pytest.raises(ValueError, match=r"custom\.trainer_variant.*pipeline\.id"):
        DetectionTrainingConfig.from_mapping(payload)


def test_target_hierarchy_requires_pipeline_id() -> None:
    payload = _detection_payload()
    payload.pop("pipeline")

    with pytest.raises(ValueError, match=r"Missing detection config sections.*pipeline"):
        DetectionTrainingConfig.from_mapping(payload)

