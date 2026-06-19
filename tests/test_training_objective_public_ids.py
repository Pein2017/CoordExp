from __future__ import annotations

import pytest

from src.config.schema import DetectionTrainingConfig
from test_detection_training_config_contract import _detection_payload


def test_standard_ce_is_public_stage1_objective_id() -> None:
    payload = _detection_payload()
    payload["pipeline"]["id"] = "stage1_standard_sft"
    payload["objective"] = {
        "id": "standard_ce",
    }

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.pipeline.id == "stage1_standard_sft"
    assert cfg.objective is not None
    assert cfg.objective.id == "standard_ce"


def test_research_teacher_forcing_is_public_stage1_objective_id() -> None:
    payload = _detection_payload()
    payload["pipeline"]["id"] = "stage1_research_teacher_forcing"
    payload["objective"] = {"id": "research_teacher_forcing"}

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.pipeline.id == "stage1_research_teacher_forcing"
    assert cfg.objective is not None
    assert cfg.objective.id == "research_teacher_forcing"


@pytest.mark.parametrize("legacy_id", ["teacher_forcing", "token_ce"])
def test_legacy_or_internal_objective_ids_are_not_public(legacy_id: str) -> None:
    payload = _detection_payload()
    payload["objective"]["id"] = legacy_id

    with pytest.raises(ValueError, match=legacy_id):
        DetectionTrainingConfig.from_mapping(payload)
