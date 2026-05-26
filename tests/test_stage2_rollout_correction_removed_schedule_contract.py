from __future__ import annotations

import pytest

from src.config.schema import TrainingConfig


def _minimal_stage2_payload() -> dict[str, object]:
    return {
        "model": {"model": "x"},
        "template": {"template": "qwen3_vl"},
        "custom": {
            "trainer_variant": "stage2_rollout_correction",
            "train_jsonl": "train.jsonl",
            "val_jsonl": "val.jsonl",
        },
        "training": {"output_dir": "out"},
        "stage2_rollout_correction": {
            "pipeline": {
                "objective": [
                    {
                        "name": "residual_set_correction",
                        "enabled": True,
                        "weight": 1.0,
                        "application": {"preset": "rollout_self_prefix"},
                        "config": {},
                    }
                ],
                "diagnostics": [],
            },
            "correction": {},
        },
    }


@pytest.mark.parametrize("removed_key", ["schedule", "b_ratio"])
def test_stage2_rollout_correction_rejects_ab_scheduler_keys(
    removed_key: str,
) -> None:
    payload = _minimal_stage2_payload()
    payload["stage2_rollout_correction"][removed_key] = (  # type: ignore[index]
        {"b_ratio": 0.85} if removed_key == "schedule" else 0.85
    )

    with pytest.raises(ValueError) as exc_info:
        TrainingConfig.from_dict(payload)

    message = str(exc_info.value)
    assert "stage2_rollout_correction" in message
    assert removed_key in message
    assert "has no A/B scheduler" in message
