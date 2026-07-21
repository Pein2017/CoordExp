from __future__ import annotations

import pytest

from scripts.research.evaluate_frozen_coordinate_state_bank import (
    EvaluationArgumentError,
    aggregate_coordinate_evaluation,
    validate_eval_event_set,
)


def _row(event_id: str, margin: float, legal_mass: float) -> dict[str, object]:
    return {
        "event_id": event_id,
        "coordinate_target_margin": margin,
        "legal_coordinate_token_mass": legal_mass,
        "raw_losses": {
            "rollout_coordinate_boundary": margin + 1.0,
            "rollout_site_token_type_gate": legal_mass + 2.0,
            "total": margin + legal_mass + 3.0,
        },
    }


def test_validate_eval_event_set_rejects_wrong_count_and_duplicates() -> None:
    with pytest.raises(EvaluationArgumentError, match="exactly match"):
        validate_eval_event_set(("e1",), expected_count=2)
    with pytest.raises(EvaluationArgumentError, match="duplicate"):
        validate_eval_event_set(("e1", "e1"), expected_count=2)


def test_validate_eval_event_set_accepts_train_and_rejects_unknown_split() -> None:
    assert validate_eval_event_set(("e1",), expected_count=1, split="train") == ("e1",)
    with pytest.raises(EvaluationArgumentError, match="split='train' or split='eval'"):
        validate_eval_event_set(("e1",), expected_count=1, split="test")
    with pytest.raises(EvaluationArgumentError, match="event-id set"):
        validate_eval_event_set(
            ("e1",), expected_event_ids=("e2",), expected_count=1
        )


def test_aggregate_coordinate_evaluation_is_event_mean_and_raw_loss_mean() -> None:
    aggregate = aggregate_coordinate_evaluation(
        (_row("e2", 2.0, 0.8), _row("e1", 4.0, 0.6))
    )
    assert aggregate["event_ids"] == ["e1", "e2"]
    assert aggregate["event_count"] == 2
    assert aggregate["coordinate_target_margin"] == pytest.approx(3.0)
    assert aggregate["coordinate_target_margin_sum"] == pytest.approx(6.0)
    assert aggregate["legal_coordinate_token_mass"] == pytest.approx(0.7)
    assert aggregate["raw_losses"]["total"]["mean"] == pytest.approx(6.7)
    assert aggregate["math_dtype"] == "float32"


def test_aggregate_coordinate_evaluation_requires_finite_raw_metrics() -> None:
    row = _row("e1", 0.0, 0.5)
    row["raw_losses"] = {
        "rollout_coordinate_boundary": 1.0,
        "rollout_site_token_type_gate": 2.0,
        "total": float("inf"),
    }
    with pytest.raises(EvaluationArgumentError, match="finite"):
        aggregate_coordinate_evaluation((row,))


def test_validate_evaluator_arguments_requires_checkpoint_payloads(tmp_path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "manifest.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter").mkdir()
    (checkpoint / "special_token_embeddings").mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    from scripts.research.evaluate_frozen_coordinate_state_bank import (
        validate_evaluator_arguments,
    )

    paths = validate_evaluator_arguments(
        bank=bank,
        source_checkpoint=checkpoint,
        config=config,
        output=tmp_path / "result.json",
    )
    assert paths["bank_manifest"] == (bank / "manifest.json").resolve()
    with pytest.raises(EvaluationArgumentError, match="checkpoint must contain"):
        validate_evaluator_arguments(
            bank=bank,
            source_checkpoint=tmp_path,
            config=config,
            output=tmp_path / "result.json",
        )
