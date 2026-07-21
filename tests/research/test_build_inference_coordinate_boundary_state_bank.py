from pathlib import Path

import pytest

from src.config.fingerprint import sha256_json
from src.rollout_calibration import CheckpointIdentity
from scripts.research import build_inference_coordinate_boundary_state_bank as builder
from scripts.research.build_inference_coordinate_boundary_state_bank import (
    checkpoint_identity_for_inference_run,
    first_wrong_coordinate,
    load_allowed_image_ids,
    split_complete_rows,
    unique_owner_match,
)


def test_split_complete_rows_keeps_only_strict_complete_rows() -> None:
    row = [151646, 42, 151647, 151648, 151671, 151672, 151673, 151674, 151649]
    assert split_complete_rows([9, *row, 10, 151646, 11]) == [row]


def test_first_wrong_coordinate_requires_accepted_earlier_prefix() -> None:
    result = first_wrong_coordinate([101, 205, 300, 400], [100, 200, 300, 400], tolerance=3)
    assert result is not None
    index, observations = result
    assert index == 1
    assert [item["coordinate"] for item in observations] == ["x1", "y1"]


def test_load_allowed_image_ids_requires_a_nonempty_unique_integer_list(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text("[7, 3]\n", encoding="utf-8")
    assert load_allowed_image_ids(cohort) == {3, 7}

    cohort.write_text("[7, 7]\n", encoding="utf-8")
    try:
        load_allowed_image_ids(cohort)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate image IDs must be rejected")


def test_unique_owner_match_refuses_a_close_second_instance() -> None:
    prediction = {
        "description": "person",
        "bbox": [0, 0, 128, 80],
        "coord_bins": [0, 0, 100, 100],
    }
    clear = [
        {"description": "person", "bbox": [0, 0, 100, 100], "object_id": "a"},
        {"description": "person", "bbox": [500, 500, 600, 600], "object_id": "b"},
    ]
    assert unique_owner_match(prediction, clear, minimum_iou=0.55, minimum_margin=0.30)[0]["object_id"] == "a"
    ambiguous = [
        {"description": "person", "bbox": [0, 0, 100, 100], "object_id": "a"},
        {"description": "person", "bbox": [5, 5, 105, 105], "object_id": "b"},
    ]
    assert unique_owner_match(prediction, ambiguous, minimum_iou=0.55, minimum_margin=0.30) is None


def test_unique_owner_match_uses_coord_bins_not_pixel_bbox() -> None:
    prediction = {
        "description": "cow",
        "bbox": [709, 337, 805, 425],
        "coord_bins": [568, 405, 645, 511],
    }
    owner = {"description": "cow", "bbox": [567, 402, 643, 511]}
    distractor = {"description": "cow", "bbox": [100, 100, 200, 200]}

    match = unique_owner_match(
        prediction,
        [owner, distractor],
        minimum_iou=0.35,
        minimum_margin=0.15,
    )

    assert match is not None
    assert match[0] is owner
    assert match[1] > 0.9


def test_refreshed_checkpoint_identity_uses_actual_inference_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter"
    embedding = tmp_path / "embedding"
    adapter.mkdir()
    embedding.mkdir()
    tokens = {"coord_token_count": 1000}
    processor = {"patch_size": 16, "merge_size": 2}
    reference = CheckpointIdentity(
        adapter_fingerprint="1" * 64,
        embedding_delta_fingerprint="2" * 64,
        base_config_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        token_identity_sha256=sha256_json(tokens),
        special_token_identity_sha256="6" * 64,
        processor_identity_sha256=sha256_json(processor),
    )
    monkeypatch.setattr(
        builder,
        "inspect_dora_adapter_payload",
        lambda path, expected_base_model_path: {"fingerprint": "8" * 64},
    )
    monkeypatch.setattr(
        builder,
        "inspect_special_token_embedding_delta_payload",
        lambda path, **kwargs: {"fingerprint": "9" * 64},
    )

    actual = checkpoint_identity_for_inference_run(
        inference_manifest={
            "adapter_identity": {"adapter_path": str(adapter)},
            "embedding_delta_identity": {
                "identity": {"delta_path": str(embedding)}
            },
            "frontend_identity": {
                "base_model_path": "/base",
                "base_config_sha256": reference.base_config_sha256,
                "tokenizer_sha256": reference.tokenizer_sha256,
                "tokens": tokens,
                "processor": processor,
            },
        },
        reference_checkpoint=reference,
        source_adapter_path=adapter,
        source_embedding_payload_path=embedding,
    )

    assert actual.adapter_fingerprint == "8" * 64
    assert actual.embedding_delta_fingerprint == "9" * 64
    assert actual.base_config_sha256 == reference.base_config_sha256
    assert actual.processor_identity_sha256 == reference.processor_identity_sha256


def test_refreshed_checkpoint_identity_rejects_inference_frontend_drift(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    embedding = tmp_path / "embedding"
    adapter.mkdir()
    embedding.mkdir()
    reference = CheckpointIdentity(
        adapter_fingerprint="1" * 64,
        embedding_delta_fingerprint="2" * 64,
        base_config_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        token_identity_sha256=sha256_json({"coord_token_count": 1000}),
        special_token_identity_sha256="6" * 64,
        processor_identity_sha256=sha256_json({"patch_size": 16}),
    )

    with pytest.raises(ValueError, match="frontend identity is incompatible"):
        checkpoint_identity_for_inference_run(
            inference_manifest={
                "adapter_identity": {"adapter_path": str(adapter)},
                "embedding_delta_identity": {
                    "identity": {"delta_path": str(embedding)}
                },
                "frontend_identity": {
                    "base_model_path": "/base",
                    "base_config_sha256": reference.base_config_sha256,
                    "tokenizer_sha256": "f" * 64,
                    "tokens": {"coord_token_count": 1000},
                    "processor": {"patch_size": 16},
                },
            },
            reference_checkpoint=reference,
            source_adapter_path=adapter,
            source_embedding_payload_path=embedding,
        )
