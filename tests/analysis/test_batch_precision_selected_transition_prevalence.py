from __future__ import annotations

import pytest

from scripts.research.run_batch_precision_selected_transition_prevalence import (
    BOX_END,
    BOX_START,
    COORDINATE_TOKEN_START,
    DEFAULT_IMAGE_IDS,
    OBJECT_REF_END,
    OBJECT_REF_START,
    build_parser,
    canonical_first_action,
    compare_action_records,
    extract_first_complete_row,
    select_source_bundles,
    validate_source_bundle,
)


def _row(description: list[int]) -> list[int]:
    return [
        OBJECT_REF_START,
        *description,
        OBJECT_REF_END,
        BOX_START,
        COORDINATE_TOKEN_START + 1,
        COORDINATE_TOKEN_START + 2,
        COORDINATE_TOKEN_START + 3,
        COORDINATE_TOKEN_START + 4,
        BOX_END,
    ]


def test_extract_first_complete_row_allows_variable_description_length() -> None:
    first = _row([41])
    second = _row([51, 52, 53])
    bundle = {
        "decode_result": {
            "prompt_token_ids": [1, 2, 3],
            "generated_token_ids": [99, *first, *second],
        }
    }

    extracted = extract_first_complete_row(bundle)

    assert extracted["row_token_ids"] == first
    assert extracted["description_token_ids"] == [41]
    assert extracted["row_token_start"] == 1
    assert extracted["row_token_end"] == 1 + len(first)


def test_extract_first_complete_row_skips_malformed_candidate() -> None:
    malformed = [
        OBJECT_REF_START,
        41,
        OBJECT_REF_END,
        BOX_START,
        COORDINATE_TOKEN_START + 1,
        BOX_END,
    ]
    valid = _row([51, 52])
    bundle = {
        "decode_result": {
            "prompt_token_ids": [1],
            "generated_token_ids": [*malformed, *valid],
        }
    }

    assert extract_first_complete_row(bundle)["row_token_ids"] == valid


def test_extract_first_complete_row_rejects_missing_complete_row() -> None:
    with pytest.raises(ValueError, match="no complete canonical object row"):
        extract_first_complete_row(
            {
                "decode_result": {
                    "prompt_token_ids": [1],
                    "generated_token_ids": [OBJECT_REF_START, 41, OBJECT_REF_END],
                }
            }
        )


def test_select_source_bundles_defaults_and_filters() -> None:
    assert [image_id for image_id, _ in select_source_bundles(None)] == list(
        DEFAULT_IMAGE_IDS
    )
    assert [image_id for image_id, _ in select_source_bundles(["7574", "8629"])] == [
        "7574",
        "8629",
    ]
    with pytest.raises(ValueError, match="unique"):
        select_source_bundles(["7574", "7574"])
    with pytest.raises(ValueError, match="unknown"):
        select_source_bundles(["999999"])


def test_validate_source_bundle_requires_frozen_arm_cell_and_image() -> None:
    bundle = {
        "scheduled_request": {
            "arm": {"arm_code": "FULL_BAG_K", "full_name": "full"},
            "cell_index": 0,
            "sampling_seed": 7,
        },
        "execution_evidence": {
            "image_id": 7574,
            "source_image_sha256": "image",
        },
    }
    assert validate_source_bundle(bundle, image_id="7574")["cell_index"] == 0
    bundle["scheduled_request"]["cell_index"] = 1
    with pytest.raises(ValueError, match="cell zero"):
        validate_source_bundle(bundle, image_id="7574")


def _record(
    *,
    description: str,
    bbox: list[float],
    trailing: int = 0,
    object_span_id: str = "request:span-0",
) -> dict:
    return {
        "first_free_action": {
            "status": "valid_row",
            "generated_order": 0,
            "object_span_id": object_span_id,
            "description": description,
            "bbox_xyxy": bbox,
            "token_start": 0,
            "token_end": 3,
            "token_ids": [11, 12, 13],
        },
        "generated_token_ids": [11, 12, 13, trailing],
    }


def test_compare_action_records_separates_primary_from_later_divergence() -> None:
    single = _record(description="cup", bbox=[1.0, 2.0, 3.0, 4.0], trailing=0)
    homogeneous = [
        _record(description="cup", bbox=[1.0, 2.0, 3.0, 4.0], trailing=1)
        for _ in range(4)
    ]

    comparison = compare_action_records(single, homogeneous)

    assert comparison["single_vs_homogeneous_first_action_exactly_equal"] is True
    assert comparison["single_vs_homogeneous_complete_generation_exactly_equal"] is False
    assert comparison["primary_first_action_divergence"] is False


def test_request_owned_span_identifier_is_not_action_semantics() -> None:
    first = _record(
        description="cup",
        bbox=[1.0, 2.0, 3.0, 4.0],
        object_span_id="request-a:span-0",
    )["first_free_action"]
    second = _record(
        description="cup",
        bbox=[1.0, 2.0, 3.0, 4.0],
        object_span_id="request-b:span-0",
    )["first_free_action"]

    assert canonical_first_action(first) == canonical_first_action(second)


def test_homogeneous_later_trajectory_difference_does_not_invalidate_first_action() -> None:
    single = _record(description="cup", bbox=[1.0, 2.0, 3.0, 4.0], trailing=0)
    homogeneous = [
        _record(
            description="cup",
            bbox=[1.0, 2.0, 3.0, 4.0],
            trailing=index,
            object_span_id=f"request-{index}:span-0",
        )
        for index in range(4)
    ]

    comparison = compare_action_records(single, homogeneous)

    assert comparison["homogeneous_first_actions_exactly_equal"] is True
    assert comparison["homogeneous_complete_generations_exactly_equal"] is False


def test_compare_action_records_detects_first_action_divergence() -> None:
    single = _record(description="cup", bbox=[1.0, 2.0, 3.0, 4.0])
    homogeneous = [
        _record(description="fork", bbox=[5.0, 6.0, 7.0, 8.0])
        for _ in range(4)
    ]

    assert compare_action_records(single, homogeneous)[
        "primary_first_action_divergence"
    ] is True


def test_compare_action_records_rejects_nonidentical_homogeneous_copies() -> None:
    single = _record(description="cup", bbox=[1.0, 2.0, 3.0, 4.0])
    homogeneous = [
        _record(description="cup", bbox=[1.0, 2.0, 3.0, 4.0])
        for _ in range(4)
    ]
    homogeneous[-1] = _record(description="fork", bbox=[5.0, 6.0, 7.0, 8.0])
    with pytest.raises(ValueError, match="homogeneous first actions disagree"):
        compare_action_records(single, homogeneous)


def test_cli_accepts_dtype_and_repeated_image_filters(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--model-dtype",
            "float32",
            "--image-id",
            "7574",
            "--image-id",
            "8629",
        ]
    )
    assert args.model_dtype == "float32"
    assert args.image_ids == ["7574", "8629"]
