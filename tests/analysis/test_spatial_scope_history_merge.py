from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from src.analysis.spatial_scope_history.merge import (
    ExpectedImageFrame,
    PredictionSet,
    merge_arm_predictions,
    merge_image_predictions,
    normalize_call_prediction,
)
from src.analysis.spatial_scope_history.parse_score_evidence import (
    build_canonical_parse_score_receipts,
)
from src.analysis.spatial_scope_history.schedule import primary_arm_definition
from src.analysis.spatial_scope_history.spatial import SpatialGrid
from src.analysis.spatial_scope_history.cohort_ledger import sha256_payload
from src.common.errors import DataContractError
from src.eval.detection_categories import COCO_80_CATEGORY_NAMESPACE_SHA256
from src.inference.backend import TokenTrace
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from spatial_scope_history_fixtures import (
    build_test_execution_evidence,
    build_test_execution_evidence_with_result,
)


def _full_prediction(
    prediction_id: str,
    bbox: tuple[float, float, float, float],
    *,
    category: str = "person",
    score: float = 0.8,
    call_id: str = "call-00",
    image_id: int = 1,
    arm: str = "FULL_BAG_K",
    source_width: int = 128,
    source_height: int = 128,
):
    cell_index = None if arm == "FULL_SINGLE" else 0
    sampling_seed = 1 + int(
        sha256_payload({"call_id": call_id, "prediction_id": prediction_id})[:12],
        16,
    )
    bins = _pixel_box_to_bins(
        bbox,
        width=source_width,
        height=source_height,
    )
    evidence, result = _object_execution(
        arm=arm,
        image_id=image_id,
        cell_index=cell_index,
        sampling_seed=sampling_seed,
        source_width=source_width,
        source_height=source_height,
        category=category,
        bins=bins,
        score=score,
    )
    receipt = build_canonical_parse_score_receipts(
        execution_evidence=evidence,
        decode_result=result,
    )[0]
    return normalize_call_prediction(
        execution_evidence=evidence,
        parse_score_receipt=receipt,
    )


def _spatial_prediction(
    prediction_id: str,
    *,
    cell_index: int,
    bins: tuple[int, int, int, int],
    arm: str = "TILE_RESET",
    score: float = 0.8,
):
    sampling_seed = 100 + int(
        sha256_payload({"prediction_id": prediction_id})[:12],
        16,
    )
    grid = SpatialGrid.build(source_width=128, source_height=128)
    mode = {
        "TILE_RESET": "tile_reset",
        "MASK_RESET": "mask_reset",
        "MASK_CUMULATIVE": "mask_cumulative",
    }[arm]
    receipt = grid.plan(cell_index=cell_index, variant_mode=mode).coordinate_receipt(
        bins
    )
    evidence, result = _object_execution(
        arm=arm,
        image_id=1,
        cell_index=cell_index,
        sampling_seed=sampling_seed,
        source_width=128,
        source_height=128,
        category="person",
        bins=bins,
        score=score,
    )
    parse_score_receipt = build_canonical_parse_score_receipts(
        execution_evidence=evidence,
        decode_result=result,
    )[0]
    return normalize_call_prediction(
        execution_evidence=evidence,
        parse_score_receipt=parse_score_receipt,
        spatial_coordinate_receipt=receipt,
    )


def _object_execution(
    *,
    arm: str,
    image_id: int,
    cell_index: int | None,
    sampling_seed: int,
    source_width: int,
    source_height: int,
    category: str,
    bins: tuple[int, int, int, int],
    score: float,
):
    coordinate_tokens = tuple(f"<|coord_{value}|>" for value in bins)
    pieces = (
        OBJECT_REF_START_TOKEN,
        category,
        OBJECT_REF_END_TOKEN,
        BOX_START_TOKEN,
        *coordinate_tokens,
        BOX_END_TOKEN,
    )
    selected_text = {
        OBJECT_REF_START_TOKEN,
        OBJECT_REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        *coordinate_tokens,
    }
    selected_logprob = math.log(score)
    token_trace = tuple(
        TokenTrace(
            step_index=index,
            token_id=1000 + index,
            token_text=text,
            logprob=(selected_logprob if text in selected_text else -0.1),
            is_stop=False,
            is_pad=False,
            backend="hf",
            backend_mode="generate",
            response_family="hf",
        )
        for index, text in enumerate(pieces)
    )
    return build_test_execution_evidence_with_result(
        arm_code=arm,
        image_id=image_id,
        cell_index=cell_index,
        sampling_seed=sampling_seed,
        source_width=source_width,
        source_height=source_height,
        raw_generated_text="".join(pieces),
        parser_text="".join(pieces),
        generated_token_ids=tuple(row.token_id for row in token_trace),
        token_trace=token_trace,
    )


def _pixel_box_to_bins(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        round(x1 * 1000 / width),
        round(y1 * 1000 / height),
        round(x2 * 1000 / width),
        round(y2 * 1000 / height),
    )


def _full_execution_and_receipt(*, seed: int):
    evidence, result = _object_execution(
        arm="FULL_SINGLE",
        image_id=1,
        cell_index=None,
        sampling_seed=seed,
        source_width=128,
        source_height=128,
        category="person",
        bins=(10, 20, 200, 300),
        score=0.8,
    )
    receipt = build_canonical_parse_score_receipts(
        execution_evidence=evidence,
        decode_result=result,
    )[0]
    return evidence, result, receipt


def test_merge_uses_schedule_owned_primary_arm_declarations() -> None:
    declaration = primary_arm_definition("MASK_CUMULATIVE")
    assert declaration.full_name == (
        "Full-Canvas Masked Region with Cumulative Accepted-Row Prefix"
    )
    assert declaration.input_policy == "full_canvas_core_plus_halo_mask"
    with pytest.raises(DataContractError, match="unknown_primary_arm"):
        build_test_execution_evidence(
            arm_code="TILE_CUMULATIVE_EXPLORATORY",
        )


def test_full_prediction_normalizes_category_and_has_no_ownership() -> None:
    prediction = _full_prediction(
        "prediction-1", (1.25, 2.5, 20.0, 30.0), category="  Traffic   Light "
    )
    assert prediction.normalized_category_name == "traffic light"
    assert prediction.ownership_status == "not_applicable"
    assert prediction.owner_cell_index is None
    assert prediction.local_bbox_xyxy is None
    assert prediction.spatial_coordinate_receipt is None
    json.dumps(prediction.to_json_dict(), allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("category_name", "evaluator_category_id", "official_coco_category_id"),
    [
        ("stop sign", 12, 13),
        ("bottle", 40, 44),
        ("toothbrush", 80, 90),
    ],
)
def test_prediction_serialization_declares_both_category_namespaces(
    category_name: str,
    evaluator_category_id: int,
    official_coco_category_id: int,
) -> None:
    prediction = _full_prediction(
        f"prediction-{category_name}",
        (1.0, 2.0, 20.0, 30.0),
        category=category_name,
    )

    assert prediction.evaluator_category_id == evaluator_category_id
    assert prediction.official_coco_category_id == official_coco_category_id
    assert prediction.category_registry_sha256 == COCO_80_CATEGORY_NAMESPACE_SHA256
    payload = prediction.to_json_dict()
    assert payload["evaluator_category_id"] == evaluator_category_id
    assert payload["official_coco_category_id"] == official_coco_category_id
    assert not _contains_mapping_key(payload, "category_id")
    merged = merge_image_predictions(
        image_id="1",
        arm_identifier="FULL_BAG_K",
        predictions=(prediction,),
    )
    assert not _contains_mapping_key(merged.to_json_dict(), "category_id")


def test_unknown_coco_category_cannot_enter_normalized_predictions() -> None:
    evidence, result = _object_execution(
        arm="FULL_SINGLE",
        image_id=1,
        cell_index=None,
        sampling_seed=900,
        source_width=128,
        source_height=128,
        category="dragon",
        bins=(10, 20, 200, 300),
        score=0.8,
    )
    receipt = build_canonical_parse_score_receipts(
        execution_evidence=evidence,
        decode_result=result,
    )[0]
    with pytest.raises(DataContractError, match="category_unknown"):
        normalize_call_prediction(
            execution_evidence=evidence,
            parse_score_receipt=receipt,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"normalized_category_name": "dog"},
        {"parsed_bbox_xyxy": (2.0, 2.0, 20.0, 30.0)},
        {"generated_row_index": 1},
        {"score": 0.25},
        {"score_policy_id": "fabricated-score-policy"},
        {"raw_generated_text_sha256": "1" * 64},
        {"full_token_trace_sha256": "2" * 64},
    ],
)
def test_parse_score_receipt_rejects_independent_semantic_mutation(mutation) -> None:
    _, _, receipt = _full_execution_and_receipt(seed=901)
    with pytest.raises(DataContractError):
        replace(receipt, **mutation)


def test_parse_score_builder_rejects_independent_raw_output_mutation() -> None:
    evidence, result, _ = _full_execution_and_receipt(seed=902)
    mutated = replace(result, raw_generated_text=result.raw_generated_text + "junk")
    with pytest.raises(DataContractError, match="raw_output_trace|parser_text"):
        build_canonical_parse_score_receipts(
            execution_evidence=evidence,
            decode_result=mutated,
        )


def test_parse_score_builder_rejects_independent_token_trace_mutation() -> None:
    evidence, result, _ = _full_execution_and_receipt(seed=903)
    mutated_trace = list(result.token_trace)
    mutated_trace[0] = replace(mutated_trace[0], logprob=-2.0)
    mutated = replace(result, token_trace=mutated_trace)
    with pytest.raises(DataContractError, match="decode_result"):
        build_canonical_parse_score_receipts(
            execution_evidence=evidence,
            decode_result=mutated,
        )


def test_parse_score_builder_rejects_token_text_mutation_not_covered_by_score_hash() -> (
    None
):
    evidence, result, _ = _full_execution_and_receipt(seed=907)
    mutated_trace = list(result.token_trace)
    mutated_trace[1] = replace(mutated_trace[1], token_text="dog")
    mutated = replace(result, token_trace=mutated_trace)
    with pytest.raises(DataContractError, match="raw_output_trace"):
        build_canonical_parse_score_receipts(
            execution_evidence=evidence,
            decode_result=mutated,
        )


def test_parse_score_receipt_rejects_cross_execution_association() -> None:
    _, _, receipt = _full_execution_and_receipt(seed=904)
    other_evidence, _, _ = _full_execution_and_receipt(seed=905)
    with pytest.raises(DataContractError, match="association"):
        normalize_call_prediction(
            execution_evidence=other_evidence,
            parse_score_receipt=receipt,
        )


def test_spatial_receipt_maps_local_to_global_and_marks_halo_only_row() -> None:
    # Cell 0 has a [0,64)x[0,64) tile. This row's center is in core cell 1.
    non_owning = _spatial_prediction("halo-row", cell_index=0, bins=(500, 0, 999, 500))
    assert non_owning.local_bbox_xyxy == (32.0, 0.0, 64.0, 32.0)
    assert non_owning.global_bbox_xyxy == (32.0, 0.0, 64.0, 32.0)
    assert non_owning.owner_cell_index == 1
    assert non_owning.ownership_status == "non_owning"

    owning = _spatial_prediction("owning-row", cell_index=0, bins=(0, 0, 250, 250))
    assert owning.ownership_status == "owned"
    assert owning.owner_cell_index == 0

    result = merge_image_predictions(
        image_id="1",
        arm_identifier="TILE_RESET",
        predictions=(non_owning, owning),
    )
    assert [item.prediction_id for item in result.raw_any_call.predictions] == [
        non_owning.prediction_id,
        owning.prediction_id,
    ]
    assert [
        item.prediction_id for item in result.non_owning_diagnostic.predictions
    ] == [non_owning.prediction_id]
    assert [item.prediction_id for item in result.pre_merge.predictions] == [
        owning.prediction_id
    ]


def test_spatial_prediction_requires_exact_reproducible_receipt() -> None:
    evidence, result = _object_execution(
        arm="TILE_RESET",
        image_id=1,
        cell_index=0,
        sampling_seed=906,
        source_width=128,
        source_height=128,
        category="person",
        bins=(0, 0, 250, 250),
        score=0.8,
    )
    parse_score_receipt = build_canonical_parse_score_receipts(
        execution_evidence=evidence,
        decode_result=result,
    )[0]
    with pytest.raises(DataContractError, match="receipt_missing"):
        normalize_call_prediction(
            execution_evidence=evidence,
            parse_score_receipt=parse_score_receipt,
        )
    wrong_receipt = (
        SpatialGrid.build(
            source_width=256,
            source_height=128,
        )
        .plan(cell_index=0, variant_mode="tile_reset")
        .coordinate_receipt((0, 0, 250, 250))
    )
    with pytest.raises(DataContractError, match="receipt_mismatch"):
        normalize_call_prediction(
            execution_evidence=evidence,
            parse_score_receipt=parse_score_receipt,
            spatial_coordinate_receipt=wrong_receipt,
        )


def test_greedy_non_maximum_suppression_is_kept_only_and_nontransitive() -> None:
    highest = _full_prediction("a", (0.0, 0.0, 10.0, 10.0), score=0.9)
    middle = _full_prediction("b", (1.0, 0.0, 11.0, 10.0), score=0.8)
    tail = _full_prediction("c", (2.0, 0.0, 12.0, 10.0), score=0.7)
    result = merge_image_predictions(
        image_id="1",
        arm_identifier="FULL_BAG_K",
        predictions=(tail, middle, highest),
    )
    assert [item.prediction_id for item in result.post_merge.predictions] == [
        highest.prediction_id,
        tail.prediction_id,
    ]
    assert [
        (edge.suppressor_prediction_id, edge.suppressed_prediction_id)
        for edge in result.suppression_edges
    ] == [(highest.prediction_id, middle.prediction_id)]
    assert result.suppression_edges[0].intersection_over_union >= 0.70


def test_non_maximum_suppression_is_class_wise_and_ties_are_deterministic() -> None:
    person_late = _full_prediction(
        "person-late",
        (0.0, 0.0, 10.0, 10.0),
        score=0.8,
    )
    person_early = _full_prediction(
        "person-early",
        (0.0, 0.0, 10.0, 10.0),
        score=0.8,
    )
    dog = _full_prediction("dog", (0.0, 0.0, 10.0, 10.0), category="dog", score=0.1)
    result = merge_image_predictions(
        image_id="1",
        arm_identifier="FULL_BAG_K",
        predictions=(person_late, dog, person_early),
    )
    expected_person = min(
        (person_late, person_early),
        key=lambda item: item.deterministic_rank_key,
    )
    assert [item.prediction_id for item in result.post_merge.predictions] == [
        expected_person.prediction_id,
        dog.prediction_id,
    ]
    assert (
        result.suppression_edges[0].suppressor_prediction_id
        == expected_person.prediction_id
    )


def test_strict_duplicate_components_are_diagnostic_and_transitive() -> None:
    first = _full_prediction("a", (0.0, 0.0, 20.0, 20.0), score=0.9)
    second = _full_prediction("b", (1.0, 0.0, 21.0, 20.0), score=0.8)
    third = _full_prediction("c", (2.0, 0.0, 22.0, 20.0), score=0.7)
    result = merge_image_predictions(
        image_id="1",
        arm_identifier="FULL_BAG_K",
        predictions=(third, second, first),
    )
    component = result.pre_merge_strict_duplicate_components[0]
    assert component.representative_prediction_id == first.prediction_id
    assert component.member_prediction_ids == (
        first.prediction_id,
        second.prediction_id,
        third.prediction_id,
    )
    assert result.pre_merge_strict_duplicate_rate == pytest.approx(2 / 3)
    assert result.post_merge_strict_duplicate_components == ()
    assert result.post_merge_strict_duplicate_rate == 0.0


def test_duplicate_prediction_identifiers_fail_before_merge() -> None:
    first = _full_prediction("duplicate", (0.0, 0.0, 10.0, 10.0))
    second = _full_prediction("duplicate", (20.0, 20.0, 30.0, 30.0))
    with pytest.raises(DataContractError, match="duplicate_prediction_id"):
        merge_image_predictions(
            image_id="1",
            arm_identifier="FULL_BAG_K",
            predictions=(first, second),
        )


def test_prediction_set_rejects_cross_image_or_cross_arm_membership() -> None:
    prediction = _full_prediction("prediction-1", (0.0, 0.0, 10.0, 10.0))
    with pytest.raises(DataContractError, match="prediction_set_image"):
        PredictionSet(
            image_id="2",
            arm=primary_arm_definition("FULL_BAG_K"),
            view_name="raw_any_call",
            predictions=(prediction,),
        )


def test_prediction_set_rejects_source_frame_mismatch() -> None:
    first = _full_prediction("prediction-1", (0.0, 0.0, 10.0, 10.0))
    second = _full_prediction(
        "prediction-2",
        (0.0, 0.0, 10.0, 10.0),
        source_width=256,
        source_height=128,
    )
    with pytest.raises(DataContractError, match="image_frame_mismatch"):
        merge_image_predictions(
            image_id="1",
            arm_identifier="FULL_BAG_K",
            predictions=(first, second),
        )


def test_arm_merge_groups_images_in_stable_order_and_is_json_safe() -> None:
    image_b = _full_prediction("b", (0.0, 0.0, 10.0, 10.0), image_id=2)
    image_a = _full_prediction("a", (0.0, 0.0, 10.0, 10.0), image_id=1)
    result = merge_arm_predictions(
        arm_identifier="FULL_BAG_K",
        predictions=(image_b, image_a),
        expected_images=(
            ExpectedImageFrame("2", 128, 128),
            ExpectedImageFrame("1", 128, 128),
            ExpectedImageFrame("3", 128, 128),
        ),
        allowed_arm_identifiers=("FULL_SINGLE", "FULL_BAG_K"),
    )
    assert [item.image_id for item in result.image_results] == [
        "1",
        "2",
        "3",
    ]
    assert result.image_results[-1].post_merge.predictions == ()
    payload = result.to_json_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "Full-Image K-Rollout Independent Bagging" in encoded


def test_empty_image_merge_returns_not_applicable_duplicate_rates() -> None:
    result = merge_image_predictions(
        image_id="image-1",
        arm_identifier="FULL_SINGLE",
        predictions=(),
        source_width=128,
        source_height=128,
    )
    assert result.pre_merge_strict_duplicate_rate is None
    assert result.post_merge_strict_duplicate_rate is None
    json.dumps(result.to_json_dict(), allow_nan=False)


def test_arm_merge_rejects_disallowed_arm() -> None:
    prediction = _full_prediction("prediction-1", (0.0, 0.0, 10.0, 10.0))
    with pytest.raises(DataContractError, match="arm_disallowed"):
        merge_arm_predictions(
            arm_identifier="FULL_BAG_K",
            predictions=(prediction,),
            expected_images=(ExpectedImageFrame("1", 128, 128),),
            allowed_arm_identifiers=("FULL_SINGLE",),
        )


def test_arm_merge_rejects_unknown_and_duplicate_universe_entries() -> None:
    prediction = _full_prediction(
        "prediction-1",
        (0.0, 0.0, 10.0, 10.0),
        image_id=4,
    )
    with pytest.raises(DataContractError, match="unknown_prediction_image"):
        merge_arm_predictions(
            arm_identifier="FULL_BAG_K",
            predictions=(prediction,),
            expected_images=(ExpectedImageFrame("1", 128, 128),),
            allowed_arm_identifiers=("FULL_BAG_K",),
        )
    with pytest.raises(DataContractError, match="expected_universe_duplicate"):
        merge_arm_predictions(
            arm_identifier="FULL_BAG_K",
            predictions=(),
            expected_images=(
                ExpectedImageFrame("1", 128, 128),
                ExpectedImageFrame("1", 128, 128),
            ),
            allowed_arm_identifiers=("FULL_BAG_K",),
        )
    with pytest.raises(DataContractError, match="expected_universe_missing"):
        merge_arm_predictions(
            arm_identifier="FULL_BAG_K",
            predictions=(),
            expected_images=(),
            allowed_arm_identifiers=("FULL_BAG_K",),
        )


def _contains_mapping_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_mapping_key(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_mapping_key(item, target) for item in value)
    return False
