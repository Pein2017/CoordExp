from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.data import (
    ObjectOrderingPlan,
    normalize_detection_row,
    parse_raw_detection_row,
)


DATASET = Path("public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl")


def _first_raw_row() -> dict:
    with DATASET.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _raw_sample():
    return parse_raw_detection_row(_first_raw_row())


def test_sorted_normalization_preserves_source_order_and_object_provenance() -> None:
    raw = _raw_sample()

    sample = normalize_detection_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(seed_source="unit:source-order"),
    )

    assert sample.image_id == raw.image_id
    assert sample.file_name == raw.file_name
    assert sample.metadata == raw.metadata
    assert len(sample.objects) == len(raw.objects)
    assert sample.realized_source_object_indices == tuple(range(len(raw.objects)))
    assert sample.object_ordering.strategy == "sorted"
    assert sample.object_ordering.seed is None
    assert sample.object_ordering.seed_source == "unit:source-order"
    assert sample.object_ordering.realized_source_object_indices == tuple(
        range(len(raw.objects))
    )

    first = sample.objects[0]
    assert first.normalized_object_index == 0
    assert first.source_object_index == 0
    assert first.desc == "clock"
    assert first.category_id == 85
    assert first.category_name == "clock"
    assert first.coco_ann_id == 1666628
    assert first.bbox_2d.tokens == raw.objects[0].bbox_2d.tokens
    assert first.object_instance_id == (
        "coco2017:val:image_id=139:source_object_index=0:coco_ann_id=1666628"
    )


def test_random_permutation_normalization_is_seeded_and_records_realized_order() -> None:
    raw = _raw_sample()
    plan = ObjectOrderingPlan.random_permutation(
        seed=17,
        seed_source="custom.seed + image_id",
    )

    first = normalize_detection_row(raw, object_ordering=plan)
    second = normalize_detection_row(raw, object_ordering=plan)

    assert first.realized_source_object_indices == second.realized_source_object_indices
    assert first.realized_source_object_indices != tuple(range(len(raw.objects)))
    assert sorted(first.realized_source_object_indices) == list(range(len(raw.objects)))
    assert first.object_ordering.strategy == "random_permutation"
    assert first.object_ordering.seed == 17
    assert first.object_ordering.seed_source == "custom.seed + image_id"
    assert (
        first.object_ordering.realized_source_object_indices
        == first.realized_source_object_indices
    )

    for normalized_index, obj in enumerate(first.objects):
        source = raw.objects[obj.source_object_index]
        assert obj.normalized_object_index == normalized_index
        assert obj.bbox_2d == source.bbox_2d
        assert obj.category_id == source.category_id
        assert obj.category_name == source.category_name
        assert obj.coco_ann_id == source.coco_ann_id
        assert f"source_object_index={source.source_object_index}" in obj.object_instance_id
        assert obj.object_instance_id.endswith(f"coco_ann_id={source.coco_ann_id}")


def test_stable_object_instance_ids_distinguish_duplicate_coco_ann_ids() -> None:
    row = json.loads(json.dumps(_first_raw_row()))
    row["objects"][1]["coco_ann_id"] = row["objects"][0]["coco_ann_id"]
    raw = parse_raw_detection_row(row)

    sample = normalize_detection_row(
        raw,
        object_ordering=ObjectOrderingPlan.sorted(seed_source="unit:duplicate-ann-id"),
    )

    first, second = sample.objects[:2]
    assert first.coco_ann_id == second.coco_ann_id
    assert first.source_object_index == 0
    assert second.source_object_index == 1
    assert first.object_instance_id != second.object_instance_id
    assert "source_object_index=0" in first.object_instance_id
    assert "source_object_index=1" in second.object_instance_id


def test_random_permutation_requires_a_seed() -> None:
    with pytest.raises(ValueError, match="requires seed"):
        ObjectOrderingPlan.random_permutation(seed=None, seed_source="missing")


def test_normalization_rejects_unknown_ordering_strategy() -> None:
    raw = _raw_sample()
    plan = ObjectOrderingPlan(
        strategy="diagonal",
        seed=None,
        seed_source="unit",
        realized_source_object_indices=(),
    )

    with pytest.raises(ValueError, match="object_ordering.strategy"):
        normalize_detection_row(raw, object_ordering=plan)


def test_sorted_normalization_rejects_source_order_that_is_not_geometry_sorted() -> None:
    row = _first_raw_row()
    row["objects"] = row["objects"][:2]
    row["objects"][0]["bbox_2d"] = [
        "<|coord_500|>",
        "<|coord_500|>",
        "<|coord_600|>",
        "<|coord_600|>",
    ]
    row["objects"][1]["bbox_2d"] = [
        "<|coord_100|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_200|>",
    ]
    raw = parse_raw_detection_row(row)

    with pytest.raises(ValueError, match="sorted object_ordering"):
        normalize_detection_row(
            raw,
            object_ordering=ObjectOrderingPlan.sorted(seed_source="unit:unsorted"),
        )
