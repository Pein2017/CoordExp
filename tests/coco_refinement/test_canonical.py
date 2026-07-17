from __future__ import annotations

import copy
from uuid import UUID

import pytest

from src.common.errors import DataContractError
from src.coco_refinement.canonical import canonicalize_objects
from src.coco_refinement.models import DraftSnapshotBinding, NativeTaskIdentity


LOCAL_KEY = "local:3f5dd17d-46ee-43dd-9fc0-51a5fd603938"


def _source_object(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "region_key": "train:coco:10",
        "bbox_2d": [101, 202, 303, 404],
        "category_name": "traffic light",
        "category_id": 10,
        "coco_ann_id": 10,
    }
    value.update(updates)
    return value


def _local_object(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "region_key": LOCAL_KEY,
        "bbox_2d": [1, 2, 998, 999],
        "category_name": "toothbrush",
        "category_id": 90,
    }
    value.update(updates)
    return value


def test_canonicalizes_native_objects_and_adapts_to_core_snapshot() -> None:
    draft = canonicalize_objects([_local_object(), _source_object()], split="train")

    assert [region["region_key"] for region in draft.to_json_regions()] == [
        LOCAL_KEY,
        "train:coco:10",
    ]
    assert draft.to_json_regions()[0] == _local_object()
    assert len(draft.semantic_hash) == 64
    assert len(draft.result_hash) == 64
    assert draft.inference_receipts == ()

    snapshot = draft.to_authoritative_snapshot(
        DraftSnapshotBinding(
            split="train",
            project_id="coco-refinement:train",
            image_id=9,
            task_id="train:9",
            annotation_id="train:9",
            draft_id="train:9:7",
            revision=7,
            updated_at="2026-07-17T00:00:00Z",
            base_row_hash="a" * 64,
            observed_generation=3,
        )
    )
    assert snapshot.annotation_revision == "7"
    assert snapshot.draft_updated_at == "2026-07-17T00:00:00Z"
    assert snapshot.semantic_hash == draft.semantic_hash
    assert [region["region_key"] for region in snapshot.regions] == [
        LOCAL_KEY,
        "train:coco:10",
    ]


@pytest.mark.parametrize(
    "bbox",
    [
        [1.0, 2, 3, 4],
        [-1, 2, 3, 4],
        [1, 2, 1000, 4],
        [1, 2, 1, 4],
        [1, 2, 3, 2],
    ],
)
def test_rejects_noncanonical_norm1000_geometry(bbox: list[object]) -> None:
    with pytest.raises(DataContractError) as exc_info:
        canonicalize_objects([_source_object(bbox_2d=bbox)], split="train")
    assert exc_info.value.code.startswith("data.bbox_")


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (_source_object(category_id=9), "label_studio.category_mismatch"),
        (_source_object(category_name="Traffic Light"), "label_studio.category_name"),
        (_source_object(region_key="train:coco:11"), "coco_refinement.source_identity"),
        (_source_object(region_key="val:coco:10"), "coco_refinement.region_split"),
        (_local_object(region_key="local:not-a-uuid"), "coco_refinement.local_key"),
        (_local_object(coco_ann_id=91), "coco_refinement.new_identity"),
        (
            _local_object(region_key="roi:receipt-1:result-0", coco_ann_id=3),
            "coco_refinement.new_identity",
        ),
    ],
)
def test_rejects_invalid_category_key_or_identity(
    value: dict[str, object], code: str
) -> None:
    with pytest.raises(DataContractError) as exc_info:
        canonicalize_objects([value], split="train")
    assert exc_info.value.code == code


def test_preserves_allocated_negative_identity_for_local_and_roi_objects() -> None:
    draft = canonicalize_objects(
        [
            _local_object(coco_ann_id=-1),
            _local_object(
                region_key="roi:receipt-1:result-0",
                coco_ann_id=-2,
                metadata={
                    "inference_origin": True,
                    "receipt_id": "receipt-1",
                    "request_id": "request-1",
                    "result_id": "result-0",
                    "draft_revision": "6",
                },
            ),
        ],
        split="train",
    )

    assert [region["coco_ann_id"] for region in draft.to_json_regions()] == [-1, -2]
    assert draft.inference_receipts == ("receipt-1",)


def test_roi_key_supports_opaque_receipt_and_result_ids_with_colons() -> None:
    receipt_id = "roi-receipt:request-1"
    result_id = "request-1:result-0"
    draft = canonicalize_objects(
        [
            _local_object(
                region_key=f"roi:{receipt_id}:{result_id}",
                metadata={
                    "inference_origin": True,
                    "receipt_id": receipt_id,
                    "request_id": "request-1",
                    "result_id": result_id,
                    "draft_revision": "6",
                },
            )
        ],
        split="train",
    )

    assert draft.to_json_regions()[0]["region_key"] == (
        "roi:roi-receipt:request-1:request-1:result-0"
    )
    assert draft.inference_receipts == (receipt_id,)


def test_metadata_is_allowlisted_and_presentation_is_not_persisted_or_hashed() -> None:
    base = _local_object()
    with_presentation = copy.deepcopy(base)
    with_presentation["presentation"] = {
        "color": "#abcdef",
        "visible": False,
        "selected": True,
    }

    plain = canonicalize_objects([base], split="train")
    decorated = canonicalize_objects([with_presentation], split="train")

    assert decorated.to_json_regions() == plain.to_json_regions()
    assert decorated.semantic_hash == plain.semantic_hash
    assert decorated.result_hash == plain.result_hash

    invalid = _local_object(metadata={"inference_origin": True, "ui_color": "red"})
    with pytest.raises(DataContractError) as exc_info:
        canonicalize_objects([invalid], split="train")
    assert exc_info.value.code == "coco_refinement.metadata_fields"


@pytest.mark.parametrize(
    "objects",
    [
        [_source_object(), _source_object()],
        [_local_object(coco_ann_id=-1), _local_object(region_key="local:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", coco_ann_id=-1)],
    ],
)
def test_rejects_duplicate_stable_keys_or_object_ids(
    objects: list[dict[str, object]],
) -> None:
    with pytest.raises(DataContractError) as exc_info:
        canonicalize_objects(objects, split="train")
    assert exc_info.value.code in {
        "coco_refinement.region_key_duplicate",
        "coco_refinement.coco_ann_id_duplicate",
    }


def test_semantic_hash_is_order_independent_but_exact_hash_preserves_order() -> None:
    first = _source_object()
    second = _local_object()
    forward = canonicalize_objects([first, second], split="train")
    reverse = canonicalize_objects([second, first], split="train")
    first["bbox_2d"] = [1, 1, 2, 2]

    assert forward.semantic_hash == reverse.semantic_hash
    assert forward.result_hash != reverse.result_hash
    assert forward.to_json_regions()[0]["bbox_2d"] == [101, 202, 303, 404]


def test_exact_hash_preserves_approved_inference_provenance() -> None:
    metadata = {
        "inference_origin": True,
        "receipt_id": "receipt-1",
        "request_id": "request-1",
        "result_id": "result-0",
        "draft_revision": "6",
    }
    original = canonicalize_objects(
        [
            _local_object(
                region_key="roi:receipt-1:result-0",
                metadata=metadata,
            )
        ],
        split="train",
    )
    changed = canonicalize_objects(
        [
            _local_object(
                region_key="roi:receipt-1:result-0",
                metadata={**metadata, "draft_revision": "7"},
            )
        ],
        split="train",
    )

    assert original.semantic_hash == changed.semantic_hash
    assert original.result_hash != changed.result_hash


def test_native_task_identity_is_compact_and_strict() -> None:
    task = NativeTaskIdentity(split="val", image_id=139, source_row_index=0)
    assert task.task_key == "val:139"
    assert UUID(LOCAL_KEY.removeprefix("local:"))

    with pytest.raises(DataContractError):
        NativeTaskIdentity(split="test", image_id=139, source_row_index=0)  # type: ignore[arg-type]
    with pytest.raises(DataContractError):
        NativeTaskIdentity(split="val", image_id=0, source_row_index=0)
    with pytest.raises(DataContractError):
        NativeTaskIdentity(split="val", image_id=139, source_row_index=-1)
