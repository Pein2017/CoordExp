from __future__ import annotations

import copy

import pytest

from src.label_studio_coco_refinement.draft_adapter import (
    DraftContractError,
    canonicalize_label_studio_draft,
)
from src.label_studio_coco_refinement.geometry import norm1000_bbox_to_label_studio_xywh


def _rectangle(
    *,
    key: str = "train:coco:10",
    label: str = "traffic light",
    bbox: tuple[int, int, int, int] = (101, 202, 303, 404),
    coco_ann_id: int | None = 10,
    creation_ordinal: int | None = None,
    inference_receipt_id: str | None = None,
    inference_request_id: str = "roi-request-1",
    inference_result_id: str = "roi-result-1",
) -> dict[str, object]:
    x, y, width, height = norm1000_bbox_to_label_studio_xywh(bbox)
    meta: dict[str, object] = {
        "coordexp_region_key": key,
        "last_committed_bbox": list(bbox),
    }
    if coco_ann_id is not None:
        meta["coco_ann_id"] = coco_ann_id
    if creation_ordinal is not None:
        meta["coordexp_creation_ordinal"] = creation_ordinal
    if inference_receipt_id is not None:
        meta["coordexp_inference_receipt_id"] = inference_receipt_id
        meta["coordexp_inference_request_id"] = inference_request_id
        meta["coordexp_inference_result_id"] = inference_result_id
    return {
        "id": key,
        "type": "rectanglelabels",
        "from_name": "bbox",
        "to_name": "image",
        "original_width": 640,
        "original_height": 480,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "meta": meta,
    }


def test_canonicalizes_unchanged_source_rectangle_without_float_drift() -> None:
    raw = _rectangle()

    draft = canonicalize_label_studio_draft(
        [raw],
        split="train",
        image_id=42,
        image_width=640,
        image_height=480,
    )

    assert draft.to_json_regions() == [
        {
            "region_key": "train:coco:10",
            "bbox_2d": [101, 202, 303, 404],
            "category_name": "traffic light",
            "category_id": 10,
            "coco_ann_id": 10,
            "creation_ordinal": 0,
            "draft_meta": raw["meta"],
            "label_studio_result": raw,
        },
    ]
    assert len(draft.semantic_hash) == 64
    assert len(draft.result_hash) == 64
    assert draft.inference_receipts == ()


def test_edited_rectangle_uses_outward_norm1000_quantization() -> None:
    raw = _rectangle()
    raw["value"] = copy.deepcopy(raw["value"])
    raw["value"]["x"] = 10.01
    raw["value"]["y"] = 20.02
    raw["value"]["width"] = 30.03
    raw["value"]["height"] = 40.04

    draft = canonicalize_label_studio_draft(
        [raw], split="train", image_id=42, image_width=640, image_height=480
    )

    assert draft.to_json_regions()[0]["bbox_2d"] == [99, 199, 400, 600]


def test_new_region_uses_label_studio_id_and_official_sparse_category() -> None:
    raw = _rectangle(
        key="drawn:stable-1",
        label="toothbrush",
        coco_ann_id=None,
        creation_ordinal=7,
        inference_receipt_id="roi-receipt-1",
    )
    raw["meta"].pop("last_committed_bbox")

    draft = canonicalize_label_studio_draft(
        [raw], split="train", image_id=42, image_width=640, image_height=480
    )

    region = draft.regions[0]
    assert region["region_key"] == "drawn:stable-1"
    assert region["category_id"] == 90
    assert "coco_ann_id" not in region
    assert region["creation_ordinal"] == 7
    assert region["metadata"] == {
        "inference_origin": True,
        "receipt_id": "roi-receipt-1",
        "request_id": "roi-request-1",
        "result_id": "roi-result-1",
    }
    assert draft.inference_receipts == ("roi-receipt-1",)


def test_result_order_does_not_change_canonical_payload_or_receipt_order() -> None:
    first = _rectangle(
        key="drawn:b",
        coco_ann_id=None,
        creation_ordinal=1,
        inference_receipt_id="receipt-b",
    )
    first["meta"].pop("last_committed_bbox")
    second = _rectangle(
        key="drawn:a",
        coco_ann_id=None,
        creation_ordinal=0,
        inference_receipt_id="receipt-a",
    )
    second["meta"].pop("last_committed_bbox")

    forward = canonicalize_label_studio_draft(
        [first, second], split="train", image_id=42, image_width=640, image_height=480
    )
    reverse = canonicalize_label_studio_draft(
        [second, first], split="train", image_id=42, image_width=640, image_height=480
    )

    assert tuple(region["region_key"] for region in forward.regions) == (
        "drawn:a",
        "drawn:b",
    )
    assert forward.regions == reverse.regions
    assert forward.semantic_hash == reverse.semantic_hash
    assert forward.result_hash != reverse.result_hash
    assert (
        forward.inference_receipts
        == reverse.inference_receipts
        == ("receipt-a", "receipt-b")
    )


def test_empty_draft_is_preserved_for_ui_but_not_claimed_commit_valid() -> None:
    draft = canonicalize_label_studio_draft(
        [], split="val", image_id=7, image_width=320, image_height=240
    )

    assert draft.regions == ()
    assert draft.inference_receipts == ()
    assert len(draft.semantic_hash) == 64


def test_canonical_payload_is_deeply_immutable_and_thaws_defensively() -> None:
    raw = _rectangle()
    draft = canonicalize_label_studio_draft(
        [raw], split="train", image_id=42, image_width=640, image_height=480
    )
    original_hashes = (draft.semantic_hash, draft.result_hash)

    raw["value"]["x"] = 99
    with pytest.raises(TypeError):
        draft.regions[0]["bbox_2d"] = (1, 2, 3, 4)
    with pytest.raises(TypeError):
        draft.regions[0]["label_studio_result"]["value"]["x"] = 99
    detached = draft.to_json_regions()
    detached[0]["bbox_2d"][0] = 999

    assert draft.to_json_regions()[0]["bbox_2d"] == [101, 202, 303, 404]
    assert (draft.semantic_hash, draft.result_hash) == original_hashes


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result.update(type="polygonlabels"), "rectanglelabels"),
        (lambda result: result.update(from_name="other"), "bbox/image"),
        (lambda result: result.update(original_width=641), "dimensions"),
        (lambda result: result.update(image_rotation=1), "image rotation"),
        (lambda result: result["value"].update(rotation=1), "rotation"),
        (lambda result: result["value"].update(rectanglelabels=["human"]), "canonical"),
        (
            lambda result: result["value"].update(rectanglelabels=["person", "car"]),
            "exactly one",
        ),
        (
            lambda result: result["meta"].update(coordexp_region_key="other"),
            "region key",
        ),
        (lambda result: result["meta"].update(coco_ann_id=True), "coco_ann_id"),
        (
            lambda result: result["meta"].update(coordexp_creation_ordinal=True),
            "creation ordinal",
        ),
    ],
)
def test_rejects_noncanonical_or_misaligned_label_studio_results(
    mutation, message: str
) -> None:
    raw = _rectangle()
    mutation(raw)

    with pytest.raises(DraftContractError, match=message):
        canonicalize_label_studio_draft(
            [raw], split="train", image_id=42, image_width=640, image_height=480
        )


def test_rejects_duplicate_region_keys_and_non_json_metadata() -> None:
    duplicate = _rectangle()
    with pytest.raises(DraftContractError, match="duplicate region key"):
        canonicalize_label_studio_draft(
            [duplicate, duplicate],
            split="train",
            image_id=42,
            image_width=640,
            image_height=480,
        )

    malformed = _rectangle()
    malformed["meta"]["not_json"] = {1, 2, 3}
    with pytest.raises(DraftContractError, match="JSON"):
        canonicalize_label_studio_draft(
            [malformed], split="train", image_id=42, image_width=640, image_height=480
        )


def test_inference_linkage_requires_exact_fields_and_cannot_be_training_overridden() -> (
    None
):
    incomplete = _rectangle(
        key="drawn:inference",
        coco_ann_id=None,
        inference_receipt_id="receipt-1",
    )
    incomplete["meta"].pop("coordexp_inference_result_id")
    with pytest.raises(DraftContractError, match="receipt, request, and result"):
        canonicalize_label_studio_draft(
            [incomplete], split="train", image_id=42, image_width=640, image_height=480
        )

    overridden = _rectangle()
    overridden["meta"]["coordexp_training_metadata"] = {
        "inference_origin": True,
        "receipt_id": "browser-forged",
    }
    with pytest.raises(DraftContractError, match="cannot override inference"):
        canonicalize_label_studio_draft(
            [overridden], split="train", image_id=42, image_width=640, image_height=480
        )
