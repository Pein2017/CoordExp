"""Regression tests for Common Objects in Context category namespaces."""

from __future__ import annotations


def test_category_namespaces_preserve_evaluator_and_official_identifiers() -> None:
    from src.eval.detection_categories import (
        COCO_80_CATEGORY_IDS,
        COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
        COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME,
        COCO_80_OFFICIAL_CATEGORY_NAME_BY_ID,
    )

    assert COCO_80_CATEGORY_IDS is COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME
    assert COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME["stop sign"] == 12
    assert COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["stop sign"] == 13
    assert COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME["bottle"] == 40
    assert COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["bottle"] == 44
    assert COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME["toothbrush"] == 80
    assert COCO_80_OFFICIAL_CATEGORY_ID_BY_NAME["toothbrush"] == 90
    assert COCO_80_OFFICIAL_CATEGORY_NAME_BY_ID[90] == "toothbrush"


def test_category_namespace_registry_has_stable_complete_fingerprint() -> None:
    from src.eval.detection_categories import (
        COCO_80_CATEGORY_NAMESPACE_ENTRIES,
        COCO_80_CATEGORY_NAMESPACE_SHA256,
        coco_80_category_namespace_payload,
        coco_80_category_namespace_sha256,
    )

    assert len(COCO_80_CATEGORY_NAMESPACE_ENTRIES) == 80
    assert len(coco_80_category_namespace_payload()) == 80
    assert COCO_80_CATEGORY_NAMESPACE_SHA256 == coco_80_category_namespace_sha256()
    assert COCO_80_CATEGORY_NAMESPACE_SHA256 == (
        "b7b3c8f2361189c88e103369fa7a6c207c81b79999f9610897cf949d546ea4fe"
    )
