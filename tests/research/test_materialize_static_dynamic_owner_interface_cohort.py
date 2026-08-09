from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from scripts.research.materialize_static_dynamic_owner_interface_cohort import (
    CohortContractError,
    FROZEN_CANDIDATES,
    LEDGER_ENVELOPE_CONTRACT,
    LEDGER_SCHEMA_VERSION,
    PanelOwner,
    UNIT_ID,
    _fractional_bbox_cell_weights,
    _normalise_image_plan_identity,
    _panel_owners,
    materialize_cohort,
    sha256_json,
)


def _bbox(candidate: dict[str, object]) -> list[int]:
    return list(candidate["pixel_bbox"])  # type: ignore[arg-type]


def _source_panel() -> list[dict[str, object]]:
    candidates_by_image: dict[int, dict[int, dict[str, object]]] = {}
    for raw in FROZEN_CANDIDATES:
        candidate = dict(raw)
        candidates_by_image.setdefault(int(candidate["image_id"]), {})[
            int(candidate["source_panel_object_index"])
        ] = candidate

    rows: list[dict[str, object]] = []
    for image, candidate_by_index in sorted(candidates_by_image.items()):
        owner_count = 46 if image == 2299 else max(candidate_by_index) + 1
        objects: list[dict[str, object]] = []
        for index in range(owner_count):
            candidate = candidate_by_index.get(index)
            if candidate is not None:
                objects.append(
                    {
                        "category_name": candidate["category"],
                        "bbox_pixel_xyxy": _bbox(candidate),
                        "coco_ann_id": f"ann-{image}-{index}",
                    }
                )
            else:
                objects.append(
                    {
                        "category_name": "background",
                        "bbox_pixel_xyxy": [1500 + index, index, 1501 + index, index + 1],
                        "coco_ann_id": f"background-{image}-{index}",
                    }
                )
        if image == 2299:
            # Exactly 19 unique background owners plus source owner 11 sort
            # before source owner 1, reproducing the real 2299 1 -> 20 case.
            background_indices = [index for index in range(owner_count) if index not in candidate_by_index]
            for ordinal, index in enumerate(background_indices[:19]):
                objects[index]["bbox_pixel_xyxy"] = [10 + ordinal, 10, 11 + ordinal, 11]
        rows.append(
            {
                "image_id": image,
                "width": 2000,
                "height": 1000,
                "objects": objects,
            }
        )
    return rows


def _derived_and_receipt(
    source: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    derived: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    owner_count = 0
    for row_index, row in enumerate(source):
        objects = row["objects"]
        assert isinstance(objects, list)
        indexed = sorted(
            enumerate(objects),
            key=lambda pair: (
                pair[1]["bbox_pixel_xyxy"][0],  # type: ignore[index]
                pair[1]["bbox_pixel_xyxy"][1],  # type: ignore[index]
                pair[0],
            ),
        )
        derived_row = copy.deepcopy(row)
        derived_row["objects"] = [copy.deepcopy(obj) for _, obj in indexed]
        derived.append(derived_row)
        mapping = [
            {
                "source_index": source_index,
                "derived_index": derived_index,
                "object_sha256": sha256_json(obj),
            }
            for derived_index, (source_index, obj) in enumerate(indexed)
        ]
        mappings.append(
            {
                "row_index": row_index,
                "image_id": row["image_id"],
                "mapping": mapping,
            }
        )
        owner_count += len(objects)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "unit_id": UNIT_ID,
        "ordering": "geo_sorted_xy",
        "sort_key": ["decoded_x1", "decoded_y1", "source_index"],
        "source_sha256": sha256_json(source),
        "derived_sha256": sha256_json(derived),
        "coordinate_arity_verified": True,
        "owner_multiset_preserved": True,
        "stable_sort_verified": True,
        "row_count": len(source),
        "owner_count": owner_count,
        "mapping_count": owner_count,
        "mapping_sha256": sha256_json(mappings),
        "source_to_derived": mappings,
    }
    return derived, receipt


def _inputs() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source = _source_panel()
    derived, receipt = _derived_and_receipt(source)
    return source, derived, receipt


def _panel_candidate(
    source: list[dict[str, object]], image_id: int, source_index: int
) -> dict[str, object]:
    row = next(item for item in source if item["image_id"] == image_id)
    objects = row["objects"]
    assert isinstance(objects, list)
    owner = objects[source_index]
    return {
        "image_id": image_id,
        "source_panel_object_index": source_index,
        "gt_owner_id": f"gt:{image_id}:{source_index}",
        "category": owner["category_name"],
        "pixel_bbox": list(owner["bbox_pixel_xyxy"]),
        "historical_labels": ["TP"],
    }


def _record(candidate: dict[str, object], **extra: object) -> dict[str, object]:
    image = int(candidate["image_id"])
    index = int(candidate["source_panel_object_index"])
    boundary = int(extra.get("natural_boundary", 7))
    result: dict[str, object] = {
        "image_id": image,
        "coco_ann_id": f"ann-{image}-{index}",
        "category_name": candidate["category"],
        "bbox_pixel_xyxy": _bbox(candidate),
        "native_tp": False,
        "native_fn": True,
        "strict_complete_row": False,
        "natural_boundary": boundary,
        "natural_boundary_valid": True,
        "support_status": "not_measured",
        "verified_support_claim": False,
    }
    result.update(extra)
    due_boundary = result["natural_boundary"]
    assert isinstance(due_boundary, int)
    native_tp = result["native_tp"] is True
    prefix_ids = [] if native_tp and due_boundary == 0 else [image, index, due_boundary]
    result.setdefault("due_boundary_index", due_boundary)
    result.setdefault("excludes_stop", True)
    result.setdefault("queried_owner_not_covered", True)
    result.setdefault("exact_prefix_token_ids", prefix_ids)
    result.setdefault("exact_prefix_sha256", sha256_json(prefix_ids))
    if native_tp:
        result.setdefault("prefix_semantics", "before_queried_owner_row")
        result.setdefault("covered_owner_ids", [])
        result.setdefault("latest_covered_owner_id", None)
        result.setdefault("boundary_disposition", "native_tp_before_queried_row")
        result.setdefault("is_earliest_eligible_boundary", None)
    else:
        result.setdefault("prefix_semantics", "after_strict_covered_row_pre_stop")
        covered_ids = result.get("covered_owner_ids")
        if not isinstance(covered_ids, list) or not covered_ids:
            fallback_a = f"gt:{image}:{0 if index != 0 else 1}"
            covered_ids = [fallback_a]
            result["covered_owner_ids"] = covered_ids
        result.setdefault("latest_covered_owner_id", covered_ids[-1])
        result.setdefault("boundary_disposition", "native_fn_after_first_covered_row")
        result.setdefault("is_earliest_eligible_boundary", True)
    prefix_end = len(prefix_ids) - 1 if prefix_ids else None
    stop_step = len(prefix_ids) + 5
    result.setdefault("generated_history_start_step", 0 if prefix_ids else None)
    result.setdefault("generated_history_end_step", prefix_end)
    result.setdefault("generated_history_stop_step", stop_step)
    result.setdefault(
        "due_boundary_evidence",
        {
            "boundary_disposition": result["boundary_disposition"],
            "queried_owner_id": candidate["gt_owner_id"],
            "covered_owner_ids": result["covered_owner_ids"],
            "covered_row_count": len(result["covered_owner_ids"]),  # type: ignore[arg-type]
            "latest_covered_owner_id": result["latest_covered_owner_id"],
            "prefix_end_step": prefix_end,
            "stop_step": stop_step,
            "queried_owner_not_covered": True,
        },
    )
    return result


def _support_record(
    h0_record: dict[str, object],
    *,
    verified_support: bool,
    **extra: object,
) -> dict[str, object]:
    result = dict(h0_record)
    result["support_status"] = "measured"
    result["verified_support_claim"] = True
    result["verified_support"] = verified_support
    result.update(extra)
    return result


def _envelope(
    records: list[dict[str, object]],
    source: list[dict[str, object]],
    derived: list[dict[str, object]],
    *,
    checkpoint: str = "S",
    config_fingerprint: str | None = None,
    **extra: object,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "checkpoint": checkpoint,
        "config_fingerprint": config_fingerprint or f"config-{checkpoint}",
        "source_panel_sha256": sha256_json(source),
        "derived_panel_sha256": sha256_json(derived),
        "run_kind": "native_h0",
        "history_complete": True,
        "records": records,
    }
    document.update(extra)
    return document


def _materialize(
    source: list[dict[str, object]],
    derived: list[dict[str, object]],
    receipt: dict[str, object],
    h0: object,
    support: object | None = None,
    **kwargs: object,
) -> dict[str, object]:
    return materialize_cohort(
        source,
        h0,  # type: ignore[arg-type]
        support,  # type: ignore[arg-type]
        derived_panel=derived,
        derived_receipt=receipt,
        **kwargs,
    )


def test_real_source_to_derived_reorder_and_deterministic_manifest(tmp_path: Path) -> None:
    source, derived, receipt = _inputs()
    mapping_2299 = next(row for row in receipt["source_to_derived"] if row["image_id"] == 2299)  # type: ignore[union-attr]
    owner_1 = next(entry for entry in mapping_2299["mapping"] if entry["source_index"] == 1)
    assert owner_1["derived_index"] == 20

    candidates = list(FROZEN_CANDIDATES)
    tp = candidates[4]
    target = candidates[5]
    h0 = _envelope(
        [
            _record(tp, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=3),
            _record(target, natural_boundary=5, covered_owner_ids=[tp["gt_owner_id"]]),
        ],
        source,
        derived,
    )
    output = tmp_path / "cohort.json"
    result = _materialize(source, derived, receipt, h0, output=output)
    cohort = result["cohort"]
    event_2299 = next(event for event in cohort["events"] if event["gt_owner_id"] == "gt:2299:1")
    assert event_2299["panel_identity"]["source_panel_object_index"] == 1
    assert event_2299["panel_identity"]["derived_panel_object_index"] == 20
    assert event_2299["panel_identity"]["mapping_method"] == "coco_ann_id"
    assert cohort["subsets"]["legacy12"]["event_count"] == 28
    assert cohort["subsets"]["image2299"]["event_count"] == 4
    assert result["manifest"]["cohort_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["cohort"]["ledger_contract"]["expected_envelope"] == LEDGER_ENVELOPE_CONTRACT
    assert result["manifest"]["ledger_envelope_contract_sha256"] == sha256_json(
        LEDGER_ENVELOPE_CONTRACT
    )


def test_checkpoint_native_tp_replaces_historical_fn_by_boundary_then_index() -> None:
    source, derived, receipt = _inputs()
    historical_tp = list(FROZEN_CANDIDATES)[4]
    later = _panel_candidate(source, 4134, 0)
    tied_first = _panel_candidate(source, 4134, 1)
    tied_second = _panel_candidate(source, 4134, 2)
    h0 = _envelope(
        [
            _record(historical_tp, natural_boundary=5),
            _record(
                later,
                native_tp=True,
                native_fn=False,
                strict_complete_row=True,
                natural_boundary=3,
            ),
            _record(
                tied_second,
                native_tp=True,
                native_fn=False,
                strict_complete_row=True,
                natural_boundary=2,
            ),
            _record(
                tied_first,
                native_tp=True,
                native_fn=False,
                strict_complete_row=True,
                natural_boundary=2,
            ),
        ],
        source,
        derived,
    )
    result = _materialize(source, derived, receipt, h0)
    events = result["cohort"]["events"]
    replacement = next(
        event for event in events if event["gt_owner_id"] == tied_first["gt_owner_id"]
    )
    assert len(events) == 32
    assert not any(
        event["gt_owner_id"] == historical_tp["gt_owner_id"] for event in events
    )
    assert replacement["primary_stratum"] == "TP"
    assert replacement["historical_labels"] == ["TP"]
    assert replacement["checkpoint_status"]["S"]["native_tp"] is True
    assert replacement["panel_identity"]["mapping_method"] in {
        "coco_ann_id",
        "category_pixel_bbox",
    }
    assert replacement["fallback"]["reason"] == "no_reestablished_native_tp"
    assert (
        replacement["fallback"]["from_owner"]["gt_owner_id"]
        == historical_tp["gt_owner_id"]
    )
    assert (
        replacement["fallback"]["to_owner"]["gt_owner_id"]
        == tied_first["gt_owner_id"]
    )
    assert replacement["fallback"]["checkpoint"] == "S"
    assert "boundary_then_source_panel_object_index" in replacement["fallback"]["rule"]


def test_missing_same_image_native_tp_retains_original_indeterminate() -> None:
    source, derived, receipt = _inputs()
    historical_tp = list(FROZEN_CANDIDATES)[4]
    result = _materialize(
        source,
        derived,
        receipt,
        _envelope([_record(historical_tp)], source, derived),
    )
    event = next(
        item
        for item in result["cohort"]["events"]
        if item["gt_owner_id"] == historical_tp["gt_owner_id"]
    )
    assert event["disposition"] == "indeterminate"
    assert event["checkpoint_status"]["S"]["native_fn"] is True
    assert (
        event["checkpoint_status"]["S"]["disposition"]
        == "no_same_image_native_tp"
    )
    assert event["fallback"]["reason"] == "no_same_image_native_tp"
    assert event["fallback"]["to_owner"] is None


def test_a_native_tp_does_not_replace_its_selected_tp_or_satisfy_s() -> None:
    source, derived, receipt = _inputs()
    selected_tp = list(FROZEN_CANDIDATES)[4]
    a_record = _record(
        selected_tp,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        natural_boundary=0,
    )
    result = _materialize(
        source,
        derived,
        receipt,
        _envelope([a_record], source, derived, checkpoint="A"),
    )
    event = next(
        item
        for item in result["cohort"]["events"]
        if item["gt_owner_id"] == selected_tp["gt_owner_id"]
    )
    assert result["cohort"]["frozen_pool"]["active_checkpoint"] == "A"
    assert event["checkpoint_status"]["A"]["native_tp"] is True
    assert event["checkpoint_status"]["S"]["native_tp"] is None
    assert event["fallback"] is None
    assert not any(
        item["checkpoint"] == "A" and item["from_owner"]["gt_owner_id"] == selected_tp["gt_owner_id"]
        for item in result["cohort"]["frozen_pool"]["native_tp_replacements"]
    )


def test_mixed_h0_checkpoints_require_independent_materializations() -> None:
    source, derived, receipt = _inputs()
    selected_tp = list(FROZEN_CANDIDATES)[4]
    record = _record(
        selected_tp,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        natural_boundary=0,
    )
    with pytest.raises(CohortContractError, match="exactly one H0 checkpoint"):
        _materialize(
            source,
            derived,
            receipt,
            [
                _envelope([record], source, derived, checkpoint="S"),
                _envelope([record], source, derived, checkpoint="A"),
            ],
        )


def test_category_bbox_is_unique_fallback_when_annotation_id_absent() -> None:
    source, derived, receipt = _inputs()
    candidate = list(FROZEN_CANDIDATES)[5]
    image = int(candidate["image_id"])
    index = int(candidate["source_panel_object_index"])
    source_row = next(row for row in source if row["image_id"] == image)
    source_row["objects"][index].pop("coco_ann_id")  # type: ignore[index,union-attr]
    derived, receipt = _derived_and_receipt(source)
    h0_record = _record(candidate)
    h0_record.pop("coco_ann_id")
    result = _materialize(source, derived, receipt, _envelope([h0_record], source, derived))
    event = next(item for item in result["cohort"]["events"] if item["gt_owner_id"] == candidate["gt_owner_id"])
    assert event["panel_identity"]["mapping_method"] == "category_pixel_bbox"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("unit_id", "wrong-unit", "unit_id mismatch"),
        ("run_kind", "intervention", "run_kind must be native_h0"),
        ("derived_panel_sha256", "0" * 64, "derived panel hash mismatch"),
        ("history_complete", False, "history_complete=true"),
        ("history_complete", "yes", "present JSON boolean"),
    ],
)
def test_rejects_invalid_h0_envelope(field: str, value: object, match: str) -> None:
    source, derived, receipt = _inputs()
    h0 = _envelope([_record(list(FROZEN_CANDIDATES)[4])], source, derived)
    h0[field] = value
    with pytest.raises(CohortContractError, match=match):
        _materialize(source, derived, receipt, h0)


@pytest.mark.parametrize(
    ("updates", "remove", "match"),
    [
        ({"native_tp": "yes"}, None, "native_tp must be a JSON boolean when present"),
        ({"strict_complete_row": "false"}, None, "strict_complete_row must be a JSON boolean when present"),
        ({"verified_support": "supported"}, None, "verified_support must be a JSON boolean when present"),
        ({"verified_support": False}, None, "must not contain a verified_support claim"),
        ({"natural_boundary_valid": None}, None, "natural_boundary_valid must be a JSON boolean when present"),
        ({}, "native_fn", "native_fn must be a present JSON boolean"),
        ({"native_tp": True, "native_fn": True}, None, "exactly one"),
        ({"native_tp": False, "native_fn": False}, None, "exactly one"),
        ({"native_tp": False, "native_fn": True, "strict_complete_row": True}, None, "contradicts native TP/FN"),
        ({"status": "supported"}, None, "status has an unsupported string value"),
        ({"support": False}, None, "must not contain a verified_support claim"),
        ({"status": {"native_tp": "yes"}}, None, "nested native_tp must be a JSON boolean"),
    ],
)
def test_rejects_non_boolean_missing_or_contradictory_record_claims(
    updates: dict[str, object], remove: str | None, match: str
) -> None:
    source, derived, receipt = _inputs()
    record = _record(list(FROZEN_CANDIDATES)[5])
    record.update(updates)
    if remove is not None:
        record.pop(remove)
    h0 = _envelope([record], source, derived)
    with pytest.raises(CohortContractError, match=match):
        _materialize(source, derived, receipt, h0)


def test_support_cannot_contradict_same_prefix_h0_claims() -> None:
    source, derived, receipt = _inputs()
    target = list(FROZEN_CANDIDATES)[5]
    h0_record = _record(target)
    h0 = _envelope([h0_record], source, derived)
    support_record = _support_record(
        h0_record,
        verified_support=True,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
    )
    support = _envelope([support_record], source, derived)
    with pytest.raises(CohortContractError, match="contradicts native TP/FN boundary"):
        _materialize(source, derived, receipt, h0, support)


def test_tp_measured_support_false_is_valid_calibration_but_never_b() -> None:
    source, derived, receipt = _inputs()
    target = list(FROZEN_CANDIDATES)[4]
    h0_record = _record(
        target,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        natural_boundary=0,
    )
    result = _materialize(
        source,
        derived,
        receipt,
        _envelope([h0_record], source, derived),
        _envelope(
            [_support_record(h0_record, verified_support=False)], source, derived
        ),
    )
    event = next(
        item
        for item in result["cohort"]["events"]
        if item["gt_owner_id"] == target["gt_owner_id"]
    )
    assert event["checkpoint_status"]["S"]["native_tp"] is True
    assert event["checkpoint_status"]["S"]["verified_support"] is False
    assert event["A_B"]["S"]["pair_status"] == "no_verified_B"
    assert event["A_B"]["S"]["B_verified_uncovered"] is None


@pytest.mark.parametrize(
    ("field", "value", "remove", "match"),
    [
        ("support_status", "measured", False, "support_status must be exactly 'not_measured'"),
        ("support_status", None, True, "support_status must be exactly 'not_measured'"),
        ("verified_support_claim", "false", False, "present JSON boolean"),
        ("verified_support_claim", True, False, "verified_support_claim must be false"),
        ("verified_support_claim", None, True, "present JSON boolean"),
    ],
)
def test_h0_support_state_is_explicitly_not_measured(
    field: str, value: object, remove: bool, match: str
) -> None:
    source, derived, receipt = _inputs()
    record = _record(list(FROZEN_CANDIDATES)[5])
    if remove:
        record.pop(field)
    else:
        record[field] = value
    with pytest.raises(CohortContractError, match=match):
        _materialize(source, derived, receipt, _envelope([record], source, derived))


@pytest.mark.parametrize(
    ("field", "value", "remove", "match"),
    [
        ("support_status", "not_measured", False, "support_status must be exactly 'measured'"),
        ("support_status", None, True, "support_status must be exactly 'measured'"),
        ("verified_support_claim", "true", False, "present JSON boolean"),
        ("verified_support_claim", False, False, "verified_support_claim must be true"),
        ("verified_support_claim", None, True, "present JSON boolean"),
        ("verified_support", "supported", False, "JSON boolean when present"),
        ("verified_support", None, True, "present JSON boolean"),
    ],
)
def test_support_source_requires_measured_boolean_claim(
    field: str, value: object, remove: bool, match: str
) -> None:
    source, derived, receipt = _inputs()
    h0_record = _record(list(FROZEN_CANDIDATES)[5])
    h0 = _envelope([h0_record], source, derived)
    support_record = _support_record(h0_record, verified_support=True)
    if remove:
        support_record.pop(field)
    else:
        support_record[field] = value
    support = _envelope([support_record], source, derived)
    with pytest.raises(CohortContractError, match=match):
        _materialize(source, derived, receipt, h0, support)


def test_rejects_tampered_derived_mapping() -> None:
    source, derived, receipt = _inputs()
    receipt["mapping_sha256"] = "0" * 64
    with pytest.raises(CohortContractError, match="mapping_sha256 mismatch"):
        _materialize(source, derived, receipt, [])


def test_support_must_match_same_h0_boundary_and_exact_prefix() -> None:
    source, derived, receipt = _inputs()
    target = list(FROZEN_CANDIDATES)[5]
    h0_record = _record(target, natural_boundary=5)
    h0 = _envelope([h0_record], source, derived)
    future_h0_shape = _record(target, natural_boundary=6)
    future_support = _support_record(future_h0_shape, verified_support=True)
    support = _envelope([future_support], source, derived)
    with pytest.raises(CohortContractError, match="same-checkpoint H0 boundary/exact-prefix"):
        _materialize(source, derived, receipt, h0, support)


def test_pair_uses_earliest_never_covered_b_and_covered_refs_priority() -> None:
    source, derived, receipt = _inputs()
    candidates = list(FROZEN_CANDIDATES)
    a_ref = candidates[5]
    later_unreferenced_a = candidates[4]
    b_target = candidates[6]
    b_record = _record(
        b_target, natural_boundary=5, covered_owner_ids=[a_ref["gt_owner_id"]]
    )
    h0 = _envelope(
        [
            _record(a_ref, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=2),
            _record(later_unreferenced_a, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=3),
            b_record,
            _record(b_target, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=9),
        ],
        source,
        derived,
    )
    support = _envelope(
        [_support_record(b_record, verified_support=True)], source, derived
    )
    result = _materialize(source, derived, receipt, h0, support)
    event = next(item for item in result["cohort"]["events"] if item["gt_owner_id"] == b_target["gt_owner_id"])
    pair = event["A_B"]["S"]
    assert pair["pair_status"] == "verified_pair"
    assert pair["B_verified_uncovered"]["natural_boundary"] == 5
    assert pair["A_latest_covered"]["gt_owner_id"] == a_ref["gt_owner_id"]
    assert pair["A_latest_covered"]["natural_boundary"] == 2


def test_image2299_requires_fresh_checkpoint_native_support_to_admit_b() -> None:
    source, derived, receipt = _inputs()
    a_owner = next(
        candidate
        for candidate in FROZEN_CANDIDATES
        if candidate["gt_owner_id"] == "gt:2299:1"
    )
    b_owner = next(
        candidate
        for candidate in FROZEN_CANDIDATES
        if candidate["gt_owner_id"] == "gt:2299:2"
    )
    identity = {
        "observed_image_grid_thw": [1, 64, 64],
        "merged_visual_tokens": 1024,
    }
    a_h0 = _record(
        a_owner,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        natural_boundary=0,
        image_plan_identity=identity,
    )
    b_h0 = _record(
        b_owner,
        natural_boundary=1,
        covered_owner_ids=[a_owner["gt_owner_id"]],
        image_plan_identity=identity,
    )
    h0 = _envelope([a_h0, b_h0], source, derived, checkpoint="A")

    h0_only = _materialize(source, derived, receipt, h0)
    h0_only_event = next(
        event
        for event in h0_only["cohort"]["events"]
        if event["gt_owner_id"] == b_owner["gt_owner_id"]
    )
    assert h0_only_event["historical_labels"] == ["NK16"]
    assert h0_only_event["A_B"]["A"] == {
        "A_latest_covered": None,
        "B_verified_uncovered": None,
        "pair_status": "no_verified_B",
    }

    fresh_support = _support_record(
        b_h0,
        verified_support=True,
        intervention="none",
        no_future_or_intervention_leakage=True,
        support_features={"behavioral_transfer": False},
    )
    supported = _materialize(
        source,
        derived,
        receipt,
        h0,
        _envelope([fresh_support], source, derived, checkpoint="A"),
    )
    event = next(
        item
        for item in supported["cohort"]["events"]
        if item["gt_owner_id"] == b_owner["gt_owner_id"]
    )
    pair = event["A_B"]["A"]
    assert event["historical_labels"] == ["NK16"]
    assert event["checkpoint_status"]["A"]["disposition"] == "established"
    assert pair["pair_status"] == "verified_pair"
    assert pair["A_latest_covered"] == {
        "gt_owner_id": a_owner["gt_owner_id"],
        "source_panel_object_index": 1,
        "natural_boundary": 0,
        "strict_complete_row": True,
    }
    assert pair["B_verified_uncovered"] == {
        "gt_owner_id": b_owner["gt_owner_id"],
        "verified_support": True,
        "strict_complete_row": False,
        "natural_boundary": 1,
        "exact_prefix_sha256": b_h0["exact_prefix_sha256"],
    }
    assert event["geometry_launch_eligible"] is True


def test_b_is_excluded_if_strict_covered_earlier() -> None:
    source, derived, receipt = _inputs()
    candidates = list(FROZEN_CANDIDATES)
    b_earlier = candidates[5]
    b_earlier_record = _record(
        b_earlier,
        strict_complete_row=False,
        native_tp=False,
        native_fn=True,
        natural_boundary=5,
    )
    h0 = _envelope(
        [
            _record(b_earlier, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=2),
            b_earlier_record,
        ],
        source,
        derived,
    )
    support = _envelope(
        [
            _support_record(b_earlier_record, verified_support=True),
        ],
        source,
        derived,
    )
    result = _materialize(source, derived, receipt, h0, support)
    events = {item["gt_owner_id"]: item for item in result["cohort"]["events"]}
    assert events[b_earlier["gt_owner_id"]]["A_B"]["S"]["pair_status"] == "no_verified_B"


def test_h0_rejects_covered_refs_that_name_queried_b() -> None:
    source, derived, receipt = _inputs()
    b_named = list(FROZEN_CANDIDATES)[6]
    record = _record(
        b_named, natural_boundary=4, covered_owner_ids=[b_named["gt_owner_id"]]
    )
    with pytest.raises(CohortContractError, match="must exclude queried owner B"):
        _materialize(source, derived, receipt, _envelope([record], source, derived))


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"natural_boundary": "<|im_end|>"}, "must not be terminal"),
        ({"excludes_stop": False}, "excludes_stop must be true"),
        ({"excludes_stop": "true"}, "present JSON boolean"),
        ({"queried_owner_not_covered": False}, "queried_owner_not_covered must be true"),
        ({"due_boundary_index": 99}, "natural boundary and due_boundary_index disagree"),
        ({"prefix_semantics": "terminal_im_end"}, "prefix_semantics contradicts"),
    ],
)
def test_h0_rejects_terminal_or_fake_due_boundaries(
    updates: dict[str, object], match: str
) -> None:
    source, derived, receipt = _inputs()
    record = _record(list(FROZEN_CANDIDATES)[5])
    record.update(updates)
    with pytest.raises(CohortContractError, match=match):
        _materialize(source, derived, receipt, _envelope([record], source, derived))


def test_invalid_native_fn_boundary_is_retained_as_indeterminate() -> None:
    source, derived, receipt = _inputs()
    target = list(FROZEN_CANDIDATES)[5]
    record = _record(target)
    record.update(
        {
            "natural_boundary": None,
            "due_boundary_index": None,
            "natural_boundary_valid": False,
            "boundary_disposition": "no_valid_post_covered_boundary",
            "prefix_semantics": "no_valid_post_covered_boundary",
            "covered_owner_ids": [],
            "latest_covered_owner_id": None,
            "is_earliest_eligible_boundary": False,
            "exact_prefix_token_ids": None,
            "exact_prefix_sha256": None,
            "generated_history_start_step": None,
            "generated_history_end_step": None,
            "due_boundary_evidence": {
                "boundary_disposition": "no_valid_post_covered_boundary",
                "queried_owner_id": target["gt_owner_id"],
                "covered_owner_ids": [],
                "covered_row_count": 0,
                "latest_covered_owner_id": None,
                "prefix_end_step": None,
                "stop_step": 8,
                "queried_owner_not_covered": True,
            },
        }
    )
    result = _materialize(
        source, derived, receipt, _envelope([record], source, derived)
    )
    event = next(
        item
        for item in result["cohort"]["events"]
        if item["gt_owner_id"] == target["gt_owner_id"]
    )
    assert (
        event["checkpoint_status"]["S"]["disposition"]
        == "indeterminate_no_valid_natural_boundary"
    )
    assert event["A_B"]["S"]["pair_status"] == "no_verified_B"

    invalid_support = _support_record(record, verified_support=True)
    with pytest.raises(
        CohortContractError,
        match="only an H0 native FN may use no_valid_post_covered_boundary",
    ):
        _materialize(
            source,
            derived,
            receipt,
            _envelope([record], source, derived),
            _envelope([invalid_support], source, derived),
        )


def test_tp_root_is_the_only_valid_empty_exact_prefix() -> None:
    source, derived, receipt = _inputs()
    target = list(FROZEN_CANDIDATES)[4]
    root = _record(
        target,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        natural_boundary=0,
    )
    _materialize(source, derived, receipt, _envelope([root], source, derived))

    non_root = _record(list(FROZEN_CANDIDATES)[5])
    non_root["exact_prefix_token_ids"] = []
    non_root["exact_prefix_sha256"] = sha256_json([])
    with pytest.raises(CohortContractError, match="only a TP root due-boundary"):
        _materialize(
            source, derived, receipt, _envelope([non_root], source, derived)
        )


def test_tp_and_fn_require_distinct_pre_stop_prefix_semantics() -> None:
    source, derived, receipt = _inputs()
    tp_record = _record(
        list(FROZEN_CANDIDATES)[4],
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        prefix_semantics="after_strict_covered_row_pre_stop",
    )
    with pytest.raises(CohortContractError, match="prefix_semantics contradicts"):
        _materialize(source, derived, receipt, _envelope([tp_record], source, derived))


def test_support_rejects_terminal_im_end_boundary_before_join() -> None:
    source, derived, receipt = _inputs()
    h0_record = _record(list(FROZEN_CANDIDATES)[5])
    support_record = _support_record(h0_record, verified_support=True)
    support_record["natural_boundary"] = "terminal_im_end"
    with pytest.raises(CohortContractError, match="must not be terminal"):
        _materialize(
            source,
            derived,
            receipt,
            _envelope([h0_record], source, derived),
            _envelope([support_record], source, derived),
        )


def test_overlap_support_is_prefix_local_and_exposes_decomposition() -> None:
    source, derived, receipt = _inputs()
    candidates = list(FROZEN_CANDIDATES)
    a_owner = candidates[12]
    target = candidates[13]
    h0_target = _record(
        target, natural_boundary=6, covered_owner_ids=[a_owner["gt_owner_id"]]
    )
    h0 = _envelope(
        [
            _record(a_owner, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=3),
            h0_target,
        ],
        source,
        derived,
    )
    support_record = _support_record(
        h0_target,
        verified_support=True,
        overlap_decomposition={
            "a_exclusive": [1, 2],
            "b_exclusive": [3],
            "shared_core": [4],
        },
    )
    support = _envelope([support_record], source, derived)
    result = _materialize(source, derived, receipt, h0, support)
    event = next(item for item in result["cohort"]["events"] if item["gt_owner_id"] == target["gt_owner_id"])
    assert event["A_B"]["S"]["pair_status"] == "verified_pair"
    assert event["SO_exclusive_shared_decomposition"]["shared_core"] == [4]


def test_custom_pool_must_preserve_images_tp_and_geometry_strata() -> None:
    source, derived, receipt = _inputs()
    no_so = [candidate for candidate in FROZEN_CANDIDATES if "SO" not in candidate["historical_labels"]]
    with pytest.raises(CohortContractError, match="missing required geometry strata: SO"):
        _materialize(source, derived, receipt, [], candidate_pool=no_so)

    no_2299_tp = [candidate for candidate in FROZEN_CANDIDATES if candidate["gt_owner_id"] != "gt:2299:1"]
    with pytest.raises(CohortContractError, match="image 2299 has no TP candidate"):
        _materialize(source, derived, receipt, [], candidate_pool=no_2299_tp)


def test_missing_h0_is_explicit_and_output_is_immutable(tmp_path: Path) -> None:
    source, derived, receipt = _inputs()
    output = tmp_path / "cohort.json"
    result = _materialize(source, derived, receipt, [], output=output)
    event = result["cohort"]["events"][0]
    assert event["disposition"] == "indeterminate"
    assert event["checkpoint_status"]["S"]["disposition"] == "indeterminate_missing_checkpoint_h0"
    assert result["cohort"]["execution_contract"]["val200_index_fallback"] is False
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CohortContractError, match="not identical"):
        _materialize(source, derived, receipt, [], output=output)


def test_geometry_derives_64_to_32_merger_cells_and_fractional_weights() -> None:
    identity = _normalise_image_plan_identity(
        {
            "image_plan_identity": {
                "observed_image_grid_thw": [1, 64, 64],
                "merged_visual_tokens": 1024,
            }
        },
        context="fixture",
    )
    assert identity is not None
    assert identity["merge_size"] == 2
    assert identity["grid_rows"] == 32
    assert identity["grid_cols"] == 32
    assert identity["merged_visual_tokens"] == 1024
    owner = PanelOwner(1, 0, "person", (31.5, 31.5, 32.5, 32.5), "ann", "gt:1:0")
    weights = _fractional_bbox_cell_weights(owner, identity={**identity, "image_width": 1024.0, "image_height": 1024.0})
    assert sorted(weights) == [0, 1, 32, 33]
    assert all(value == pytest.approx(0.000244140625) for value in weights.values())


def test_panel_owner_keeps_rounded_launch_identity_and_raw_fractional_bbox() -> None:
    owners = _panel_owners(
        [
            {
                "image_id": 1,
                "width": 1024,
                "height": 1024,
                "objects": [
                    {
                        "category_name": "person",
                        "bbox_pixel_xyxy": [210.944, 151.6, 878.49, 895.8],
                        "coco_ann_id": "ann-1",
                    }
                ],
            }
        ]
    )
    assert owners[1][0].bbox == (211, 152, 878, 896)
    assert owners[1][0].raw_bbox == pytest.approx((210.944, 151.6, 878.49, 895.8))


def test_geometry_rejects_grid_token_identity_mismatch() -> None:
    with pytest.raises(CohortContractError, match="merged_visual_tokens mismatch"):
        _normalise_image_plan_identity(
            {
                "image_plan_identity": {
                    "observed_image_grid_thw": [1, 64, 64],
                    "merge_size": 2,
                    "merged_visual_tokens": 256,
                }
            },
            context="fixture",
        )


def test_verified_pair_materializes_regions_background_and_stable_receipts() -> None:
    source, derived, receipt = _inputs()
    candidates = list(FROZEN_CANDIDATES)
    a_owner = candidates[12]
    b_owner = candidates[13]
    identity = {"observed_image_grid_thw": [1, 64, 64], "merged_visual_tokens": 1024}
    a_prior = _record(a_owner, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=3)
    b_at_b = _record(b_owner, natural_boundary=6, covered_owner_ids=[a_owner["gt_owner_id"]])
    for record in (a_prior, b_at_b):
        record["image_plan_identity"] = identity
    h0 = _envelope([a_prior, b_at_b], source, derived)
    support = _envelope(
        [_support_record(b_at_b, verified_support=True)],
        source,
        derived,
    )
    first = _materialize(source, derived, receipt, h0, support)
    second = _materialize(source, derived, receipt, h0, support)
    event = next(item for item in first["cohort"]["events"] if item["gt_owner_id"] == b_owner["gt_owner_id"])
    event_again = next(item for item in second["cohort"]["events"] if item["gt_owner_id"] == b_owner["gt_owner_id"])
    assert event["geometry_status"] == "available"
    assert event["geometry_launch_eligible"] is True
    regions = event["image_cell_regions"]
    assert regions["a_exclusive"] and regions["b_exclusive"]
    assert not set(regions["a_exclusive"]) & set(regions["b_exclusive"])
    assert len(regions["background"]) == len(regions["b_exclusive"])
    assert event["verified_support_owner_ids"] == [b_owner["gt_owner_id"]]
    assert event["exact_b_boundary_verified_support_owner_ids"] == [
        b_owner["gt_owner_id"]
    ]
    assert a_owner["gt_owner_id"] not in event["verified_support_owner_roles"]
    assert event["verified_support_owner_roles"][b_owner["gt_owner_id"]] == "target-B"
    assert event["owner_region_roles"][a_owner["gt_owner_id"]] == "covered-A"
    assert event["owner_region_evidence"][a_owner["gt_owner_id"]] == {
        "strict_complete_covered_before_B": True,
        "exact_B_boundary_verified_support": False,
        "checkpoint_owner_level_verified_support": False,
    }
    assert event["owner_region_evidence"][b_owner["gt_owner_id"]][
        "exact_B_boundary_verified_support"
    ] is True
    assert event["geometry_sha256"] == event_again["geometry_sha256"]
    assert event["image_cell_region_receipts"] == event_again["image_cell_region_receipts"]
    assert event["SO_exclusive_shared_decomposition"]["a_exclusive"] == regions["a_exclusive"]
    assert event["SO_exclusive_shared_decomposition"]["b_exclusive"] == regions["b_exclusive"]


@pytest.mark.parametrize("foreign_support", [False, True])
def test_missing_or_foreign_b_support_never_qualifies_geometry(
    foreign_support: bool,
) -> None:
    source, derived, receipt = _inputs()
    candidates = list(FROZEN_CANDIDATES)
    a_owner, b_owner, foreign_owner = candidates[12], candidates[13], candidates[14]
    identity = {
        "observed_image_grid_thw": [1, 64, 64],
        "merged_visual_tokens": 1024,
    }
    a_prior = _record(
        a_owner,
        native_tp=True,
        native_fn=False,
        strict_complete_row=True,
        natural_boundary=3,
    )
    b_at_b = _record(
        b_owner,
        natural_boundary=6,
        covered_owner_ids=[a_owner["gt_owner_id"]],
    )
    h0_records = [a_prior, b_at_b]
    support = None
    if foreign_support:
        foreign_at_b = _record(
            foreign_owner,
            native_tp=True,
            native_fn=False,
            strict_complete_row=True,
            natural_boundary=6,
        )
        foreign_at_b["exact_prefix_token_ids"] = list(
            b_at_b["exact_prefix_token_ids"]
        )
        foreign_at_b["exact_prefix_sha256"] = b_at_b["exact_prefix_sha256"]
        h0_records.append(foreign_at_b)
        support = _envelope(
            [_support_record(foreign_at_b, verified_support=True)], source, derived
        )
    for record in h0_records:
        record["image_plan_identity"] = identity
    result = _materialize(
        source,
        derived,
        receipt,
        _envelope(h0_records, source, derived),
        support,
    )
    event = next(
        item
        for item in result["cohort"]["events"]
        if item["gt_owner_id"] == b_owner["gt_owner_id"]
    )
    assert event["A_B"]["S"]["pair_status"] == "no_verified_B"
    assert event["geometry_launch_eligible"] is False
    assert event["image_cell_regions"]["a_exclusive"] == []
    assert event["image_cell_regions"]["b_exclusive"] == []


def test_support_grid_identity_mismatch_is_rejected_before_pair_materialization() -> None:
    source, derived, receipt = _inputs()
    target = list(FROZEN_CANDIDATES)[5]
    h0_record = _record(target)
    h0_record["image_plan_identity"] = {
        "observed_image_grid_thw": [1, 64, 64],
        "merged_visual_tokens": 1024,
    }
    support_record = _support_record(h0_record, verified_support=True)
    support_record["image_plan_identity"] = {
        "observed_image_grid_thw": [1, 32, 32],
        "merged_visual_tokens": 256,
    }
    with pytest.raises(CohortContractError, match="image plan identity mismatch"):
        _materialize(
            source,
            derived,
            receipt,
            _envelope([h0_record], source, derived),
            _envelope([support_record], source, derived),
        )


def test_verified_pair_without_native_image_identity_is_mechanically_indeterminate() -> None:
    source, derived, receipt = _inputs()
    candidates = list(FROZEN_CANDIDATES)
    a_owner, b_owner = candidates[12], candidates[13]
    a_prior = _record(a_owner, native_tp=True, native_fn=False, strict_complete_row=True, natural_boundary=3)
    b_record = _record(b_owner, natural_boundary=6, covered_owner_ids=[a_owner["gt_owner_id"]])
    h0 = _envelope([a_prior, b_record], source, derived)
    support = _envelope([_support_record(b_record, verified_support=True)], source, derived)
    result = _materialize(source, derived, receipt, h0, support)
    event = next(item for item in result["cohort"]["events"] if item["gt_owner_id"] == b_owner["gt_owner_id"])
    assert event["A_B"]["S"]["pair_status"] == "verified_pair"
    assert event["geometry_status"] == "indeterminate"
    assert event["geometry_launch_eligible"] is False
    assert event["geometry_mechanical_disposition"] == "indeterminate_missing_image_plan_identity"
