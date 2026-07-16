from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.run_fixed_encoding_query_scoped_object_centered_spatial_eligibility import (
    CANONICAL_ROWS_RECEIPT_SHA256,
    PARENT_RECEIPT_SHA256,
    _effect_floor,
    build_query_scoped_key_eligibility_mask,
    classify_panel,
    classify_query_scoped_case,
    compare_frozen_case_contract,
    compare_parent_canonical_rows,
    inspect_query_scoped_mask_structure,
    query_row_range,
    validate_canonical_rows_panel,
    validate_canonical_rows_receipt,
    validate_parent_receipt,
    validate_requested_image_ids,
)


def _phase(*, crossover: float, target: float, competitor: float, release: float = 0.1) -> dict[str, float]:
    return {
        "crossover": crossover,
        "gamma_target": target,
        "gamma_competitor": competitor,
        "target_release": release,
        "competitor_release": 0.0,
    }


def _complete(*, full: float = 0.2, geometry: float = 0.2, description: float = 0.2) -> dict[str, dict[str, float]]:
    return {
        "full_row": _phase(crossover=full, target=0.2, competitor=-0.2),
        "geometry": _phase(crossover=geometry, target=0.2, competitor=-0.2),
        "description": _phase(crossover=description, target=0.2, competitor=-0.2),
    }


def test_query_row_range_is_exact_and_inclusive() -> None:
    assert query_row_range(prefix_length=5, row_length=4) == (4, 7)
    assert query_row_range(prefix_length=5, row_length=7) == (4, 10)
    with pytest.raises(ValueError):
        query_row_range(prefix_length=0, row_length=4)
    with pytest.raises(ValueError):
        query_row_range(prefix_length=5, row_length=0)


def test_requested_image_ids_allow_smoke_subset_but_reject_drift() -> None:
    assert validate_requested_image_ids(["139"]) == ["139"]
    assert validate_requested_image_ids(["139", "632", "12120", "12639"]) == [
        "139",
        "632",
        "12120",
        "12639",
    ]
    with pytest.raises(ValueError, match="at least one"):
        validate_requested_image_ids([])
    with pytest.raises(ValueError, match="duplicate"):
        validate_requested_image_ids(["139", "139"])
    with pytest.raises(ValueError, match="unexpected"):
        validate_requested_image_ids(["9400"])


def test_query_scoped_mask_only_changes_exact_query_rows() -> None:
    mask = build_query_scoped_key_eligibility_mask(
        sequence_length=12,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        row_length=4,
    )[0, 0]
    # Exact [P-1, P+K-2] = [4, 7].
    for query in range(12):
        for key in (1, 2, 3):
            expected = key <= query and not (4 <= query <= 7 and key in {1, 3})
            assert bool(mask[query, key]) is expected
    assert bool(mask[3, 1])  # off-scope query remains ordinary causal visibility
    assert bool(mask[8, 1])  # q=8 sees the blocked key unchanged
    assert not bool(mask[4, 1])  # first scored row is restricted
    assert bool(mask[4, 2])  # selected image key remains eligible


def test_different_row_lengths_use_independent_query_ranges() -> None:
    target = build_query_scoped_key_eligibility_mask(
        sequence_length=16,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        row_length=3,
    )[0, 0]
    competitor = build_query_scoped_key_eligibility_mask(
        sequence_length=16,
        image_key_positions=[1, 2, 3],
        eligible_image_positions=[2],
        prefix_length=5,
        row_length=7,
    )[0, 0]
    assert not bool(target[6, 1])
    assert bool(target[7, 1])
    assert not bool(competitor[10, 1])
    assert bool(competitor[11, 1])


def test_non_image_keys_and_future_blocking_are_invariant() -> None:
    mask = build_query_scoped_key_eligibility_mask(
        sequence_length=10,
        image_key_positions=[1, 2],
        eligible_image_positions=[2],
        prefix_length=4,
        row_length=3,
    )[0, 0]
    for query in range(10):
        for key in (0, 4, 8):
            assert bool(mask[query, key]) == (key <= query)
        for key in range(query + 1, 10):
            assert not bool(mask[query, key])


def test_structural_receipt_covers_blocked_scope_and_invariants() -> None:
    full = build_query_scoped_key_eligibility_mask(
        sequence_length=10,
        image_key_positions=[1, 2],
        eligible_image_positions=[2],
        prefix_length=4,
        row_length=3,
    )
    receipt = inspect_query_scoped_mask_structure(
        full,
        sequence_length=10,
        image_key_positions=[1, 2],
        eligible_image_positions=[2],
        prefix_length=4,
        row_length=3,
    )
    assert receipt["passed"]
    assert receipt["query_range"]["indices"] == [3, 4, 5]
    assert receipt["blocked_image_cell_count"] == 3
    assert receipt["changed_non_image_key_cell_count"] == 0
    assert receipt["changed_off_scope_query_cell_count"] == 0
    assert receipt["future_key_blocking_unchanged"]


def test_structural_receipt_rejects_no_blocked_image_cell() -> None:
    full = build_query_scoped_key_eligibility_mask(
        sequence_length=8,
        image_key_positions=[1, 2],
        eligible_image_positions=[1, 2],
        prefix_length=3,
        row_length=2,
    )
    receipt = inspect_query_scoped_mask_structure(
        full,
        sequence_length=8,
        image_key_positions=[1, 2],
        eligible_image_positions=[1, 2],
        prefix_length=3,
        row_length=2,
    )
    assert not receipt["passed"]


def test_structural_receipt_rejects_corrupted_blocked_cell() -> None:
    mask = build_query_scoped_key_eligibility_mask(
        sequence_length=10, image_key_positions=[1, 2], eligible_image_positions=[2], prefix_length=4, row_length=3
    )
    mask[0, 0, 3, 1] = True
    receipt = inspect_query_scoped_mask_structure(
        mask, sequence_length=10, image_key_positions=[1, 2], eligible_image_positions=[2], prefix_length=4, row_length=3
    )
    assert not receipt["passed"]
    assert not receipt["exact_expected_mask_match"]


def test_structural_receipt_rejects_corrupted_eligible_cell() -> None:
    mask = build_query_scoped_key_eligibility_mask(
        sequence_length=10, image_key_positions=[1, 2], eligible_image_positions=[2], prefix_length=4, row_length=3
    )
    mask[0, 0, 4, 2] = False
    receipt = inspect_query_scoped_mask_structure(
        mask, sequence_length=10, image_key_positions=[1, 2], eligible_image_positions=[2], prefix_length=4, row_length=3
    )
    assert not receipt["passed"]
    assert not receipt["eligible_image_cells_unchanged"]


def test_structural_receipt_rejects_corrupted_off_scope_cell() -> None:
    mask = build_query_scoped_key_eligibility_mask(
        sequence_length=10, image_key_positions=[1, 2], eligible_image_positions=[2], prefix_length=4, row_length=3
    )
    mask[0, 0, 8, 1] = False
    receipt = inspect_query_scoped_mask_structure(
        mask, sequence_length=10, image_key_positions=[1, 2], eligible_image_positions=[2], prefix_length=4, row_length=3
    )
    assert not receipt["passed"]
    assert receipt["changed_off_scope_query_cell_count"] > 0


def test_parent_digest_is_checked_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_parent_receipt(path)
    assert len(PARENT_RECEIPT_SHA256) == 64


def test_parent_receipt_validator_accepts_frozen_receipt() -> None:
    path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/"
        "cohort-six-float32-20260715b/receipt.json"
    )
    payload, digest = validate_parent_receipt(path)
    assert payload["unit_id"].endswith("spatial-eligibility-crossover")
    assert digest == PARENT_RECEIPT_SHA256


def test_canonical_rows_receipt_validator_accepts_frozen_receipt() -> None:
    path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/"
        "cohort-four-float32-20260715a/receipt.json"
    )
    payload, digest = validate_canonical_rows_receipt(path, hard_parent_sha256=PARENT_RECEIPT_SHA256)
    assert payload["unit_id"].endswith("dose-response")
    assert digest == CANONICAL_ROWS_RECEIPT_SHA256


def test_canonical_rows_panel_rejects_source_contract_mismatch() -> None:
    parent = {
        "source_jsonl_sha256": "source",
        "config_sha256": "config",
        "audit_ledger_sha256": "ledger",
        "results": [],
    }
    with pytest.raises(ValueError, match="source/config/ledger"):
        validate_canonical_rows_panel(
            {"source_jsonl_sha256": "wrong", "config_sha256": "config", "audit_ledger_sha256": "ledger", "results": []},
            parent=parent,
        )


def test_exact_rows_are_checked_before_scoring() -> None:
    canonical = {"target_row_token_ids": [1, 2], "competitor_row_token_ids": [3, 4], "canonical_rows_sha256": "bad"}
    comparison = compare_parent_canonical_rows(parent_result=canonical, target_row=[1, 9], competitor_row=[3, 4])
    assert not comparison["passed"]
    assert not comparison["target_equal"]
    frozen = compare_frozen_case_contract(
        parent_result={"image_id": "139", "target_annotation_id": "a", "competitor_annotation_id": "b", "target_mask_indices": [1], "competitor_mask_indices": [2]},
        canonical_result={"image_id": "139", "target_annotation_id": "a", "competitor_annotation_id": "b", "target_mask_indices": [1], "competitor_mask_indices": [2], **canonical},
        image_id="139", target_row=[1, 2], competitor_row=[3, 4],
    )
    assert not frozen["passed"]


def test_effect_floor_has_absolute_minimum() -> None:
    assert _effect_floor(0.0) == pytest.approx(0.01)
    assert _effect_floor(0.02) == pytest.approx(0.2)


def test_image139_requires_complete_semantic_geometry_owner_reversal() -> None:
    parent = _complete()
    result = classify_query_scoped_case(
        image_id="139", crossover=_complete(), parent_crossover=parent,
        no_op_drift=0.0, different_category=True, owner_rows_not_destructive=True,
    )
    assert result["classification"] == "retain_primary_complete_switch"
    missing_description = _complete(description=0.01)
    result = classify_query_scoped_case(
        image_id="139", crossover=missing_description, parent_crossover=parent,
        no_op_drift=0.0, different_category=True, owner_rows_not_destructive=True,
    )
    assert result["classification"] == "inconclusive"


def test_image12120_allows_geometry_and_full_row_without_description() -> None:
    result = classify_query_scoped_case(
        image_id="12120", crossover=_complete(description=0.01), parent_crossover=_complete(),
        no_op_drift=0.0, different_category=True, owner_rows_not_destructive=True,
    )
    assert result["classification"] == "retain_secondary_anchor"


def test_image12639_requires_half_parent_recovery_target_tolerance_and_positive_release() -> None:
    parent = _complete(full=1.0, geometry=1.2)
    current = {
        "full_row": _phase(crossover=0.5, target=0.2, competitor=-0.2, release=0.1),
        "geometry": _phase(crossover=0.6, target=0.2, competitor=-0.2),
    }
    result = classify_query_scoped_case(
        image_id="12639", crossover=current, parent_crossover=parent,
        no_op_drift=0.0, different_category=False, owner_rows_not_destructive=True,
    )
    assert result["classification"] == "retain_secondary_anchor"
    assert result["predicate"]["half_parent_recovery_passed"]
    current["full_row"]["target_release"] = 0.0
    result = classify_query_scoped_case(
        image_id="12639", crossover=current, parent_crossover=parent,
        no_op_drift=0.0, different_category=False, owner_rows_not_destructive=True,
    )
    assert result["classification"] == "inconclusive"


def test_classifier_rejects_destructive_owner_row() -> None:
    result = classify_query_scoped_case(
        image_id="139", crossover=_complete(), parent_crossover=_complete(),
        no_op_drift=0.0, different_category=True, owner_rows_not_destructive=False,
    )
    assert result["classification"] == "inconclusive"


def _panel_case(
    image_id: str,
    classification: str,
    *,
    complete: bool = False,
    semantic: bool = False,
    target_release: float | None = None,
) -> dict:
    release = (0.1 if complete else 0.0) if target_release is None else float(target_release)
    return {
        "image_id": image_id,
        "classification": classification,
        "structural_mask_gate_passed": True,
        "no_op_trust_gate": {"passed": True},
        "parent_continuity": {"passed": True},
        "feature_continuity": {"passed": True},
        "crossover": {},
        "predicate": {
            "complete_switch": complete,
            "semantic_switch": semantic,
            "target_release": release,
            "full_row_recovery": 1.0 if complete else 0.0,
            "geometry_recovery": 1.0 if complete else 0.0,
            "owner_rows_not_destructive": True,
        },
    }


def test_panel_requires_exact_anchor_set_and_never_negates_subset() -> None:
    subset = [_panel_case("139", "retain_primary_complete_switch", complete=True, semantic=True)]
    decision = classify_panel(subset)
    assert decision["classification"].startswith("incomplete_panel_")
    assert "support_dependence_on_earlier_query_computation" not in decision["classification"]


def test_panel_invalid_decision_critical_anchor_never_negates() -> None:
    results = [
        _panel_case("139", "invalid_structural_mask_gate"),
        _panel_case("632", "inconclusive"),
        _panel_case("12120", "retain_secondary_anchor", complete=True),
        _panel_case("12639", "inconclusive"),
    ]
    decision = classify_panel(results)
    assert decision["classification"] == "incomplete_panel_invalid_anchor_cases"
    assert "support_dependence_on_earlier_query_computation" not in decision["classification"]


def test_panel_promotes_only_after_two_complete_switches_and_semantic_switch() -> None:
    results = [
        _panel_case("139", "retain_primary_complete_switch", complete=True, semantic=True),
        _panel_case("632", "inconclusive"),
        _panel_case("12120", "retain_secondary_anchor", complete=True),
        _panel_case("12639", "inconclusive"),
    ]
    decision = classify_panel(results)
    assert decision["classification"] == "promote_one_bounded_free_row_replay"


def test_panel_counts_second_complete_switch_without_release_when_semantic_case_releases() -> None:
    results = [
        _panel_case(
            "139",
            "retain_primary_complete_switch",
            complete=True,
            semantic=True,
            target_release=0.1,
        ),
        _panel_case("632", "inconclusive"),
        _panel_case(
            "12120",
            "retain_secondary_anchor",
            complete=True,
            target_release=0.0,
        ),
        _panel_case("12639", "inconclusive"),
    ]
    decision = classify_panel(results)
    assert decision["classification"] == "promote_one_bounded_free_row_replay"
    assert decision["complete_switch_case_ids"] == ["139", "12120"]
    assert decision["semantic_complete_switch_with_release_present"]


def test_panel_does_not_promote_without_semantic_complete_switch() -> None:
    results = [
        _panel_case("139", "retain_primary_complete_switch", complete=True),
        _panel_case("632", "inconclusive"),
        _panel_case("12120", "retain_secondary_anchor", complete=True),
        _panel_case("12639", "inconclusive"),
    ]
    decision = classify_panel(results)
    assert decision["classification"] == "support_direct_row_read_contribution"
