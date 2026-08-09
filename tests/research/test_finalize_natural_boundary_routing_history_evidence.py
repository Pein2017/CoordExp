from __future__ import annotations

import pytest

from scripts.research.finalize_natural_boundary_routing_history_evidence import (
    EvidenceContractError,
    classify_endpoint,
    finalize_crossover,
    validate_admission_mode,
)


def _natural(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "admission_mode": "pre_opener_natural",
        "opener_token_id": 151646,
        "initial_prefix_last_token_id": 42,
        "opener_injected": False,
        "first_generated_token_id": 151646,
        "opener_generated_by_model": True,
        "native_parse": {"valid": True, "parse_status": "accepted"},
        "generation_status": "complete",
        "complete_row": True,
        "stop_reason": "box_end",
        "owner_match": {
            "status": "unique",
            "owner_id": "gt:1:2",
            "source_specific": True,
            "physical_match": True,
        },
        "owner_bookkeeping": {
            "G": ["gt:1:2"],
            "K": [],
            "L": [],
            "net": 1,
            "parse": {"unmatched_rows": 0, "duplicate_rows": 0},
        },
    }
    value.update(updates)
    return value


def test_seeded_row_marked_natural_is_rejected_fail_closed() -> None:
    row = _natural(
        natural=True,
        initial_prefix_last_token_id=151646,
        opener_injected=True,
        opener_generated_by_model=False,
        first_generated_token_id=8987,
    )
    with pytest.raises(EvidenceContractError, match="already ends with opener"):
        validate_admission_mode(row)
    assert classify_endpoint(row)["technical_status"] == "technical_invalid"


def test_opener_identity_is_never_defaulted() -> None:
    row = _natural()
    row.pop("opener_token_id")
    with pytest.raises(EvidenceContractError, match="lacks explicit opener_token_id"):
        validate_admission_mode(row)
    assert classify_endpoint(row)["technical_status"] == "technical_invalid"


def test_mechanically_valid_unmatched_is_scientific_unmatched() -> None:
    row = _natural(
        owner_match={"status": "unmatched"},
        owner_bookkeeping={
            "G": [],
            "K": [],
            "L": [],
            "net": 0,
            "parse": {"unmatched_rows": 1, "duplicate_rows": 0},
        },
    )
    result = classify_endpoint(row)
    assert result["mechanically_valid"] is True
    assert result["scientific_status"] == "unmatched"
    assert result["technical_status"] == "valid"
    assert result["source_specific_match"] is False


def test_crossover_keeps_neutral_and_charged_tau_and_unqualifies_unmatched_cell() -> None:
    cells = {
        "Y00": _natural(),
        "Y10": _natural(owner_match={"status": "unmatched"}, owner_bookkeeping={"G": [], "K": [], "L": [], "net": 0, "parse": {"unmatched_rows": 1, "duplicate_rows": 0}}),
        "Y01": _natural(owner_bookkeeping={"G": [], "K": [], "L": ["gt:1:0"], "net": -1, "parse": {"unmatched_rows": 0, "duplicate_rows": 0}}),
        "Y11": _natural(owner_bookkeeping={"G": ["gt:1:2"], "K": [], "L": ["gt:1:0"], "net": 0, "parse": {"unmatched_rows": 0, "duplicate_rows": 0}}),
    }
    result = finalize_crossover(cells)["evidence"]
    assert result["source_specific_crossover_status"] == "unqualified"
    assert result["cells"]["Y10"]["scientific_status"] == "unmatched"
    assert result["descriptive_matched_net_tau"] != result["net_charged_tau"]
    assert result["net_charged"]["unmatched_rows"]["Y10"] == 1


def test_post_opener_seeded_mode_is_explicit_and_not_primary_strict_match() -> None:
    row = _natural(
        admission_mode="post_opener_seeded",
        initial_prefix_last_token_id=151646,
        opener_injected=True,
        first_generated_token_id=8987,
        opener_generated_by_model=False,
        seed_provenance={"source": "test"},
    )
    admission = validate_admission_mode(row)
    assert admission["admission_mode"] == "post_opener_seeded"
    assert classify_endpoint(row)["source_specific_match"] is False


def test_invalid_primary_cell_never_contributes_imputed_zero_tau() -> None:
    cells = {cell: _natural() for cell in ("Y00", "Y10", "Y01", "Y11")}
    cells["Y11"] = {**cells["Y11"], "mechanically_valid": False}
    result = finalize_crossover(cells)["evidence"]
    assert result["cells"]["Y11"]["mechanically_valid"] is False
    assert result["descriptive_matched_neutral"]["cell_net"]["Y11"] is None
    assert result["descriptive_matched_net_tau"] is None
    assert result["descriptive_matched_net_tau_status"] == "not_computable"
    assert result["net_charged_tau"] is None
    assert result["net_charged_tau_status"] == "not_computable"
    assert result["contrast_summary"]["matched_neutral"]["Delta_static"] is None
