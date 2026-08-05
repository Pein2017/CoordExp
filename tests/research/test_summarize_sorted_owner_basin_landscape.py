"""CPU contract tests for the Sorted owner-basin summarizer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.research import sorted_owner_basin_landscape as core
from scripts.research import score_sorted_owner_basin_landscape as score_producer
from scripts.research import summarize_sorted_owner_basin_landscape as sut


JsonDict = dict[str, Any]


def _rules(*, contract_mode: str = "test_fixture") -> JsonDict:
    coordinate_max = 3 if contract_mode == "test_fixture" else 999
    coordinate_count = coordinate_max + 1
    document: JsonDict = {
        "schema_version": core.RULES_SCHEMA_VERSION,
        "contract_mode": contract_mode,
        "geometry_identity": {"schema": core.GEOMETRY_IDENTITY_SCHEMA, "coordinate_denominator": 1000},
        "coordinate_bins": {"min": 0, "max": coordinate_max},
        "scoring_contract": {
            "coordinate_token_id_start": 100,
            "coordinate_token_id_end_exclusive": 100 + coordinate_count,
            "foil_set_digests": {"smoke-foils": "f" * 64},
        },
        "target_anchor": {"margin_fraction": 0.0, "min_margin_bins": 0, "max_margin_bins": 0},
        "bank_order": ["target", "covered", "background", "scan"],
        "proposal_measures": {
            "matched": {
                "comparability_group": "same-domain",
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": {"target": 1.0, "covered": 1.0, "background": 1.0, "scan": 1.0},
            }
        },
        "bank_proposal_measure": {
            "target": "matched",
            "covered": "matched",
            "background": "matched",
            "scan": "matched",
        },
        "spatial_clustering": {
            "owner_link_iou_min": 0.1,
            "owner_link_center_distance_max": 0.1,
            "extent_submode_iou_min": 0.75,
        },
        "shape": {
            "near_peak_logprob_delta": 0.2,
            "wide_ridge_min_candidates": 2,
            "multi_submode_min": 2,
            "merged_extent_submodes": [],
            "scan_bank_names": ["scan"],
        },
        "declared_extent_submodes": ["whole"],
        "registered_basin_roles": {
            "target_owner": {
                "kind": "target",
                "foil_set_id": "smoke-foils",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["target"],
            },
            "covered_owner": {
                "kind": "foil",
                "foil_set_id": "smoke-foils",
                "identity_kind": "reviewed_physical_owner",
                "allowed_bank_names": ["covered"],
            },
            "background_geometry": {
                "kind": "foil",
                "foil_set_id": "smoke-foils",
                "identity_kind": "registered_geometry",
                "allowed_bank_names": ["background"],
            },
            "scan_geometry": {
                "kind": "foil",
                "foil_set_id": "smoke-foils",
                "identity_kind": "registered_geometry",
                "allowed_bank_names": ["scan"],
            },
        },
        "prominence": {"functional": "peak_height_difference"},
    }
    if contract_mode == "production":
        calibration = {
            "algorithm": "fixture",
            "quantile": 0.5,
            "minimum_roles": 1,
            "metric_reducers": [],
            "control_ids": [],
        }
        document.update(
            {
                "free_coordinate_tree": {
                    "budget": {"fixture": True},
                    "selector": {"fixture": True},
                    "candidate_membership_surface": "canonical_description_free",
                },
                "ablation": {},
                "calibration": calibration,
                "calibration_contract": dict(calibration),
            }
        )
        semantic_payload = core.build_semantic_core_payload(document)
        document["semantic_core"] = {
            "schema_version": core.SEMANTIC_CORE_SCHEMA_VERSION,
            "payload": semantic_payload,
            "sha256": sut.sha256_json(semantic_payload),
            "excluded_outer_execution_fields": list(
                core.SEMANTIC_CORE_OUTER_EXECUTION_FIELDS
            ),
        }
    return document


def _vocab_attestation(rule_digest: str) -> JsonDict:
    return {
        "vocab_size": 1000,
        "domain_digest": "vocab-domain",
        "filtered": False,
        "tokenizer_identity_digest": "tokenizer-digest",
        "model_identity_digest": "model-digest",
        "rule_digest": rule_digest,
        "runtime_receipt_id": "runtime-1",
    }


def _geometry(
    box: tuple[int, int, int, int], rules: core.LandscapeRules
) -> JsonDict:
    receipt = core.canonical_geometry_identity(
        core.CoordinateBox.from_values(*box), image_width=1000, image_height=500, rules=rules
    )
    return cast(JsonDict, core.json_serializable_receipt(receipt))


def _common_row(
    *, candidate_id: str, rule_digest: str, request_kind: str, owner_status: str = "gt", review: str = "reviewed"
) -> JsonDict:
    return {
        "schema_version": sut.SCORE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "diagnostic_owner_id": "gt:7511:22",
        "gt_owner_id": "gt:7511:22",
        "owner_status": owner_status,
        "image_id": "7511",
        "image_identity": "image:7511:sha256",
        "context_id": "root",
        "landscape_surface": "restricted_gt_target",
        "role": "target",
        "physical_owner_hint": "gt:7511:22",
        "review_status": review,
        "candidate_kind": "target_anchor",
        "foil_set_id": "smoke-foils",
        "foil_set_digest": "f" * 64,
        "rule_digest": rule_digest,
        "request_kind": request_kind,
        "native_repetition_penalty_stratum": 1.0,
        "basin_id": None,
        "prefix_token_count": 1,
        "prefix_token_ids_sha256": "p" * 64,
        "proposal_verification": {"self_consistency": "passed", "pure_core_recomputation": "passed"},
        "upstream_adjudication": {"global_ambiguity_status": "clear"},
        "likelihood_channel_note": "raw is model likelihood; policy is auxiliary",
    }


def _complete_row(
    *,
    candidate_id: str,
    box: tuple[int, int, int, int],
    total: float,
    bank: str,
    role_id: str,
    identity_kind: str,
    identity_id: str,
    rules: core.LandscapeRules,
) -> JsonDict:
    rule_digest = sut._core_rule_digest(rules)  # noqa: SLF001
    row = _common_row(
        candidate_id=candidate_id,
        rule_digest=rule_digest,
        request_kind="complete_box",
    )
    row["coord_token_ids"] = [100 + value for value in box]
    row["coord_token_ids_sha256"] = sut.sha256_json(row["coord_token_ids"])
    factors = {f"{name}_logprob": total / 4 for name in sut.COORDINATE_NAMES}
    row["raw_model_logprob"] = {
        **factors,
        "complete_box_logprob_sum": total,
        "vocab_attestation": {
            name: _vocab_attestation(rule_digest) for name in sut.COORDINATE_NAMES
        },
    }
    row["auxiliary_policy_scores"] = {}
    row["core_candidate"] = {
        "bank_name": bank,
        "source_id": f"source:{candidate_id}",
        "extent_submode": "whole",
        "proposal_measure_id": "matched",
        "coordinate_bins": list(box),
        "role_id": role_id,
        "identity_kind": identity_kind,
        "identity_id": identity_id,
        "geometry_identity": _geometry(box, rules),
    }
    return row


def _dense_row(*, rules: core.LandscapeRules, x1: int = 1) -> JsonDict:
    row = _common_row(
        candidate_id=f"dense-y1:{x1}",
        rule_digest=sut._core_rule_digest(rules),  # noqa: SLF001
        request_kind="dense_scan",
    )
    row.update(
        {
            "fixed_coord_token_ids": [100 + x1],
            "scan_slot": "y1",
            "raw_bin_scan": {"bin_logprobs": [-1.0] * (rules.coordinate_max + 1)},
            "auxiliary_policy_bin_scan": {},
        }
    )
    return row


def _attestation(rules: core.LandscapeRules) -> JsonDict:
    gt_box = core.CoordinateBox.from_values(1, 1, 2, 2)
    scores = {
        core.CoordinateBin(1): tuple(
            core.ConditionalY1ScoreReceipt(
                x1=entry.x1,
                y1=entry.y1,
                raw_selected_token_logprob=-1.0,
                can_form_valid_box=entry.can_form_valid_box,
                invalid_box_reason=entry.invalid_box_reason,
            )
            for entry in core.enumerate_complete_conditional_y1(core.CoordinateBin(1), gt_box, rules)
        )
    }
    value = core.attest_complete_conditional_y1_scores(
        diagnostic_owner_id="gt:7511:22",
        gt_owner_id="gt:7511:22",
        image_identity="image:7511:sha256",
        context_id="root",
        canonical_description_text="person",
        canonical_description_token_digest=hashlib.sha256(b"person-tokens").hexdigest(),
        context_token_digest=hashlib.sha256(b"root-context").hexdigest(),
        tokenizer_identity="tokenizer:fixture",
        model_identity="model:fixture",
        runtime_identity="runtime:fixture",
        gt_box=gt_box,
        scores_by_x1=scores,
        rules=rules,
    )
    receipt = core.json_serializable_receipt(value)
    assert isinstance(receipt, dict)
    receipt["landscape_surface"] = "restricted_gt_target"
    return receipt


def _fixture(
    tmp_path: Path, *, contract_mode: str = "test_fixture"
) -> JsonDict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rule_document = _rules(contract_mode=contract_mode)
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rule_document), encoding="utf-8")
    rules = core.validate_rule_mapping(rule_document)
    rows = [
        _complete_row(
            candidate_id="target",
            box=(1, 1, 2, 2),
            total=-1.0,
            bank="target",
            role_id="target_owner",
            identity_kind="reviewed_physical_owner",
            identity_id="gt:7511:22",
            rules=rules,
        ),
        _complete_row(
            candidate_id="covered",
            box=(0, 0, 1, 1),
            total=-3.0,
            bank="covered",
            role_id="covered_owner",
            identity_kind="reviewed_physical_owner",
            identity_id="gt:7511:26",
            rules=rules,
        ),
        _complete_row(
            candidate_id="background",
            box=(2, 2, 3, 3),
            total=-4.0,
            bank="background",
            role_id="background_geometry",
            identity_kind="registered_geometry",
            identity_id="geometry:bg:1",
            rules=rules,
        ),
        _dense_row(rules=rules),
    ]
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source_path = tmp_path / "owner-ledger.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    receipt = {
        "schema_version": sut.SCORE_RECEIPT_SCHEMA_VERSION,
        "runtime_execution_status": "test_fixture_scorer_shaped_output",
        "implementation_provenance": {
            "git_head": "a" * 40,
            "git_dirty": True,
            "git_dirty_diff_sha256": "b" * 64,
            "relevant_file_digests": {"fixture-scorer.py": "c" * 64},
        },
        "command_environment": {
            "cwd": str(tmp_path),
            "python_executable": "python",
            "python_version": "fixture",
            "torch_version": "fixture",
            "transformers_version": "fixture",
        },
        "source_digests": {
            "owner_ledger": {
                "path": str(source_path),
                "sha256": sut.sha256_file(source_path),
                "manifest_expected": sut.sha256_file(source_path),
                "match": True,
            }
        },
        "decision_rules": {
            "file_sha256": sut.sha256_file(rules_path),
            "core_rule_digest": sut._core_rule_digest(rules),  # noqa: SLF001
        },
        "mandatory_cache_parity_gate": {
            "status": "passed",
            "atol": score_producer.CACHE_PARITY_ATOL,
            "rtol": score_producer.CACHE_PARITY_RTOL,
        },
        "conditional_y1_attestations": [_attestation(rules)],
    }
    receipt["scoring_backend_admission"] = score_producer.build_scoring_backend_admission(
        parity_gate=receipt["mandatory_cache_parity_gate"],
        selection=score_producer.select_scoring_backend_from_parity(
            receipt["mandatory_cache_parity_gate"]
        ),
        score_row_count=len(rows),
        per_context_group_accounting=[
            {
                "scoring_backend": score_producer.KV_CACHE_SCORING_BACKEND,
                "context_id": "context:fixture",
                "group_id": "group:fixture",
                "root_prefix_length": 1,
                "root_calls": 1,
                "logical_token_step_requests": 0,
                "logical_token_step_requests_by_depth": {},
                "actual_forward_calls": 1,
                "actual_forward_calls_by_depth": {"0": 1},
                "memo_hits_by_depth": {},
                "memo_entries_by_depth": {},
                "retained_relative_depths": [],
            }
        ],
    )
    receipt = score_producer.seal_execution_receipt(
        receipt,
        scores_path=scores_path,
        rows=rows,
    )
    receipt_path = tmp_path / "scores-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {
        "rules_path": rules_path,
        "scores_path": scores_path,
        "receipt_path": receipt_path,
        "rows": rows,
        "receipt": receipt,
    }


def _rewrite_scores(fixture: JsonDict, rows: list[JsonDict]) -> None:
    path = fixture["scores_path"]
    assert isinstance(path, Path)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    receipt["output_artifacts"]["landscape_scores"] = {
        "sha256": sut.sha256_file(path),
        "row_count": len(rows),
    }
    by_surface = sorted(rows, key=lambda row: row["candidate_id"])
    receipt["landscape_surface_receipts"]["restricted_gt_target"] = {
        "status": "passed",
        "row_count": len(by_surface),
        "score_rows_sha256": sut.sha256_json(by_surface),
    }
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    fixture["rows"] = rows
    fixture["receipt"] = receipt


def _bounded_null_free_execution_receipt(request_id: str) -> JsonDict:
    receipt: JsonDict = {
        "schema_version": "sorted_owner_basin_free_tree_execution.v1",
        "surface": "free_coordinate_tree",
        "request_id": request_id,
        "status": "executed",
        "counts": {"complete_box_count": 0},
        "null_semantics": "bounded_free_search_null_is_non_evidence_for_absence",
    }
    receipt["receipt_sha256"] = score_producer.sha256_json(receipt)
    return receipt


def _reviewed_candidate_lineage() -> JsonDict:
    lineage: JsonDict = {
        "source_pred_row_id": "pred:7511:3",
        "review_status": "reviewed_candidate",
        "registry_id": "control:b2:7511:22",
        "foreign_keys": {
            "gt_owner_id": "gt:7511:22",
            "pred_row_id": "pred:7511:3",
        },
        "source_binding": {
            "trajectory_id": "trajectory:7511",
            "source_artifact_sha256": "a" * 64,
        },
    }
    lineage["lineage_sha256"] = sut.sha256_json(lineage)
    return lineage


def _mark_reviewed_candidate(rows: list[JsonDict]) -> None:
    for row in rows:
        row["review_status"] = "reviewed_candidate"
        row["upstream_adjudication"][
            "source_review_foreign_key_lineage"
        ] = _reviewed_candidate_lineage()


def _sealed_context_lineage() -> JsonDict:
    lineage: JsonDict = {
        "source_pred_row_id": None,
        "review_status": "sealed_context",
        "registry_id": "control:smoke:strict-visible:7511:22",
        "foreign_keys": {
            "gt_owner_id": "gt:7511:22",
            "pred_row_id": None,
        },
        "source_binding": {
            "trajectory_id": "trajectory:7511",
            "source_artifact_sha256": "a" * 64,
        },
    }
    lineage["lineage_sha256"] = sut.sha256_json(lineage)
    return lineage


def _mark_sealed_context(rows: list[JsonDict]) -> None:
    for row in rows:
        row["review_status"] = "sealed_context"
        row["upstream_adjudication"][
            "source_review_foreign_key_lineage"
        ] = _sealed_context_lineage()


def _admit_live_runtime(fixture: JsonDict) -> JsonDict:
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    runtime_identity_sha256 = "d" * 64
    receipt["runtime_execution_status"] = (
        "live_model_scoring_completed_artifacts_sealed"
    )
    receipt["runtime_identity_admission"] = {
        "config": {
            "status": "passed",
            "model_dtype": "fp32",
            "source_jsonl": "fixture-source.jsonl",
            "binding_sha256": "c" * 64,
        },
        "preload": {
            "status": "passed",
            "identity_file_sha256": "1" * 64,
            "identity_receipt_digest": "2" * 64,
            "resolved_infer_fingerprint": "3" * 64,
            "source_panel_sha256": "4" * 64,
            "tokenizer_identity_sha256": "5" * 64,
            "model_identity_sha256": "6" * 64,
            "runtime_identity_sha256": runtime_identity_sha256,
            "admission_sha256": "7" * 64,
        },
        "postload": {
            "status": "passed",
            "projection_sha256": "8" * 64,
            "runtime_identity_sha256": runtime_identity_sha256,
            "observed_runtime_identity_sha256": runtime_identity_sha256,
            "expected_projection": {"fixture": "exact"},
            "observed_projection": {"fixture": "exact"},
        },
    }
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    fixture["receipt"] = receipt
    return receipt


def _rewrite_backend_admission(
    fixture: JsonDict, *, parity_status: str
) -> JsonDict:
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    rows = fixture["rows"]
    assert isinstance(rows, list)
    parity = {
        "status": parity_status,
        "atol": score_producer.CACHE_PARITY_ATOL,
        "rtol": score_producer.CACHE_PARITY_RTOL,
    }
    selection = score_producer.select_scoring_backend_from_parity(parity)
    selected = selection["selected_backend"]
    fallback = selected == score_producer.FULL_REFORWARD_SCORING_BACKEND
    receipt["mandatory_cache_parity_gate"] = parity
    receipt["scoring_backend_admission"] = score_producer.build_scoring_backend_admission(
        parity_gate=parity,
        selection=selection,
        score_row_count=len(rows),
        per_context_group_accounting=[
            {
                "scoring_backend": selected,
                "context_id": "context:fixture",
                "group_id": "group:fixture",
                "root_prefix_length": 1,
                "root_calls": 1,
                "logical_token_step_requests": 7 if fallback else 0,
                "logical_token_step_requests_by_depth": (
                    {"1": 2, "2": 3, "3": 2} if fallback else {}
                ),
                "actual_forward_calls": 6 if fallback else 1,
                "actual_forward_calls_by_depth": (
                    {"0": 1, "1": 1, "2": 2, "3": 2}
                    if fallback
                    else {"0": 1}
                ),
                "memo_hits_by_depth": {"1": 1, "2": 1} if fallback else {},
                "memo_entries_by_depth": {"0": 1, "1": 1, "2": 2}
                if fallback
                else {},
                "retained_relative_depths": [0, 1, 2] if fallback else [],
            }
        ],
    )
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    fixture["receipt"] = receipt
    return receipt["scoring_backend_admission"]


def _mutate_uncached_depth_accounting(admission: JsonDict, case: str) -> None:
    entry = admission["forward_accounting"]["per_context_group"][0]
    if case == "missing_logical_depth":
        entry["logical_token_step_requests_by_depth"].pop("3")
        entry["logical_token_step_requests"] = 5
    elif case == "missing_actual_depth":
        entry["actual_forward_calls_by_depth"].pop("3")
        entry["actual_forward_calls"] = 4
    elif case == "missing_memo_depth":
        entry["memo_hits_by_depth"].pop("2")
    elif case == "extra_logical_depth":
        entry["logical_token_step_requests_by_depth"]["4"] = 1
        entry["logical_token_step_requests"] = 8
    elif case == "extra_actual_depth":
        entry["actual_forward_calls_by_depth"]["4"] = 1
        entry["actual_forward_calls"] = 7
    elif case == "count_identity_mismatch":
        entry["logical_token_step_requests_by_depth"]["1"] = 3
        entry["logical_token_step_requests"] = 8
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unknown accounting mutation: {case}")


def test_summary_reconstructs_core_peaks_prominence_mass_and_background_neutrality(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary, receipt = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )
    assert receipt["independent_reconstruction"] == "passed"
    entry = summary["per_owner_context"][0]
    assert entry["decision_status"] == "measured_no_conclusion"
    assert len(entry["peak_prominence"]) == 2
    assert all(basin["basin_mass"]["status"] == "comparable" for basin in entry["basins"])
    background = next(basin for basin in entry["basins"] if basin["registered_geometry_id"] == "geometry:bg:1")
    assert background["identity_kind"] == "registered_geometry"
    assert background["evidence_polarity"] == "neutral"
    assert summary["scientific_conclusion"] is None


def test_scorer_sealed_output_is_consumed_without_contract_translation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = fixture["receipt"]
    assert isinstance(receipt, dict)
    assert set(receipt["landscape_surface_receipts"]) == set(sut.LANDSCAPE_SURFACES)
    summary, output_receipt = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )
    assert output_receipt["independent_reconstruction"] == "passed"
    assert summary["per_owner_context"][0]["decision_status"] == "measured_no_conclusion"


def test_failed_cache_parity_admits_only_complete_uncached_reference_accounting(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    admission = _rewrite_backend_admission(fixture, parity_status="failed")
    summary, receipt = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )
    assert admission["cache_score_row_count"] == 0
    accounting = admission["forward_accounting"]["per_context_group"][0]
    assert accounting["logical_token_step_requests_by_depth"] == {
        "1": 2,
        "2": 3,
        "3": 2,
    }
    assert accounting["actual_forward_calls_by_depth"] == {
        "0": 1,
        "1": 1,
        "2": 2,
        "3": 2,
    }
    assert accounting["memo_hits_by_depth"] == {"1": 1, "2": 1}
    assert summary["scoring_backend_admission"] == admission
    assert receipt["mandatory_cache_parity_gate"] == "failed"
    assert (
        receipt["scoring_backend_gate"]
        == "cache_parity_failed_uncached_reference_used"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda admission: admission.update(cache_score_row_count=1),
            "cache_score_row_count",
        ),
        (
            lambda admission: admission["forward_accounting"][
                "per_context_group"
            ][0].update(scoring_backend=score_producer.KV_CACHE_SCORING_BACKEND),
            "mixes scoring backends",
        ),
        (
            lambda admission: admission["forward_accounting"][
                "per_context_group"
            ][0]["memo_entries_by_depth"].update({"3": 1}),
            "exact depths 0, 1, and 2",
        ),
    ],
)
def test_rejects_inconsistent_uncached_backend_admission(
    tmp_path: Path, mutation, message: str
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_backend_admission(fixture, parity_status="failed")
    receipt = deepcopy(fixture["receipt"])
    admission = receipt["scoring_backend_admission"]
    mutation(admission)
    admission_without_digest = dict(admission)
    admission_without_digest.pop("sha256")
    admission["sha256"] = sut.sha256_json(admission_without_digest)
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match=message):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


def test_rejects_probe_only_relaxed_cache_from_decision_bearing_summary(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = deepcopy(fixture["receipt"])
    admission = receipt["scoring_backend_admission"]
    admission["decision_use"] = "probe_only_not_decision_bearing"
    admission["cache_admission_policy"] = {
        "effective_mode": "relaxed_coordinate_behavior"
    }
    admission_without_digest = dict(admission)
    admission_without_digest.pop("sha256")
    admission["sha256"] = sut.sha256_json(admission_without_digest)
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(sut.SummaryContractError, match="probe-only"):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )

    admission.pop("decision_use")
    admission_without_digest = dict(admission)
    admission_without_digest.pop("sha256")
    admission["sha256"] = sut.sha256_json(admission_without_digest)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match="legacy relaxed-cache"):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_logical_depth", "identity failed at depth 3"),
        ("missing_actual_depth", "identity failed at depth 3"),
        ("missing_memo_depth", "identity failed at depth 2"),
        ("extra_logical_depth", "identity failed at depth 4"),
        ("extra_actual_depth", "identity failed at depth 4"),
        ("count_identity_mismatch", "identity failed at depth 1"),
    ],
)
def test_rejects_uncached_per_depth_forward_accounting_mismatch(
    tmp_path: Path, case: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_backend_admission(fixture, parity_status="failed")
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    admission = receipt["scoring_backend_admission"]
    _mutate_uncached_depth_accounting(admission, case)
    admission_without_digest = dict(admission)
    admission_without_digest.pop("sha256")
    admission["sha256"] = sut.sha256_json(admission_without_digest)
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match=message):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


def test_production_rules_reject_fixture_runtime_execution_status(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, contract_mode="production")
    with pytest.raises(
        sut.SummaryContractError,
        match="production rules require sealed live scoring",
    ):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


def test_producer_to_consumer_accepts_executed_bounded_free_null_as_non_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, contract_mode="production")
    receipt = _admit_live_runtime(fixture)
    rows = fixture["rows"]
    assert isinstance(receipt, dict)
    assert isinstance(rows, list)
    request_id = "free-root:fixture"
    free_execution = _bounded_null_free_execution_receipt(request_id)
    receipt["surfaces"] = {
        "free_coordinate_tree": {
            "declared_request_ids": [request_id],
            "receipts": [free_execution],
        },
        "restricted_candidate_bank": {"status": "executed"},
    }
    sealed = score_producer.seal_execution_receipt(
        receipt,
        scores_path=fixture["scores_path"],
        rows=rows,
    )
    free_surface = sealed["landscape_surface_receipts"][
        "canonical_description_free"
    ]
    assert free_surface["status"] == "executed_bounded_null_non_evidence"
    assert free_surface["row_count"] == 0
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(sealed), encoding="utf-8")
    summary, output_receipt = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=receipt_path,
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )
    assert summary["contract_mode"] == "production"
    assert sealed["runtime_execution_status"] == (
        "live_model_scoring_completed_artifacts_sealed"
    )
    assert output_receipt["independent_reconstruction"] == "passed"


def test_reviewed_candidate_with_clear_digest_bound_registry_lineage_is_measured(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    _mark_reviewed_candidate(rows)
    _rewrite_scores(fixture, rows)

    summary, _ = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )

    entry = summary["per_owner_context"][0]
    assert entry["decision_status"] == "measured_no_conclusion"
    assert entry["neutral_reasons"] == []


def test_sealed_root_context_with_null_source_row_and_registry_lineage_is_measured(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    _mark_sealed_context(rows)
    _rewrite_scores(fixture, rows)

    summary, _ = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )

    entry = summary["per_owner_context"][0]
    assert entry["decision_status"] == "measured_no_conclusion"
    assert entry["neutral_reasons"] == []


def test_sealed_context_rejects_registry_lineage_with_a_different_review_status(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    _mark_sealed_context(rows)
    for row in rows:
        lineage = row["upstream_adjudication"][
            "source_review_foreign_key_lineage"
        ]
        lineage["review_status"] = "reviewed_candidate"
        lineage_payload = dict(lineage)
        lineage_payload.pop("lineage_sha256")
        lineage["lineage_sha256"] = sut.sha256_json(lineage_payload)
    _rewrite_scores(fixture, rows)

    with pytest.raises(sut.SummaryContractError, match="relabels its review status"):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("neutral_kind", ["unresolved", "globally_ambiguous"])
def test_sealed_context_does_not_override_owner_neutrality(
    tmp_path: Path, neutral_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    _mark_sealed_context(rows)
    for row in rows:
        if neutral_kind == "unresolved":
            row["owner_status"] = "unresolved"
        else:
            row["upstream_adjudication"]["global_ambiguity_status"] = (
                "globally_ambiguous"
            )
    _rewrite_scores(fixture, rows)

    summary, _ = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )

    entry = summary["per_owner_context"][0]
    assert entry["decision_status"] == "neutral_raw_only"
    assert entry["neutral_reasons"] == [neutral_kind]
    assert entry["basins"] == []


@pytest.mark.parametrize("neutral_kind", ["unresolved", "globally_ambiguous"])
def test_reviewed_candidate_does_not_override_owner_neutrality(
    tmp_path: Path, neutral_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    _mark_reviewed_candidate(rows)
    for row in rows:
        if neutral_kind == "unresolved":
            row["owner_status"] = "unresolved"
        else:
            row["upstream_adjudication"]["global_ambiguity_status"] = (
                "globally_ambiguous"
            )
    _rewrite_scores(fixture, rows)

    summary, _ = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )

    entry = summary["per_owner_context"][0]
    assert entry["decision_status"] == "neutral_raw_only"
    assert entry["neutral_reasons"] == [neutral_kind]
    assert entry["basins"] == []


@pytest.mark.parametrize("neutral_kind", ["unresolved", "unreviewed", "globally_ambiguous"])
def test_unknown_unreviewed_and_global_ambiguity_are_raw_only_neutral(tmp_path: Path, neutral_kind: str) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    for row in rows:
        if neutral_kind == "unresolved":
            row["owner_status"] = "unresolved"
        elif neutral_kind == "unreviewed":
            row["review_status"] = "unreviewed"
        else:
            row["upstream_adjudication"]["global_ambiguity_status"] = "globally_ambiguous"
    _rewrite_scores(fixture, rows)
    summary, _ = sut.summarize(
        scores_path=fixture["scores_path"],  # type: ignore[arg-type]
        score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        rules_path=fixture["rules_path"],  # type: ignore[arg-type]
    )
    entry = summary["per_owner_context"][0]
    assert entry["decision_status"] == "neutral_raw_only"
    assert neutral_kind in entry["neutral_reasons"]
    assert entry["basins"] == []


def test_rejects_policy_mixing_in_one_owner_context(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = deepcopy(fixture["rows"])
    assert isinstance(rows, list)
    rows[0]["native_repetition_penalty_stratum"] = 1.1
    _rewrite_scores(fixture, rows)
    with pytest.raises(sut.SummaryContractError, match="decode-policy strata"):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


def test_rejects_tampered_score_artifact_and_stale_conditional_y1_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    scores_path = fixture["scores_path"]
    assert isinstance(scores_path, Path)
    scores_path.write_text(scores_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match="digest"):
        sut.summarize(
            scores_path=scores_path,
            score_receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )

    fixture = _fixture(tmp_path / "stale")
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    receipt["conditional_y1_attestations"][0]["completeness_digest"] = "0" * 64
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match="does not independently reconstruct"):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


def test_current_scorer_receipt_fails_with_exact_missing_contract(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    receipt.pop("conditional_y1_attestations")
    receipt["conditional_y1_completeness"] = [{"status": "complete", "completeness_digest": "x"}]
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match="fully identity-bound"):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(runtime_execution_status="implemented_not_gpu_smoke_tested_this_session"), "runtime_execution_status"),
        (
            lambda receipt: receipt["implementation_provenance"].update(
                git_dirty_diff_sha256="0" * 63
            ),
            "dirty-diff digest",
        ),
        (
            lambda receipt: receipt["landscape_surface_receipts"][
                "restricted_gt_target"
            ].update(score_rows_sha256="0" * 64),
            "receipt cardinality/digest",
        ),
    ],
)
def test_rejects_execution_provenance_and_surface_receipt_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    fixture = _fixture(tmp_path)
    receipt = deepcopy(fixture["receipt"])
    assert isinstance(receipt, dict)
    mutation(receipt)
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt_path, Path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SummaryContractError, match=message):
        sut.summarize(
            scores_path=fixture["scores_path"],  # type: ignore[arg-type]
            score_receipt_path=receipt_path,
            rules_path=fixture["rules_path"],  # type: ignore[arg-type]
        )


def test_reduced_coordinate_ranges_require_explicit_test_fixture_mode() -> None:
    document = _rules(contract_mode="production")
    document["coordinate_bins"] = {"min": 0, "max": 3}
    document["scoring_contract"]["coordinate_token_id_end_exclusive"] = 104
    semantic_payload = core.build_semantic_core_payload(document)
    document["semantic_core"]["payload"] = semantic_payload
    document["semantic_core"]["sha256"] = sut.sha256_json(semantic_payload)
    with pytest.raises(ValueError, match="production rules require"):
        core.validate_rule_mapping(document)


def test_cli_is_write_once(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "output"
    argv = [
        "--scores",
        str(fixture["scores_path"]),
        "--score-receipt",
        str(fixture["receipt_path"]),
        "--decision-rules",
        str(fixture["rules_path"]),
        "--output-dir",
        str(output),
    ]
    assert sut.main(argv) == 0
    with pytest.raises(sut.SummaryContractError, match="refusing to overwrite"):
        sut.main(argv)
