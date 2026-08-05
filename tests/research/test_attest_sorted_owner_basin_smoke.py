"""CPU tests for control-only calibration and blinded smoke attestation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research import attest_sorted_owner_basin_smoke as sut
from scripts.research import score_sorted_owner_basin_landscape as score_producer
from scripts.research import sorted_owner_basin_landscape as core
from scripts.research import summarize_sorted_owner_basin_landscape as summary_sut
from scripts.research.prepare_sorted_owner_basin_inputs import _structural_policy


def _core_rules() -> dict:
    return {
        "schema_version": core.RULES_SCHEMA_VERSION,
        "contract_mode": "test_fixture",
        "geometry_identity": {"schema": core.GEOMETRY_IDENTITY_SCHEMA, "coordinate_denominator": 1000},
        "coordinate_bins": {"min": 0, "max": 3},
        "target_anchor": {"margin_fraction": 0.0, "min_margin_bins": 0, "max_margin_bins": 0},
        "bank_order": ["target", "covered", "background", "scan"],
        "proposal_measures": {
            "matched": {
                "comparability_group": "matched",
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": {"target": 1.0, "covered": 1.0, "background": 1.0, "scan": 1.0},
            }
        },
        "bank_proposal_measure": {bank: "matched" for bank in ("target", "covered", "background", "scan")},
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
        },
        "prominence": {"functional": "peak_height_difference"},
        "structural_status": "draft_pre_smoke",
        "task6_context_selection": {
            "plan_membership": "task4_control_only",
            "emitted_context_ids": ["root"],
        },
        "free_coordinate_tree": {
            "budget": {"fixture": True},
            "selector": {"fixture": True},
            "candidate_membership_surface": "separate_from_restricted_candidate_bank",
        },
        "global_foil_role_and_description_registry": {
            "fixture": "global-registry"
        },
        "candidate_weights": {
            "target": 1.0,
            "covered": 1.0,
            "background": 1.0,
            "scan": 1.0,
        },
        "score_channels": {"raw_fp32": {"role": "primary_model_likelihood"}},
        "registered_sampling_policy": {"null_semantics": "absence_neutral"},
        "loose_support": {"rule_owner": "fixture"},
        "ablation": {
            "operator": "fixture-target-ablation",
            "equal_area_background_operator": "fixture-background-ablation",
        },
        "calibration": {
            "algorithm": "empirical_quantile_lower.v1",
            "quantile": 0.5,
            "minimum_roles": ["strict_visible_true_positive", "b1_loose_only"],
            "control_ids": list(sut.REQUIRED_SMOKE_CONTROLS[:2]),
            "metric_reducers": {
                "peak_height": "maximum_target_peak",
                "peak_prominence": "minimum_registered_foil_prominence",
            },
        },
        "invalidation_rule": "any semantic or source identity change invalidates",
        "control_hierarchy": {"fixture": list(sut.REQUIRED_SMOKE_CONTROLS)},
        "canonical_description_source": "fixture owner ledger",
        "canonical_alias_policy": "fixture frozen aliases",
        "model_tokenizer_runtime_vocabulary_identity": {
            "model_identity_sha256": "6" * 64,
            "tokenizer_identity_sha256": "7" * 64,
            "runtime_identity_sha256": "8" * 64,
            "model_vocab_size": 4,
        },
    }


def _bind_semantic_core(document: dict) -> None:
    payload = core.build_semantic_core_payload(document)
    document["semantic_core"] = {
        "schema_version": core.SEMANTIC_CORE_SCHEMA_VERSION,
        "payload": payload,
        "sha256": summary_sut.sha256_json(payload),
        "excluded_outer_execution_fields": list(
            core.SEMANTIC_CORE_OUTER_EXECUTION_FIELDS
        ),
    }


def _entry(
    gt_owner_id: str,
    *,
    peak: float,
    prominence: float,
    neutral: bool = False,
    surface: str = "restricted_gt_target",
) -> dict:
    return {
        "diagnostic_owner_id": gt_owner_id,
        "gt_owner_id": gt_owner_id,
        "image_id": "7511",
        "image_identity": "image:7511",
        "context_id": "root",
        "landscape_surface": surface,
        "native_repetition_penalty_stratum": 1.0,
        "foil_set_id": "smoke-foils",
        "decision_status": "neutral_raw_only" if neutral else "measured_no_conclusion",
        "neutral_reasons": ["globally_ambiguous"] if neutral else [],
        "basins": []
        if neutral
        else [
            {
                "basin_id": "target",
                "role_kind": "target",
                "shape": "localized_peak",
                "peak_height": peak,
            }
        ],
        "peak_prominence": [] if neutral else [{"peak_prominence": prominence}],
        "scientific_conclusion": None,
    }


def _control_surface_pair(gt_owner_id: str, *, peak: float, prominence: float) -> list[dict]:
    return [
        _entry(gt_owner_id, peak=peak, prominence=prominence),
        _entry(
            gt_owner_id,
            peak=peak,
            prominence=prominence,
            neutral=True,
            surface="canonical_description_free",
        ),
    ]


def test_b2_production_suppression_requires_prospectively_bound_physical_foil() -> None:
    before = {
        "peak_prominence": [
            {
                "foil_identity_kind": "reviewed_physical_owner",
                "foil_reviewed_physical_owner_id": "gt:7511:24",
                "peak_prominence": -1.0,
            },
            {
                "foil_identity_kind": "registered_geometry",
                "foil_registered_geometry_id": "scan",
                "peak_prominence": 10.0,
            },
        ]
    }
    after = {
        "peak_prominence": [
            {
                "foil_identity_kind": "reviewed_physical_owner",
                "foil_reviewed_physical_owner_id": "gt:7511:24",
                "peak_prominence": -2.0,
            },
            {
                "foil_identity_kind": "registered_geometry",
                "foil_registered_geometry_id": "scan",
                "peak_prominence": 5.0,
            },
        ]
    }
    status, reason, owner_ids = sut._b2_matched_physical_suppression(  # noqa: SLF001
        before_entry=before,
        after_entry=after,
        b2_control={
            "target_gt_owner_id": "gt:7511:26",
            "covering_gt_owner_id": "gt:7511:22",
        },
    )
    assert status == "failed"
    assert reason == "required_matched_physical_foil_absent"
    assert owner_ids == []

    status, reason, owner_ids = sut._b2_matched_physical_suppression(  # noqa: SLF001
        before_entry=before,
        after_entry=after,
        b2_control={
            "target_gt_owner_id": "gt:7511:26",
            "covering_gt_owner_id": "gt:7511:22",
            "matched_same_description_nonoverlap_gt_owner_id": "gt:7511:22",
        },
    )
    assert status == "failed"
    assert reason == "matched_physical_foil_reuses_covering_owner"
    assert owner_ids == []

    status, reason, owner_ids = sut._b2_matched_physical_suppression(  # noqa: SLF001
        before_entry=before,
        after_entry=after,
        b2_control={
            "target_gt_owner_id": "gt:7511:26",
            "covering_gt_owner_id": "gt:7511:22",
            "matched_same_description_nonoverlap_gt_owner_id": "gt:7511:24",
        },
    )
    assert status == "passed"
    assert reason == "none"
    assert owner_ids == ["gt:7511:24"]

    status, reason, owner_ids = sut._b2_matched_physical_suppression(  # noqa: SLF001
        before_entry=before,
        after_entry=after,
        b2_control={
            "target_gt_owner_id": "gt:7511:26",
            "covering_gt_owner_id": "gt:7511:22",
            "matched_same_description_nonoverlap_gt_owner_id": "gt:7511:24",
            "matched_different_description_gt_owner_id": "gt:7511:24",
        },
    )
    assert status == "failed"
    assert reason == "invalid_optional_matched_physical_foil"
    assert owner_ids == []


def test_attestor_rejects_probe_only_relaxed_cache_summary() -> None:
    parity = {
        "status": "passed",
        "atol": score_producer.CACHE_PARITY_ATOL,
        "rtol": score_producer.CACHE_PARITY_RTOL,
    }
    admission = score_producer.build_scoring_backend_admission(
        parity_gate=parity,
        selection=score_producer.select_scoring_backend_from_parity(parity),
        score_row_count=1,
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
    admission["decision_use"] = score_producer.PROBE_ONLY_SCORE_USE
    admission["cache_admission_policy"] = {
        "effective_mode": score_producer.RELAXED_CACHE_ADMISSION_MODE
    }
    admission_without_digest = dict(admission)
    admission_without_digest.pop("sha256")
    admission["sha256"] = sut.sha256_json(admission_without_digest)
    summary = {"scoring_backend_admission": admission}
    receipt = {
        "scoring_backend_admission": admission,
        "mandatory_cache_parity_gate": "passed",
        "scoring_backend_gate": "cache_parity_passed",
    }

    with pytest.raises(sut.SmokeAttestationError, match="probe-only"):
        sut._summary_scoring_backend_gate(summary, receipt)  # noqa: SLF001

    admission.pop("decision_use")
    admission_without_digest = dict(admission)
    admission_without_digest.pop("sha256")
    admission["sha256"] = sut.sha256_json(admission_without_digest)
    with pytest.raises(sut.SmokeAttestationError, match="legacy relaxed-cache"):
        sut._summary_scoring_backend_gate(summary, receipt)  # noqa: SLF001


def _write_summary(
    path: Path,
    receipt_path: Path,
    *,
    entries: list[dict],
    rules: core.LandscapeRules,
    rules_sha: str,
    parity_status: str = "passed",
) -> None:
    parity = {
        "status": parity_status,
        "atol": score_producer.CACHE_PARITY_ATOL,
        "rtol": score_producer.CACHE_PARITY_RTOL,
    }
    selection = score_producer.select_scoring_backend_from_parity(parity)
    selected_backend = selection["selected_backend"]
    fallback = selected_backend == score_producer.FULL_REFORWARD_SCORING_BACKEND
    admission = score_producer.build_scoring_backend_admission(
        parity_gate=parity,
        selection=selection,
        score_row_count=len(entries),
        per_context_group_accounting=[
            {
                "scoring_backend": selected_backend,
                "context_id": "context:fixture",
                "group_id": "group:fixture",
                "root_prefix_length": 1,
                "root_calls": 1,
                "logical_token_step_requests": 0,
                "logical_token_step_requests_by_depth": {},
                "actual_forward_calls": 1,
                "actual_forward_calls_by_depth": {"0": 1},
                "memo_hits_by_depth": {"1": 0, "2": 0} if fallback else {},
                "memo_entries_by_depth": {"0": 1, "1": 0, "2": 0}
                if fallback
                else {},
                "retained_relative_depths": [0, 1, 2] if fallback else [],
            }
        ],
    )
    scoring_gate = (
        "cache_parity_passed"
        if parity_status == "passed"
        else "cache_parity_failed_uncached_reference_used"
    )
    summary = {
        "schema_version": summary_sut.SUMMARY_SCHEMA_VERSION,
        "contract_mode": "test_fixture",
        "rule_digest": rules.rule_digest,
        "rules_file_sha256": rules_sha,
        "score_artifact_sha256": "9" * 64,
        "score_receipt_sha256": "a" * 64,
        "scoring_backend_admission": admission,
        "per_owner_context": entries,
        "scientific_conclusion": None,
    }
    receipt = {
        "schema_version": summary_sut.RECEIPT_SCHEMA_VERSION,
        "contract_mode": "test_fixture",
        "rule_digest": rules.rule_digest,
        "rules_file_sha256": rules_sha,
        "summary_payload_sha256": summary_sut.sha256_json(summary),
        "independent_reconstruction": "passed",
        "mandatory_cache_parity_gate": parity_status,
        "scoring_backend_gate": scoring_gate,
        "scoring_backend_admission": admission,
        "scientific_conclusion": None,
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def _fixture(tmp_path: Path, *, disposition: str = "proceed_census_only") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": sut.SENTINEL_SELECTION_SCHEMA_VERSION,
        "anti_leakage_contract": {"landscape_scores_used_for_selection": False},
    }
    selection_path = tmp_path / "sentinel-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    task0_hashes = {
        "task0_census_artifact_manifest_sha256": "1" * 64,
        "task0_execution_receipt_content_sha256": "2" * 64,
        "task0_execution_receipt_file_sha256": "3" * 64,
        "owner_ledger_sha256": "4" * 64,
        "owner_trajectory_matrix_sha256": "5" * 64,
    }
    confirmation = {
        "schema_version": sut.SENTINEL_CONFIRMATION_SCHEMA_VERSION,
        "anti_leakage_contract": {
            "landscape_scores_used_for_original_selection": False,
            "landscape_scores_used_for_v2_confirmation": False,
            "selection_membership_changed": False,
        },
        "final_task0_v2": {
            "artifact_manifest_sha256": task0_hashes["task0_census_artifact_manifest_sha256"],
            "execution_receipt_content_sha256": task0_hashes["task0_execution_receipt_content_sha256"],
            "execution_receipt_file_sha256": task0_hashes["task0_execution_receipt_file_sha256"],
            "owner_ledger_sha256": task0_hashes["owner_ledger_sha256"],
            "owner_trajectory_matrix_sha256": task0_hashes["owner_trajectory_matrix_sha256"],
        },
        "confirmed_sentinels": [
            {
                "gt_owner_id": "gt:7511:6",
                "decision_eligible": True,
                "strict_match_count": 0,
                "max_semantic_compatible_iou": 0.0,
            }
        ],
    }
    confirmation_path = tmp_path / "sentinel-confirmation.json"
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    sentinel_registry = {
        "schema_version": sut.SENTINEL_REGISTRY_SCHEMA_VERSION,
        "selection_receipt": {"sha256": summary_sut.sha256_file(selection_path)},
        "selection_confirmation_receipt": {"sha256": summary_sut.sha256_file(confirmation_path)},
        "sentinels": [
            {"sentinel_id": "sentinel:far-person:7511:6", "gt_owner_id": "gt:7511:6", "image_id": "7511"}
        ],
    }
    sentinel_registry_path = tmp_path / "sentinel-registry.json"
    sentinel_registry_path.write_text(json.dumps(sentinel_registry), encoding="utf-8")
    registry_controls = [
        {
            "control_id": sut.REQUIRED_SMOKE_CONTROLS[0],
            "role": "strict_visible_true_positive",
            "gt_owner_id": "gt:7511:22",
            "strata": ["representative_smoke", "far_person_calibration"],
        },
        {
            "control_id": sut.REQUIRED_SMOKE_CONTROLS[1],
            "role": "b1_loose_only",
            "gt_owner_id": "gt:7511:26",
            "strata": ["representative_smoke", "far_person_calibration"],
        },
        {
            "control_id": sut.REQUIRED_SMOKE_CONTROLS[2],
            "role": "b2_distinct_same_description_pair",
            "covering_gt_owner_id": "gt:7511:22",
            "target_gt_owner_id": "gt:7511:26",
            "physical_identity_review": "distinct_people_confirmed_by_lead_visual_inspection",
            "strata": ["representative_smoke", "far_person_calibration"],
        },
        {
            "control_id": "control:far-person:b1:7511:15",
            "role": "b1_loose_only",
            "gt_owner_id": "gt:7511:15",
            "strata": ["far_person_calibration"],
        },
        {
            "control_id": "control:wall-bowl:strict-visible:13923:5",
            "role": "strict_visible_true_positive",
            "gt_owner_id": "gt:13923:5",
            "strata": ["wall_bowl_fallback_calibration"],
        },
        {
            "control_id": "control:wall-bowl:b1:16228:47",
            "role": "b1_loose_only",
            "gt_owner_id": "gt:16228:47",
            "strata": ["wall_bowl_fallback_calibration"],
        },
    ]
    control_registry = {
        "schema_version": sut.CONTROL_REGISTRY_SCHEMA_VERSION,
        "status": "lead_frozen_before_scoring_and_resealed_to_final_task0_v2",
        "source_digests": {
            **task0_hashes,
            "sentinel_selection_confirmation_receipt_sha256": summary_sut.sha256_file(confirmation_path),
        },
        "selection_rules": {"no_c_outcome_use": True},
        "controls": registry_controls,
    }
    control_registry_path = tmp_path / "control-registry.json"
    control_registry_path.write_text(json.dumps(control_registry), encoding="utf-8")
    rules_document = _core_rules()
    prepare_policy = _structural_policy(
        "draft_pre_smoke", controls=registry_controls, contract_mode="production"
    )
    rules_document["calibration"] = prepare_policy["calibration"]
    rules_document["sealed_inputs"] = {
        **task0_hashes,
        "sentinel_selection_confirmation_receipt_sha256": summary_sut.sha256_file(confirmation_path),
        "control_registry_sha256": summary_sut.sha256_file(control_registry_path),
        "sentinel_registry_sha256": summary_sut.sha256_file(sentinel_registry_path),
        "sentinel_selection_receipt_sha256": summary_sut.sha256_file(selection_path),
        "identity_receipt_sha256": "b" * 64,
    }
    rules_document["calibration_contract"] = prepare_policy["calibration"]
    rules_document["smoke_attestation"] = {
        "disposition_rules": [
            {"when": {"stop_rule_3_positive_control_peak": "clear"}, "disposition": disposition}
        ]
    }
    _bind_semantic_core(rules_document)
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules_document), encoding="utf-8")
    rules = core.validate_rule_mapping(rules_document)
    rules_sha = summary_sut.sha256_file(rules_path)
    control_summary_path = tmp_path / "control-summary.json"
    control_receipt_path = tmp_path / "control-summary-receipt.json"
    _write_summary(
        control_summary_path,
        control_receipt_path,
        entries=[
            *_control_surface_pair("gt:7511:22", peak=-1.0, prominence=2.0),
            *_control_surface_pair("gt:7511:26", peak=-2.0, prominence=1.0),
        ],
        rules=rules,
        rules_sha=rules_sha,
    )
    sentinel_summary_path = tmp_path / "sentinel-summary.json"
    sentinel_receipt_path = tmp_path / "sentinel-summary-receipt.json"
    _write_summary(
        sentinel_summary_path,
        sentinel_receipt_path,
        entries=[_entry("gt:7511:6", peak=1000.0, prominence=1000.0)],
        rules=rules,
        rules_sha=rules_sha,
    )
    return {
        "rules": rules_path,
        "control_registry": control_registry_path,
        "control_summary": control_summary_path,
        "control_receipt": control_receipt_path,
        "sentinel_registry": sentinel_registry_path,
        "selection": selection_path,
        "confirmation": confirmation_path,
        "sentinel_summary": sentinel_summary_path,
        "sentinel_receipt": sentinel_receipt_path,
    }


def _attest(paths: dict[str, Path], output: Path, *, sentinel: bool) -> tuple[dict, dict]:
    optional: dict = {}
    if sentinel:
        stage1_output = output.parent / f"{output.name}-stage1"
        sut.attest(
            control_summary_path=paths["control_summary"],
            control_summary_receipt_path=paths["control_receipt"],
            control_registry_path=paths["control_registry"],
            rules_path=paths["rules"],
            output_dir=stage1_output,
        )
        freeze_path = stage1_output / sut.NON_C_SMOKE_FREEZE_NAME
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        sentinel_rules = json.loads(paths["rules"].read_text(encoding="utf-8"))
        sentinel_rules["structural_status"] = "sealed_non_c_smoke"
        sentinel_rules["task6_context_selection"] = {
            "plan_membership": "task4_sentinel_only",
            "emitted_context_ids": ["root"],
        }
        sentinel_rules["non_c_smoke_freeze_receipt"] = {
            "sha256": summary_sut.sha256_file(freeze_path),
            "control_decision_rules_sha256": summary_sut.sha256_file(paths["rules"]),
            "semantic_core_sha256": freeze["semantic_core_sha256"],
        }
        sentinel_rules_path = paths["rules"].parent / f"sentinel-rules-{output.name}.json"
        sentinel_rules_path.write_text(json.dumps(sentinel_rules), encoding="utf-8")
        sentinel_rules_validated = core.validate_rule_mapping(sentinel_rules)
        sentinel_rules_sha = summary_sut.sha256_file(sentinel_rules_path)
        sentinel_summary = json.loads(
            paths["sentinel_summary"].read_text(encoding="utf-8")
        )
        sentinel_summary["rule_digest"] = sentinel_rules_validated.rule_digest
        if sentinel_summary.get("rules_file_sha256") != "0" * 64:
            sentinel_summary["rules_file_sha256"] = sentinel_rules_sha
        paths["sentinel_summary"].write_text(
            json.dumps(sentinel_summary), encoding="utf-8"
        )
        sentinel_receipt = json.loads(
            paths["sentinel_receipt"].read_text(encoding="utf-8")
        )
        sentinel_receipt["rule_digest"] = sentinel_rules_validated.rule_digest
        sentinel_receipt["rules_file_sha256"] = sentinel_rules_sha
        sentinel_receipt["summary_payload_sha256"] = summary_sut.sha256_json(
            sentinel_summary
        )
        paths["sentinel_receipt"].write_text(
            json.dumps(sentinel_receipt), encoding="utf-8"
        )
        optional = {
            "sentinel_rules_path": sentinel_rules_path,
            "sentinel_summary_path": paths["sentinel_summary"],
            "sentinel_summary_receipt_path": paths["sentinel_receipt"],
            "sentinel_registry_path": paths["sentinel_registry"],
            "sentinel_selection_path": paths["selection"],
            "sentinel_confirmation_path": paths["confirmation"],
        }
    return sut.attest(
        control_summary_path=paths["control_summary"],
        control_summary_receipt_path=paths["control_receipt"],
        control_registry_path=paths["control_registry"],
        rules_path=paths["rules"],
        output_dir=output,
        **optional,
    )


def test_control_only_calibration_ignores_optional_sentinel_and_disposition_is_declared(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture", disposition="narrow")
    control_only, lead_without = _attest(paths, tmp_path / "without", sentinel=False)
    with_sentinel, lead_with = _attest(paths, tmp_path / "with", sentinel=True)
    assert control_only == with_sentinel
    assert control_only["thresholds"] == {"peak_height": -2.0, "peak_prominence": 1.0}
    assert control_only["anti_leakage"]["only_sealed_non_c_controls_read"] is True
    assert lead_without["disposition"] == "narrow"
    assert lead_with["disposition"] == "narrow"
    assert lead_with["sentinel_attestation"]["case_disposition"] == "unresolved"
    assert lead_with["scientific_conclusion"] is None


def test_failed_positive_control_emits_hold_without_non_c_freeze(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "fixture")
    rules = json.loads(paths["rules"].read_text(encoding="utf-8"))
    rules["smoke_attestation"] = {
        "disposition_rules": [
            {
                "when": {"stop_rule_3_positive_control_peak": "clear"},
                "disposition": "proceed_census_only",
            },
            {
                "when": {"stop_rule_3_positive_control_peak": "triggered"},
                "disposition": "hold",
            },
        ]
    }
    _bind_semantic_core(rules)
    paths["rules"].write_text(json.dumps(rules), encoding="utf-8")
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    summary["rules_file_sha256"] = summary_sut.sha256_file(paths["rules"])
    for entry in summary["per_owner_context"]:
        if (
            entry["gt_owner_id"] == "gt:7511:22"
            and entry["landscape_surface"] == "restricted_gt_target"
        ):
            entry["basins"][0]["shape"] = "part_or_whole_lobes"
    paths["control_summary"].write_text(json.dumps(summary), encoding="utf-8")
    receipt = json.loads(paths["control_receipt"].read_text(encoding="utf-8"))
    receipt["rules_file_sha256"] = summary_sut.sha256_file(paths["rules"])
    receipt["rule_digest"] = core.validate_rule_mapping(rules).rule_digest
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(summary)
    paths["control_receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    _, lead = _attest(paths, tmp_path / "held", sentinel=False)

    assert lead["disposition"] == "hold"
    assert lead["status"] == "held_before_non_c_smoke_freeze"
    assert lead["stop_rule_states"]["stop_rule_3_positive_control_peak"] == (
        "triggered"
    )
    assert lead["non_c_smoke_freeze_receipt_sha256"] is None
    assert lead["non_c_smoke_freeze_status"] == "not_emitted_control_gate_failed"
    assert not (tmp_path / "held" / sut.NON_C_SMOKE_FREEZE_NAME).exists()


def test_attester_freeze_and_lead_preserve_failed_parity_uncached_admission(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "fixture")
    rules_document = json.loads(paths["rules"].read_text(encoding="utf-8"))
    rules = core.validate_rule_mapping(rules_document)
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    _write_summary(
        paths["control_summary"],
        paths["control_receipt"],
        entries=summary["per_owner_context"],
        rules=rules,
        rules_sha=summary_sut.sha256_file(paths["rules"]),
        parity_status="failed",
    )
    output = tmp_path / "out"
    _calibration, lead = _attest(paths, output, sentinel=False)
    freeze = json.loads((output / sut.NON_C_SMOKE_FREEZE_NAME).read_text(encoding="utf-8"))
    assert freeze["gates"]["mandatory_cache_parity"] == "failed"
    assert (
        freeze["gates"]["scoring_backend_admission"]
        == "cache_parity_failed_uncached_reference_used"
    )
    assert (
        lead["control_scoring_backend_gate"]
        == "cache_parity_failed_uncached_reference_used"
    )


def test_attester_rejects_relabelled_failed_parity_backend_gate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    rules = core.validate_rule_mapping(
        json.loads(paths["rules"].read_text(encoding="utf-8"))
    )
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    _write_summary(
        paths["control_summary"],
        paths["control_receipt"],
        entries=summary["per_owner_context"],
        rules=rules,
        rules_sha=summary_sut.sha256_file(paths["rules"]),
        parity_status="failed",
    )
    receipt = json.loads(paths["control_receipt"].read_text(encoding="utf-8"))
    receipt["scoring_backend_gate"] = "cache_parity_passed"
    paths["control_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SmokeAttestationError, match="does not match its admission"):
        _attest(paths, tmp_path / "out", sentinel=False)


def test_globally_ambiguous_control_is_rejected_from_calibration(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    summary["per_owner_context"][0] = _entry("gt:7511:22", peak=-1.0, prominence=2.0, neutral=True)
    paths["control_summary"].write_text(json.dumps(summary), encoding="utf-8")
    receipt = json.loads(paths["control_receipt"].read_text(encoding="utf-8"))
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(summary)
    paths["control_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SmokeAttestationError, match="globally ambiguous"):
        _attest(paths, tmp_path / "out", sentinel=False)


def test_summary_tamper_is_rejected_by_independent_reconstruction_digest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    summary["per_owner_context"][0]["basins"][0]["peak_height"] = 99.0
    paths["control_summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(sut.SmokeAttestationError, match="does not independently reconstruct"):
        _attest(paths, tmp_path / "out", sentinel=False)


@pytest.mark.parametrize(
    "missing_surface", ["canonical_description_free", "restricted_gt_target"]
)
def test_control_calibration_requires_both_executed_surfaces(
    tmp_path: Path, missing_surface: str
) -> None:
    paths = _fixture(tmp_path / "fixture")
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    summary["per_owner_context"] = [
        entry
        for entry in summary["per_owner_context"]
        if not (
            entry["gt_owner_id"] == "gt:7511:22"
            and entry["landscape_surface"] == missing_surface
        )
    ]
    paths["control_summary"].write_text(json.dumps(summary), encoding="utf-8")
    receipt = json.loads(paths["control_receipt"].read_text(encoding="utf-8"))
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(summary)
    paths["control_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SmokeAttestationError, match="exactly one executed free"):
        _attest(paths, tmp_path / "out", sentinel=False)


def test_attester_rejects_extra_non_smoke_calibration_control_id(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "fixture")
    rules_document = json.loads(paths["rules"].read_text(encoding="utf-8"))
    extra_id = "control:far-person:b1:7511:15"
    rules_document["calibration"]["control_ids"].append(extra_id)
    rules_document["calibration_contract"]["control_ids"].append(extra_id)
    _bind_semantic_core(rules_document)
    paths["rules"].write_text(json.dumps(rules_document), encoding="utf-8")
    rules = core.validate_rule_mapping(rules_document)
    rules_sha = summary_sut.sha256_file(paths["rules"])
    summary = json.loads(paths["control_summary"].read_text(encoding="utf-8"))
    summary["rule_digest"] = rules.rule_digest
    summary["rules_file_sha256"] = rules_sha
    paths["control_summary"].write_text(json.dumps(summary), encoding="utf-8")
    receipt = json.loads(paths["control_receipt"].read_text(encoding="utf-8"))
    receipt["rule_digest"] = rules.rule_digest
    receipt["rules_file_sha256"] = rules_sha
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(summary)
    paths["control_receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(sut.SmokeAttestationError, match="may use only the sealed"):
        _attest(paths, tmp_path / "out", sentinel=False)


def test_rule_change_after_score_invalidates_sentinel_and_leaves_only_sealed_calibration(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    sentinel = json.loads(paths["sentinel_summary"].read_text(encoding="utf-8"))
    sentinel["rules_file_sha256"] = "0" * 64
    paths["sentinel_summary"].write_text(json.dumps(sentinel), encoding="utf-8")
    receipt = json.loads(paths["sentinel_receipt"].read_text(encoding="utf-8"))
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(sentinel)
    paths["sentinel_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(sut.SmokeAttestationError, match="stale decision-rule file digest"):
        _attest(paths, output, sentinel=True)
    assert (output / sut.CALIBRATION_NAME).is_file()
    assert (output / sut.NON_C_SMOKE_FREEZE_NAME).is_file()
    assert not (output / sut.LEAD_REVIEW_NAME).exists()


def test_optional_sentinel_must_be_exactly_one_sealed_owner(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    summary = json.loads(paths["sentinel_summary"].read_text(encoding="utf-8"))
    summary["per_owner_context"].append(_entry("gt:7511:4", peak=-3.0, prominence=-1.0))
    paths["sentinel_summary"].write_text(json.dumps(summary), encoding="utf-8")
    receipt = json.loads(paths["sentinel_receipt"].read_text(encoding="utf-8"))
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(summary)
    paths["sentinel_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(sut.SmokeAttestationError, match="exactly one GT owner"):
        _attest(paths, tmp_path / "out", sentinel=True)


def test_attestor_outputs_are_write_once(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    output = tmp_path / "out"
    _attest(paths, output, sentinel=False)
    with pytest.raises(sut.SmokeAttestationError, match="refusing to overwrite"):
        _attest(paths, output, sentinel=False)


def test_calibration_and_freeze_are_durable_before_first_sentinel_outcome_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "fixture")
    output = tmp_path / "out"
    original = sut._read_json
    sentinel_reads: list[Path] = []

    def tracking_read(path: Path):
        if path == paths["sentinel_summary"]:
            sentinel_reads.append(path)
            assert (output / sut.CALIBRATION_NAME).is_file()
            assert (output / sut.NON_C_SMOKE_FREEZE_NAME).is_file()
        return original(path)

    monkeypatch.setattr(sut, "_read_json", tracking_read)
    _attest(paths, output, sentinel=True)
    assert sentinel_reads


def test_two_process_freeze_lifecycle_recomputes_identical_phase_a_receipts(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "fixture")
    final_output = tmp_path / "final"

    calibration, lead = _attest(paths, final_output, sentinel=True)

    phase_a_output = tmp_path / "final-stage1"
    phase_a_calibration = phase_a_output / sut.CALIBRATION_NAME
    phase_a_freeze = phase_a_output / sut.NON_C_SMOKE_FREEZE_NAME
    final_calibration = final_output / sut.CALIBRATION_NAME
    final_freeze = final_output / sut.NON_C_SMOKE_FREEZE_NAME
    sentinel_rules_path = paths["rules"].parent / "sentinel-rules-final.json"
    sentinel_rules = json.loads(sentinel_rules_path.read_text(encoding="utf-8"))

    assert phase_a_calibration.read_bytes() == final_calibration.read_bytes()
    assert phase_a_freeze.read_bytes() == final_freeze.read_bytes()
    assert summary_sut.sha256_file(phase_a_freeze) == summary_sut.sha256_file(
        final_freeze
    )
    assert sentinel_rules["non_c_smoke_freeze_receipt"]["sha256"] == (
        summary_sut.sha256_file(phase_a_freeze)
    )
    assert sentinel_rules["non_c_smoke_freeze_receipt"][
        "control_decision_rules_sha256"
    ] == summary_sut.sha256_file(paths["rules"])
    assert sentinel_rules["non_c_smoke_freeze_receipt"][
        "semantic_core_sha256"
    ] == calibration["semantic_core_sha256"]
    assert lead["non_c_smoke_freeze_receipt_sha256"] == summary_sut.sha256_file(
        final_freeze
    )
    assert lead["sentinel_attestation"]["case_disposition"] == "unresolved"
    assert (final_output / sut.LEAD_REVIEW_NAME).is_file()


def test_failed_phase_c_never_writes_a_final_lead_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    sentinel = json.loads(paths["sentinel_summary"].read_text(encoding="utf-8"))
    sentinel["rules_file_sha256"] = "0" * 64
    paths["sentinel_summary"].write_text(json.dumps(sentinel), encoding="utf-8")
    receipt = json.loads(paths["sentinel_receipt"].read_text(encoding="utf-8"))
    receipt["summary_payload_sha256"] = summary_sut.sha256_json(sentinel)
    paths["sentinel_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    final_output = tmp_path / "final"

    with pytest.raises(sut.SmokeAttestationError, match="stale decision-rule file digest"):
        _attest(paths, final_output, sentinel=True)

    assert (final_output / sut.CALIBRATION_NAME).is_file()
    assert (final_output / sut.NON_C_SMOKE_FREEZE_NAME).is_file()
    assert not (final_output / sut.LEAD_REVIEW_NAME).exists()
