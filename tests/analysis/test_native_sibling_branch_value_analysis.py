from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.analyze_native_sibling_branch_value import (
    BOOTSTRAP_SEED_ROOT,
    CONFIRMATION_SEED_ROOT,
    _all_exact_variant_pairs_nonreversing,
    _candidate_clusters,
    _candidate_for_prediction,
    _classify_path,
    _crossover_primitives,
    _execution_contract_ok,
    _exact_variant_crossover_primitives,
    _formal_greedy_gap,
    _freeze_review,
    _joint_bootstrap_intervals,
    _load_variants,
    _match_physical,
    _owner_completeness,
    _outcome_summary,
    _resolve_greedy_control,
    _review_candidate_key,
    _tree_digest,
    _unmatched_rows,
    _validate_execution_contract_lineage,
    _validate_call_bundle_identity,
)


def _ledger(owner: str, category: str, box: list[float]) -> dict[str, object]:
    return {
        "final_state": "accepted",
        "image_id": 1,
        "object_identifier": owner,
        "normalized_category_name": category,
        "source_canvas_box_xyxy": box,
    }


def test_blind_candidate_ids_are_deterministic_and_omit_provenance() -> None:
    rows = [
        {"image_id": "1", "seed": "9", "record": {"owner_id": "arm-a"}, "row_index": 0, "prediction": {"description": "cup", "bbox": [10, 10, 30, 30]}},
        {"image_id": "1", "seed": "10", "record": {"owner_id": "arm-b"}, "row_index": 1, "prediction": {"description": "cup", "bbox": [11, 11, 31, 31]}},
    ]
    candidates = _candidate_clusters(rows)
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == _review_candidate_key("1", "cup", [10, 10, 30, 30])
    assert "seed" not in candidates[0]
    assert "record" not in candidates[0]
    assert "arm" not in candidates[0]


def test_freeze_review_requires_every_candidate_and_stable_approval_id() -> None:
    candidates = [{"candidate_id": "candidate:a", "image_id": "1", "category": "cup", "bbox_xyxy": [1, 1, 4, 4]}]
    with pytest.raises(ValueError, match="approved review candidate lacks"):
        _freeze_review(candidates, [{"candidate_id": "candidate:a", "verdict": "approve"}])
    with pytest.raises(ValueError, match="incomplete"):
        _freeze_review(candidates, [])
    frozen = _freeze_review(candidates, [{"candidate_id": "candidate:a", "verdict": "approve", "entity_ref": "human:1:0001", "comment": "real"}])
    assert frozen["mapping"]["candidate:a"]["entity_ref"] == "human:1:0001"


def test_stateful_classification_does_not_reassign_duplicates_to_unused_owner() -> None:
    ledger = [_ledger("covered", "cup", [10, 10, 30, 30]), _ledger("other", "cup", [100, 100, 130, 130])]
    record = {
        "branch_hash": "branch",
        "owner_id": "new",
        "call": {
            "seed": "7",
            "rows": [
                {"generated_order": 0, "description": "cup", "bbox": [10, 10, 30, 30]},
                {"generated_order": 1, "description": "cup", "bbox": [10, 10, 30, 30]},
            ],
            "termination": {"termination_classification": "horizon_reached"},
        },
    }
    result = _classify_path(record, image_id="1", parent_covered={"covered"}, ledger=ledger, candidates=[], frozen_review={"mapping": {}}, discovered_owner_ids={"new", "other"})
    assert [item["classification"] for item in result["outcomes"]] == ["new_supported_predeclared", "duplicate_parent_covered", "duplicate_within_suffix"]
    assert all(item.get("owner_id") == "covered" for item in result["outcomes"][1:])


def test_unknown_world_changes_only_unknown_support_count() -> None:
    outcomes = [{"classification": "new_supported_predeclared", "owner_id": "owner"}, {"classification": "new_supported_not_predeclared", "owner_id": "human:owner"}, {"classification": "unknown", "unknown_id": "candidate:x"}, {"classification": "unsupported"}, {"classification": "terminal"}]
    lower = _outcome_summary(outcomes, unknown_as_supported=False)
    upper = _outcome_summary(outcomes, unknown_as_supported=True)
    assert lower["new_supported_count"] == 1
    assert upper["new_supported_count"] == 3
    assert lower["unsupported_event"] == upper["unsupported_event"] == 1
    assert lower["unknown_event"] == upper["unknown_event"] == 1


def test_outcome_summary_exposes_required_descriptive_duplicate_contrasts() -> None:
    outcomes = [
        {"classification": "duplicate_current", "owner_id": "owner"},
        {"classification": "duplicate_parent_covered", "owner_id": "covered"},
        {"classification": "duplicate_within_suffix", "owner_id": "other"},
        {"classification": "unknown", "unknown_id": "candidate:x"},
        {"classification": "terminal"},
    ]
    summary = _outcome_summary(outcomes, unknown_as_supported=False)
    assert summary["duplicate_current_event"] == 1
    assert summary["duplicate_covered_event"] == 1
    assert summary["duplicate_event"] == 1
    assert summary["unknown_event"] == 1
    assert summary["terminal_event"] == 1


def test_execution_contract_rejects_runtime_or_policy_drift() -> None:
    contract = {
        "physical_batch_size": 1,
        "runtime_dtype_mode": "config",
        "model_config_dtype": "bf16",
        "actual_model_parameter_dtypes": {"parameter_dtype_names": ["torch.bfloat16", "torch.float32"]},
        "attention_implementation": "sdpa",
        "decode_generation_policy": {"mode": "sampled", "temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.0},
    }
    _execution_contract_ok(contract)
    with pytest.raises(ValueError, match="physical batch"):
        _execution_contract_ok({**contract, "physical_batch_size": 2})
    with pytest.raises(ValueError, match="sampling policy"):
        _execution_contract_ok({**contract, "decode_generation_policy": {**contract["decode_generation_policy"], "temperature": 0.7}})


def test_full_model_fp32_runtime_requires_explicit_opt_in() -> None:
    fp32_contract = {
        "physical_batch_size": 1,
        "runtime_dtype_mode": "fp32",
        "model_config_dtype": "bf16",
        "actual_model_parameter_dtypes": {"parameter_dtype_names": ["torch.float32"]},
        "attention_implementation": "sdpa",
        "decode_generation_policy": {"mode": "sampled", "temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.0},
    }
    with pytest.raises(ValueError, match="expected source-consistent config mode"):
        _execution_contract_ok(fp32_contract)
    _execution_contract_ok(fp32_contract, expected_runtime_dtype="fp32")


def test_fp32_runtime_rejects_mixed_or_malformed_actual_parameter_dtypes() -> None:
    base = {
        "physical_batch_size": 1,
        "runtime_dtype_mode": "fp32",
        "model_config_dtype": "bf16",
        "attention_implementation": "sdpa",
        "decode_generation_policy": {"mode": "sampled", "temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.0},
    }
    with pytest.raises(ValueError, match="all model parameters as torch.float32"):
        _execution_contract_ok({**base, "actual_model_parameter_dtypes": {"parameter_dtype_names": ["torch.bfloat16", "torch.float32"]}}, expected_runtime_dtype="fp32")
    with pytest.raises(ValueError, match="malformed actual model parameter dtype"):
        _execution_contract_ok({**base, "actual_model_parameter_dtypes": {"parameter_dtype_names": "torch.float32"}}, expected_runtime_dtype="fp32")


def test_execution_contract_lineage_rejects_mixed_runtime_and_contract_drift() -> None:
    config = {
        "physical_batch_size": 1,
        "runtime_dtype_mode": "config",
        "model_config_dtype": "bf16",
        "actual_model_parameter_dtypes": {"parameter_dtype_names": ["torch.bfloat16", "torch.float32"]},
        "attention_implementation": "sdpa",
        "decode_generation_policy": {"mode": "sampled", "temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.0},
        "model_identity": {"family": "test"},
    }
    fp32 = {
        **config,
        "runtime_dtype_mode": "fp32",
        "actual_model_parameter_dtypes": {"parameter_dtype_names": ["torch.float32"]},
    }
    with pytest.raises(ValueError, match="expected source-consistent config mode"):
        _validate_execution_contract_lineage([config, fp32])
    drifted = {**config, "model_identity": {"family": "different"}}
    with pytest.raises(ValueError, match="execution contracts disagree"):
        _validate_execution_contract_lineage([config, drifted])


def test_tree_digest_is_stable_for_json_inputs(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text('{"b":2}\n', encoding="utf-8")
    (tmp_path / "a.json").write_text('{"a":1}\n', encoding="utf-8")
    first = _tree_digest(tmp_path)
    second = _tree_digest(tmp_path)
    assert first == second
    assert first
    assert BOOTSTRAP_SEED_ROOT == 2026071703000001
    assert CONFIRMATION_SEED_ROOT == 2026071702000001


def test_load_variants_excludes_greedy_and_counts_discovery_frequency(tmp_path: Path) -> None:
    path = tmp_path / "variants.jsonl"
    rows = [
        {"decode_mode": "greedy", "sampling_seed": None, "owner_id": "owner-a", "row_token_ids_sha256": "greedy"},
        {"decode_mode": "sampled", "sampling_seed": "1", "owner_id": "owner-a", "row_token_ids_sha256": "a"},
        {"decode_mode": "sampled", "sampling_seed": "2", "owner_id": "owner-a", "row_token_ids_sha256": "a"},
        {"decode_mode": "sampled", "sampling_seed": "3", "owner_id": "owner-b", "row_token_ids_sha256": "b"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    variants = _load_variants(path)
    assert set(variants) == {"a", "b"}
    assert variants["a"]["discovery_support_count"] == 2
    assert variants["b"]["discovery_support_count"] == 1
    assert all(row["discovery_denominator"] == 3 for row in variants.values())


def test_accepted_ledger_owner_is_predeclared_even_without_admission_arm() -> None:
    ledger = [_ledger("accepted-extra", "cup", [40, 40, 60, 60])]
    record = {
        "branch_hash": "branch",
        "owner_id": "admitted-branch",
        "call": {"seed": "7", "rows": [{"generated_order": 0, "description": "cup", "bbox": [40, 40, 60, 60]}], "termination": {}},
    }
    result = _classify_path(record, image_id="1", parent_covered=set(), ledger=ledger, candidates=[], frozen_review={"mapping": {}}, discovered_owner_ids={"admitted-branch", "accepted-extra"})
    assert result["outcomes"][1]["classification"] == "new_supported_predeclared"
    assert result["outcomes"][1]["owner_id"] == "accepted-extra"
    assert _outcome_summary(result["outcomes"], unknown_as_supported=False)["new_supported_count"] == 2


def test_review_approved_absent_ledger_entity_is_not_predeclared_and_neutral() -> None:
    prediction = {"description": "dog", "bbox": [40, 40, 60, 60]}
    candidates = _candidate_clusters([{"image_id": "1", "row_index": 0, "prediction": prediction}])
    frozen = _freeze_review(candidates, [{"candidate_id": candidates[0]["candidate_id"], "verdict": "approve", "entity_ref": "human:1:dog-0001", "comment": "real"}])
    record = {
        "branch_hash": "branch",
        "owner_id": "admitted-branch",
        "call": {"seed": "7", "rows": [prediction], "termination": {}},
    }
    result = _classify_path(record, image_id="1", parent_covered=set(), ledger=[], candidates=candidates, frozen_review=frozen, discovered_owner_ids={"admitted-branch"})
    assert result["outcomes"][1]["classification"] == "new_supported_not_predeclared"
    assert result["outcomes"][1]["physical"]["physical_status"] == "review_approved"
    assert _outcome_summary(result["outcomes"], unknown_as_supported=False)["good_event"] == 0.0


def test_frozen_review_can_resolve_ambiguous_parent_ledger_match() -> None:
    prediction = {"description": "cup", "bbox": [1, 1, 11, 11]}
    ledger = [_ledger("left", "cup", [0, 0, 10, 10]), _ledger("right", "cup", [2, 2, 12, 12])]
    candidates = _candidate_clusters([{"image_id": "1", "row_index": 0, "prediction": prediction}])
    frozen = _freeze_review(candidates, [{"candidate_id": candidates[0]["candidate_id"], "verdict": "approve", "entity_ref": "left", "comment": "left entity"}])
    result = _match_physical(prediction, image_id="1", ledger=ledger, candidates=candidates, frozen_review=frozen)
    assert result["physical_status"] == "review_approved"
    assert result["owner_id"] == "left"


def test_review_approved_accepted_ledger_owner_remains_predeclared() -> None:
    prediction = {"description": "cup", "bbox": [1, 1, 11, 11]}
    ledger = [_ledger("left", "cup", [0, 0, 10, 10]), _ledger("right", "cup", [2, 2, 12, 12])]
    candidates = _candidate_clusters([{"image_id": "1", "row_index": 0, "prediction": prediction}])
    frozen = _freeze_review(candidates, [{"candidate_id": candidates[0]["candidate_id"], "verdict": "approve", "entity_ref": "left", "comment": "accepted ledger owner"}])
    record = {"branch_hash": "branch", "owner_id": "new", "call": {"seed": "7", "rows": [prediction], "termination": {}}}
    result = _classify_path(record, image_id="1", parent_covered=set(), ledger=ledger, candidates=candidates, frozen_review=frozen, discovered_owner_ids=set(), accepted_owner_ids={"left", "right"})
    assert result["outcomes"][1]["classification"] == "new_supported_predeclared"


def test_automatic_ambiguous_suffix_rows_enter_blind_candidate_clustering() -> None:
    prediction = {"description": "cup", "bbox": [1, 0, 11, 10]}
    records = [{"branch_hash": "branch", "owner_id": "owner", "calls_by_seed": {"1": {"rows": [prediction]}}}]
    ledger = [_ledger("left", "cup", [0, 0, 10, 10]), _ledger("right", "cup", [2, 0, 12, 10])]
    rows = _unmatched_rows(records, ledger=ledger, image_id="1")
    assert len(rows) == 1
    assert rows[0]["physical_match_status"] == "ambiguous"
    assert len(_candidate_clusters(rows)) == 1


def test_candidate_clusters_are_connected_components_and_retain_member_signatures() -> None:
    rows = [
        {"image_id": "1", "row_index": 0, "prediction": {"description": "person", "bbox": [0, 0, 10, 10]}},
        {"image_id": "1", "row_index": 1, "prediction": {"description": "person", "bbox": [2, 0, 12, 10]}},
        {"image_id": "1", "row_index": 2, "prediction": {"description": "person", "bbox": [4, 0, 14, 10]}},
    ]
    candidates = _candidate_clusters(rows)
    assert len(candidates) == 1
    assert len(candidates[0]["member_prediction_signatures"]) == 3
    assert _candidate_for_prediction(rows[-1]["prediction"], candidates, "1")["candidate_id"] == candidates[0]["candidate_id"]


def test_crossover_evaluates_every_unordered_three_owner_pair() -> None:
    seeds = ["s1", "s2", "s3"]
    owners = ["a", "b", "c"]
    panel = {}
    for owner in owners:
        panel[owner] = {}
        for seed in seeds:
            panel[owner][seed] = {
                "next_owner:a": 0.1 if owner == "a" else 0.7,
                "next_owner:b": 0.1 if owner == "b" else 0.7,
                "next_owner:c": 0.1 if owner == "c" else 0.7,
                "good_event": 0.8,
                "bad_event": 0.1,
            }
    result = _crossover_primitives(panel, owners, seed_order=seeds, replicates=200, upper_owner_seed=panel)
    assert result["pair_count"] == 3
    assert set(result["pairs"]) == {"a__vs__b", "a__vs__c", "b__vs__c"}


def test_aggregate_crossover_uses_interval_not_each_seed_for_no_reversal() -> None:
    seeds = [f"s{index}" for index in range(32)]
    panel = {"a": {}, "b": {}}
    for index, seed in enumerate(seeds):
        if index == 0:
            panel["a"][seed] = {"next_owner:a": 1.0, "next_owner:b": 0.0, "good_event": 0.8, "bad_event": 0.1}
            panel["b"][seed] = {"next_owner:a": 0.0, "next_owner:b": 1.0, "good_event": 0.8, "bad_event": 0.1}
        else:
            panel["a"][seed] = {"next_owner:a": 0.0, "next_owner:b": 1.0, "good_event": 0.8, "bad_event": 0.1}
            panel["b"][seed] = {"next_owner:a": 1.0, "next_owner:b": 0.0, "good_event": 0.8, "bad_event": 0.1}
    result = _crossover_primitives(panel, ["a", "b"], seed_order=seeds, replicates=500, upper_owner_seed=panel)
    pair = result["pairs"]["a__vs__b"]
    assert any(value < 0 for value in pair["per_seed_contrasts"]["C_a"].values())
    assert pair["no_reversal"] is True
    assert pair["status"] == "identified"


def test_exact_variant_crossover_emits_paired_intervals_for_every_pair() -> None:
    seeds = [f"s{index}" for index in range(32)]
    rows = []
    for owner, next_owner in (("a", "b"), ("b", "a")):
        for variant in ("v1", "v2"):
            per_seed = []
            for index, seed in enumerate(seeds):
                # One paired seed is a deliberately negative observation for
                # both directions; the mean/interval remains positive. This
                # must not be treated as a variant reversal.
                if index == 0:
                    value = owner
                else:
                    value = next_owner
                per_seed.append({"seed": seed, "next_owner": value})
            rows.append({"owner_id": owner, "branch_hash": f"{owner}-{variant}", "per_seed": per_seed})
    result = _exact_variant_crossover_primitives(rows, ["a", "b"], seed_order=seeds, replicates=500)
    pair = result["pairs"]["a__vs__b"]
    assert result["status"] == "identified"
    assert pair["variant_count"] == 4
    for variant in pair["variants"].values():
        assert variant["intervals"]["paired_unit"] == "confirmation_seed_index"
        assert variant["intervals"]["simultaneous"] is True
        assert any(value < 0 for value in variant["per_seed_contrasts"]["C_a"].values())
        assert variant["no_reversal"] is True


def test_zero_exact_variant_lower_bound_is_non_reversing_but_not_identified() -> None:
    seeds = [f"s{index}" for index in range(8)]
    rows = [
        {"owner_id": owner, "branch_hash": owner, "per_seed": [{"seed": seed, "next_owner": None} for seed in seeds]}
        for owner in ("a", "b")
    ]
    result = _exact_variant_crossover_primitives(rows, ["a", "b"], seed_order=seeds, replicates=200)
    pair = result["pairs"]["a__vs__b"]
    variant = next(iter(pair["variants"].values()))
    assert all(metric["lower_95"] == 0.0 for metric in variant["intervals"]["metrics"].values())
    assert variant["no_reversal"] is True
    assert pair["no_reversal_across_exact_variants"] is True
    assert pair["status"] == "not_identified"
    assert result["status"] == "not_identified"
    assert _all_exact_variant_pairs_nonreversing(result) is True


def test_formal_greedy_gap_includes_zero_when_all_alternatives_are_negative() -> None:
    comparisons = {
        "owner-a": {"Q_H_lower": -0.25},
        "owner-b": {"Q_H_lower": -0.10},
    }
    result = _formal_greedy_gap(comparisons, greedy_owner_id="greedy", value_key="Q_H_lower", comparison_evaluated=True)
    assert result["formal_value"] == 0.0
    assert result["formal_owner_id"] == "greedy"
    assert result["maximum_alternative_value"] == -0.10
    assert result["maximum_alternative_owner_id"] == "owner-b"


def test_refused_empty_greedy_comparison_does_not_emit_numeric_zero() -> None:
    result = _formal_greedy_gap({}, greedy_owner_id="", value_key="Q_H_lower", comparison_evaluated=False)
    assert result["formal_value"] is None
    assert result["formal_owner_id"] is None
    assert result["maximum_alternative_value"] is None
    assert result["maximum_alternative_owner_id"] is None
    assert result["greedy_zero_baseline_included"] is False
    assert result["status"] == "not_applicable"


def test_unknown_upper_bound_counts_unique_candidate_ids_not_rows() -> None:
    outcomes = [
        {"classification": "new_supported_predeclared", "owner_id": "owner"},
        {"classification": "unknown", "unknown_id": "candidate:x"},
        {"classification": "unknown", "unknown_id": "candidate:x"},
    ]
    assert _outcome_summary(outcomes, unknown_as_supported=False)["new_supported_count"] == 1
    assert _outcome_summary(outcomes, unknown_as_supported=True)["new_supported_count"] == 2
    assert _outcome_summary(outcomes, unknown_as_supported=True)["unknown_unique_count"] == 1


def test_absent_admitted_owner_is_incomplete() -> None:
    result = _owner_completeness({"variant-a"}, set(), {})
    assert result["complete"] is False
    assert result["missing_exact_variants"] == ["variant-a"]


def test_malformed_first_action_does_not_compress_later_rows() -> None:
    record = {
        "branch_hash": "branch",
        "owner_id": "branch-owner",
        "call": {"seed": "7", "chronology_valid": False, "chronology_reason": "parser_dropped_prediction", "rows": [{"generated_order": 1, "description": "cup", "bbox": [1, 1, 3, 3]}], "termination": {}},
    }
    result = _classify_path(record, image_id="1", parent_covered=set(), ledger=[], candidates=[], frozen_review={"mapping": {}}, discovered_owner_ids={"branch-owner"})
    assert [item["classification"] for item in result["outcomes"]] == ["new_supported_predeclared", "invalid"]
    assert result["outcomes"][1]["row_index"] == 0


def test_cross_linked_bundle_is_rejected() -> None:
    contract = {"decode_generation_policy": {"mode": "sampled", "temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.0}}
    receipt = {"image_id": "1", "prompt": {"prompt_token_ids_sha256": "prompt"}, "execution_contract": contract}
    call = {"request_id": "request", "decode_mode": "sampled"}
    bundle = {"image_id": "1", "sampling_seed": 7, "request_id": "request", "executed_call_attestation": {"request_id": "request"}, "donor": {"branch_row": {"token_ids_sha256": "wrong"}}, "prompt": {"prompt_token_ids_sha256": "prompt"}, "runtime": {"decode_mode": "sampled", "decode_generation_policy": contract["decode_generation_policy"]}}
    with pytest.raises(ValueError, match="branch hash"):
        _validate_call_bundle_identity(receipt, call, bundle, receipt_branch_hash="expected", seed=7, receipt_path=Path("receipt.json"))


def test_joint_bootstrap_is_simultaneous_and_non_studentized() -> None:
    result = _joint_bootstrap_intervals({"value": {"a": 1.0, "b": 2.0}, "safety": {"a": 0.0, "b": 0.1}}, seed_order=["a", "b"], replicates=100, family="test")
    assert result["simultaneous"] is True
    assert result["interval_method"] == "non_studentized_max_deviation"
    assert set(result["metrics"]) == {"value", "safety"}


def test_parent_covered_greedy_owner_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "greedy.json"
    source.write_text(json.dumps({"image_id": "1", "request_id": "request", "parse_result": {"predictions": [{"description": "cup", "bbox": [1, 1, 4, 4]}]}}), encoding="utf-8")
    admission = {"admitted_owner_ids": ["owner"], "greedy_control": [{"ledger_status": "unique", "owner_admitted": True, "owner_id": "owner", "source_bundle": str(source)}]}
    result = _resolve_greedy_control(admission, image_id="1", ledger=[_ledger("owner", "cup", [1, 1, 4, 4])], parent_covered={"owner"})
    assert result["status"] == "refused"
    assert result["reason"] == "greedy_owner_parent_covered"


def test_unresolved_greedy_row_is_review_resolved_then_admitted_and_uncovered(tmp_path: Path) -> None:
    source = tmp_path / "greedy.json"
    prediction = {"description": "cup", "bbox": [1, 1, 4, 4]}
    source.write_text(json.dumps({"image_id": "1", "request_id": "request", "parse_result": {"predictions": [prediction]}}), encoding="utf-8")
    candidates = _candidate_clusters([{"image_id": "1", "prediction": prediction, "source": "greedy_control"}])
    frozen = _freeze_review(candidates, [{"candidate_id": candidates[0]["candidate_id"], "verdict": "approve", "entity_ref": "owner", "comment": "greedy owner"}])
    admission = {"admitted_owner_ids": ["owner"], "greedy_control": [{"ledger_status": "unmatched", "owner_admitted": None, "owner_id": None, "source_bundle": str(source)}]}
    result = _resolve_greedy_control(admission, image_id="1", ledger=[_ledger("owner", "cup", [1, 1, 4, 4])], parent_covered=set(), candidates=candidates, frozen_review=frozen)
    assert result["status"] == "validated"
    assert result["owner_id"] == "owner"
    assert result["physical_status"] == "review_approved"
