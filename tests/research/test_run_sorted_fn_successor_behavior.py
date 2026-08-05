"""Focused CPU contracts for the sorted FN successor behavior runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.analyze_sorted_fn_mechanisms import (
    FIXED_BUDGET_SCHEMA_VERSION,
    PLANNER_RECEIPT_SCHEMA_VERSION,
    analyze_sorted_fn_mechanisms,
)
from scripts.research.attest_sorted_fn_successor_score_run import (
    ATTESTATION_NAME,
    attest,
)
from scripts.research.merge_sorted_fn_successor_score_shards import (
    MERGE_SCHEMA_VERSION,
    PREDECESSOR_PRIMITIVES_FILE,
)
from scripts.research.run_sorted_fn_successor_behavior import (
    BehaviorContractError,
    CONTRACT_RECEIPT_SCHEMA_VERSION,
    DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    LANDSCAPE_ADMISSION_SCHEMA_VERSION,
    LANDSCAPE_RECEIPT_SCHEMA_VERSION,
    LANDSCAPE_DECISION_CHANNEL,
    REGISTRY_SCHEMA_VERSION,
    SAMPLING_ADMISSION_SCHEMA_VERSION,
    UNIT_ID,
    _owner_pixel_bbox_to_entity_norm1000,
    _write_create_or_identical,
    _git_code_provenance,
    build_contract,
    canonical_json_bytes,
    classify_semantic_relation,
    execute_behavior_contract,
    sha256_file,
    sha256_json,
    validate_registry,
    validate_live_runtime_identity,
    validate_sampling_admission,
)
from scripts.research.run_same_covered_set_prefix_order_probe import (
    match_predictions_to_entities,
)


_REAL_RUNTIME = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-01-sorted-owner-basin-landscape-and-repair/runtime-identity.json"
)
_REAL_INFER_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32.yaml"
).resolve()
_REAL_SOURCE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/"
    "evaluation-inputs/human-refined-12.coord.jsonl"
)


def _json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _role(
    role_id: str,
    kind: str,
    tokens: list[int],
    pred_ids: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "role_id": role_id,
        "role_kind": kind,
        "gt_owner_id": "gt:1:1",
        "trajectory": {"image_id": "1", "decode_mode": "greedy", "seed": 0},
        "prefix": {
            "token_ids": tokens,
            "token_ids_sha256": sha256_json(tokens),
            "token_count": len(tokens),
            "prefix_pred_row_ids": pred_ids,
        },
        "provenance": {},
        **extra,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    owner = tmp_path / "owners.jsonl"
    pred = tmp_path / "predictions.jsonl"
    cohort = tmp_path / "cohorts.jsonl"
    rollout = tmp_path / "rollout.json"
    _jsonl(
        owner,
        [
            {"gt_owner_id": "gt:1:1", "image_id": "1", "normalized_description": "person", "bbox_xyxy": [1, 1, 2, 2]},
            {"gt_owner_id": "gt:1:2", "image_id": "1", "normalized_description": "person", "bbox_xyxy": [3, 3, 4, 4]},
            {"gt_owner_id": "gt:1:9", "image_id": "1", "normalized_description": "bus", "bbox_xyxy": [5, 5, 6, 6]},
        ],
    )
    _jsonl(cohort, [{"gt_owner_id": "gt:1:1"}])
    row_a = [151646, 90, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    row_d = [151646, 91, 151647, 151648, 151674, 151675, 151676, 151677, 151649]
    _json(
        rollout,
        {
            "rollouts": [
                {
                    "image_id": "1",
                    "decode_mode": "greedy",
                    "seed": 0,
                    "stop_reason": "im_end",
                    "prompt_token_ids": [10, 11],
                    "generated_token_ids": [*row_a, *row_d],
                    "predictions": {
                        "predictions": [
                            {
                                "generated_order": 0,
                                "raw_span_sha256": "d" * 64,
                                "bbox": [5, 5, 6, 6],
                            },
                            {
                                "generated_order": 1,
                                "raw_span_sha256": "e" * 64,
                                "bbox": [3, 3, 4, 4],
                            },
                        ],
                        "dropped_predictions": [],
                    },
                }
            ]
        },
    )
    rollout_ref = _ref(rollout)
    common_lineage = {
        "image_id": "1",
        "decode_mode": "greedy",
        "seed": 0,
        "source_artifact_path": rollout_ref["path"],
        "source_artifact_sha256": rollout_ref["sha256"],
    }
    _jsonl(
        pred,
        [
            {
                "pred_row_id": "pred:sorted:greedy:0:1:0",
                "strict_match_gt_owner_id": "gt:1:9",
                **common_lineage,
            },
            {
                "pred_row_id": "pred:sorted:greedy:0:1:1",
                "strict_match_gt_owner_id": "gt:1:2",
                **common_lineage,
            },
        ],
    )
    pre = _role(
        "first_skip:P_pre",
        "first_skip_pre",
        [10, 11, *row_a],
        ["pred:sorted:greedy:0:1:0"],
    )
    post = _role(
        "first_skip:P_post",
        "first_skip_post",
        [10, 11, *row_a, *row_d],
        ["pred:sorted:greedy:0:1:0", "pred:sorted:greedy:0:1:1"],
        successor_gt_owner_id="gt:1:2",
        successor_pred_row_id="pred:sorted:greedy:0:1:1",
    )
    for role in (pre, post):
        role["provenance"] = {
            "source_artifact_path": rollout_ref["path"],
            "source_artifact_sha256": rollout_ref["sha256"],
        }
    registry_content = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "sources": {
            "owner_ledger": _ref(owner),
            "prediction_row_ledger": _ref(pred),
            "cohort_assignments": _ref(cohort),
            "rollout_artifacts": [rollout_ref],
        },
        "mechanism_cohort": {"targets": []},
        "context_control_registry": {"bound_non_targets": []},
        "smoke": {
            "roles": [pre, post],
            "null_pair_envelope": {"pairs": []},
        },
    }
    registry = tmp_path / "registry.json"
    _json(registry, {**registry_content, "registry_digest": sha256_json(registry_content)})

    runtime = _REAL_RUNTIME
    runtime_document = json.loads(runtime.read_text(encoding="utf-8"))

    forced = [151646, 77, 151647, 151648]
    token_registry_content = {
        "schema_tokens": runtime_document["schema_tokens"],
        "model_vocab_size": runtime_document["model_vocab_size"],
        "identity_receipt_digest": runtime_document["receipt_digest"],
    }
    token_registry = {
        **token_registry_content,
        "registry_sha256": sha256_json(token_registry_content),
    }
    ledger_rows = []
    for role, self_prefix in ((pre, row_a), (post, [*row_a, *row_d])):
        full = role["prefix"]["token_ids"]
        ledger_rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "diagnostic_owner_id": "diagnostic:gt:1:1",
                "gt_owner_id": "gt:1:1",
                "context_id": f"ctx:fn:{role['role_id']}",
                "image_id": "1",
                "ground_truth": {"box": [1, 1, 2, 2]},
                "canonical_description": {
                    "text": "person",
                    "forced_row_prefix_through_box_start_token_ids": forced,
                    "forced_row_prefix_through_box_start_sha256": sha256_json(forced),
                },
                "context_tokens": {
                    "token_ids": full,
                    "token_ids_sha256": sha256_json(full),
                    "prompt_prefix_token_count": 2,
                    "prompt_token_ids_sha256": sha256_json([10, 11]),
                    "self_prefix_generated_token_ids_sha256": sha256_json(self_prefix),
                },
                "context_provenance": {
                    "source_binding": {"registry_digest": registry_content and sha256_json(registry_content)}
                },
                "upstream_digests": {"fn_mechanism_registry_sha256": sha256_json(registry_content)},
                "token_registry": token_registry,
                "runtime_vocabulary_receipt": {
                    "identity_receipt_digest": runtime_document["receipt_digest"],
                    "model_vocab_size": runtime_document["model_vocab_size"],
                    "model_identity_sha256": runtime_document["model"]["identity_sha256"],
                    "tokenizer_identity_sha256": runtime_document["tokenizer"]["identity_sha256"],
                    "runtime_identity_sha256": runtime_document["runtime"]["identity_sha256"],
                    "token_registry_sha256": token_registry["registry_sha256"],
                },
                "other_owner_reference_status": (
                    "no_same_description_non_overlapping_owner"
                ),
            }
        )
    ledger = tmp_path / "owner-context-ledger.jsonl"
    _jsonl(ledger, ledger_rows)

    landscape = tmp_path / "landscape-receipt.json"
    decision_rules = tmp_path / "landscape-decision-rules.json"
    _json(decision_rules, {"fixture": "execution-rules"})
    fixed_budget = tmp_path / "fixed-budget-candidates.jsonl"
    _jsonl(fixed_budget, [{"fixture": "candidate-parent"}])
    rules_template = tmp_path / "rules-template.json"
    _json(rules_template, {"fixture": "rules-template"})
    mechanism_content = {
        "schema_version": "sorted-fn-mechanism-decision-rules.v1",
        "unit_id": UNIT_ID,
        "conditional_sampling": {
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "null_semantics": "finite_sampling_null_is_absence_neutral",
        },
        "upstream_digests": {
            "execution_landscape_decision_rules_sha256": sha256_file(
                decision_rules
            ),
            "fn_mechanism_registry_sha256": sha256_json(registry_content),
            "rules_template_sha256": sha256_file(rules_template),
        },
    }
    mechanism_rules = tmp_path / "mechanism-decision-rules.json"
    _json(
        mechanism_rules,
        {**mechanism_content, "self_digest": sha256_json(mechanism_content)},
    )
    planner_content = {
        "schema_version": "sorted-fn-successor-input-plan-receipt.v2",
        "unit_id": UNIT_ID,
        "sources": {
            "registry": _ref(registry),
            "rules_template": _ref(rules_template),
        },
        "outputs": {
            "owner_context_ledger": {**_ref(ledger), "row_count": 2},
            "landscape_decision_rules": _ref(decision_rules),
            "mechanism_decision_rules": _ref(mechanism_rules),
            "fixed_budget_candidates": {**_ref(fixed_budget), "row_count": 1},
        },
        "neighborhood_consistency": {
            "multi_context_groups_checked": 0,
            "status": "consistent",
        },
    }
    planner = tmp_path / "planner-receipt.json"
    _json(
        planner,
        {**planner_content, "receipt_digest": sha256_json(planner_content)},
    )
    mechanism_binding = {
        "path": str(mechanism_rules),
        "sha256": sha256_file(mechanism_rules),
        "self_digest": json.loads(mechanism_rules.read_text())["self_digest"],
        "parent_execution_rules_sha256": sha256_file(decision_rules),
    }
    _json(
        landscape,
        {
            "schema_version": LANDSCAPE_RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "generic_arbitrary_role_merger": True,
            "decision_channel": {
                "name": LANDSCAPE_DECISION_CHANNEL,
                "primary_repetition_penalty_stratum": 1.0,
                "auxiliary_policy_is_not_a_model_likelihood": True,
            },
            "successor_scorer_provenance": {
                "row_schema_version": "sorted_fn_fixed_budget_scores.v1",
                "receipt_schema_version": "sorted_fn_fixed_budget_scores_receipt.v1",
                "unit_id": UNIT_ID,
            },
            "source_digests": {
                "fn_mechanism_registry": _ref(registry),
                "owner_context_ledger": _ref(ledger),
                "decision_rules": _ref(decision_rules),
                "mechanism_decision_rules": mechanism_binding,
                "fixed_budget_candidates": _ref(fixed_budget),
            },
            "planner_receipt": {
                **_ref(planner),
                "receipt_digest": json.loads(planner.read_text())["receipt_digest"],
            },
            "context_selection": {
                "selected_context_ids": ["ctx:fn:first_skip:P_pre", "ctx:fn:first_skip:P_post"]
            },
            "selected_rungs": ["L1"],
        },
    )
    landscape_admission = tmp_path / "landscape-admission.json"
    _json(
        landscape_admission,
        {
            "schema_version": LANDSCAPE_ADMISSION_SCHEMA_VERSION,
            "status": "passed",
            "registry_digest": sha256_json(registry_content),
            "landscape_receipt_sha256": sha256_file(landscape),
            "mechanism_decision_rules": {
                "sha256": sha256_file(mechanism_rules),
                "self_digest": mechanism_binding["self_digest"],
                "parent_execution_rules_sha256": sha256_file(decision_rules),
                "planner_receipt_digest": json.loads(planner.read_text())[
                    "receipt_digest"
                ],
                "registry_digest": sha256_json(registry_content),
                "fixed_budget_candidates_sha256": sha256_file(fixed_budget),
            },
            "admitted_roles": [
                {
                    "role_id": role["role_id"],
                    "gt_owner_id": role["gt_owner_id"],
                    "context_id": f"ctx:fn:{role['role_id']}",
                    "status": "passed",
                    "landscape_condition": {
                        "name": "usable_target_strict_region_peak",
                        "status": "passed",
                        "rung": "L1",
                    },
                }
                for role in (pre, post)
            ],
        },
    )
    return {
        "owner": owner,
        "pred": pred,
        "registry": registry,
        "runtime": runtime,
        "ledger": ledger,
        "landscape": landscape,
        "landscape_admission": landscape_admission,
        "mechanism_rules": mechanism_rules,
        "infer": _REAL_INFER_CONFIG,
        "source": _REAL_SOURCE,
    }


def _build(paths: dict[str, Path], **overrides: Any) -> dict[str, Any]:
    kwargs = {
        "registry_path": paths["registry"],
        "owner_context_ledger_path": paths["ledger"],
        "landscape_receipt_path": paths["landscape"],
        "landscape_admission_path": paths["landscape_admission"],
        "runtime_identity_path": paths["runtime"],
        "infer_config_path": paths["infer"],
        "source_jsonl_path": paths["source"],
        "suffix_horizon_rows": 1,
    }
    kwargs.update(overrides)
    return build_contract(**kwargs)


def _analyze_v2_landscape_for_behavior(
    tmp_path: Path, paths: dict[str, Path]
) -> tuple[Path, Path, Path]:
    rules_path = tmp_path / "rules.json"
    coordinate_tokens = list(range(1000, 2000))
    _json(
        rules_path,
        {
            "numeric_tolerance": 1e-5,
            "token_registry": {
                "coordinate_bin_to_token_id": {
                    "coordinate_bin_token_ids": coordinate_tokens
                }
            },
        },
    )
    rules_template_path = tmp_path / "analysis-rules-template.json"
    _json(rules_template_path, {"fixture": "analysis-rules-template"})
    registry_document = json.loads(paths["registry"].read_text(encoding="utf-8"))
    mechanism_content = {
        "schema_version": "sorted-fn-mechanism-decision-rules.v1",
        "unit_id": UNIT_ID,
        "geometry": {"iou_thresholds": [0.4, 0.5, 0.6]},
        "calibration": {
            "quantile_algorithm": "type7",
            "lower_quantile": 0.10,
            "upper_quantile": 0.90,
        },
        "populations": {"target": "fixture", "decoy": "fixture", "reference": "fixture"},
        "rung_quotas": {
            "scalar_smoke": {
                "claim_direction": "positive_only",
                "target_count": 30,
                "decoy_count": "equal_to_target",
                "reference_count": 7,
            },
            "L1": {
                "target_count": 256,
                "decoy_count": "equal_to_target",
                "reference_count": 65,
            },
        },
        "collision": {"statistics": {"F1": "fixture", "F2": "fixture", "F3": "fixture"}},
        "neighborhood": {
            "eligible_family": "near_gt_micro",
            "eligible_population": "target",
            "member_field": "candidate_neighborhood_member",
            "id_field": "candidate_neighborhood_id",
        },
        "exact_gt_singleton": {
            "member_field": "exact_gt_singleton_member",
            "id_field": "exact_gt_singleton_id",
            "id_rule": "exact-gt:<gt_owner_id>:<rung>",
            "eligible_family": "near_gt_micro",
            "eligible_population": "target",
        },
        "conditional_sampling": {
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "null_semantics": "finite_sampling_null_is_absence_neutral",
        },
        "upstream_digests": {
            "execution_landscape_decision_rules_sha256": sha256_file(rules_path),
            "fn_mechanism_registry_sha256": registry_document["registry_digest"],
            "rules_template_sha256": sha256_file(rules_template_path),
        },
    }
    mechanism_rules_path = tmp_path / "analysis-mechanism-decision-rules.json"
    _json(
        mechanism_rules_path,
        {**mechanism_content, "self_digest": sha256_json(mechanism_content)},
    )
    mechanism_rules_sha256 = sha256_file(mechanism_rules_path)
    fixed_path = tmp_path / "fixed-budget.jsonl"
    fixed_rows: list[dict[str, Any]] = []
    contexts = ["ctx:fn:first_skip:P_pre", "ctx:fn:first_skip:P_post"]
    for context_index, context_id in enumerate(contexts):
        for candidate_index, (population, family, box, iou) in enumerate(
            (
                ("target", "near_gt_micro", [1, 1, 2, 2], 1.0),
                ("target", "b", [1, 1, 2, 3], 0.5),
                ("decoy", "near_gt_micro", [500, 500, 510, 510], 0.0),
                ("decoy", "b", [520, 520, 528, 528], 0.0),
            )
        ):
            candidate_id = f"cand:{context_index}:{candidate_index}"
            fixed_rows.append(
                {
                    "schema_version": FIXED_BUDGET_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "coord_token_ids": [coordinate_tokens[value] for value in box],
                    "source_digest": sha256_json([context_id, candidate_id]),
                    "owner_context_id": context_id,
                    "rung": "L1",
                    "region": (
                        "target_strict" if population == "target" else "background"
                    ),
                    "population": population,
                    "is_control": context_index == 1,
                    "control_kind": (
                        "strict_positive" if context_index == 1 else None
                    ),
                    "matched_control_group": "person-small-crowded",
                    "description_size_crowding_stratum": "person-small-crowded",
                    "family_id": family,
                    "iou_to_target": iou,
                    "candidate_neighborhood_member": (
                        population == "target" and family == "near_gt_micro"
                    ),
                    "candidate_neighborhood_id": (
                        "nbh:gt:1:1:L1:near_gt_micro:0"
                        if population == "target" and family == "near_gt_micro"
                        else None
                    ),
                    "exact_gt_singleton_member": (
                        population == "target" and family == "near_gt_micro"
                    ),
                    "exact_gt_singleton_id": (
                        "exact-gt:gt:1:1:L1"
                        if population == "target" and family == "near_gt_micro"
                        else None
                    ),
                    "mechanism_decision_rules_sha256": mechanism_rules_sha256,
                }
            )
    _jsonl(fixed_path, fixed_rows)
    planner_path = tmp_path / "planner.json"
    planner_content = {
        "schema_version": PLANNER_RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "sources": {
            "registry": _ref(paths["registry"]),
            "rules_template": _ref(rules_template_path),
        },
        "outputs": {
            "owner_context_ledger": {**_ref(paths["ledger"]), "row_count": 2},
            "landscape_decision_rules": _ref(rules_path),
            "mechanism_decision_rules": _ref(mechanism_rules_path),
            "fixed_budget_candidates": {
                **_ref(fixed_path),
                "row_count": len(fixed_rows),
            },
        },
        "neighborhood_consistency": {
            "multi_context_groups_checked": 0,
            "status": "consistent",
        },
    }
    _json(
        planner_path,
        {**planner_content, "receipt_digest": sha256_json(planner_content)},
    )
    scores_path = tmp_path / "merged-scores.jsonl"
    score_rows = [
        {
            "schema_version": "sorted_fn_fixed_budget_scores.v1",
            "unit_id": UNIT_ID,
            "candidate_id": row["candidate_id"],
            "context_id": row["owner_context_id"],
            "rung": row["rung"],
            "region": row["region"],
            "population": row["population"],
            "candidate_neighborhood_member": row[
                "candidate_neighborhood_member"
            ],
            "candidate_neighborhood_id": row["candidate_neighborhood_id"],
            "exact_gt_singleton_member": row["exact_gt_singleton_member"],
            "exact_gt_singleton_id": row["exact_gt_singleton_id"],
            "mechanism_decision_rules_sha256": mechanism_rules_sha256,
            "native_repetition_penalty_stratum": 1.0,
            "raw_model_logprob": {
                "complete_box_logprob_sum": (
                    -1.0
                    if row["population"] == "target" and row["family_id"] == "a"
                    else -1.2
                    if row["population"] == "target"
                    else -3.0
                    if row["family_id"] == "a"
                    else -4.0
                )
            },
        }
        for row in fixed_rows
    ]
    _jsonl(scores_path, score_rows)
    shard_receipt_path = tmp_path / "score-shard-receipt.json"
    _json(
        shard_receipt_path,
        {
            "likelihood_channels": {"raw": "unmodified fp32 lm-head channel"},
            "scoring_backend_admission": {
                "selected_backend": "full_reforward_fp32",
                "cache_enabled": False,
                "use_cache": False,
                "atol": 1e-5,
                "rtol": 1e-5,
                "batched_reforward_admission": {
                    "status": "passed",
                    "requested_batch_size": 1,
                    "effective_batch_size": 1,
                },
            },
        },
    )
    merge_path = tmp_path / "merge-v2.json"
    merge = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "generic_arbitrary_role_merger": True,
        "decision_channel": {
            "name": LANDSCAPE_DECISION_CHANNEL,
            "primary_repetition_penalty_stratum": 1.0,
            "auxiliary_policy_is_not_a_model_likelihood": True,
        },
        "successor_scorer_provenance": {
            "row_schema_version": "sorted_fn_fixed_budget_scores.v1",
            "receipt_schema_version": "sorted_fn_fixed_budget_scores_receipt.v1",
            "unit_id": UNIT_ID,
        },
        "imported_predecessor_primitive_provenance": {
            "predecessor_primitives_file": PREDECESSOR_PRIMITIVES_FILE,
            "predecessor_primitives_file_sha256": "a" * 64,
        },
        "source_digests": {
            "fn_mechanism_registry": {
                **_ref(paths["registry"]),
                "registry_digest": registry_document["registry_digest"],
            },
            "owner_context_ledger": _ref(paths["ledger"]),
            "fixed_budget_candidates": _ref(fixed_path),
            "decision_rules": _ref(rules_path),
            "mechanism_decision_rules": {
                **_ref(mechanism_rules_path),
                "self_digest": json.loads(
                    mechanism_rules_path.read_text(encoding="utf-8")
                )["self_digest"],
                "parent_execution_rules_sha256": sha256_file(rules_path),
            },
        },
        "planner_receipt": {
            **_ref(planner_path),
            "receipt_digest": json.loads(planner_path.read_text())["receipt_digest"],
        },
        "context_selection": {"selected_context_ids": contexts},
        "selected_rungs": ["L1"],
        "identity_projection_sha256": "b" * 64,
        "shards": [{"receipt": _ref(shard_receipt_path)}],
        "output_artifacts": {
            "merged_scores": {**_ref(scores_path), "row_count": len(score_rows)}
        },
    }
    _json(merge_path, merge)
    attestation_dir = tmp_path / "attestation"
    attest(
        merge_receipt_path=merge_path,
        decision_rules_path=rules_path,
        mechanism_decision_rules_path=mechanism_rules_path,
        run_mode="scale",
        output_dir=attestation_dir,
    )
    analysis_dir = tmp_path / "analysis"
    analyze_sorted_fn_mechanisms(
        fn_mechanism_registry=paths["registry"],
        planner_receipt=planner_path,
        fixed_budget_candidates=fixed_path,
        merged_scores=scores_path,
        run_attestation=attestation_dir / ATTESTATION_NAME,
        output_dir=analysis_dir,
    )
    return merge_path, analysis_dir / "behavior-landscape-admission.json", analysis_dir


def _fake_generator(calls: list[dict[str, Any]]):
    def generate(**kwargs: Any) -> dict[str, Any]:
        calls.append(deepcopy(kwargs))
        role_id = kwargs["role"]["role_id"]
        forced = kwargs["forced_row_prefix_token_ids"] is not None
        released_forced_suffix = (
            kwargs["row_index"] > 0
            and str(kwargs["arm_id"]).startswith("forced_description")
        )
        if released_forced_suffix and role_id == "first_skip:P_pre":
            strict, loose, token, desc = ["gt:1:1"], ["gt:1:1"], 121, "person"
        elif released_forced_suffix:
            strict, loose, token, desc = [], ["gt:1:1"], 122, "person"
        elif role_id == "first_skip:P_pre" and not forced:
            strict, loose, token, desc = ["gt:1:2"], ["gt:1:2"], 202, "person"
        elif role_id == "first_skip:P_pre":
            strict, loose, token, desc = ["gt:1:1"], ["gt:1:1"], 101, "person"
        elif not forced:
            strict, loose, token, desc = ["gt:1:9"], ["gt:1:9"], 909, "bus"
        else:
            strict, loose, token, desc = [], ["gt:1:1"], 111, "person"
        return {
            "status": "success",
            "row_stop": {"stop_reason": "complete_row"},
            "raw_generated_token_ids": [151646, token, 151649],
            "parsed_predictions": [{"description": desc, "bbox": [1, 2, 3, 4]}],
            "strict_matched_owner_ids": strict,
            "loose_matched_owner_ids": loose,
            "semantic_drift_evidence": (
                [{"observed_description": "bus", "target_gt_owner_id": "gt:1:1"}]
                if desc == "bus"
                else []
            ),
        }

    return generate


def test_exact_token_prefix_use_pre_post_asymmetry_and_arm_separation(tmp_path: Path) -> None:
    contract = _build(_fixture(tmp_path), repetition_penalties=[1.0, 1.1])
    calls: list[dict[str, Any]] = []
    document = execute_behavior_contract(contract, _fake_generator(calls))

    rp_primary = document["policy_views"][0]
    rp_policy = document["policy_views"][1]
    assert [view["policy_view_id"] for view in document["policy_views"]] == ["rp_1_00", "rp_1_10"]
    assert "raw likelihood remains unchanged" in rp_policy["likelihood_semantics"]
    pre, post = rp_primary["roles"]
    pre_prefix = contract["roles"][0]["ledger"]["self_prefix_token_ids"]
    post_prefix = contract["roles"][1]["ledger"]["self_prefix_token_ids"]
    assert pre["exact_self_prefix_token_ids_sha256"] == sha256_json(pre_prefix)
    assert post["exact_self_prefix_token_ids_sha256"] == sha256_json(post_prefix)

    primary_calls = calls[:4]
    assert [call["prefix_token_ids"] for call in primary_calls] == [
        pre_prefix,
        pre_prefix,
        post_prefix,
        post_prefix,
    ]
    assert primary_calls[0]["forced_row_prefix_token_ids"] is None
    assert primary_calls[1]["forced_row_prefix_token_ids"] == [151646, 77, 151647, 151648]
    assert pre["arms"]["free_next_row"]["arm_kind"] == "free_next_row"
    assert pre["arms"]["forced_description_greedy"]["arm_kind"] == "forced_description"
    assert pre["comparison"]["free_vs_forced_first_divergence"]["index"] == 1

    row = pre["arms"]["free_next_row"]["rows"][0]
    assert row["pred_row_id"].startswith("pred:sorted:successor:")
    assert row["description_box_evidence"] == [
        {"prediction_index": 0, "description": "person", "bbox": [1, 2, 3, 4]}
    ]
    assert row["raw_generated_token_ids"] == [151646, 202, 151649]


def test_v2_attestor_analyzer_admission_builds_behavior_contract_cpu_only(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    merge_path, admission_path, _analysis_dir = _analyze_v2_landscape_for_behavior(
        tmp_path, paths
    )
    paths["landscape"] = merge_path
    paths["landscape_admission"] = admission_path

    contract = _build(paths)

    assert contract["schema_version"] == CONTRACT_RECEIPT_SCHEMA_VERSION
    assert MERGE_SCHEMA_VERSION == LANDSCAPE_RECEIPT_SCHEMA_VERSION
    assert contract["landscape"]["schema_version"] == (
        "sorted_fn_successor_score_shard_merge.v2"
    )
    assert contract["landscape"]["decision_bearing_channel"] == (
        "raw_model_logprob.complete_box_logprob_sum"
    )
    assert contract["landscape_admission"]["admitted_role_ids"] == [
        "first_skip:P_post",
        "first_skip:P_pre",
    ]
    assert contract["static_live_config_binding"]["status"] == (
        "passed_cpu_before_live_load"
    )


def test_downstream_owner_accounting_target_recovery_exchange_and_successor(tmp_path: Path) -> None:
    document = execute_behavior_contract(
        _build(_fixture(tmp_path), suffix_horizon_rows=2), _fake_generator([])
    )
    pre, post = document["policy_views"][0]["roles"]
    pre_accounting = pre["comparison"]["downstream_owner_accounting"]
    assert pre_accounting["gained_owner_ids"] == ["gt:1:1"]
    assert pre_accounting["lost_owner_ids"] == ["gt:1:2"]
    assert pre_accounting["target_recovery_exchange"] is True
    assert pre["comparison"]["successor"]["lost"] is True
    assert pre["comparison"]["intervention_target_recovered_strict"] is True
    assert pre["comparison"]["autonomous_suffix_target_recovered_strict"] is True
    assert pre["comparison"]["target_recovered_strict"] is True
    assert pre["arms"]["forced_description_greedy"][
        "autonomous_suffix_unique_strict_owner_ids"
    ] == ["gt:1:1"]

    # P_post already contains D in its exact prefix.  Loose target recovery
    # therefore retains D even though the free next row differs.
    assert post["comparison"]["target_recovered_loose"] is True
    assert post["comparison"]["target_recovered_strict"] is False
    assert post["comparison"]["successor"]["retained"] is True
    assert post["arms"]["free_next_row"]["semantic_drift_evidence"]


def test_forced_intervention_row_is_conditioned_recovery_not_autonomous_suffix(
    tmp_path: Path,
) -> None:
    document = execute_behavior_contract(_build(_fixture(tmp_path)), _fake_generator([]))
    role = document["policy_views"][0]["roles"][0]
    forced = role["arms"]["forced_description_greedy"]
    assert forced["intervention"]["target_realized_strict"] is True
    assert forced["intervention"]["accounting_status"] == (
        "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set"
    )
    assert forced["released_suffix_rows"] == []
    assert forced["intervention_target_recovered_strict"] is True
    assert forced["autonomous_suffix_target_recovered_strict"] is False
    assert forced["target_recovered_strict"] is True
    assert forced["autonomous_suffix_unique_strict_owner_ids"] == []
    assert forced["final_intervention_conditioned_unique_strict_owner_ids"] == [
        "gt:1:1",
        "gt:1:9",
    ]
    assert role["comparison"]["downstream_owner_accounting"]["gained_owner_ids"] == [
        "gt:1:1"
    ]


def _minimal_sampling_admission(
    registry_digest: str, landscape_sha: str
) -> dict[str, Any]:
    return {
        "schema_version": SAMPLING_ADMISSION_SCHEMA_VERSION,
        "status": "passed",
        "registry_digest": registry_digest,
        "landscape_receipt_sha256": landscape_sha,
        "landscape_condition": {
            "name": "usable_target_strict_region_peak",
            "status": "passed",
        },
        "admitted_role_ids": ["first_skip:P_pre"],
        "decode_parameters": {
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "k": 2,
            "seeds": [17, 23],
            "horizon_rows": 3,
        },
    }


def _sampling_admission(
    tmp_path: Path,
    *,
    registry_digest: str,
    landscape_sha: str,
    mechanism_rules_path: Path,
    recovered: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = "gt:1:1"
    intervention_row = {
        "pred_row_id": "prior:intervention",
        "strict_matched_owner_ids": [],
        "loose_matched_owner_ids": [],
    }
    released_rows = (
        [
            {
                "pred_row_id": "prior:released:0",
                "strict_matched_owner_ids": [target],
                "loose_matched_owner_ids": [target],
            }
        ]
        if recovered
        else []
    )
    behavior_role = {
        "role_id": "first_skip:P_pre",
        "gt_owner_id": target,
        "arms": {
            "forced_description_greedy": {
                "arm_kind": "forced_description",
                "rows": [intervention_row, *released_rows],
                "released_suffix_rows": released_rows,
                "autonomous_evidence_row_ids": [
                    row["pred_row_id"] for row in released_rows
                ],
                "intervention": {
                    "row": intervention_row,
                    "accounting_status": "excluded_from_autonomous_suffix_but_included_in_final_conditioned_set",
                },
                "autonomous_suffix_target_recovered_strict": recovered,
                "autonomous_suffix_target_recovered_loose": recovered,
            }
        },
    }
    behavior_content = {
        "schema_version": "sorted-fn-successor-behavior.v1",
        "unit_id": UNIT_ID,
        "contract": {
            "registry": {"registry_digest": registry_digest},
            "landscape": {"sha256": landscape_sha},
            "sampling_admission": None,
        },
        "policy_views": [
            {
                "repetition_penalty": 1.0,
                "roles": [behavior_role],
            }
        ],
    }
    behavior_document = {
        **behavior_content,
        "output_content_sha256": sha256_json(behavior_content),
    }
    behavior_path = tmp_path / "prior-greedy-behavior.json"
    _json(behavior_path, behavior_document)
    mechanism_document = json.loads(mechanism_rules_path.read_text())
    conditional_sampling = mechanism_document["conditional_sampling"]
    mechanism_binding = {
        "path": str(mechanism_rules_path.resolve()),
        "sha256": sha256_file(mechanism_rules_path),
        "self_digest": mechanism_document["self_digest"],
    }
    admission = {
        "schema_version": SAMPLING_ADMISSION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "passed",
        "registry_digest": registry_digest,
        "landscape_receipt_sha256": landscape_sha,
        "mechanism_decision_rules": {
            **mechanism_binding,
            "conditional_sampling": {
                key: conditional_sampling[key]
                for key in (
                    "temperature",
                    "top_p",
                    "repetition_penalty",
                    "null_semantics",
                )
            },
        },
        "behavior_output": {
            "path": str(behavior_path.resolve()),
            "sha256": sha256_file(behavior_path),
            "output_content_sha256": behavior_document[
                "output_content_sha256"
            ],
        },
        "landscape_condition": {
            "name": "per_role_localized_landscape_with_greedy_failure",
            "status": "passed",
            "greedy_canonical_description_recovery_failed": True,
            "condition_by_role": {
                "first_skip:P_pre": {
                    "role_id": "first_skip:P_pre",
                    "context_id": "ctx:fn:first_skip:P_pre",
                    "gt_owner_id": target,
                    "landscape_condition": {
                        "name": "usable_target_strict_region_peak",
                        "status": "passed",
                        "rung": "L1",
                    },
                    "greedy_canonical_description_recovery": {
                        "status": "failed",
                        "arm": "forced_description_greedy",
                        "target_recovered": False,
                        "behavior_role_content_sha256": sha256_json(
                            behavior_role
                        ),
                    },
                }
            },
        },
        "admitted_role_ids": ["first_skip:P_pre"],
        "decode_parameters": {
            "temperature": conditional_sampling["temperature"],
            "top_p": conditional_sampling["top_p"],
            "repetition_penalty": conditional_sampling["repetition_penalty"],
            "k": 2,
            "seeds": [17, 23],
            "horizon_rows": 3,
        },
        "finite_sampling_failure_semantics": conditional_sampling[
            "null_semantics"
        ],
    }
    return admission, mechanism_binding, behavior_document


def test_low_temperature_admission_rejection_and_acceptance(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text())
    registry_digest = registry["registry_digest"]
    landscape_sha = sha256_file(paths["landscape"])
    selected_roles = [
        {"role_id": "first_skip:P_pre", "gt_owner_id": "gt:1:1"}
    ]
    mechanism_document = json.loads(paths["mechanism_rules"].read_text())
    mechanism_binding = {
        "path": str(paths["mechanism_rules"].resolve()),
        "sha256": sha256_file(paths["mechanism_rules"]),
        "self_digest": mechanism_document["self_digest"],
    }
    with pytest.raises(BehaviorContractError, match="sampling flags require"):
        validate_sampling_admission(
            None,
            sampling_flags={"temperature": 0.4},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=mechanism_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    with pytest.raises(BehaviorContractError):
        validate_sampling_admission(
            _minimal_sampling_admission(registry_digest, landscape_sha),
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=mechanism_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    admission, mechanism_binding, _ = _sampling_admission(
        tmp_path,
        registry_digest=registry_digest,
        landscape_sha=landscape_sha,
        mechanism_rules_path=paths["mechanism_rules"],
    )
    accepted = validate_sampling_admission(
        admission,
        sampling_flags={
            "temperature": 0.4,
            "top_p": 0.95,
            "k": 2,
            "seeds": [17, 23],
            "horizon_rows": 3,
        },
        registry_digest=registry_digest,
        landscape_receipt_sha256=landscape_sha,
        mechanism_decision_rules_binding=mechanism_binding,
        selected_roles=selected_roles,
        selected_rungs=["L1"],
    )
    assert accepted is not None
    assert accepted["decode_parameters"]["seeds"] == [17, 23]

    wrong_unit = deepcopy(admission)
    wrong_unit["unit_id"] = "wrong-unit"
    with pytest.raises(BehaviorContractError, match="schema/status"):
        validate_sampling_admission(
            wrong_unit,
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=mechanism_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    missing_behavior = deepcopy(admission)
    missing_behavior.pop("behavior_output")
    with pytest.raises(BehaviorContractError):
        validate_sampling_admission(
            missing_behavior,
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=mechanism_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    missing_role_failure = deepcopy(admission)
    del missing_role_failure["landscape_condition"]["condition_by_role"][
        "first_skip:P_pre"
    ]["greedy_canonical_description_recovery"]
    with pytest.raises(BehaviorContractError):
        validate_sampling_admission(
            missing_role_failure,
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=mechanism_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    wrong_owner = deepcopy(admission)
    wrong_owner["landscape_condition"]["condition_by_role"][
        "first_skip:P_pre"
    ]["gt_owner_id"] = "gt:1:999"
    with pytest.raises(BehaviorContractError, match="prior greedy-failure evidence"):
        validate_sampling_admission(
            wrong_owner,
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=mechanism_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    recovered_admission, recovered_binding, _ = _sampling_admission(
        tmp_path,
        registry_digest=registry_digest,
        landscape_sha=landscape_sha,
        mechanism_rules_path=paths["mechanism_rules"],
        recovered=True,
    )
    with pytest.raises(BehaviorContractError, match="prior greedy-failure evidence"):
        validate_sampling_admission(
            recovered_admission,
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=recovered_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )

    stale_admission, stale_binding, stale_behavior = _sampling_admission(
        tmp_path,
        registry_digest=registry_digest,
        landscape_sha=landscape_sha,
        mechanism_rules_path=paths["mechanism_rules"],
    )
    _json(
        Path(stale_admission["behavior_output"]["path"]),
        {**stale_behavior, "tampered": True},
    )
    with pytest.raises(BehaviorContractError, match="output digest mismatch"):
        validate_sampling_admission(
            stale_admission,
            sampling_flags={},
            registry_digest=registry_digest,
            landscape_receipt_sha256=landscape_sha,
            mechanism_decision_rules_binding=stale_binding,
            selected_roles=selected_roles,
            selected_rungs=["L1"],
        )


def test_sampling_execution_stays_in_own_arm_and_rp_primary(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text())
    admission_path = tmp_path / "admission.json"
    admission, _, _ = _sampling_admission(
        tmp_path,
        registry_digest=registry["registry_digest"],
        landscape_sha=sha256_file(paths["landscape"]),
        mechanism_rules_path=paths["mechanism_rules"],
    )
    _json(admission_path, admission)
    contract = _build(
        paths,
        repetition_penalties=[1.0, 1.1],
        sampling_admission_path=admission_path,
    )
    calls: list[dict[str, Any]] = []
    document = execute_behavior_contract(contract, _fake_generator(calls))
    primary_pre = document["policy_views"][0]["roles"][0]["arms"]
    policy_pre = document["policy_views"][1]["roles"][0]["arms"]
    assert [arm["producer"]["seed"] for arm in primary_pre["forced_description_low_temperature_samples"]] == [17, 23]
    assert policy_pre["forced_description_low_temperature_samples"] == []
    assert primary_pre["free_next_row"]["producer"]["mode"] == "greedy"
    assert primary_pre["forced_description_greedy"]["producer"]["mode"] == "greedy"


def test_create_or_identical_and_refuse_different(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    document = {"schema_version": "fixture", "value": 1}
    assert _write_create_or_identical(output, document)["status"] == "created"
    assert _write_create_or_identical(output, document)["status"] == "identical_existing"
    with pytest.raises(BehaviorContractError, match="different content"):
        _write_create_or_identical(output, {"schema_version": "fixture", "value": 2})


def test_registry_digest_mismatch_fails_before_runtime(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text())
    registry["registry_digest"] = "0" * 64
    _json(paths["registry"], registry)
    with pytest.raises(BehaviorContractError, match="digest mismatch"):
        validate_registry(paths["registry"])


def test_made_up_successor_identity_is_rejected_even_when_only_pre_selected(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text())
    post = next(
        role
        for role in registry["smoke"]["roles"]
        if role["role_id"] == "first_skip:P_post"
    )
    post["successor_gt_owner_id"] = "gt:1:999"
    content = {key: value for key, value in registry.items() if key != "registry_digest"}
    registry["registry_digest"] = sha256_json(content)
    _json(paths["registry"], registry)
    with pytest.raises(BehaviorContractError, match="successor owner"):
        validate_registry(paths["registry"], ["first_skip:P_pre"])


def test_landscape_gate_requires_each_selected_role(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    admission = json.loads(paths["landscape_admission"].read_text())
    admission["admitted_roles"] = [
        entry
        for entry in admission["admitted_roles"]
        if entry["role_id"] != "first_skip:P_pre"
    ]
    _json(paths["landscape_admission"], admission)
    with pytest.raises(BehaviorContractError, match="lack passed landscape admission"):
        _build(paths, include_role_ids=["first_skip:P_pre"])


def test_landscape_admission_rung_must_be_selected_by_merge(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    landscape = json.loads(paths["landscape"].read_text())
    landscape["selected_rungs"] = ["L0"]
    _json(paths["landscape"], landscape)
    admission = json.loads(paths["landscape_admission"].read_text())
    admission["landscape_receipt_sha256"] = sha256_file(paths["landscape"])
    _json(paths["landscape_admission"], admission)
    with pytest.raises(
        BehaviorContractError,
        match="owner-matched passed landscape condition",
    ):
        _build(paths, include_role_ids=["first_skip:P_pre"])


def test_stale_v1_landscape_receipt_is_rejected_before_runtime(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["landscape"].read_text(encoding="utf-8"))
    receipt["schema_version"] = "sorted_fn_successor_score_shard_merge.v1"
    _json(paths["landscape"], receipt)
    with pytest.raises(BehaviorContractError, match="schema_version"):
        _build(paths)


def test_landscape_missing_mechanism_rules_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["landscape"].read_text(encoding="utf-8"))
    receipt["source_digests"].pop("mechanism_decision_rules")
    _json(paths["landscape"], receipt)
    with pytest.raises(BehaviorContractError, match="mechanism-decision-rules"):
        _build(paths)


def _live_identity_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    model = {"family": "base-plus-adapter-plus-delta", "checkpoint": "fixture"}
    tokenizer = {"tokenizer_sha256": "a" * 64, "vocab_size": 152670}
    processor = {"processor_class": "FixtureProcessor"}
    runtime = {
        "backend": "hf",
        "backend_mode": "generate",
        "backend_version": "fixture-transformers",
        "effective_settings": {
            "backend_options": {
                "hf": {
                    "attn_implementation": "sdpa",
                    "patch_embed_linearization": "enabled",
                }
            },
            "batch_size": 1,
            "device": "cuda",
            "output_scores": True,
        },
        "generation_config_fingerprint": "b" * 64,
        "likelihood_semantics": {
            "policy": "fixture-policy",
            "raw": "fixture-raw",
            "score_owned_channel": "policy_logprob",
        },
        "processor_identity": processor,
        "response_family": "hf",
    }
    observed = {
        **runtime,
        "effective_settings": {
            **runtime["effective_settings"],
            "observed_attn_implementation": "sdpa",
            "observed_model_dtype": {
                "parameter_dtype_counts": {"torch.float32": 17},
                "parameter_dtype_names": ["torch.float32"],
            },
            "performance": {"load_seconds": 1.25},
        },
        "model_identity": model,
        "tokenizer_identity": tokenizer,
    }
    frozen = {
        "model": {
            "identity_source": {
                "model_identity": model,
                "model_identity_fingerprint": sha256_json(model),
            }
        },
        "tokenizer": {"identity_source": {"tokenizer_identity": tokenizer}},
        "runtime": {
            "identity_source": {
                **runtime,
                "precision": "float32",
                "processor_identity_fingerprint": sha256_json(processor),
                "resolved_config_fingerprints": {"infer_config": "c" * 64},
            }
        },
    }
    return observed, frozen


def test_live_identity_exact_match_and_mismatches_fail_closed() -> None:
    observed, frozen = _live_identity_fixture()
    receipt = validate_live_runtime_identity(
        observed_receipt=observed,
        frozen_identity=frozen,
        resolved_config_fingerprint="c" * 64,
        generation_config_fingerprint="b" * 64,
    )
    assert receipt["status"] == "passed_before_generation"
    mismatched = deepcopy(observed)
    mismatched["tokenizer_identity"] = {"tokenizer_sha256": "f" * 64}
    with pytest.raises(BehaviorContractError, match="tokenizer_identity"):
        validate_live_runtime_identity(
            observed_receipt=mismatched,
            frozen_identity=frozen,
            resolved_config_fingerprint="c" * 64,
            generation_config_fingerprint="b" * 64,
        )
    with pytest.raises(BehaviorContractError, match="resolved_config_fingerprint"):
        validate_live_runtime_identity(
            observed_receipt=observed,
            frozen_identity=frozen,
            resolved_config_fingerprint="d" * 64,
            generation_config_fingerprint="b" * 64,
        )


def test_live_identity_postload_observations_fail_closed() -> None:
    observed, frozen = _live_identity_fixture()

    wrong_attn = deepcopy(observed)
    wrong_attn["effective_settings"]["observed_attn_implementation"] = "eager"
    with pytest.raises(BehaviorContractError, match="observed_attn_implementation"):
        validate_live_runtime_identity(
            observed_receipt=wrong_attn,
            frozen_identity=frozen,
            resolved_config_fingerprint="c" * 64,
            generation_config_fingerprint="b" * 64,
        )

    for names, counts in (
        (
            ["torch.float32", "torch.bfloat16"],
            {"torch.float32": 16, "torch.bfloat16": 1},
        ),
        (["torch.bfloat16"], {"torch.bfloat16": 17}),
        (["torch.float32"], {"torch.float32": 0}),
    ):
        wrong_dtype = deepcopy(observed)
        wrong_dtype["effective_settings"]["observed_model_dtype"] = {
            "parameter_dtype_counts": counts,
            "parameter_dtype_names": names,
        }
        with pytest.raises(BehaviorContractError, match="observed_model_dtype"):
            validate_live_runtime_identity(
                observed_receipt=wrong_dtype,
                frozen_identity=frozen,
                resolved_config_fingerprint="c" * 64,
                generation_config_fingerprint="b" * 64,
            )

    malformed_dtype = deepcopy(observed)
    malformed_dtype["effective_settings"]["observed_model_dtype"]["unexpected"] = True
    with pytest.raises(BehaviorContractError, match="observed_model_dtype"):
        validate_live_runtime_identity(
            observed_receipt=malformed_dtype,
            frozen_identity=frozen,
            resolved_config_fingerprint="c" * 64,
            generation_config_fingerprint="b" * 64,
        )

    unknown_extra = deepcopy(observed)
    unknown_extra["effective_settings"]["unrecognized_postload_field"] = True
    with pytest.raises(BehaviorContractError, match="effective_settings_extra_fields"):
        validate_live_runtime_identity(
            observed_receipt=unknown_extra,
            frozen_identity=frozen,
            resolved_config_fingerprint="c" * 64,
            generation_config_fingerprint="b" * 64,
        )

    config_drift = deepcopy(observed)
    config_drift["effective_settings"]["batch_size"] = 2
    with pytest.raises(BehaviorContractError, match="runtime_identity"):
        validate_live_runtime_identity(
            observed_receipt=config_drift,
            frozen_identity=frozen,
            resolved_config_fingerprint="c" * 64,
            generation_config_fingerprint="b" * 64,
        )


def test_semantic_drift_uses_frozen_alias_equivalence_and_high_overlap() -> None:
    equivalence = {
        "schema_version": DESCRIPTION_EQUIVALENCE_SCHEMA_VERSION,
        "alias_to_category": {
            "person": "person",
            "man": "person",
            "woman": "person",
            "bus": "bus",
        },
    }
    alias = classify_semantic_relation(
        observed_description="man",
        canonical_description="person",
        iou_to_target=0.9,
        equivalence=equivalence,
    )
    assert alias["relation"] == "equivalent"
    assert alias["semantic_drift_supported"] is False
    mismatch = classify_semantic_relation(
        observed_description="bus",
        canonical_description="person",
        iou_to_target=0.9,
        equivalence=equivalence,
    )
    assert mismatch["relation"] == "semantic_mismatch"
    assert mismatch["semantic_drift_supported"] is True
    low_overlap = classify_semantic_relation(
        observed_description="bus",
        canonical_description="person",
        iou_to_target=0.2,
        equivalence=equivalence,
    )
    assert low_overlap["semantic_drift_supported"] is False
    neutral = classify_semantic_relation(
        observed_description="bus",
        canonical_description="person",
        iou_to_target=0.9,
        equivalence=None,
    )
    assert neutral["relation"].startswith("unknown_neutral")


def test_owner_pixel_bbox_converts_to_norm1000_and_strict_matches_target_owner() -> None:
    """Live evidence: owner gt:7511:17 bbox_xyxy is decoded-pixel space.

    Previously this pixel box was copied verbatim into the entity ledger's
    ``bbox_norm1000`` field, so the matcher compared a norm1000-scaled
    prediction against an unconverted pixel target and recorded IoU 0.0 for
    every candidate.  This proves the fixed conversion recovers the real
    strict match instead.
    """

    decoded_width, decoded_height = 1152, 864
    owner_bbox_norm1000 = _owner_pixel_bbox_to_entity_norm1000(
        [607, 492, 619, 508],
        decoded_width=decoded_width,
        decoded_height=decoded_height,
        label="owner 'gt:7511:17' image '7511'",
    )
    assert owner_bbox_norm1000 == pytest.approx(
        [526.9097222, 569.4444444, 537.3263889, 587.962963], rel=1e-6
    )
    entities = [
        {
            "entity_id": "gt:7511:17",
            "description": "person",
            "bbox_norm1000": owner_bbox_norm1000,
        }
    ]
    # Real generated coord bins [525, 569, 536, 591] decoded to this pixel box.
    predictions = [{"description": "person", "bbox_xyxy": [605, 492, 617, 511]}]
    matches = match_predictions_to_entities(
        predictions,
        entities,
        image_width=decoded_width,
        image_height=decoded_height,
        restrict_to_person=False,
    )
    assert matches[0]["status"] == "matched"
    assert matches[0]["matched_entity_id"] == "gt:7511:17"
    assert matches[0]["candidates"][0]["iou"] == pytest.approx(0.6153846, rel=1e-6)


def test_owner_pixel_bbox_conversion_preserves_ambiguity_between_close_owners() -> None:
    """Two nearly-identical owner boxes stay ambiguous after the fix.

    Guards against a per-image conversion that only happens to work for a
    single isolated owner while breaking the ambiguity margin used to avoid
    over-claiming a matched owner.
    """

    decoded_width, decoded_height = 1152, 864
    entities = [
        {
            "entity_id": "gt:7511:17",
            "description": "person",
            "bbox_norm1000": _owner_pixel_bbox_to_entity_norm1000(
                [607, 492, 619, 508],
                decoded_width=decoded_width,
                decoded_height=decoded_height,
                label="owner 'gt:7511:17' image '7511'",
            ),
        },
        {
            "entity_id": "gt:7511:99",
            "description": "person",
            "bbox_norm1000": _owner_pixel_bbox_to_entity_norm1000(
                [605, 494, 619, 507],
                decoded_width=decoded_width,
                decoded_height=decoded_height,
                label="owner 'gt:7511:99' image '7511'",
            ),
        },
    ]
    predictions = [{"description": "person", "bbox_xyxy": [605, 492, 617, 511]}]
    matches = match_predictions_to_entities(
        predictions,
        entities,
        image_width=decoded_width,
        image_height=decoded_height,
        restrict_to_person=False,
    )
    assert matches[0]["status"] == "ambiguous"
    assert matches[0]["matched_entity_id"] is None


def test_owner_pixel_bbox_domain_confusion_fails_closed() -> None:
    with pytest.raises(BehaviorContractError, match="outside the decoded"):
        _owner_pixel_bbox_to_entity_norm1000(
            [607, 492, 619, 508],
            decoded_width=1152,
            decoded_height=480,
            label="owner 'gt:7511:17' image '7511'",
        )
    with pytest.raises(BehaviorContractError, match="outside the decoded"):
        _owner_pixel_bbox_to_entity_norm1000(
            [607, 492, 619, 508],
            decoded_width=600,
            decoded_height=864,
            label="owner 'gt:7511:17' image '7511'",
        )
    with pytest.raises(BehaviorContractError, match="four values"):
        _owner_pixel_bbox_to_entity_norm1000(
            [607, 492, 619],
            decoded_width=1152,
            decoded_height=864,
            label="owner 'gt:7511:17' image '7511'",
        )
    with pytest.raises(BehaviorContractError, match="numeric"):
        _owner_pixel_bbox_to_entity_norm1000(
            [607, 492, 619, "not-a-number"],
            decoded_width=1152,
            decoded_height=864,
            label="owner 'gt:7511:17' image '7511'",
        )


def test_code_provenance_binds_dirty_content_and_backend_parser_helpers() -> None:
    receipt = _git_code_provenance()
    assert len(receipt["dirty_diff_content_sha256"]) == 64
    assert "src/inference/backend.py" in receipt["helper_sha256"]
    assert "src/inference/hf_backend.py" in receipt["helper_sha256"]
    assert "src/inference/parsing.py" in receipt["helper_sha256"]
    assert receipt["helper_sha256"]["src/inference/parsing.py"] == sha256_file(
        Path("src/inference/parsing.py").resolve()
    )


def test_default_contract_is_greedy_only_and_role_filter_is_exact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    contract = _build(paths, include_role_ids=["first_skip:P_pre"])
    assert contract["schema_version"] == CONTRACT_RECEIPT_SCHEMA_VERSION
    assert contract["selected_role_ids"] == ["first_skip:P_pre"]
    assert contract["roles"][0]["successor_gt_owner_id"] == "gt:1:2"
    assert contract["sampling_admission"] is None
    assert contract["policy_repetition_penalties"] == [1.0]
    with pytest.raises(BehaviorContractError, match="unknown registered role"):
        _build(paths, include_role_ids=["made-up-role"])


def test_contract_only_rejects_incomplete_infer_config(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    incomplete = tmp_path / "incomplete-infer.yaml"
    incomplete.write_text("backend:\n  type: hf\n", encoding="utf-8")
    with pytest.raises(BehaviorContractError, match="infer config is incomplete or invalid"):
        _build(paths, infer_config_path=incomplete)


@pytest.mark.parametrize("mutation", ["identity_source", "component_sha256"])
def test_contract_only_rejects_mutated_runtime_identity_internals(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _fixture(tmp_path)
    runtime = json.loads(paths["runtime"].read_text(encoding="utf-8"))
    model_source = runtime["model"]["identity_source"]
    if mutation == "identity_source":
        model_source["model_identity_fingerprint"] = "f" * 64
    else:
        model_source["component_files"][0]["sha256"] = "f" * 64
    runtime["model"]["identity_sha256"] = sha256_json(model_source)
    content = {key: value for key, value in runtime.items() if key != "receipt_digest"}
    runtime["receipt_digest"] = sha256_json(content)
    mutated = tmp_path / f"runtime-{mutation}.json"
    _json(mutated, runtime)
    with pytest.raises(
        BehaviorContractError,
        match="differs from exact source recomputation",
    ):
        _build(paths, runtime_identity_path=mutated)


_REAL_REGISTRY = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-02-sorted-fn-mechanism-decomposition-smoke-v1/input/"
    "fn-mechanism-registry.json"
)


@pytest.mark.skipif(not _REAL_REGISTRY.is_file(), reason="real frozen registry unavailable")
def test_current_real_registry_accepts_exact_first_skip_lineage_cpu_only() -> None:
    _registry, roles, receipt = validate_registry(
        _REAL_REGISTRY, ["first_skip:P_pre"]
    )
    assert [role["role_id"] for role in roles] == ["first_skip:P_pre"]
    assert receipt["registry_digest"]
    assert receipt["first_skip_successor_binding"]["gt_owner_id"] == "gt:7511:2"
