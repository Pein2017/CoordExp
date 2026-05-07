from __future__ import annotations

import runpy
from pathlib import Path

from src.trainers.stage1_set_continuation.metrics import (
    EMITTED_STAGE1_SET_CONTINUATION_METRICS,
    STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION,
    numeric_metric_payload,
)


_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_stage1_set_continuation_trainer_smoke.py"))
)

PHASE1_SETCONT_RMP_V3_KEYS = {
    "setcont/rmp/branch_node_count",
    "setcont/rmp/valid_child_mass_mean",
    "setcont/rmp/valid_child_mass_p10",
    "setcont/rmp/invalid_child_mass_mean",
    "setcont/rmp/valid_invalid_margin_mean",
    "setcont/rmp/top1_invalid_rate",
    "setcont/rmp/top1_valid_not_teacher_rate",
    "setcont/rmp/positive_child_rank_mean",
    "setcont/rmp/teacher_path_child_prob_mean",
    "setcont/rmp/teacher_path_child_rank_mean",
    "setcont/rmp/valid_child_effective_count_mean",
    "setcont/rmp/effective_count_node_count",
    "setcont/rmp/balance_node_count",
    "setcont/rmp/balance_kl_mean",
    "setcont/rmp/support_loss_desc_text_mean",
    "setcont/rmp/support_loss_coord_mean",
    "setcont/rmp/support_loss_structural_mean",
    "setcont/rmp/support_loss_other_mean",
    "setcont/rmp/balance_kl_desc_text_mean",
    "setcont/rmp/balance_kl_coord_mean",
    "setcont/rmp/balance_kl_structural_mean",
    "setcont/rmp/balance_kl_other_mean",
    "setcont/rmp/desc_text_branch_node_count",
    "setcont/rmp/coord_branch_node_count",
    "setcont/rmp/structural_branch_node_count",
    "setcont/rmp/other_branch_node_count",
    "setcont/rmp/desc_text_balance_node_count",
    "setcont/rmp/coord_balance_node_count",
    "setcont/rmp/structural_balance_node_count",
    "setcont/rmp/other_balance_node_count",
}


def test_set_continuation_emits_train_forward_runtime_metric_keys() -> None:
    cfg = _HELPERS["_cfg"](
        bidirectional_token_gate={
            "enabled": True,
            "coord_gate_weight": 0.5,
            "text_gate_weight": 0.1,
        },
        train_forward={
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_candidates": 2},
                "fallback": {
                    "mode": "approximate_uniform_subsample",
                    "max_candidates": 1,
                    "estimator": "uniform_importance",
                },
            }
        },
    )
    trainer = _HELPERS["_trainer"](cfg)
    model = _HELPERS["_FakeModel"]()
    trainer.model = model

    trainer.compute_loss(
        model,
        _HELPERS["_batch"](
            [
                _HELPERS["OBJECT_A"],
                _HELPERS["OBJECT_B"],
                _HELPERS["OBJECT_C"],
            ]
        ),
        return_outputs=False,
    )

    keys = set(trainer.custom_metrics["train"].keys())
    assert {
        "loss/candidate_balanced",
        "loss/coord_gate",
        "loss/schema_open",
        "loss/text_gate",
        "loss/json_structural",
        "loss/anti_close_start",
        "loss/weak_schema_close",
        "gate/coord_slot_coord_mass_mean",
        "gate/text_slot_coord_mass_mean",
        "gate/coord_tokens_count",
        "gate/text_tokens_count",
        "mp/num_prefix_objects",
        "mp/num_remaining_objects",
        "mp/num_candidates_scored",
        "mp/candidate_tokens_scored_mean",
        "mp/schema_open_tokens_scored_mean",
        "mp/json_structural_tokens_scored_mean",
        "mp/annotation_completeness_weight_mean",
        "mp/final_close_weight_mean",
        "mp/tail_positive_samples",
        "mp/final_gt_object_scored_samples",
        "mp/objective_fidelity_exact_samples",
        "mp/fallback_applied_samples",
        "mp/selected_mode_empty_prefix",
        "mp/selected_mode_full_prefix",
        "mp/objective_contributing_samples",
        "stop/p_close_start_when_remaining_exists",
        "stop/p_continue_start_when_remaining_exists",
    }.issubset(keys)
    assert {
        "mp/branch_runtime_mode",
        "mp/checkpointed_branch_forwards",
        "mp/retained_graph_branch_forwards",
        "mp/smart_batched_branch_forwards",
        "mp/branch_batch_count",
        "mp/logZ_estimator",
        "mp/repeated_forward_token_ratio_vs_baseline",
        "mp/prefix_encoding_cache_hits",
        "mp/configured_ratio_empty_prefix",
        "stop/p_stop_when_remaining_exists",
    }.isdisjoint(keys)


def test_set_continuation_emits_compact_entry_trie_rmp_metric_keys() -> None:
    assert {
        "loss/rmp",
        "loss/rmp_branch_support",
        "loss/rmp_branch_balance",
        "loss/rmp_branch_total",
        "loss/rmp_branch_ce",
        "loss/rmp_unique_ce",
        "loss/rmp_coord_branch_ce",
        "loss/rmp_desc_text_branch_ce",
        "loss/rmp_boundary_ce",
        "loss/rmp_close_ce",
        "loss/rmp_eos_ce",
        "rmp/branch_nodes",
        "rmp/branch_nodes_desc_text",
        "rmp/branch_nodes_coord",
        "rmp/branch_nodes_structural",
        "rmp/branch_nodes_other",
        "rmp/valid_children_mean",
        "rmp/target_entropy_mean",
        "rmp/valid_child_mass_mean",
        "rmp/valid_child_mass_min",
        "rmp/valid_child_mass_p10",
        "rmp/valid_child_mass_p50",
        "rmp/valid_child_mass_p90",
        "rmp/valid_child_mass_desc_text",
        "rmp/valid_child_mass_coord",
        "rmp/valid_child_mass_structural",
        "rmp/valid_child_mass_other",
        "rmp/teacher_branch_top1_acc",
        "rmp/valid_child_top1_acc",
        "rmp/gt_count_ge7_samples",
    }.issubset(EMITTED_STAGE1_SET_CONTINUATION_METRICS)


def test_set_continuation_metric_schema_v3_allow_lists_phase1_rmp_keys() -> None:
    assert (
        STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION
        == "stage1_set_continuation_metrics_v3"
    )
    assert PHASE1_SETCONT_RMP_V3_KEYS.issubset(
        EMITTED_STAGE1_SET_CONTINUATION_METRICS
    )


def test_set_continuation_numeric_payload_preserves_v3_keys_without_internal_leaks() -> None:
    source = {
        key: float(index)
        for index, key in enumerate(sorted(PHASE1_SETCONT_RMP_V3_KEYS), start=1)
    }
    source.update(
        {
            "loss/rmp": 0.25,
            "rmp/valid_child_mass_mean": 0.75,
            "mp/num_candidates_scored": 3,
            "stop/p_close_start_when_remaining_exists": 0.1,
            "batch_loss": 9.0,
            "batch_size": 2,
            "setcont/rmp/private_scratch": 1.0,
            "setcont/rmp/non_numeric": "not-a-scalar",
            "setcont/rmp/not_finite": float("nan"),
        }
    )

    payload = numeric_metric_payload(source)

    assert PHASE1_SETCONT_RMP_V3_KEYS.issubset(payload)
    assert payload["loss/rmp"] == 0.25
    assert payload["rmp/valid_child_mass_mean"] == 0.75
    assert payload["mp/num_candidates_scored"] == 3.0
    assert payload["stop/p_close_start_when_remaining_exists"] == 0.1
    assert "batch_loss" not in payload
    assert "batch_size" not in payload
    assert "setcont/rmp/private_scratch" not in payload
    assert "setcont/rmp/non_numeric" not in payload
    assert "setcont/rmp/not_finite" not in payload
