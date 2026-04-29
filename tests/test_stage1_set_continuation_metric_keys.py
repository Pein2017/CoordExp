from __future__ import annotations

import runpy
from pathlib import Path

from src.trainers.stage1_set_continuation.metrics import (
    EMITTED_STAGE1_SET_CONTINUATION_METRICS,
)


_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_stage1_set_continuation_trainer_smoke.py"))
)


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
        "rmp/teacher_branch_top1_acc",
        "rmp/valid_child_top1_acc",
        "rmp/gt_count_ge7_samples",
    }.issubset(EMITTED_STAGE1_SET_CONTINUATION_METRICS)
