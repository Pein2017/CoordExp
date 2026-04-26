from __future__ import annotations

import pytest

from src.config.schema import PromptOverrides, TrainingConfig


def _payload() -> dict:
    return {
        "template": {"truncation_strategy": "raise"},
        "training": {"packing": False, "eval_packing": False},
        "custom": {
            "train_jsonl": "train.coord.jsonl",
            "user_prompt": "prompt",
            "emit_norm": "none",
            "json_format": "standard",
            "object_field_order": "desc_first",
            "trainer_variant": "stage1_set_continuation",
            "coord_tokens": {"enabled": True, "skip_bbox_norm": True},
            "stage1_set_continuation": {
                "subset_sampling": {
                    "empty_prefix_ratio": 1.0,
                    "random_subset_ratio": 0.0,
                    "leave_one_out_ratio": 0.0,
                    "full_prefix_ratio": 0.0,
                    "prefix_order": "dataset",
                },
                "candidates": {"mode": "exact"},
                "structural_close": {
                    "close_start_suppression_weight": 0.0,
                    "final_schema_close_weight": 0.0,
                },
                "positive_evidence_margin": {"objective": "disabled"},
            },
        },
    }


def test_train_forward_defaults_preserve_current_retained_graph_behavior() -> None:
    cfg = TrainingConfig.from_mapping(_payload(), PromptOverrides())

    train_forward = cfg.custom.stage1_set_continuation.train_forward
    assert train_forward.branch_runtime.mode == "retained_graph"
    assert train_forward.logits.mode == "full"
    assert train_forward.ddp_sync.candidate_padding == "max_count"
    assert train_forward.branch_runtime.checkpoint_use_reentrant is False
    assert train_forward.branch_runtime.preserve_rng_state is True
    assert train_forward.budget_policy.enabled is False
    assert train_forward.budget_policy.fallback.mode == "disabled"
    assert train_forward.prefix_reuse.encoding_cache is False
    assert train_forward.prefix_reuse.kv_cache.mode == "disabled"


def test_train_forward_accepts_checkpointed_exact_with_approx_fallback() -> None:
    payload = _payload()
    payload["training"]["ddp_broadcast_buffers"] = False
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {
            "mode": "checkpointed_exact",
            "checkpoint_use_reentrant": False,
            "preserve_rng_state": True,
        },
        "budget_policy": {
            "enabled": True,
            "exact_until": {
                "max_candidates": 8,
                "max_branch_tokens_per_sample": 24000,
                "min_free_memory_gib": 4.0,
            },
            "fallback": {
                "mode": "approximate_uniform_subsample",
                "max_candidates": 4,
                "estimator": "uniform_importance",
                "require_telemetry": True,
            },
        },
        "prefix_reuse": {
            "encoding_cache": True,
            "kv_cache": {"mode": "disabled"},
        },
        "logits": {"mode": "supervised_suffix"},
        "ddp_sync": {"candidate_padding": "none"},
        "telemetry": {
            "per_rank_memory": True,
            "branch_budget": True,
            "objective_fidelity": True,
        },
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    train_forward = cfg.custom.stage1_set_continuation.train_forward

    assert train_forward.branch_runtime.mode == "checkpointed_exact"
    assert train_forward.budget_policy.enabled is True
    assert train_forward.budget_policy.exact_until.max_candidates == 8
    assert train_forward.budget_policy.fallback.mode == "approximate_uniform_subsample"
    assert train_forward.budget_policy.fallback.max_candidates == 4
    assert train_forward.budget_policy.fallback.estimator == "uniform_importance"
    assert train_forward.prefix_reuse.encoding_cache is True
    assert train_forward.prefix_reuse.kv_cache.mode == "disabled"
    assert train_forward.logits.mode == "supervised_suffix"
    assert train_forward.ddp_sync.candidate_padding == "none"


def test_train_forward_accepts_smart_batched_exact_branch_runtime() -> None:
    payload = _payload()
    payload["training"]["ddp_broadcast_buffers"] = False
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {"mode": "smart_batched_exact"},
        "branch_batching": {
            "enabled": True,
            "strategy": "ms_swift_constant_volume_buckets",
            "max_branch_rows": 8,
            "max_branch_tokens": 65536,
            "min_fill_ratio": 0.70,
            "padding_waste_warn_fraction": 0.40,
        },
        "logits": {"mode": "supervised_suffix"},
        "ddp_sync": {"candidate_padding": "none"},
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    train_forward = cfg.custom.stage1_set_continuation.train_forward

    assert train_forward.branch_runtime.mode == "smart_batched_exact"
    assert train_forward.branch_batching.enabled is True
    assert train_forward.branch_batching.strategy == "ms_swift_constant_volume_buckets"
    assert train_forward.branch_batching.max_branch_rows == 8
    assert train_forward.branch_batching.max_branch_tokens == 65536
    assert train_forward.branch_batching.min_fill_ratio == pytest.approx(0.70)
    assert train_forward.branch_batching.padding_waste_warn_fraction == pytest.approx(
        0.40
    )


def test_train_forward_rejects_invalid_branch_batching_caps() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_batching": {"enabled": True, "max_branch_rows": 0}
    }

    with pytest.raises(ValueError, match="branch_batching.max_branch_rows"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_train_forward_rejects_unknown_nested_key() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {"mode": "checkpointed_exact", "mystery": 1}
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.stage1_set_continuation.train_forward.branch_runtime.mystery" in str(
        exc.value
    )


def test_budget_fallback_requires_positive_max_candidates() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "budget_policy": {
            "enabled": True,
            "fallback": {
                "mode": "approximate_uniform_subsample",
                "max_candidates": None,
            },
        }
    }

    with pytest.raises(ValueError, match="fallback.max_candidates"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_kv_cache_is_disabled_in_immediate_bridge() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "prefix_reuse": {"kv_cache": {"mode": "detached_no_grad"}}
    }

    with pytest.raises(ValueError, match="kv_cache.mode"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_train_forward_rejects_invalid_logits_mode() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "logits": {"mode": "prefix_cache"}
    }

    with pytest.raises(ValueError, match="train_forward.logits.mode"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_train_forward_rejects_invalid_ddp_padding_policy() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "ddp_sync": {"candidate_padding": "always"}
    }

    with pytest.raises(ValueError, match="ddp_sync.candidate_padding"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_ddp_no_padding_requires_broadcast_buffers_disabled() -> None:
    payload = _payload()
    payload["training"]["ddp_broadcast_buffers"] = True
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "ddp_sync": {"candidate_padding": "none"}
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    message = str(exc.value)
    assert "train_forward.ddp_sync.candidate_padding=none" in message
    assert "training.ddp_broadcast_buffers=false" in message


def test_checkpointed_exact_rejects_branch_local_aux_in_initial_bridge() -> None:
    payload = _payload()
    payload["custom"]["coord_soft_ce_w1"] = {"enabled": True}
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {"mode": "checkpointed_exact"}
    }

    with pytest.raises(ValueError, match="checkpointed_exact.*coord_soft_ce_w1"):
        TrainingConfig.from_mapping(payload, PromptOverrides())


def test_smart_batched_exact_rejects_branch_local_aux_in_initial_bridge() -> None:
    payload = _payload()
    payload["custom"]["coord_soft_ce_w1"] = {"enabled": True}
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {"mode": "smart_batched_exact"},
        "branch_batching": {"enabled": True},
    }

    with pytest.raises(ValueError, match="smart_batched_exact.*coord_soft_ce_w1"):
        TrainingConfig.from_mapping(payload, PromptOverrides())
