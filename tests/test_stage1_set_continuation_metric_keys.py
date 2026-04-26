from __future__ import annotations

import runpy
from pathlib import Path


_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_stage1_set_continuation_trainer_smoke.py"))
)


def test_set_continuation_emits_train_forward_runtime_metric_keys() -> None:
    cfg = _HELPERS["_cfg"](
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
        }
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
        "mp/branch_runtime_mode",
        "mp/checkpointed_branch_forwards",
        "mp/retained_graph_branch_forwards",
        "mp/smart_batched_branch_forwards",
        "mp/branch_batch_count",
        "mp/branch_batch_rows_mean",
        "mp/branch_batch_rows_max",
        "mp/branch_batch_tokens_mean",
        "mp/branch_batch_tokens_max",
        "mp/branch_batch_padding_fraction",
        "mp/branch_batch_scheduler",
        "mp/objective_fidelity_exact_samples",
        "mp/objective_fidelity_approx_samples",
        "mp/fallback_applied_samples",
        "mp/fallback_reason_candidate_budget",
        "mp/fallback_reason_token_budget",
        "mp/fallback_reason_memory_budget",
        "mp/prefix_encoding_cache_hits",
        "mp/prefix_encoding_cache_misses",
    }.issubset(keys)
