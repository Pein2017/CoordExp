# Stage-1 Train-Forward Runtime Stabilization Implementation Plan

> Archived / superseded on 2026-05-02.
> Historical provenance only for the pre-refactor Stage-1 set-continuation family.
> Do not use this file as an execution source.
> Active execution sources:
> - `docs/superpowers/specs/2026-05-02-training-infra-template-mode-refactor-design.md`
> - `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`

## Historical Execution Notes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status update, 2026-04-26:** This plan began as a checkpoint/recompute memory bridge. It is now historical context only. The active 2026-05-02 greenfield recursive detection CE spec/plan listed above supersedes this production-speed bridge for new training-infrastructure work.

**Goal:** Stabilize Stage-1 set-continuation training against long-sample CUDA OOM and severe throughput loss by adding a train-forward runtime that materializes only supervised suffix logits, avoids DDP padding forwards in production, preserves exact MP/PEM whenever all authored candidates are scored, and makes cap-8 approximate fallback explicit.

**Architecture:** Keep the verified set-continuation trainer structure intact, but extract branch planning, budget policy, branch scoring, and telemetry into focused helpers. Preserve the current retained-graph loop as the immediate production runtime, add exact `supervised_suffix` logit cropping, make DDP candidate padding configurable, keep checkpoint/recompute exact scoring as optional fallback context, and add deterministic approximate fallback only when an authored budget policy requires it.

**Tech Stack:** Python, Qwen3-VL `logits_to_keep` no-cache model forwards, optional PyTorch activation checkpointing, Hugging Face / ms-swift trainer stack, CoordExp strict config dataclasses, existing Stage-1 set-continuation tests, pytest, YAML production config, JSON runtime artifacts, and `conda run -n ms python -m pytest` verification.

---

## Scope Decision

This is not a new training paradigm. The current Stage-1 set-continuation structure is already verified; this plan addresses the production OOM and throughput components caused by full-prefix/full-logit repeated branch scoring and candidate-padding waste for a single sample. It does not implement prefix KV/cache reuse, packing, or branch attention masks.

Implement only the train-step forward runtime:

- no train-time eval decoding redesign;
- no upstream HF model edits;
- no packing enablement;
- no GPU KV prefix cache in the immediate bridge;
- no branch attention mask in the immediate bridge.

The implementation must preserve current behavior when `train_forward.logits.mode: full` and `train_forward.ddp_sync.candidate_padding: max_count`, and must make any approximate fallback visible through metrics and artifacts.
When `custom.stage1_set_continuation.train_forward` is omitted, resolved config must preserve current verified behavior: retained-graph branch scoring, full logits, max-count DDP candidate padding, budget fallback disabled, prefix encoding cache disabled, GPU KV cache disabled, and branch attention masks disabled.

## Historical Source Contracts

When this plan was active, these files were the source contracts:

- `openspec/changes/add-stage1-set-continuation-training/proposal.md`
- `openspec/changes/add-stage1-set-continuation-training/design.md`
- `openspec/changes/add-stage1-set-continuation-training/specs/stage1-set-continuation-training/spec.md`
- `openspec/changes/add-stage1-set-continuation-training/specs/trainer-metrics-components/spec.md`
- `openspec/changes/add-stage1-set-continuation-training/tasks.md`
- `docs/training/STAGE1_OBJECTIVE.md`
- `configs/stage1/set_continuation/production.yaml`

Use the current code as the compatibility baseline:

- `src/trainers/stage1_set_continuation/trainer.py`
- `src/trainers/stage1_set_continuation/branch_encoder.py`
- `src/trainers/stage1_set_continuation/losses.py`
- `src/trainers/stage1_set_continuation/sampling.py`
- `src/trainers/stage1_set_continuation/metrics.py`
- `src/config/schema.py`
- `src/sft.py`

## File Structure

### New files

- Create: `src/trainers/stage1_set_continuation/runtime.py`
  - Typed runtime plan payloads: branch runtime mode, objective fidelity, fallback reason, candidate execution plan, score bundle, telemetry atoms.
- Create: `src/trainers/stage1_set_continuation/budget.py`
  - Budget policy helpers that decide exact versus approximate candidate selection before expensive branch scoring.
- Create: `src/trainers/stage1_set_continuation/branch_scorer.py`
  - Retained-graph and checkpointed-exact branch scoring helpers that wrap existing branch encoding/forward/loss primitives without changing candidate semantics.
- Create: `tests/test_stage1_set_continuation_train_forward_config.py`
  - Strict config/default/production-profile tests for `custom.stage1_set_continuation.train_forward`.
- Create: `tests/test_stage1_set_continuation_runtime_policy.py`
  - Pure-Python tests for budget policy, fallback labels, objective fidelity, and deterministic candidate selection.
- Create: `tests/test_stage1_set_continuation_branch_runtime.py`
  - Tiny-model tests proving retained-graph and checkpointed-exact scoring produce matching loss/gradient behavior.
- Create: `tests/test_stage1_set_continuation_metric_keys.py`
  - Variant-specific metric-key coverage for train-forward runtime metrics.

### Existing files to modify

- Modify: `src/config/schema.py`
  - Add strict `Stage1SetContinuationTrainForwardConfig` and nested branch-runtime, budget, fallback, prefix-reuse, and telemetry dataclasses.
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
  - Replace inline candidate-loop policy with a runtime plan + branch scorer while preserving current retained-graph behavior.
- Modify: `src/trainers/stage1_set_continuation/metrics.py`
  - Register new train-forward metric keys and aggregation semantics.
- Modify: `src/sft.py`
  - Record train-forward runtime config and objective-fidelity/fallback surfaces in `effective_runtime.json`.
- Modify: `configs/stage1/set_continuation/production.yaml`
  - Opt production into `supervised_suffix`, no DDP candidate padding,
    `ddp_find_unused_parameters=false`, `ddp_broadcast_buffers=false`, and
    authored cap-8 approximate fallback thresholds.
- Modify: `tests/test_stage1_set_continuation_config.py`
  - Extend existing strict parsing coverage only if helper fixtures should remain centralized there.
- Modify: `tests/test_stage1_set_continuation_trainer_smoke.py`
  - Add retained-graph parity and checkpointed-exact smoke coverage.
- Modify: `tests/test_stage1_set_continuation_benchmark_profiles.py`
  - Assert production config and effective runtime record branch runtime, budget policy, fallback, prefix reuse, and fidelity fields.
- Do not modify ordinary Stage-1 metric parity expectations in
  `tests/test_stage1_metric_key_parity.py` except to preserve the assertion that
  ordinary Stage-1 does not emit `mp/*` keys.

### Files to avoid

- Do not edit upstream HF files such as `modeling_qwen3_vl.py`.
- Do not modify eval decoding modules for this slice.
- Do not enable packing for `stage1_set_continuation`.

---

### Task 1: Add Strict Train-Forward Config

**Files:**
- Modify: `src/config/schema.py`
- Create: `tests/test_stage1_set_continuation_train_forward_config.py`
- Modify: `tests/test_stage1_set_continuation_benchmark_profiles.py`

- [ ] **Step 1: Write config tests first**

Create `tests/test_stage1_set_continuation_train_forward_config.py`:

```python
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
    assert train_forward.branch_runtime.checkpoint_use_reentrant is False
    assert train_forward.branch_runtime.preserve_rng_state is True
    assert train_forward.budget_policy.enabled is False
    assert train_forward.budget_policy.fallback.mode == "disabled"
    assert train_forward.prefix_reuse.encoding_cache is False
    assert train_forward.prefix_reuse.kv_cache.mode == "disabled"


def test_train_forward_accepts_checkpointed_exact_with_approx_fallback() -> None:
    payload = _payload()
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


def test_train_forward_rejects_unknown_nested_key() -> None:
    payload = _payload()
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {"mode": "checkpointed_exact", "mystery": 1}
    }

    with pytest.raises(ValueError) as exc:
        TrainingConfig.from_mapping(payload, PromptOverrides())

    assert "custom.stage1_set_continuation.train_forward.branch_runtime.mystery" in str(exc.value)


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


def test_checkpointed_exact_rejects_branch_local_aux_in_initial_bridge() -> None:
    payload = _payload()
    payload["custom"]["coord_soft_ce_w1"] = {"enabled": True}
    payload["custom"]["stage1_set_continuation"]["train_forward"] = {
        "branch_runtime": {"mode": "checkpointed_exact"}
    }

    with pytest.raises(ValueError, match="checkpointed_exact.*coord_soft_ce_w1"):
        TrainingConfig.from_mapping(payload, PromptOverrides())
```

- [ ] **Step 2: Run the config test and confirm it fails**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_train_forward_config.py -q
```

Expected before implementation:

```text
FAILED tests/test_stage1_set_continuation_train_forward_config.py::test_train_forward_defaults_preserve_current_retained_graph_behavior
AttributeError: 'Stage1SetContinuationConfig' object has no attribute 'train_forward'
```

- [ ] **Step 3: Add config dataclasses**

In `src/config/schema.py`, add dataclasses near the existing `Stage1SetContinuation*Config` classes:

```python
@dataclass(frozen=True)
class Stage1SetContinuationBranchRuntimeConfig:
    mode: Literal["retained_graph", "checkpointed_exact"] = "retained_graph"
    checkpoint_use_reentrant: bool = False
    preserve_rng_state: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"retained_graph", "checkpointed_exact"}:
            raise ValueError(
                "custom.stage1_set_continuation.train_forward.branch_runtime.mode "
                "must be one of {'retained_graph', 'checkpointed_exact'}"
            )


@dataclass(frozen=True)
class Stage1SetContinuationExactUntilConfig:
    max_candidates: Optional[int] = None
    max_branch_tokens_per_sample: Optional[int] = None
    min_free_memory_gib: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_branch_tokens_per_sample"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(
                    f"custom.stage1_set_continuation.train_forward.budget_policy.exact_until.{name} must be > 0"
                )
            if value is not None:
                object.__setattr__(self, name, int(value))
        if self.min_free_memory_gib is not None and float(self.min_free_memory_gib) < 0:
            raise ValueError(
                "custom.stage1_set_continuation.train_forward.budget_policy.exact_until.min_free_memory_gib must be >= 0"
            )
        if self.min_free_memory_gib is not None:
            object.__setattr__(self, "min_free_memory_gib", float(self.min_free_memory_gib))


@dataclass(frozen=True)
class Stage1SetContinuationFallbackConfig:
    mode: Literal["disabled", "approximate_uniform_subsample"] = "disabled"
    max_candidates: Optional[int] = None
    estimator: Literal["uniform_importance"] = "uniform_importance"
    require_telemetry: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "approximate_uniform_subsample"}:
            raise ValueError(
                "custom.stage1_set_continuation.train_forward.budget_policy.fallback.mode "
                "must be one of {'disabled', 'approximate_uniform_subsample'}"
            )
        if self.estimator != "uniform_importance":
            raise ValueError(
                "custom.stage1_set_continuation.train_forward.budget_policy.fallback.estimator "
                "must be 'uniform_importance'"
            )
        if self.mode == "approximate_uniform_subsample":
            if self.max_candidates is None or int(self.max_candidates) <= 0:
                raise ValueError(
                    "custom.stage1_set_continuation.train_forward.budget_policy.fallback.max_candidates "
                    "must be > 0 when fallback.mode=approximate_uniform_subsample"
                )
            object.__setattr__(self, "max_candidates", int(self.max_candidates))
        elif self.max_candidates is not None:
            raise ValueError(
                "custom.stage1_set_continuation.train_forward.budget_policy.fallback.max_candidates "
                "requires fallback.mode=approximate_uniform_subsample"
            )


@dataclass(frozen=True)
class Stage1SetContinuationBudgetPolicyConfig:
    enabled: bool = False
    exact_until: Stage1SetContinuationExactUntilConfig = field(
        default_factory=Stage1SetContinuationExactUntilConfig
    )
    fallback: Stage1SetContinuationFallbackConfig = field(
        default_factory=Stage1SetContinuationFallbackConfig
    )


@dataclass(frozen=True)
class Stage1SetContinuationKVCacheConfig:
    mode: Literal["disabled"] = "disabled"

    def __post_init__(self) -> None:
        if self.mode != "disabled":
            raise ValueError(
                "custom.stage1_set_continuation.train_forward.prefix_reuse.kv_cache.mode "
                "must be 'disabled' in the immediate bridge"
            )


@dataclass(frozen=True)
class Stage1SetContinuationPrefixReuseConfig:
    encoding_cache: bool = False
    kv_cache: Stage1SetContinuationKVCacheConfig = field(
        default_factory=Stage1SetContinuationKVCacheConfig
    )


@dataclass(frozen=True)
class Stage1SetContinuationTelemetryConfig:
    per_rank_memory: bool = True
    branch_budget: bool = True
    objective_fidelity: bool = True


@dataclass(frozen=True)
class Stage1SetContinuationTrainForwardConfig:
    branch_runtime: Stage1SetContinuationBranchRuntimeConfig = field(
        default_factory=Stage1SetContinuationBranchRuntimeConfig
    )
    budget_policy: Stage1SetContinuationBudgetPolicyConfig = field(
        default_factory=Stage1SetContinuationBudgetPolicyConfig
    )
    prefix_reuse: Stage1SetContinuationPrefixReuseConfig = field(
        default_factory=Stage1SetContinuationPrefixReuseConfig
    )
    telemetry: Stage1SetContinuationTelemetryConfig = field(
        default_factory=Stage1SetContinuationTelemetryConfig
    )
```

Add `train_forward: Stage1SetContinuationTrainForwardConfig` to `Stage1SetContinuationConfig`, and ensure strict parsing reports dotted paths under `custom.stage1_set_continuation.train_forward`.

Also extend `TrainingConfig.from_mapping` validation for the immediate bridge:

```python
if (
    custom.trainer_variant == "stage1_set_continuation"
    and custom.stage1_set_continuation is not None
    and custom.stage1_set_continuation.train_forward.branch_runtime.mode == "checkpointed_exact"
):
    if custom.coord_soft_ce_w1.enabled:
        raise ValueError(
            "custom.stage1_set_continuation.train_forward.branch_runtime.mode=checkpointed_exact "
            "does not yet support branch-local coord_soft_ce_w1; use retained_graph or disable coord_soft_ce_w1"
        )
    if custom.bbox_geo.enabled:
        raise ValueError(
            "custom.stage1_set_continuation.train_forward.branch_runtime.mode=checkpointed_exact "
            "does not yet support branch-local bbox_geo; use retained_graph or disable bbox_geo"
        )
    if custom.bbox_size_aux.enabled:
        raise ValueError(
            "custom.stage1_set_continuation.train_forward.branch_runtime.mode=checkpointed_exact "
            "does not yet support branch-local bbox_size_aux; use retained_graph or disable bbox_size_aux"
        )
```

- [ ] **Step 4: Re-run the config test**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_train_forward_config.py tests/test_stage1_set_continuation_config.py -q
```

Expected after implementation:

```text
passed
```

- [ ] **Step 5: Commit**

```bash
git add src/config/schema.py tests/test_stage1_set_continuation_train_forward_config.py tests/test_stage1_set_continuation_config.py
git commit -m "feat: add stage1 set-continuation train-forward config"
```

---

### Task 2: Add Runtime Plan and Budget Policy Helpers

**Files:**
- Create: `src/trainers/stage1_set_continuation/runtime.py`
- Create: `src/trainers/stage1_set_continuation/budget.py`
- Create: `tests/test_stage1_set_continuation_runtime_policy.py`

- [ ] **Step 1: Write budget policy tests first**

Create `tests/test_stage1_set_continuation_runtime_policy.py`:

```python
from __future__ import annotations

from src.config.schema import Stage1SetContinuationConfig
from src.trainers.stage1_set_continuation.budget import plan_candidate_execution
from src.trainers.stage1_set_continuation.sampling import Stage1SetContinuationSample


def _sample(candidate_count: int) -> Stage1SetContinuationSample:
    candidates = tuple(range(candidate_count))
    return Stage1SetContinuationSample(
        selected_mode="empty_prefix",
        configured_mixture={"empty_prefix": 1.0},
        resolved_valid_mixture={"empty_prefix": 1.0},
        prefix_indices=(),
        remaining_indices=candidates,
        candidate_indices=candidates,
        candidate_scoring_mode="exact",
        scored_candidate_fraction=1.0,
    )


def _cfg(payload: dict) -> Stage1SetContinuationConfig:
    base = {
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
    }
    base["train_forward"] = payload
    return Stage1SetContinuationConfig.from_mapping(base)


def test_disabled_budget_keeps_exact_candidate_plan() -> None:
    cfg = _cfg({"budget_policy": {"enabled": False}})

    plan = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 0),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )

    assert plan.objective_fidelity == "exact"
    assert plan.authored_candidate_scoring_mode == "exact"
    assert plan.selected_candidate_indices == (0, 1, 2, 3, 4)
    assert plan.fallback_applied is False
    assert plan.logz_estimator == "exact"


def test_candidate_budget_falls_back_to_uniform_subset() -> None:
    cfg = _cfg(
        {
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_candidates": 3},
                "fallback": {
                    "mode": "approximate_uniform_subsample",
                    "max_candidates": 2,
                    "estimator": "uniform_importance",
                },
            }
        }
    )

    first = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 17),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )
    second = plan_candidate_execution(
        sample=_sample(5),
        cfg=cfg,
        sample_seed_parts=("unit", 17),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10, 10, 10, 10],
        memory_free_gib=None,
    )

    assert first.selected_candidate_indices == second.selected_candidate_indices
    assert len(first.selected_candidate_indices) == 2
    assert first.remaining_candidate_count == 5
    assert first.authored_candidate_scoring_mode == "exact"
    assert first.objective_fidelity == "approximate_uniform_subsample"
    assert first.fallback_applied is True
    assert first.fallback_reason == "candidate_budget"
    assert first.logz_estimator == "uniform_importance"


def test_token_budget_fallback_records_token_reason() -> None:
    cfg = _cfg(
        {
            "budget_policy": {
                "enabled": True,
                "exact_until": {"max_branch_tokens_per_sample": 110},
                "fallback": {
                    "mode": "approximate_uniform_subsample",
                    "max_candidates": 1,
                    "estimator": "uniform_importance",
                },
            }
        }
    )

    plan = plan_candidate_execution(
        sample=_sample(3),
        cfg=cfg,
        sample_seed_parts=("unit", 3),
        prefix_tokens=100,
        candidate_token_lengths=[20, 20, 20],
        memory_free_gib=None,
    )

    assert plan.objective_fidelity == "approximate_uniform_subsample"
    assert plan.fallback_reason == "token_budget"
    assert len(plan.selected_candidate_indices) == 1


def test_authored_uniform_subsample_remains_approximate_without_budget_fallback() -> None:
    sample = _sample(2)
    sample = Stage1SetContinuationSample(
        selected_mode=sample.selected_mode,
        configured_mixture=sample.configured_mixture,
        resolved_valid_mixture=sample.resolved_valid_mixture,
        prefix_indices=sample.prefix_indices,
        remaining_indices=(0, 1, 2, 3),
        candidate_indices=(1, 3),
        candidate_scoring_mode="uniform_subsample",
        scored_candidate_fraction=0.5,
    )
    cfg = _cfg({"budget_policy": {"enabled": False}})

    plan = plan_candidate_execution(
        sample=sample,
        cfg=cfg,
        sample_seed_parts=("unit", 5),
        prefix_tokens=100,
        candidate_token_lengths=[10, 10],
        memory_free_gib=None,
    )

    assert plan.authored_candidate_scoring_mode == "uniform_subsample"
    assert plan.objective_fidelity == "approximate_uniform_subsample"
    assert plan.fallback_applied is False
    assert plan.logz_estimator == "sampled_raw"
```

- [ ] **Step 2: Run the policy test and confirm it fails**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_runtime_policy.py -q
```

Expected before implementation:

```text
FAILED tests/test_stage1_set_continuation_runtime_policy.py
ModuleNotFoundError: No module named 'src.trainers.stage1_set_continuation.budget'
```

- [ ] **Step 3: Implement runtime payloads**

Create `src/trainers/stage1_set_continuation/runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ObjectiveFidelity = Literal["exact", "approximate_uniform_subsample"]
FallbackReason = Literal["none", "candidate_budget", "token_budget", "memory_budget"]
LogZEstimator = Literal["exact", "sampled_raw", "uniform_importance"]


@dataclass(frozen=True)
class CandidateExecutionPlan:
    selected_candidate_indices: tuple[int, ...]
    remaining_candidate_count: int
    authored_candidate_scoring_mode: str
    objective_fidelity: ObjectiveFidelity
    fallback_applied: bool
    fallback_reason: FallbackReason
    logz_estimator: LogZEstimator
    prefix_tokens: int
    planned_candidate_tokens: int
    exact_candidate_count: int
```

- [ ] **Step 4: Implement budget planning**

Create `src/trainers/stage1_set_continuation/budget.py`:

```python
from __future__ import annotations

import hashlib
import random
from typing import Sequence

from src.config.schema import Stage1SetContinuationConfig

from .runtime import CandidateExecutionPlan
from .sampling import Stage1SetContinuationSample


def _stable_rng(parts: Sequence[object]) -> random.Random:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return random.Random(seed)


def _budget_reason(
    *,
    exact_candidate_count: int,
    prefix_tokens: int,
    candidate_token_lengths: Sequence[int],
    max_candidates: int | None,
    max_branch_tokens_per_sample: int | None,
    min_free_memory_gib: float | None,
    memory_free_gib: float | None,
    enabled_budget_kinds: Sequence[str],
) -> str:
    enabled = set(enabled_budget_kinds)
    if "candidate" in enabled and max_candidates is not None and exact_candidate_count > max_candidates:
        return "candidate_budget"
    planned_tokens = int(prefix_tokens) * max(1, exact_candidate_count) + sum(
        int(length) for length in candidate_token_lengths
    )
    if (
        "token" in enabled
        and max_branch_tokens_per_sample is not None
        and planned_tokens > max_branch_tokens_per_sample
    ):
        return "token_budget"
    if (
        "memory" in enabled
        and min_free_memory_gib is not None
        and memory_free_gib is not None
        and memory_free_gib < min_free_memory_gib
    ):
        return "memory_budget"
    return "none"


def plan_candidate_execution(
    *,
    sample: Stage1SetContinuationSample,
    cfg: Stage1SetContinuationConfig,
    sample_seed_parts: Sequence[object],
    prefix_tokens: int,
    candidate_token_lengths: Sequence[int],
    memory_free_gib: float | None,
    enabled_budget_kinds: Sequence[str] = ("candidate", "token", "memory"),
) -> CandidateExecutionPlan:
    train_forward = cfg.train_forward
    exact_indices = tuple(int(idx) for idx in sample.candidate_indices)
    authored_mode = str(sample.candidate_scoring_mode)
    pem_enabled = cfg.positive_evidence_margin.objective == "threshold_loss"
    authored_is_exact = (
        authored_mode == "exact"
        and len(exact_indices) == len(sample.remaining_indices)
    )
    no_fallback_fidelity = "exact" if authored_is_exact else "approximate_uniform_subsample"
    no_fallback_estimator = (
        "exact"
        if authored_is_exact
        else ("uniform_importance" if pem_enabled else "sampled_raw")
    )
    planned_tokens = int(prefix_tokens) * max(1, len(exact_indices)) + sum(
        int(length) for length in candidate_token_lengths
    )
    if not train_forward.budget_policy.enabled or not exact_indices:
        return CandidateExecutionPlan(
            selected_candidate_indices=exact_indices,
            remaining_candidate_count=len(sample.remaining_indices),
            authored_candidate_scoring_mode=authored_mode,
            objective_fidelity=no_fallback_fidelity,
            fallback_applied=False,
            fallback_reason="none",
            logz_estimator=no_fallback_estimator,
            prefix_tokens=int(prefix_tokens),
            planned_candidate_tokens=planned_tokens,
            exact_candidate_count=len(exact_indices),
        )

    exact_until = train_forward.budget_policy.exact_until
    reason = _budget_reason(
        exact_candidate_count=len(exact_indices),
        prefix_tokens=int(prefix_tokens),
        candidate_token_lengths=candidate_token_lengths,
        max_candidates=exact_until.max_candidates,
        max_branch_tokens_per_sample=exact_until.max_branch_tokens_per_sample,
        min_free_memory_gib=exact_until.min_free_memory_gib,
        memory_free_gib=memory_free_gib,
        enabled_budget_kinds=enabled_budget_kinds,
    )
    if reason == "none":
        return CandidateExecutionPlan(
            selected_candidate_indices=exact_indices,
            remaining_candidate_count=len(sample.remaining_indices),
            authored_candidate_scoring_mode=authored_mode,
            objective_fidelity=no_fallback_fidelity,
            fallback_applied=False,
            fallback_reason="none",
            logz_estimator=no_fallback_estimator,
            prefix_tokens=int(prefix_tokens),
            planned_candidate_tokens=planned_tokens,
            exact_candidate_count=len(exact_indices),
        )

    fallback = train_forward.budget_policy.fallback
    if fallback.mode != "approximate_uniform_subsample" or fallback.max_candidates is None:
        raise RuntimeError(
            "stage1_set_continuation exact candidate plan exceeded train_forward budget "
            f"without an enabled approximate fallback: reason={reason}"
        )

    rng = _stable_rng((*sample_seed_parts, reason, len(exact_indices)))
    selected = tuple(sorted(rng.sample(list(exact_indices), min(fallback.max_candidates, len(exact_indices)))))
    return CandidateExecutionPlan(
        selected_candidate_indices=selected,
        remaining_candidate_count=len(sample.remaining_indices),
        authored_candidate_scoring_mode=authored_mode,
        objective_fidelity="approximate_uniform_subsample",
        fallback_applied=True,
        fallback_reason=reason,  # type: ignore[arg-type]
        logz_estimator="uniform_importance" if pem_enabled else "sampled_raw",
        prefix_tokens=int(prefix_tokens),
        planned_candidate_tokens=planned_tokens,
        exact_candidate_count=len(exact_indices),
    )
```

- [ ] **Step 5: Re-run policy tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_runtime_policy.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/trainers/stage1_set_continuation/runtime.py src/trainers/stage1_set_continuation/budget.py tests/test_stage1_set_continuation_runtime_policy.py
git commit -m "feat: add stage1 set-continuation train-forward budget policy"
```

---

### Task 3: Extract Branch Scoring With Retained-Graph Parity

**Files:**
- Create: `src/trainers/stage1_set_continuation/branch_scorer.py`
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
- Modify: `tests/test_stage1_set_continuation_trainer_smoke.py`

- [ ] **Step 1: Add retained-graph parity assertions to the smoke test**

In `tests/test_stage1_set_continuation_trainer_smoke.py`, add a test that forces retained mode and asserts the existing call count and metrics:

```python
def test_retained_graph_runtime_preserves_current_exact_branch_behavior() -> None:
    cfg = _cfg()
    trainer = _trainer(cfg)
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch(), return_outputs=False)

    assert torch.isfinite(loss)
    assert len(model.calls) == 3
    assert _latest_metric(trainer, "mp/branch_runtime_mode") >= 0.0
    assert _latest_metric(trainer, "mp/retained_graph_branch_forwards") == pytest.approx(2.0)
    assert _latest_metric(trainer, "mp/checkpointed_branch_forwards") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/objective_fidelity_exact_samples") == pytest.approx(1.0)
    assert _latest_metric(trainer, "mp/objective_fidelity_approx_samples") == pytest.approx(0.0)
```

- [ ] **Step 2: Run the smoke test and confirm it fails only on new metrics**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_trainer_smoke.py::test_retained_graph_runtime_preserves_current_exact_branch_behavior -q
```

Expected before implementation:

```text
FAILED tests/test_stage1_set_continuation_trainer_smoke.py::test_retained_graph_runtime_preserves_current_exact_branch_behavior
AssertionError: metric 'mp/branch_runtime_mode' was not emitted
```

- [ ] **Step 3: Implement branch scorer retained mode**

Create `src/trainers/stage1_set_continuation/branch_scorer.py` with a retained scorer that delegates to existing trainer internals:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .branch_encoder import EncodedSetContinuationBranch
from .losses import CandidateLogProbResult, compute_candidate_full_entry_logprob


@dataclass(frozen=True)
class BranchScoreBundle:
    score: torch.Tensor
    logprob: CandidateLogProbResult
    outputs: Any
    branch_inputs: dict[str, Any]


def score_branch_retained_graph(
    *,
    trainer: Any,
    model: Any,
    branch: EncodedSetContinuationBranch,
    coord_token_ids: torch.Tensor,
) -> BranchScoreBundle:
    outputs, branch_inputs = trainer._forward_branch(model=model, branch=branch)
    logits = outputs.logits
    logprob = compute_candidate_full_entry_logprob(
        logits=logits,
        labels=branch_inputs["labels"],
        candidate_entry_label_mask=branch.candidate_entry_label_mask,
        coord_label_mask=branch.coord_label_mask,
        coord_token_ids=coord_token_ids.to(logits.device),
    )
    return BranchScoreBundle(
        score=logprob.score,
        logprob=logprob,
        outputs=outputs,
        branch_inputs=branch_inputs,
    )
```

- [ ] **Step 4: Route `_score_candidate_branch` through the branch scorer**

In `src/trainers/stage1_set_continuation/trainer.py`, keep the public method name but delegate retained mode:

```python
from .branch_scorer import score_branch_retained_graph


def _score_candidate_branch(
    self,
    *,
    model: Any,
    branch: EncodedSetContinuationBranch,
    coord_token_ids: torch.Tensor,
) -> tuple[CandidateLogProbResult, Any, dict[str, Any]]:
    runtime_mode = self._cfg().train_forward.branch_runtime.mode
    if runtime_mode == "retained_graph":
        bundle = score_branch_retained_graph(
            trainer=self,
            model=model,
            branch=branch,
            coord_token_ids=coord_token_ids,
        )
        return bundle.logprob, bundle.outputs, bundle.branch_inputs
    raise ValueError(f"unsupported branch runtime mode: {runtime_mode}")
```

Keep the return shape compatible with current `_process_sample`.

- [ ] **Step 5: Emit retained runtime metrics**

Inside `_process_sample`, initialize and update:

```python
metrics["mp/branch_runtime_mode"] = metric_code(
    self._cfg().train_forward.branch_runtime.mode,
    metric_name="mp/branch_runtime_mode",
)
metrics["mp/retained_graph_branch_forwards"] = float(
    len(candidate_indices) if self._cfg().train_forward.branch_runtime.mode == "retained_graph" else 0
)
metrics["mp/checkpointed_branch_forwards"] = float(0)
metrics["mp/objective_fidelity_exact_samples"] = 1.0
metrics["mp/objective_fidelity_approx_samples"] = 0.0
metrics["mp/fallback_applied_samples"] = 0.0
metrics["mp/fallback_reason_candidate_budget"] = 0.0
metrics["mp/fallback_reason_token_budget"] = 0.0
metrics["mp/fallback_reason_memory_budget"] = 0.0
```

Task 5 will replace these initial exact/no-fallback constants with
execution-plan values after the budget policy is integrated.

- [ ] **Step 6: Re-run retained parity smoke**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_trainer_smoke.py::test_retained_graph_runtime_preserves_current_exact_branch_behavior -q
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/trainers/stage1_set_continuation/branch_scorer.py src/trainers/stage1_set_continuation/trainer.py tests/test_stage1_set_continuation_trainer_smoke.py
git commit -m "refactor: preserve retained set-continuation branch runtime"
```

---

### Task 4: Add Checkpointed Exact Branch Runtime

**Files:**
- Modify: `src/trainers/stage1_set_continuation/branch_scorer.py`
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
- Create: `tests/test_stage1_set_continuation_branch_runtime.py`

- [ ] **Step 1: Write deterministic checkpoint parity tests**

Create `tests/test_stage1_set_continuation_branch_runtime.py`:

```python
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn


class _TinyBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16)
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self.calls += 1
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return type("Output", (), {"logits": logits})()


def test_checkpointed_exact_matches_retained_graph_gradient_on_tiny_branch() -> None:
    from src.trainers.stage1_set_continuation.branch_scorer import (
        score_tensor_checkpointed,
        score_tensor_retained,
    )

    torch.manual_seed(11)
    model_a = _TinyBranchModel()
    model_b = copy.deepcopy(model_a)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = input_ids.clone()
    candidate_mask = torch.tensor([[False, True, True, True]])
    coord_mask = torch.tensor([[False, False, False, False]])
    coord_token_ids = torch.tensor([10, 11, 12], dtype=torch.long)

    retained = score_tensor_retained(
        model=model_a,
        model_inputs={"input_ids": input_ids},
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
    )
    checkpointed = score_tensor_checkpointed(
        model=model_b,
        model_inputs={"input_ids": input_ids},
        labels=labels,
        candidate_entry_label_mask=candidate_mask,
        coord_label_mask=coord_mask,
        coord_token_ids=coord_token_ids,
        use_reentrant=False,
        preserve_rng_state=True,
    )

    (-retained.score).backward()
    (-checkpointed.score).backward()

    assert torch.allclose(retained.score.detach(), checkpointed.score.detach(), atol=1e-6)
    assert torch.allclose(retained.coord_score.detach(), checkpointed.coord_score.detach(), atol=1e-6)
    assert torch.allclose(retained.non_coord_score.detach(), checkpointed.non_coord_score.detach(), atol=1e-6)
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert param_a.grad is not None
        assert param_b.grad is not None
        assert torch.allclose(param_a.grad, param_b.grad, atol=1e-6)
    assert model_b.calls >= 2
```

- [ ] **Step 2: Run the checkpoint parity test and confirm it fails**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_branch_runtime.py -q
```

Expected before implementation:

```text
FAILED tests/test_stage1_set_continuation_branch_runtime.py::test_checkpointed_exact_matches_retained_graph_gradient_on_tiny_branch
ImportError: cannot import name 'score_tensor_checkpointed'
```

- [ ] **Step 3: Implement tensor-level retained and checkpointed scorers**

In `src/trainers/stage1_set_continuation/branch_scorer.py`, add tensor-level functions that can be tested without a full trainer:

```python
from torch.utils.checkpoint import checkpoint


def _score_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
) -> torch.Tensor:
    return compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        coord_token_ids=coord_token_ids.to(logits.device),
    ).score


def _result_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
) -> CandidateLogProbResult:
    return compute_candidate_full_entry_logprob(
        logits=logits,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        coord_token_ids=coord_token_ids.to(logits.device),
    )


def _token_counts(
    *,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
) -> tuple[int, int, int]:
    tokens = int(candidate_entry_label_mask.sum().item())
    coord_tokens = int((candidate_entry_label_mask & coord_label_mask).sum().item())
    return tokens, coord_tokens, tokens - coord_tokens


def score_tensor_retained(
    *,
    model: Any,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
) -> CandidateLogProbResult:
    outputs = model(**model_inputs)
    return _result_from_logits(
        logits=outputs.logits,
        labels=labels,
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
        coord_token_ids=coord_token_ids,
    )


def score_tensor_checkpointed(
    *,
    model: Any,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    candidate_entry_label_mask: torch.Tensor,
    coord_label_mask: torch.Tensor,
    coord_token_ids: torch.Tensor,
    use_reentrant: bool,
    preserve_rng_state: bool,
) -> CandidateLogProbResult:
    input_names = tuple(model_inputs.keys())
    input_values = tuple(model_inputs[name] for name in input_names)

    def _forward_scores(*flat_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = dict(zip(input_names, flat_inputs))
        outputs = model(**kwargs)
        result = _result_from_logits(
            logits=outputs.logits,
            labels=labels,
            candidate_entry_label_mask=candidate_entry_label_mask,
            coord_label_mask=coord_label_mask,
            coord_token_ids=coord_token_ids,
        )
        return result.score, result.coord_score, result.non_coord_score

    score, coord_score, non_coord_score = checkpoint(
        _forward_scores,
        *input_values,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )
    tokens, coord_tokens, non_coord_tokens = _token_counts(
        candidate_entry_label_mask=candidate_entry_label_mask,
        coord_label_mask=coord_label_mask,
    )
    return CandidateLogProbResult(
        score=score,
        coord_score=coord_score,
        non_coord_score=non_coord_score,
        tokens=tokens,
        coord_tokens=coord_tokens,
        non_coord_tokens=non_coord_tokens,
    )
```

- [ ] **Step 4: Implement branch-level checkpointed scorer**

Add a branch-level wrapper that prepares model inputs once, then checkpoints the
candidate score components. The immediate bridge requires branch-local aux
objectives to be disabled in config when `checkpointed_exact` is active, so the
wrapper does not return grad-bearing full logits for aux losses.

```python
def score_branch_checkpointed_exact(
    *,
    trainer: Any,
    model: Any,
    branch: EncodedSetContinuationBranch,
    coord_token_ids: torch.Tensor,
    use_reentrant: bool,
    preserve_rng_state: bool,
) -> BranchScoreBundle:
    branch_inputs = trainer._prepare_branch_inputs(model=model, branch=branch)
    _core_model, inputs_for_model, _model_type = prepare_forward_inputs(
        model=model,
        inputs=branch_inputs,
        ignored_keys=("labels", "compute_loss_func", "loss_scale", "text_position_ids", "channel", "logits_to_keep"),
        packing_enabled=False,
        where="stage1-set-continuation-checkpointed",
    )
    tensor_inputs = {
        key: value
        for key, value in inputs_for_model.items()
        if isinstance(value, torch.Tensor)
    }
    logprob = score_tensor_checkpointed(
        model=model,
        model_inputs=tensor_inputs,
        labels=branch_inputs["labels"],
        candidate_entry_label_mask=branch.candidate_entry_label_mask,
        coord_label_mask=branch.coord_label_mask,
        coord_token_ids=coord_token_ids.to(branch_inputs["labels"].device),
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )
    return BranchScoreBundle(
        score=logprob.score,
        logprob=logprob,
        outputs=None,
        branch_inputs=branch_inputs,
    )
```

If branch-local aux objectives need checkpointed support later, add a separate
task that computes aux-bearing atoms from the same checkpointed branch
computation. Do not return no-grad logits as `outputs` for aux loss code.

- [ ] **Step 5: Route trainer runtime mode**

First, update `_compute_candidate_aux_atoms` so it returns before reading
`outputs.logits` when all branch-local aux configs are disabled:

```python
def _branch_aux_enabled(self) -> bool:
    return any(
        bool(getattr(getattr(self, attr, None), "enabled", False))
        for attr in ("coord_soft_ce_w1_cfg", "bbox_geo_cfg", "bbox_size_aux_cfg")
    )


def _compute_candidate_aux_atoms(
    self,
    *,
    branch: EncodedSetContinuationBranch,
    branch_inputs: Mapping[str, Any],
    outputs: Any,
    aux_accum: dict[str, dict[str, Any]],
) -> None:
    if not self._branch_aux_enabled():
        return
    logits = getattr(outputs, "logits", None)
    labels = branch_inputs.get("labels")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("branch-local aux losses require logits and labels tensors")
    # Keep the existing aux implementation after this guard unchanged.
```

Then update `Stage1SetContinuationTrainer._score_candidate_branch` to dispatch
without changing its external return contract:

```python
runtime = self._cfg().train_forward.branch_runtime
if runtime.mode == "retained_graph":
    bundle = score_branch_retained_graph(
        trainer=self,
        model=model,
        branch=branch,
        coord_token_ids=coord_token_ids,
    )
    return bundle.logprob, bundle.outputs, bundle.branch_inputs
if runtime.mode == "checkpointed_exact":
    bundle = score_branch_checkpointed_exact(
        trainer=self,
        model=model,
        branch=branch,
        coord_token_ids=coord_token_ids,
        use_reentrant=runtime.checkpoint_use_reentrant,
        preserve_rng_state=runtime.preserve_rng_state,
    )
    return bundle.logprob, bundle.outputs, bundle.branch_inputs
raise ValueError(f"unsupported branch runtime mode: {runtime.mode}")
```

- [ ] **Step 6: Route DDP padding branches through the selected runtime**

In `_process_sample`, replace the padding path that directly calls
`_forward_branch(...)` with a helper that honors `train_forward.branch_runtime`.
For retained mode, preserve the existing zero-loss behavior. For checkpointed
mode, build a zero-valued differentiable scalar through the checkpointed branch
runtime instead of retaining a full padding branch graph:

```python
padding_result, padding_outputs, _padding_inputs = self._score_candidate_branch(
    model=model,
    branch=padding_branch,
    coord_token_ids=coord_token_ids,
)
padding_loss = padding_result.score * 0.0
total_loss = padding_loss if total_loss is None else total_loss + padding_loss
if padding_outputs is not None:
    last_outputs = padding_outputs
sync_contributes = True
```

Add a trainer smoke or runtime-policy test that forces one local sample to have
fewer candidates than `max_candidate_forwards` and asserts the emitted
`mp/checkpointed_branch_forwards` includes padding branches when checkpointed
mode is active.

- [ ] **Step 7: Run branch runtime and smoke tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_branch_runtime.py \
  tests/test_stage1_set_continuation_trainer_smoke.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 8: Commit**

```bash
git add src/trainers/stage1_set_continuation/branch_scorer.py src/trainers/stage1_set_continuation/trainer.py tests/test_stage1_set_continuation_branch_runtime.py tests/test_stage1_set_continuation_trainer_smoke.py
git commit -m "feat: add checkpointed exact set-continuation branch runtime"
```

---

### Task 5: Integrate Budgeted Approximate Fallback Into Trainer

**Files:**
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
- Modify: `src/trainers/stage1_set_continuation/budget.py`
- Modify: `tests/test_stage1_set_continuation_trainer_smoke.py`
- Modify: `tests/test_stage1_set_continuation_loss.py`

- [ ] **Step 1: Add trainer fallback smoke test**

In `tests/test_stage1_set_continuation_trainer_smoke.py`, add:

```python
def test_budget_fallback_scores_subset_and_reports_approximate_fidelity() -> None:
    cfg = Stage1SetContinuationConfig.from_mapping(
        {
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
            "train_forward": {
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
        }
    )
    trainer = _trainer(cfg)
    model = _FakeModel()
    trainer.model = model

    loss = trainer.compute_loss(model, _batch([OBJECT_A, OBJECT_B, OBJECT_C]), return_outputs=False)

    assert torch.isfinite(loss)
    assert _latest_metric(trainer, "mp/num_remaining_objects") == pytest.approx(3.0)
    assert _latest_metric(trainer, "mp/num_candidates_scored") == pytest.approx(1.0)
    assert _latest_metric(trainer, "mp/objective_fidelity_exact_samples") == pytest.approx(0.0)
    assert _latest_metric(trainer, "mp/objective_fidelity_approx_samples") == pytest.approx(1.0)
    assert _latest_metric(trainer, "mp/fallback_applied_samples") == pytest.approx(1.0)
    assert _latest_metric(trainer, "mp/fallback_reason_candidate_budget") == pytest.approx(1.0)
```

- [ ] **Step 2: Run fallback smoke and confirm it fails**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_trainer_smoke.py::test_budget_fallback_scores_subset_and_reports_approximate_fidelity -q
```

Expected before implementation:

```text
FAILED tests/test_stage1_set_continuation_trainer_smoke.py::test_budget_fallback_scores_subset_and_reports_approximate_fidelity
assert 3.0 == 1.0 for metric 'mp/num_candidates_scored'
```

- [ ] **Step 3: Split planning into candidate-count and encoded-token phases**

In `_process_sample`, do not use string-length guesses. Build the plan in two
explicit phases:

1. Candidate-count phase, before branch encoding:

```python
candidate_count_plan = plan_candidate_execution(
    sample=sample,
    cfg=cfg,
    sample_seed_parts=(*self._seed_parts(meta=meta, sample_offset=sample_offset), "candidate_budget"),
    prefix_tokens=0,
    candidate_token_lengths=[],
    memory_free_gib=None,
    enabled_budget_kinds=("candidate",),
)
candidate_indices = candidate_count_plan.selected_candidate_indices
```

2. Encoded-token phase, after encoding the candidate branches selected by the
count phase and before model forwards:

```python
encoded_candidates: list[tuple[int, EncodedSetContinuationBranch]] = []
candidate_token_lengths: list[int] = []
for candidate_index in candidate_indices:
    branch = self._encode_branch(
        meta=meta,
        prefix_indices=prefix_indices,
        candidate_index=candidate_index,
    )
    encoded_candidates.append((candidate_index, branch))
    candidate_token_lengths.append(int(branch.candidate_entry_label_mask.sum().item()))

execution_plan = plan_candidate_execution(
    sample=sample,
    cfg=cfg,
    sample_seed_parts=(*self._seed_parts(meta=meta, sample_offset=sample_offset), "token_budget"),
    prefix_tokens=_prefix_tokens_from_encoded_candidates(encoded_candidates),
    candidate_token_lengths=candidate_token_lengths,
    memory_free_gib=_cuda_free_memory_gib_or_none(),
    enabled_budget_kinds=("token", "memory"),
)
encoded_candidates = [
    (candidate_index, branch)
    for candidate_index, branch in encoded_candidates
    if candidate_index in set(execution_plan.selected_candidate_indices)
]
candidate_indices = execution_plan.selected_candidate_indices
```

Implement `_prefix_tokens_from_encoded_candidates(...)` locally in the trainer
as the first available candidate branch prefix length, falling back to the close
branch prefix length when there are no candidates. Implement
`_cuda_free_memory_gib_or_none()` as a best-effort helper that returns `None`
when CUDA is unavailable or memory telemetry errors.

- [ ] **Step 4: Use the estimator from the execution plan**

Use `execution_plan.logz_estimator` directly:

```python
mp_result = compute_mp_pem_losses(
    scores=score_tensor,
    pem_mode=str(cfg.positive_evidence_margin.objective),
    rho=cfg.positive_evidence_margin.rho,
    log_rho=cfg.positive_evidence_margin.log_rho,
    estimator=execution_plan.logz_estimator,
    remaining_count=execution_plan.remaining_candidate_count,
    scored_count=len(candidate_indices),
    candidate_lengths=score_tensor.new_tensor(candidate_lengths),
)
```

This preserves the current rule:

```text
exact mode -> exact estimator
authored or fallback uniform subsample with PEM disabled -> sampled_raw
authored or fallback uniform subsample with PEM threshold_loss -> uniform_importance
```

- [ ] **Step 5: Recompute sampler-derived metrics from the final execution plan**

After the final execution plan is known, overwrite metrics initialized from the
sampler so they reflect the actual runtime:

```python
metrics["mp/num_candidates_scored"] = float(len(candidate_indices))
metrics["mp/scored_candidate_fraction"] = float(
    len(candidate_indices) / max(1, execution_plan.remaining_candidate_count)
)
metrics["mp/candidate_scoring_mode"] = metric_code(
    execution_plan.authored_candidate_scoring_mode,
    CANDIDATE_SCORING_MODE_CODES,
    metric_name="mp/candidate_scoring_mode",
)
metrics["mp/logZ_estimator"] = metric_code(
    execution_plan.logz_estimator,
    LOGZ_ESTIMATOR_CODES,
    metric_name="mp/logZ_estimator",
)
```

- [ ] **Step 6: Emit fallback metrics**

Add metrics:

```python
metrics["mp/objective_fidelity_exact_samples"] = float(execution_plan.objective_fidelity == "exact")
metrics["mp/objective_fidelity_approx_samples"] = float(execution_plan.objective_fidelity != "exact")
metrics["mp/fallback_applied_samples"] = float(execution_plan.fallback_applied)
metrics["mp/fallback_reason_candidate_budget"] = float(execution_plan.fallback_reason == "candidate_budget")
metrics["mp/fallback_reason_token_budget"] = float(execution_plan.fallback_reason == "token_budget")
metrics["mp/fallback_reason_memory_budget"] = float(execution_plan.fallback_reason == "memory_budget")
```

- [ ] **Step 7: Run fallback and loss tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_runtime_policy.py \
  tests/test_stage1_set_continuation_loss.py \
  tests/test_stage1_set_continuation_trainer_smoke.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 8: Commit**

```bash
git add src/trainers/stage1_set_continuation/trainer.py src/trainers/stage1_set_continuation/budget.py tests/test_stage1_set_continuation_trainer_smoke.py tests/test_stage1_set_continuation_loss.py
git commit -m "feat: add budgeted approximate fallback for set-continuation MP"
```

---

### Task 6: Add Safe Prefix Reuse Boundary

**Files:**
- Modify: `src/trainers/stage1_set_continuation/branch_encoder.py`
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
- Modify: `tests/test_stage1_set_continuation_serialization.py`
- Modify: `tests/test_stage1_set_continuation_trainer_smoke.py`

- [ ] **Step 1: Add parity test before enabling any token-id prefix concatenation**

In `tests/test_stage1_set_continuation_trainer_smoke.py`, add a test that compares cached versus uncached branch construction:

```python
from src.trainers.stage1_set_continuation.branch_encoder import (
    PrefixEncodingCache,
    encode_set_continuation_branch,
    encode_set_continuation_branch_with_prefix_cache,
)


def _assert_encoded_branch_equal(left, right) -> None:
    assert torch.equal(left.branch_inputs["input_ids"], right.branch_inputs["input_ids"])
    assert torch.equal(left.labels, right.labels)
    assert torch.equal(left.candidate_entry_label_mask, right.candidate_entry_label_mask)
    assert torch.equal(left.coord_label_mask, right.coord_label_mask)
    assert torch.equal(left.non_coord_label_mask, right.non_coord_label_mask)
    assert torch.equal(left.structural_close_start_mask, right.structural_close_start_mask)
    assert torch.equal(left.structural_close_sequence_mask, right.structural_close_sequence_mask)
    assert left.rendered_text == right.rendered_text
    assert left.prefix_text == right.prefix_text
    assert left.continuation_text == right.continuation_text
    assert left.prefix_indices == right.prefix_indices
    assert left.candidate_index == right.candidate_index


def test_prefix_encoding_cache_matches_uncached_branch_inputs() -> None:
    meta = _raw_sample([OBJECT_A, OBJECT_B, OBJECT_C])
    template = _FakeTemplate()
    prefix_cache = PrefixEncodingCache()

    uncached_a = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=(0,),
        candidate_index=1,
        object_field_order="desc_first",
    )
    cached_a = encode_set_continuation_branch_with_prefix_cache(
        meta=meta,
        template=template,
        prefix_indices=(0,),
        candidate_index=1,
        object_field_order="desc_first",
        prefix_cache=prefix_cache,
    )
    cached_b = encode_set_continuation_branch_with_prefix_cache(
        meta=meta,
        template=template,
        prefix_indices=(0,),
        candidate_index=2,
        object_field_order="desc_first",
        prefix_cache=prefix_cache,
    )
    uncached_b = encode_set_continuation_branch(
        meta=meta,
        template=template,
        prefix_indices=(0,),
        candidate_index=2,
        object_field_order="desc_first",
    )

    _assert_encoded_branch_equal(cached_a, uncached_a)
    _assert_encoded_branch_equal(cached_b, uncached_b)
    assert prefix_cache.hits >= 1
    assert prefix_cache.misses == 1
```

- [ ] **Step 2: Implement exact render/span prefix reuse only**

Implement a small in-sample render/span cache. Do not implement token-id prefix
concatenation in this bridge, because tokenizer/template compositionality has
not been proven for every active Qwen3-VL chat-template surface.

```text
cache selected prefix render state, object spans, and reusable branch metadata
tokenize each full candidate branch as before
emit cache hit/miss metrics from the render/span cache
```

Keep `prefix_reuse.encoding_cache` as this exact render/span cache and emit:

```text
mp/prefix_encoding_cache_hits
mp/prefix_encoding_cache_misses
```

with exact behavior preserved.

- [ ] **Step 3: Keep GPU KV cache disabled**

Do not add `use_cache=True` to model forwards. Do not detach prefix K/V. Any future detached KV cache must be a separate approximate mode.

- [ ] **Step 4: Run serialization and smoke tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_serialization.py \
  tests/test_stage1_set_continuation_trainer_smoke.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 5: Commit**

```bash
git add src/trainers/stage1_set_continuation/branch_encoder.py src/trainers/stage1_set_continuation/trainer.py tests/test_stage1_set_continuation_serialization.py tests/test_stage1_set_continuation_trainer_smoke.py
git commit -m "feat: add exact prefix reuse boundary for set-continuation branches"
```

---

### Task 7: Add Telemetry, Runtime Artifacts, and Production Config

**Files:**
- Modify: `src/trainers/stage1_set_continuation/metrics.py`
- Modify: `src/trainers/stage1_set_continuation/trainer.py`
- Modify: `src/sft.py`
- Modify: `configs/stage1/set_continuation/production.yaml`
- Create: `tests/test_stage1_set_continuation_metric_keys.py`
- Modify: `tests/test_stage1_set_continuation_benchmark_profiles.py`

- [ ] **Step 1: Add metric/artifact tests first**

Extend `tests/test_stage1_set_continuation_benchmark_profiles.py::test_effective_runtime_records_production_set_continuation_provenance` with:

```python
assert sc_runtime["train_forward"]["branch_runtime"]["mode"] == "checkpointed_exact"
assert sc_runtime["train_forward"]["budget_policy"]["enabled"] is True
assert sc_runtime["train_forward"]["budget_policy"]["fallback"]["mode"] == "approximate_uniform_subsample"
assert sc_runtime["train_forward"]["prefix_reuse"]["kv_cache"]["mode"] == "disabled"
assert sc_runtime["objective_fidelity"]["exact_metric"] == "mp/objective_fidelity_exact_samples"
assert sc_runtime["objective_fidelity"]["approx_metric"] == "mp/objective_fidelity_approx_samples"
```

Create `tests/test_stage1_set_continuation_metric_keys.py` with a real
`Stage1SetContinuationTrainer` fixture, following the helper style in
`tests/test_stage1_set_continuation_trainer_smoke.py`, and assert that the
set-continuation variant emits:

```python
{
    "mp/branch_runtime_mode",
    "mp/checkpointed_branch_forwards",
    "mp/retained_graph_branch_forwards",
    "mp/objective_fidelity_exact_samples",
    "mp/objective_fidelity_approx_samples",
    "mp/fallback_applied_samples",
    "mp/fallback_reason_candidate_budget",
    "mp/fallback_reason_token_budget",
    "mp/fallback_reason_memory_budget",
    "mp/prefix_encoding_cache_hits",
    "mp/prefix_encoding_cache_misses",
}
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  tests/test_stage1_set_continuation_metric_keys.py \
  -q
```

Expected before implementation:

```text
FAILED tests/test_stage1_set_continuation_benchmark_profiles.py::test_effective_runtime_records_production_set_continuation_provenance
KeyError: 'train_forward'
```

- [ ] **Step 3: Update metrics and artifact payload**

In `src/sft.py`, extend the existing set-continuation runtime payload with:

```python
"train_forward": {
    "branch_runtime": asdict(sc_cfg.train_forward.branch_runtime),
    "budget_policy": asdict(sc_cfg.train_forward.budget_policy),
    "prefix_reuse": asdict(sc_cfg.train_forward.prefix_reuse),
    "telemetry": asdict(sc_cfg.train_forward.telemetry),
},
"objective_fidelity": {
    "exact_metric": "mp/objective_fidelity_exact_samples",
    "approx_metric": "mp/objective_fidelity_approx_samples",
    "fallback_metric": "mp/fallback_applied_samples",
},
```

Use the repo's existing dataclass-to-mapping helper if one is already present near the runtime payload code.

- [ ] **Step 4: Update production config**

In `configs/stage1/set_continuation/production.yaml`, under `custom.stage1_set_continuation`, add:

```yaml
    train_forward:
      branch_runtime:
        mode: checkpointed_exact
        checkpoint_use_reentrant: false
        preserve_rng_state: true
      budget_policy:
        enabled: true
        exact_until:
          max_candidates: 8
          max_branch_tokens_per_sample: 24000
          min_free_memory_gib: 4.0
        fallback:
          mode: approximate_uniform_subsample
          max_candidates: 4
          estimator: uniform_importance
          require_telemetry: true
      prefix_reuse:
        encoding_cache: true
        kv_cache:
          mode: disabled
      telemetry:
        per_rank_memory: true
        branch_budget: true
        objective_fidelity: true
```

Also update `experiment.key_deviations` from the old repeated-forward/no-prefix-cache wording to:

```yaml
    - "execution: checkpointed exact repeated-branch semantics with explicit approximate fallback for over-budget samples"
    - "prefix reuse: exact CPU/render-prefix cache only; GPU KV cache and branch attention mask disabled"
```

- [ ] **Step 5: Re-run profile and metric tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  tests/test_stage1_set_continuation_metric_keys.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/trainers/stage1_set_continuation/metrics.py src/trainers/stage1_set_continuation/trainer.py src/sft.py configs/stage1/set_continuation/production.yaml tests/test_stage1_set_continuation_metric_keys.py tests/test_stage1_set_continuation_benchmark_profiles.py
git commit -m "feat: record set-continuation train-forward runtime provenance"
```

---

### Task 8: Verification and Production Relaunch Gate

**Files:**
- Modify only if tests reveal a focused gap:
  - `tests/test_stage1_set_continuation_*.py`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `docs/training/METRICS.md`

- [ ] **Step 1: Run focused Stage-1 set-continuation tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_set_continuation_config.py \
  tests/test_stage1_set_continuation_train_forward_config.py \
  tests/test_stage1_set_continuation_runtime_policy.py \
  tests/test_stage1_set_continuation_branch_runtime.py \
  tests/test_stage1_set_continuation_loss.py \
  tests/test_stage1_set_continuation_sampler.py \
  tests/test_stage1_set_continuation_serialization.py \
  tests/test_stage1_set_continuation_collator.py \
  tests/test_stage1_set_continuation_aux_adapters.py \
  tests/test_stage1_set_continuation_trainer_smoke.py \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run config/artifact adjacent tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage1_metric_key_parity.py \
  tests/test_stage1_set_continuation_metric_keys.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_training_config_strict_unknown_keys.py \
  -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run code checks on touched Python paths**

Run:

```bash
conda run -n ms python -m py_compile \
  src/config/schema.py \
  src/sft.py \
  src/trainers/stage1_set_continuation/runtime.py \
  src/trainers/stage1_set_continuation/budget.py \
  src/trainers/stage1_set_continuation/branch_scorer.py \
  src/trainers/stage1_set_continuation/trainer.py \
  src/trainers/stage1_set_continuation/metrics.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 4: Run a tiny train-forward smoke**

Use the existing tiny trainer smoke first. If a production-like CLI smoke already exists for `configs/stage1/set_continuation/production.yaml`, add a sample-limited overlay only after the unit smoke passes. The smoke must exercise:

```text
custom.trainer_variant=stage1_set_continuation
train_forward.branch_runtime.mode=checkpointed_exact
budget_policy.enabled=true
fallback.mode=approximate_uniform_subsample
training.packing=false
eval decoding disabled or outside the assertion scope
```

Expected artifacts:

```text
resolved_config.json
effective_runtime.json with train_forward block
experiment_manifest.json with runtime summary pointers
logging.jsonl or trainer metrics containing branch runtime/fidelity keys
```

- [ ] **Step 5: Run OpenSpec validation if the CLI is available**

Run:

```bash
command -v openspec
openspec validate add-stage1-set-continuation-training --strict
```

Expected when `openspec` is installed:

```text
valid
```

If `openspec` is not installed, record the blocker in the final implementation summary and do not claim strict OpenSpec validation passed.

- [ ] **Step 6: Commit final verification/doc adjustments**

```bash
git add \
  docs/training/STAGE1_OBJECTIVE.md \
  docs/training/METRICS.md \
  tests/test_stage1_set_continuation_train_forward_config.py \
  tests/test_stage1_set_continuation_runtime_policy.py \
  tests/test_stage1_set_continuation_branch_runtime.py \
  tests/test_stage1_set_continuation_metric_keys.py
git commit -m "test: verify stage1 set-continuation train-forward runtime"
```

---

## Production Relaunch Criteria

Do not relaunch the full 8xA100 production run until all of these are true:

- retained-graph mode passes the existing verified tests;
- checkpointed-exact mode passes loss/gradient parity on deterministic tiny fixtures with branch-local aux disabled;
- checkpointed-exact mode rejects enabled branch-local aux objectives until aux-bearing checkpoint support is implemented;
- DDP padding branches honor the selected branch runtime;
- fallback tests prove approximate mode is deterministic, preserve authored sampled-mode semantics, and never report approximate samples as exact;
- production config resolves with `checkpointed_exact` and explicit fallback thresholds;
- `effective_runtime.json` records train-forward runtime policy and objective-fidelity metric handles;
- smoke metrics include `mp/branch_runtime_mode`, `mp/objective_fidelity_*`, and fallback counters;
- no code path enables GPU KV prefix cache or edits upstream HF model files.

## Implementation Notes

- Keep `retained_graph` as the default for omitted `train_forward` config so existing verified configs are not silently changed.
- Opt the canonical production config into `checkpointed_exact`.
- Use `use_reentrant=False` for checkpointing unless tests prove the current environment requires otherwise.
- Preserve RNG state in checkpointed mode so stochastic layers do not silently turn exact MP into approximate MP.
- Keep `run_no_cache_forward`; do not enable model `use_cache`.
- Implement only render/span prefix caching in this bridge. Treat prefix token-id concatenation as a future optimization that requires its own parity tests.
- If the coord-offset adapter allocation remains a separate peak after checkpointing, address it in a separate targeted change with its own tests. Do not mix that optimization into the branch-runtime refactor.
