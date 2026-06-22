# Stage-2 Trie Forward Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage-2 Channel-B trie multiple-positive hard CE so rollout-aware repaired candidates train like Stage-1 entry-trie supervision instead of ordinary single-target SFT.

**Architecture:** Keep Stage-2 rollout, greedy IoU assignment, false-negative insertion, and false-positive policy in the Stage-2 target-building layer. Compile one per-example trie target sidecar for Channel-B, then consume it from a new `stage2_trie_ce` teacher-forcing objective module that uses float32-safe pure hard trie CE math with `token_mean` normalization and emits structured diagnostics. Keep stable docs and OpenSpec updates out of the first implementation slice until tiny train-side evidence is collected.

**Tech Stack:** Python, PyTorch, CoordExp teacher-forcing objective pipeline, compact-full Stage-2 two-channel trainer, pytest under `conda run -n ms`.

---

Date: 2026-05-18

Spec: `docs/superpowers/specs/2026-05-18-stage2-trie-forward-supervision-design.md`

Worktree root: `/data/CoordExp/.worktrees/unified-training-infra-refactor`

## Execution Policy

Use one worker per task group when possible. Each worker must inspect the current target files before editing and must preserve unrelated untracked ablation configs. Do not edit upstream Hugging Face or Qwen3-VL model files.

Commit after each task group if the user asks for commits. Otherwise leave changes unstaged and report the exact file set.

## File Map

| Path | Responsibility |
|---|---|
| `src/trainers/stage2_two_channel/trie_supervision.py` | Stage-2 candidate annotations, trie target compiler, FP/FN/fallback role accounting, deterministic insertion helpers. |
| `src/trainers/stage2_two_channel/types.py` | TypedDict sidecar fields for Stage-2 trie metadata. |
| `src/trainers/stage2_two_channel/target_builder.py` | Build Channel-B repaired candidates and attach trie supervision sidecars to segment metadata. |
| `src/trainers/teacher_forcing/modules/stage2_trie_ce.py` | Objective module that consumes Stage-2 trie sidecars and computes hard trie CE. |
| `src/trainers/teacher_forcing/module_registry.py` | Register `stage2_trie_ce` in the objective catalog. |
| `src/trainers/teacher_forcing/objective_pipeline.py` | Route module config/context into the new objective module. |
| `src/config/schema.py` | Accept `stage2_trie_ce`, reject Channel-B double supervision, and validate FP policy fields. |
| `src/trainers/stage2_two_channel.py` | Aggregate Stage-2 trie metrics and write span-score diagnostics from Channel-B metadata. |
| `tests/test_stage2_trie_supervision.py` | Unit tests for trie target compilation and candidate policy semantics. |
| `tests/test_stage2_trie_ce_module.py` | Unit tests for hard trie CE numerical behavior and dtype boundary. |
| `tests/test_stage2_ab_config_contract.py` | Config acceptance and rejection tests. |
| `tests/test_stage2_ab_training.py` | Trainer wiring tests for Channel-B metadata, metrics, and fallback accounting. |
| `configs/stage2_two_channel/smoke/*.yaml` | Stage-2 trie tiny-overfit ablation configs. |

## Required Behavior Summary

- Channel-B `stage2_trie_ce` trains on one merged per-example trie across K rollout-derived repaired candidates.
- K defaults to 4 through `stage2_ab.channel_b.triage_posterior.num_rollouts`.
- Stage-2 trie v0 is pure hard CE only with `token_mean` normalization.
- `support_weight`, `balance_weight`, semantic role weights, and semantic image bucket balancing are reserved future knobs, not active v0 behavior.
- `token_ce` and `stage2_trie_ce` cannot both target Channel-B.
- False-negative insertion policies remain `tail_append`, `fn_slot_shuffle`, and `sorted`.
- False-positive policies are `zero_loss_context` and `weak_positive_context`.
- Weak-positive false positives require explorer support in v0.
- Fallback GT/FN append-only supervision is downweighted with default `fallback_loss_weight=0.25`.
- Span score records are diagnostic-only in v0.
- Loss internals use float32 for log-probability and denominator aggregation.

## Task 1: Add Config Contract Tests First

**Files:**

- Modify: `tests/test_stage2_ab_config_contract.py`
- Modify: `src/config/schema.py`

- [ ] **Step 1: Inspect existing Stage-2 config schema and tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
rg -n "Stage2AB|stage2_ab|objective|token_ce|bbox_geo|coord_reg|fallback_loss_weight|fp_policy" src/config/schema.py tests/test_stage2_ab_config_contract.py
```

Expected: command prints the existing Stage-2 schema owners and current objective validation tests.

- [ ] **Step 2: Add failing config acceptance test for Stage-2 trie**

Add this test near the existing Stage-2 objective/config tests in `tests/test_stage2_ab_config_contract.py`:

```python
def test_stage2_pipeline_accepts_channel_b_trie_objective() -> None:
    config = {
        "stage2_ab": {
            "channel_b": {
                "fallback_loss_weight": 0.25,
                "insertion_order": "fn_slot_shuffle",
                "triage_posterior": {"num_rollouts": 4},
                "fp_policy": {
                    "mode": "weak_positive_context",
                    "weak_positive_weight": 0.05,
                    "require_explorer_support": True,
                    "min_support_count": 1,
                    "require_token_score": False,
                },
            },
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A"],
                        "application": {"preset": "anchor_text_only"},
                    },
                    {
                        "name": "stage2_trie_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["B"],
                        "application": {"preset": "rollout_trie_hard_ce"},
                        "config": {
                            "support_weight": 1.0,
                            "balance_weight": 1.0,
                            "struct_weight": 1.0,
                            "desc_weight": 1.0,
                            "coord_hard_ce_weight": 1.0,
                            "eos_weight": 1.0,
                            "normalization": "token_mean",
                        },
                    },
                ]
            },
        }
    }

    loaded = Stage2ABConfig.model_validate(config["stage2_ab"])

    objective_names = [entry.name for entry in loaded.pipeline.objective]
    assert objective_names == ["token_ce", "stage2_trie_ce"]
    assert loaded.channel_b.fallback_loss_weight == 0.25
    assert loaded.channel_b.triage_posterior.num_rollouts == 4
    assert loaded.channel_b.fp_policy.mode == "weak_positive_context"
```

If this project uses a different config loader helper in this test file, adapt only the construction wrapper and keep the assertions.

- [ ] **Step 3: Add failing rejection test for double Channel-B supervision**

Add:

```python
def test_stage2_pipeline_rejects_token_ce_and_trie_ce_for_channel_b() -> None:
    config = {
        "stage2_ab": {
            "pipeline": {
                "objective": [
                    {
                        "name": "token_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["A", "B"],
                        "application": {"preset": "anchor_and_rollout_text"},
                    },
                    {
                        "name": "stage2_trie_ce",
                        "enabled": True,
                        "weight": 1.0,
                        "channels": ["B"],
                        "application": {"preset": "rollout_trie_hard_ce"},
                        "config": {
                            "support_weight": 1.0,
                            "balance_weight": 1.0,
                            "struct_weight": 1.0,
                            "desc_weight": 1.0,
                            "coord_hard_ce_weight": 1.0,
                            "eos_weight": 1.0,
                            "normalization": "token_mean",
                        },
                    },
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="Channel-B.*token_ce.*stage2_trie_ce"):
        Stage2ABConfig.model_validate(config["stage2_ab"])
```

- [ ] **Step 4: Add failing rejection test for unsupported FP policy**

Add:

```python
def test_stage2_fp_policy_rejects_unknown_mode() -> None:
    config = {
        "stage2_ab": {
            "channel_b": {
                "fp_policy": {
                    "mode": "score_threshold_context",
                    "weak_positive_weight": 0.05,
                    "require_explorer_support": True,
                    "min_support_count": 1,
                    "require_token_score": False,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="zero_loss_context|weak_positive_context"):
        Stage2ABConfig.model_validate(config["stage2_ab"])
```

- [ ] **Step 5: Run config tests and confirm failure**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py -q
```

Expected: the new tests fail because `stage2_trie_ce` and `fp_policy` schema support are not implemented.

- [ ] **Step 6: Implement schema additions**

In `src/config/schema.py`, add or extend Pydantic models equivalent to:

```python
class Stage2ABChannelBFalsePositivePolicyConfig(BaseModel):
    mode: Literal["zero_loss_context", "weak_positive_context"] = "zero_loss_context"
    weak_positive_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    require_explorer_support: bool = True
    min_support_count: int = Field(default=1, ge=1)
    require_token_score: bool = False


class Stage2TrieCEObjectiveConfig(BaseModel):
    support_weight: float = Field(default=1.0, ge=0.0)
    balance_weight: float = Field(default=1.0, ge=0.0)
    struct_weight: float = Field(default=1.0, ge=0.0)
    desc_weight: float = Field(default=1.0, ge=0.0)
    coord_hard_ce_weight: float = Field(default=1.0, ge=0.0)
    eos_weight: float = Field(default=1.0, ge=0.0)
    normalization: Literal["token_mean"] = "token_mean"

Reject non-default `support_weight`, `balance_weight`, `struct_weight`,
`desc_weight`, `coord_hard_ce_weight`, and `eos_weight` values with an explicit
Stage-2 trie CE pure hard CE v0/reserved future knob error.
```

Then update the Stage-2 objective validator to allow `stage2_trie_ce` and reject simultaneous enabled Channel-B `token_ce` and `stage2_trie_ce`. Use the existing model-validator style in the file. The validator should effectively perform:

```python
enabled_channel_b = {
    entry.name
    for entry in self.pipeline.objective
    if entry.enabled and "B" in set(entry.channels)
}
if {"token_ce", "stage2_trie_ce"}.issubset(enabled_channel_b):
    raise ValueError(
        "Channel-B cannot enable token_ce and stage2_trie_ce together; "
        "use stage2_trie_ce for trie Stage-2 training"
    )
```

- [ ] **Step 7: Run config tests and confirm pass**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py -q
```

Expected: the Stage-2 config contract tests pass.

## Task 2: Register The Objective Module Shell

**Files:**

- Create: `src/trainers/teacher_forcing/modules/stage2_trie_ce.py`
- Modify: `src/trainers/teacher_forcing/module_registry.py`
- Modify: `src/trainers/teacher_forcing/objective_pipeline.py`
- Test: `tests/test_stage2_trie_ce_module.py`

- [ ] **Step 1: Add failing catalog test**

Create `tests/test_stage2_trie_ce_module.py` with:

```python
from __future__ import annotations

import torch

from src.trainers.teacher_forcing.module_registry import OBJECTIVE_MODULE_CATALOG
from src.trainers.teacher_forcing.modules.stage2_trie_ce import Stage2TrieCEConfig


def test_stage2_trie_ce_is_registered_as_text_objective() -> None:
    entry = OBJECTIVE_MODULE_CATALOG["stage2_trie_ce"]

    assert entry.name == "stage2_trie_ce"
    assert entry.emission_group == "text"
    assert Stage2TrieCEConfig().normalization == "token_mean"
```

- [ ] **Step 2: Run the new catalog test and confirm failure**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_ce_module.py::test_stage2_trie_ce_is_registered_as_text_objective -q
```

Expected: import or catalog lookup fails because the module is not registered.

- [ ] **Step 3: Create the objective module dataclass shell**

Create `src/trainers/teacher_forcing/modules/stage2_trie_ce.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Stage2TrieCEConfig:
    """Configuration for hard-CE Stage-2 trie supervision."""

    support_weight: float = 1.0
    balance_weight: float = 1.0
    struct_weight: float = 1.0
    desc_weight: float = 1.0
    coord_hard_ce_weight: float = 1.0
    eos_weight: float = 1.0
    normalization: str = "token_mean"


@dataclass(frozen=True)
class Stage2TrieCELossResult:
    """Loss and diagnostic scalars emitted by Stage-2 trie CE."""

    loss: torch.Tensor
    metrics: dict[str, float]
    atom_losses: dict[str, torch.Tensor]


def build_stage2_trie_ce_config(raw_config: Any | None) -> Stage2TrieCEConfig:
    """Build the Stage-2 trie CE config from a schema or dict object."""

    if raw_config is None:
        return Stage2TrieCEConfig()
    if isinstance(raw_config, Stage2TrieCEConfig):
        return raw_config
    if hasattr(raw_config, "model_dump"):
        return Stage2TrieCEConfig(**raw_config.model_dump())
    if isinstance(raw_config, dict):
        return Stage2TrieCEConfig(**raw_config)
    raise TypeError(f"Unsupported stage2_trie_ce config type: {type(raw_config)!r}")


def compute_stage2_trie_ce_loss(
    *,
    logits: torch.Tensor,
    metadata: list[dict[str, Any]],
    config: Stage2TrieCEConfig,
) -> Stage2TrieCELossResult:
    """Compute hard trie CE from Stage-2 Channel-B metadata."""

    zero = logits.float().sum() * 0.0
    return Stage2TrieCELossResult(
        loss=zero,
        metrics={"stage2_trie/target_positions": 0.0},
        atom_losses={"trie_ce": zero},
    )
```

This shell intentionally returns zero until Task 4 supplies compiled trie targets and Task 5 replaces the body with real trie CE.

- [ ] **Step 4: Register `stage2_trie_ce` in the catalog**

In `src/trainers/teacher_forcing/module_registry.py`, add an entry equivalent to:

```python
ObjectiveModuleCatalogEntry(
    name="stage2_trie_ce",
    emission_group="text",
    description="Stage-2 Channel-B hard trie multiple-positive CE.",
)
```

Use the exact dataclass or constructor already used by `token_ce`.

- [ ] **Step 5: Wire objective pipeline dispatch**

In `src/trainers/teacher_forcing/objective_pipeline.py`, route `stage2_trie_ce` to `compute_stage2_trie_ce_loss`. The dispatch must pass full logits and segment metadata. The minimal branch should follow the local style and behave like:

```python
if objective.name == "stage2_trie_ce":
    from src.trainers.teacher_forcing.modules.stage2_trie_ce import (
        build_stage2_trie_ce_config,
        compute_stage2_trie_ce_loss,
    )

    result = compute_stage2_trie_ce_loss(
        logits=logits,
        metadata=metadata,
        config=build_stage2_trie_ce_config(objective.config),
    )
    emissions.append(
        ObjectiveEmission(
            name="stage2_trie_ce",
            loss=result.loss * objective.weight,
            metrics=result.metrics,
            atom_losses=result.atom_losses,
        )
    )
```

Adapt names to the existing emission class in the file.

- [ ] **Step 6: Run catalog test and py-compile**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_ce_module.py::test_stage2_trie_ce_is_registered_as_text_objective -q
conda run -n ms python -m py_compile src/trainers/teacher_forcing/modules/stage2_trie_ce.py src/trainers/teacher_forcing/module_registry.py src/trainers/teacher_forcing/objective_pipeline.py
```

Expected: test and py-compile pass.

## Task 3: Build Stage-2 Trie Supervision Compiler

**Files:**

- Create: `src/trainers/stage2_two_channel/trie_supervision.py`
- Modify: `src/trainers/stage2_two_channel/types.py`
- Test: `tests/test_stage2_trie_supervision.py`

- [ ] **Step 1: Add failing tests for deterministic trie compilation**

Create `tests/test_stage2_trie_supervision.py`:

```python
from __future__ import annotations

from src.trainers.stage2_two_channel.trie_supervision import (
    Stage2TrieCandidate,
    Stage2TrieObjectSpan,
    Stage2TrieTokenTarget,
    compile_stage2_trie_targets,
)


def test_compile_stage2_trie_merges_multiple_positive_next_tokens() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 11, 13, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
    ]

    targets = compile_stage2_trie_targets(candidates, segment_start=5)

    branch_by_position = {target.position: target for target in targets.token_targets}
    assert branch_by_position[7].positive_token_ids == (12, 13)
    assert branch_by_position[7].source_weights == (1.0, 1.0)
    assert targets.summary.candidate_count == 2
    assert targets.summary.branch_points == 1


def test_compile_stage2_trie_downweights_fallback_candidate() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=-1,
            source="fallback_gt_fn_append_only",
            token_ids=[101, 21, 102],
            loss_weight=0.25,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="fallback_fn",
                    token_start=1,
                    token_end=2,
                    object_iou=None,
                    support_count=0,
                    loss_weight=0.25,
                )
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, segment_start=0)

    assert targets.summary.fallback_candidate_count == 1
    assert targets.summary.fallback_loss_weight_sum == 0.25
    assert targets.span_score_records[0].span_role == "fallback_fn"


def test_compile_stage2_trie_records_weak_fp_without_coord_supervision() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=2,
            source="valid_rollout",
            token_ids=[101, 31, 32, 102],
            loss_weight=1.0,
            object_spans=[
                Stage2TrieObjectSpan(
                    role="weak_positive_fp",
                    token_start=1,
                    token_end=3,
                    object_iou=None,
                    support_count=2,
                    loss_weight=0.05,
                )
            ],
        )
    ]

    targets = compile_stage2_trie_targets(candidates, segment_start=10)

    assert targets.summary.weak_positive_fp_count == 1
    assert targets.span_score_records[0].loss_weight == 0.05
    assert all(target.position >= 11 for target in targets.token_targets)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_supervision.py -q
```

Expected: import fails because the compiler module does not exist.

- [ ] **Step 3: Implement trie supervision dataclasses and compiler**

Create `src/trainers/stage2_two_channel/trie_supervision.py`:

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal


Stage2TrieCandidateSource = Literal[
    "valid_rollout",
    "fallback_gt_fn_append_only",
]

Stage2TrieSpanRole = Literal[
    "matched_clean",
    "inserted_fn",
    "recovered_fn",
    "neutral_fp",
    "weak_positive_fp",
    "fallback_fn",
]


@dataclass(frozen=True)
class Stage2TrieObjectSpan:
    """Object-level token span annotation for Stage-2 trie diagnostics."""

    role: Stage2TrieSpanRole
    token_start: int
    token_end: int
    object_iou: float | None
    support_count: int
    loss_weight: float


@dataclass(frozen=True)
class Stage2TrieCandidate:
    """One repaired Channel-B continuation candidate."""

    sample_id: str
    rollout_index: int
    source: Stage2TrieCandidateSource
    token_ids: list[int]
    loss_weight: float
    object_spans: list[Stage2TrieObjectSpan]


@dataclass(frozen=True)
class Stage2TrieTokenTarget:
    """Multiple-positive next-token target for one label position."""

    position: int
    positive_token_ids: tuple[int, ...]
    source_weights: tuple[float, ...]
    semantic_role: str


@dataclass(frozen=True)
class Stage2TrieSpanScoreRecord:
    """Serializable span-score diagnostic record."""

    sample_id: str
    rollout_index: int
    candidate_source: str
    span_role: str
    object_role: str
    token_start: int
    token_end: int
    mean_token_logprob: float | None
    min_token_logprob: float | None
    object_iou: float | None
    support_count: int
    loss_weight: float


@dataclass(frozen=True)
class Stage2TrieSummary:
    """Aggregate target statistics used for metrics."""

    candidate_count: int
    fallback_candidate_count: int
    fallback_loss_weight_sum: float
    weak_positive_fp_count: int
    target_positions: int
    branch_points: int
    max_branching_factor: int


@dataclass(frozen=True)
class Stage2TrieTargets:
    """Compiled Stage-2 trie supervision sidecar."""

    token_targets: tuple[Stage2TrieTokenTarget, ...]
    span_score_records: tuple[Stage2TrieSpanScoreRecord, ...]
    summary: Stage2TrieSummary


def _semantic_role_for_position(_: int) -> str:
    """Default semantic bucket until compact-full token-role mapping is attached."""

    return "text"


def compile_stage2_trie_targets(
    candidates: list[Stage2TrieCandidate],
    *,
    segment_start: int,
) -> Stage2TrieTargets:
    """Compile repaired Stage-2 candidates into merged next-token trie targets."""

    positive_by_position: dict[int, dict[int, float]] = defaultdict(dict)

    for candidate in candidates:
        for local_index, token_id in enumerate(candidate.token_ids):
            position = segment_start + local_index
            previous = positive_by_position[position].get(token_id, 0.0)
            positive_by_position[position][token_id] = max(previous, candidate.loss_weight)

    token_targets: list[Stage2TrieTokenTarget] = []
    for position in sorted(positive_by_position):
        token_weights = positive_by_position[position]
        ordered = sorted(token_weights.items(), key=lambda item: item[0])
        token_targets.append(
            Stage2TrieTokenTarget(
                position=position,
                positive_token_ids=tuple(token_id for token_id, _ in ordered),
                source_weights=tuple(weight for _, weight in ordered),
                semantic_role=_semantic_role_for_position(position),
            )
        )

    records: list[Stage2TrieSpanScoreRecord] = []
    for candidate in candidates:
        for span in candidate.object_spans:
            records.append(
                Stage2TrieSpanScoreRecord(
                    sample_id=candidate.sample_id,
                    rollout_index=candidate.rollout_index,
                    candidate_source=candidate.source,
                    span_role=span.role,
                    object_role=span.role,
                    token_start=segment_start + span.token_start,
                    token_end=segment_start + span.token_end,
                    mean_token_logprob=None,
                    min_token_logprob=None,
                    object_iou=span.object_iou,
                    support_count=span.support_count,
                    loss_weight=span.loss_weight,
                )
            )

    branching_factors = [len(target.positive_token_ids) for target in token_targets]
    summary = Stage2TrieSummary(
        candidate_count=len(candidates),
        fallback_candidate_count=sum(
            1 for candidate in candidates if candidate.source == "fallback_gt_fn_append_only"
        ),
        fallback_loss_weight_sum=sum(
            candidate.loss_weight
            for candidate in candidates
            if candidate.source == "fallback_gt_fn_append_only"
        ),
        weak_positive_fp_count=sum(
            1
            for candidate in candidates
            for span in candidate.object_spans
            if span.role == "weak_positive_fp"
        ),
        target_positions=len(token_targets),
        branch_points=sum(1 for factor in branching_factors if factor > 1),
        max_branching_factor=max(branching_factors, default=0),
    )

    return Stage2TrieTargets(
        token_targets=tuple(token_targets),
        span_score_records=tuple(records),
        summary=summary,
    )
```

This first compiler treats every candidate token position as supervised. Later tasks may attach compact-full token-role mapping for role-specific metrics, but the target shape remains the same.

- [ ] **Step 4: Add typed metadata fields**

In `src/trainers/stage2_two_channel/types.py`, add optional keys to the Channel-B metadata type:

```python
stage2_trie_targets: NotRequired[Any]
stage2_trie_span_scores: NotRequired[list[dict[str, Any]]]
stage2_trie_candidate_summary: NotRequired[dict[str, Any]]
```

If `NotRequired` is not already imported, import it from `typing_extensions` or the existing project-preferred typing source.

- [ ] **Step 5: Run supervision compiler tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_supervision.py -q
```

Expected: all Stage-2 trie supervision tests pass.

## Task 4: Compute Real Hard Trie CE

**Files:**

- Modify: `src/trainers/teacher_forcing/modules/stage2_trie_ce.py`
- Test: `tests/test_stage2_trie_ce_module.py`

- [ ] **Step 1: Add failing numerical trie CE tests**

Append to `tests/test_stage2_trie_ce_module.py`:

```python
from src.trainers.stage2_two_channel.trie_supervision import (
    Stage2TrieSummary,
    Stage2TrieTargets,
    Stage2TrieTokenTarget,
)
from src.trainers.teacher_forcing.modules.stage2_trie_ce import (
    compute_stage2_trie_ce_loss,
)


def test_stage2_trie_ce_uses_multi_positive_logsumexp() -> None:
    logits = torch.full((1, 3, 8), -20.0, dtype=torch.bfloat16)
    logits[0, 1, 2] = 3.0
    logits[0, 1, 5] = 3.0
    metadata = [
        {
            "stage2_channel": "B",
            "stage2_trie_targets": Stage2TrieTargets(
                token_targets=(
                    Stage2TrieTokenTarget(
                        position=1,
                        positive_token_ids=(2, 5),
                        source_weights=(1.0, 1.0),
                        semantic_role="text",
                    ),
                ),
                span_score_records=(),
                summary=Stage2TrieSummary(
                    candidate_count=2,
                    fallback_candidate_count=0,
                    fallback_loss_weight_sum=0.0,
                    weak_positive_fp_count=0,
                    target_positions=1,
                    branch_points=1,
                    max_branching_factor=2,
                ),
            ),
        }
    ]

    result = compute_stage2_trie_ce_loss(
        logits=logits,
        metadata=metadata,
        config=Stage2TrieCEConfig(),
    )

    assert result.loss.dtype == torch.float32
    assert result.loss.item() < 0.01
    assert result.metrics["stage2_trie/branch_points"] == 1.0


def test_stage2_trie_ce_ignores_non_channel_b_segments() -> None:
    logits = torch.zeros((1, 2, 4), dtype=torch.float32)
    metadata = [{"stage2_channel": "A"}]

    result = compute_stage2_trie_ce_loss(
        logits=logits,
        metadata=metadata,
        config=Stage2TrieCEConfig(),
    )

    assert result.loss.item() == 0.0
    assert result.metrics["stage2_trie/target_positions"] == 0.0
```

- [ ] **Step 2: Run numerical tests and confirm failure**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_ce_module.py -q
```

Expected: the multi-positive loss test fails because the shell returns zero without using targets.

- [ ] **Step 3: Implement float32 hard trie CE**

Replace the body of `compute_stage2_trie_ce_loss` with logic equivalent to:

```python
logits_f32 = logits.float()
loss_terms: list[torch.Tensor] = []
weights: list[torch.Tensor] = []
metric_sums: dict[str, float] = {
    "stage2_trie/target_positions": 0.0,
    "stage2_trie/branch_points": 0.0,
    "stage2_trie/max_branching_factor": 0.0,
    "stage2_trie/candidate_count_mean": 0.0,
    "stage2_trie/fallback_candidate_share": 0.0,
    "stage2_trie/fallback_loss_share": 0.0,  # compatibility alias for candidate share
    "stage2_trie/fallback_dominance_warning": 0.0,
    "stage2_trie/fp_policy_weak_positive_count": 0.0,
}
active_segments = 0

for row_index, item in enumerate(metadata):
    if item.get("stage2_channel") != "B":
        continue
    targets = item.get("stage2_trie_targets")
    if targets is None:
        continue
    active_segments += 1
    summary = targets.summary
    metric_sums["stage2_trie/target_positions"] += float(summary.target_positions)
    metric_sums["stage2_trie/branch_points"] += float(summary.branch_points)
    metric_sums["stage2_trie/max_branching_factor"] = max(
        metric_sums["stage2_trie/max_branching_factor"],
        float(summary.max_branching_factor),
    )
    metric_sums["stage2_trie/candidate_count_mean"] += float(summary.candidate_count)
    metric_sums["stage2_trie/fp_policy_weak_positive_count"] += float(
        summary.weak_positive_fp_count
    )

    for target in targets.token_targets:
        if target.position < 0 or target.position >= logits_f32.shape[1]:
            raise ValueError(
                f"Stage-2 trie target position {target.position} outside logits length "
                f"{logits_f32.shape[1]}"
            )
        token_ids = torch.tensor(
            target.positive_token_ids,
            device=logits_f32.device,
            dtype=torch.long,
        )
        token_weights = torch.tensor(
            target.source_weights,
            device=logits_f32.device,
            dtype=torch.float32,
        )
        token_weights = token_weights / token_weights.sum().clamp_min(1e-8)
        log_probs = torch.log_softmax(logits_f32[row_index, target.position], dim=-1)
        positive_log_prob = torch.logsumexp(
            log_probs[token_ids] + torch.log(token_weights.clamp_min(1e-8)),
            dim=0,
        )
        loss_terms.append(-positive_log_prob)
        weights.append(torch.ones((), device=logits_f32.device, dtype=torch.float32))

if not loss_terms:
    zero = logits_f32.sum() * 0.0
    return Stage2TrieCELossResult(
        loss=zero,
        metrics=metric_sums,
        atom_losses={"trie_ce": zero},
    )

stacked_losses = torch.stack(loss_terms)
stacked_weights = torch.stack(weights)
loss = (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-8)

if active_segments > 0:
    metric_sums["stage2_trie/candidate_count_mean"] /= float(active_segments)
metric_sums["stage2_trie/fallback_dominance_warning"] = float(
    metric_sums["stage2_trie/fallback_candidate_share"] > 0.35
)

return Stage2TrieCELossResult(
    loss=loss,
    metrics=metric_sums | {"loss/B/stage2_trie_ce": float(loss.detach().cpu())},
    atom_losses={"trie_ce": loss},
)
```

Adapt row indexing if current teacher-forcing metadata uses packed segment views. The invariant is that `target.position` must index the logits row used by the segment.

- [ ] **Step 4: Run numerical tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_ce_module.py -q
```

Expected: all Stage-2 trie CE module tests pass.

## Task 5: Attach Trie Sidecars In Channel-B Target Builder

**Files:**

- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel/types.py`
- Test: `tests/test_stage2_ab_training.py`
- Test: `tests/test_stage2_trie_supervision.py`

- [ ] **Step 1: Inspect Channel-B target builder return shape**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
rg -n "_build_channel_b_supervision_targets|_ChannelBSupervisionTargets|fallback_gt_fn_append_only|insertion_order|fn_slot_shuffle|tail_append|sorted|clean_target_text|y_train_ids" src/trainers/stage2_two_channel/target_builder.py tests/test_stage2_ab_training.py
```

Expected: command prints the target-builder functions and current insertion/fallback tests.

- [ ] **Step 2: Add failing test for Channel-B trie metadata presence**

Add to `tests/test_stage2_ab_training.py` near current Channel-B target-builder tests:

```python
def test_channel_b_target_builder_attaches_stage2_trie_sidecar() -> None:
    result = _build_channel_b_supervision_targets(
        sample_id="img-1",
        rollout_index=0,
        rollout_context="valid_rollout",
        y_train_ids=[101, 11, 12, 102],
        clean_target_text="<|object_ref_start|> cat <|box_start|>",
        fallback_loss_weight=0.25,
        insertion_order="tail_append",
        fp_policy={
            "mode": "zero_loss_context",
            "weak_positive_weight": 0.05,
            "require_explorer_support": True,
            "min_support_count": 1,
            "require_token_score": False,
        },
    )

    assert result.stage2_trie_targets.summary.candidate_count == 1
    assert result.stage2_trie_targets.token_targets
    assert result.stage2_trie_candidate_summary["candidate_count"] == 1
```

If `_build_channel_b_supervision_targets` is not directly callable with these keyword names, create the smallest existing fixture-driven call that reaches the same builder and assert on the returned metadata. Keep the assertion names exact: `stage2_trie_targets` and `stage2_trie_candidate_summary`.

- [ ] **Step 3: Run test and confirm failure**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_ab_training.py::test_channel_b_target_builder_attaches_stage2_trie_sidecar -q
```

Expected: failure because no sidecar exists yet.

- [ ] **Step 4: Extend target result structure**

In `src/trainers/stage2_two_channel/target_builder.py`, extend `_ChannelBSupervisionTargets` with:

```python
stage2_trie_targets: Stage2TrieTargets | None = None
stage2_trie_span_scores: list[dict[str, Any]] = field(default_factory=list)
stage2_trie_candidate_summary: dict[str, Any] = field(default_factory=dict)
```

Import the required dataclasses from `src.trainers.stage2_two_channel.trie_supervision`.

- [ ] **Step 5: Compile one-candidate trie sidecar for the current repaired target**

Inside `_build_channel_b_supervision_targets`, after `y_train_ids` and object span groups are finalized, construct:

```python
candidate_source = (
    "fallback_gt_fn_append_only"
    if rollout_context == "fallback_gt_fn_append_only"
    else "valid_rollout"
)
candidate_loss_weight = (
    fallback_loss_weight
    if candidate_source == "fallback_gt_fn_append_only"
    else 1.0
)
candidate = Stage2TrieCandidate(
    sample_id=str(sample_id),
    rollout_index=int(rollout_index),
    source=candidate_source,
    token_ids=list(y_train_ids),
    loss_weight=float(candidate_loss_weight),
    object_spans=build_stage2_trie_object_spans_from_channel_b_groups(...),
)
stage2_trie_targets = compile_stage2_trie_targets([candidate], segment_start=0)
```

The helper `build_stage2_trie_object_spans_from_channel_b_groups` should map existing prefix/tail description and coordinate groups into roles:

```python
def build_stage2_trie_object_spans_from_channel_b_groups(
    *,
    rollout_context: str,
    prefix_desc_groups: Sequence[Any],
    tail_desc_groups: Sequence[Any],
    prefix_coord_groups: Sequence[Any],
    tail_coord_groups: Sequence[Any],
    fp_policy_mode: str,
    fallback_loss_weight: float,
) -> list[Stage2TrieObjectSpan]:
    """Map Channel-B object groups into Stage-2 trie diagnostic spans."""

    spans: list[Stage2TrieObjectSpan] = []
    if rollout_context == "fallback_gt_fn_append_only":
        role = "fallback_fn"
        loss_weight = fallback_loss_weight
    else:
        role = "matched_clean"
        loss_weight = 1.0
    for group in list(prefix_desc_groups) + list(tail_desc_groups):
        spans.append(
            Stage2TrieObjectSpan(
                role=role,
                token_start=int(group.start),
                token_end=int(group.end),
                object_iou=getattr(group, "iou", None),
                support_count=int(getattr(group, "support_count", 0)),
                loss_weight=float(loss_weight),
            )
        )
    return spans
```

Adapt group attribute names to current dataclasses. Use explicit conversion and fail fast if a required token span is absent.

- [ ] **Step 6: Attach sidecar to metadata**

Where the trainer builds per-segment metadata for Channel-B, add:

```python
metadata["stage2_trie_targets"] = targets.stage2_trie_targets
metadata["stage2_trie_span_scores"] = [
    record.__dict__ for record in targets.stage2_trie_targets.span_score_records
]
metadata["stage2_trie_candidate_summary"] = dataclasses.asdict(
    targets.stage2_trie_targets.summary
)
```

Use `dataclasses.asdict` for dataclasses. Keep sidecars out of model-forward tensors.

- [ ] **Step 7: Run Channel-B sidecar tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_ab_training.py::test_channel_b_target_builder_attaches_stage2_trie_sidecar tests/test_stage2_trie_supervision.py -q
```

Expected: tests pass.

## Task 6: Merge K Rollout Candidates Into One Per-Example Trie

**Files:**

- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel/trie_supervision.py`
- Test: `tests/test_stage2_trie_supervision.py`
- Test: `tests/test_stage2_ab_training.py`

- [ ] **Step 1: Add failing test for K candidate merge**

Append to `tests/test_stage2_ab_training.py`:

```python
def test_channel_b_stage2_trie_merges_k_rollout_candidates_per_example() -> None:
    candidates = [
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=0,
            source="valid_rollout",
            token_ids=[101, 11, 12, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=1,
            source="valid_rollout",
            token_ids=[101, 11, 13, 102],
            loss_weight=1.0,
            object_spans=[],
        ),
        Stage2TrieCandidate(
            sample_id="img-1",
            rollout_index=-1,
            source="fallback_gt_fn_append_only",
            token_ids=[101, 21, 102],
            loss_weight=0.25,
            object_spans=[],
        ),
    ]

    targets = compile_stage2_trie_targets(candidates, segment_start=0)

    assert targets.summary.candidate_count == 3
    assert targets.summary.fallback_candidate_count == 1
    assert targets.summary.branch_points >= 1
```

- [ ] **Step 2: Run test and confirm current behavior**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_ab_training.py::test_channel_b_stage2_trie_merges_k_rollout_candidates_per_example -q
```

Expected: the test passes if Task 3 compiler is already generic. If it fails due to import placement, fix imports in the test file only.

- [ ] **Step 3: Build candidate aggregation seam**

In `src/trainers/stage2_two_channel/target_builder.py`, add a small function:

```python
def compile_stage2_trie_targets_for_rollout_group(
    candidates: Sequence[Stage2TrieCandidate],
    *,
    segment_start: int,
) -> Stage2TrieTargets:
    """Compile all repaired candidates for one original Stage-2 example."""

    if not candidates:
        raise ValueError("Stage-2 trie requires at least one repaired candidate")
    sample_ids = {candidate.sample_id for candidate in candidates}
    if len(sample_ids) != 1:
        raise ValueError(f"Stage-2 trie candidates must share one sample_id, got {sample_ids}")
    return compile_stage2_trie_targets(list(candidates), segment_start=segment_start)
```

- [ ] **Step 4: Route K rollouts into the aggregation seam**

Find the code that iterates over Channel-B anchor and explorer rollouts. Replace any per-rollout-only sidecar attachment with:

```python
grouped_candidates[sample_id].append(stage2_trie_candidate)
```

After all K rollouts and fallback candidates for the sample are known, compile one trie and attach it to the segment metadata for that sample.

If the current training batch still materializes one Channel-B segment per rollout instead of per original sample, keep the segment shape unchanged for now but attach the same merged trie target to the primary Channel-B segment and mark non-primary sibling segments with:

```python
metadata["stage2_trie_skip_loss"] = True
```

The objective module must ignore `stage2_trie_skip_loss`.

- [ ] **Step 5: Add objective skip handling**

In `compute_stage2_trie_ce_loss`, before reading targets:

```python
if item.get("stage2_trie_skip_loss"):
    continue
```

- [ ] **Step 6: Run K aggregation tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_supervision.py tests/test_stage2_ab_training.py -k "stage2_trie or channel_b" -q
```

Expected: Stage-2 trie and relevant Channel-B tests pass.

## Task 7: Implement FP Policy Semantics

**Files:**

- Modify: `src/trainers/stage2_two_channel/trie_supervision.py`
- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Test: `tests/test_stage2_trie_supervision.py`
- Test: `tests/test_stage2_ab_training.py`

- [ ] **Step 1: Add failing test for zero-loss FP context**

Append to `tests/test_stage2_trie_supervision.py`:

```python
from src.trainers.stage2_two_channel.trie_supervision import build_fp_object_span


def test_zero_loss_fp_context_records_span_without_positive_weight() -> None:
    span = build_fp_object_span(
        token_start=3,
        token_end=7,
        policy_mode="zero_loss_context",
        support_count=3,
        weak_positive_weight=0.05,
    )

    assert span.role == "neutral_fp"
    assert span.loss_weight == 0.0
    assert span.support_count == 3
```

- [ ] **Step 2: Add failing test for weak-positive FP context**

Append:

```python
def test_weak_positive_fp_requires_support() -> None:
    unsupported = build_fp_object_span(
        token_start=3,
        token_end=7,
        policy_mode="weak_positive_context",
        support_count=0,
        weak_positive_weight=0.05,
    )
    supported = build_fp_object_span(
        token_start=3,
        token_end=7,
        policy_mode="weak_positive_context",
        support_count=2,
        weak_positive_weight=0.05,
    )

    assert unsupported.role == "neutral_fp"
    assert unsupported.loss_weight == 0.0
    assert supported.role == "weak_positive_fp"
    assert supported.loss_weight == 0.05
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_supervision.py -k "fp_context or weak_positive" -q
```

Expected: import fails because the helper does not exist.

- [ ] **Step 4: Implement FP helper**

Add to `src/trainers/stage2_two_channel/trie_supervision.py`:

```python
def build_fp_object_span(
    *,
    token_start: int,
    token_end: int,
    policy_mode: str,
    support_count: int,
    weak_positive_weight: float,
) -> Stage2TrieObjectSpan:
    """Create a Stage-2 trie diagnostic span for an unmatched prediction."""

    if policy_mode == "weak_positive_context" and support_count >= 1:
        return Stage2TrieObjectSpan(
            role="weak_positive_fp",
            token_start=token_start,
            token_end=token_end,
            object_iou=None,
            support_count=support_count,
            loss_weight=weak_positive_weight,
        )
    return Stage2TrieObjectSpan(
        role="neutral_fp",
        token_start=token_start,
        token_end=token_end,
        object_iou=None,
        support_count=support_count,
        loss_weight=0.0,
    )
```

- [ ] **Step 5: Use FP helper in target builder**

In the existing unmatched prediction handling path, attach `neutral_fp` or `weak_positive_fp` spans through `build_fp_object_span`. Weak-positive spans should affect candidate loss only for object-entry continuation tokens, not description or coordinate content.

If the current target builder cannot isolate object-entry tokens yet, apply the weak-positive weight to the full object span and add this metric:

```python
metadata["stage2_trie_weak_fp_span_level_fallback"] = True
```

This makes the approximation visible in diagnostics.

- [ ] **Step 6: Run FP tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_trie_supervision.py -k "fp_context or weak_positive" -q
```

Expected: FP policy tests pass.

## Task 8: Add Span Score Diagnostic Dump

**Files:**

- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `src/trainers/stage2_two_channel/trie_supervision.py`
- Test: `tests/test_stage2_ab_training.py`

- [ ] **Step 1: Add failing test for sidecar dump path records**

Add to `tests/test_stage2_ab_training.py`:

```python
def test_stage2_trie_span_score_records_are_json_serializable() -> None:
    record = Stage2TrieSpanScoreRecord(
        sample_id="img-1",
        rollout_index=0,
        candidate_source="valid_rollout",
        span_role="matched_clean",
        object_role="matched_clean",
        token_start=4,
        token_end=8,
        mean_token_logprob=-0.2,
        min_token_logprob=-0.5,
        object_iou=0.8,
        support_count=1,
        loss_weight=1.0,
    )

    payload = stage2_trie_span_score_record_to_json(record)

    assert payload["sample_id"] == "img-1"
    assert payload["span_role"] == "matched_clean"
    assert payload["mean_token_logprob"] == -0.2
```

- [ ] **Step 2: Implement JSON conversion helper**

Add to `src/trainers/stage2_two_channel/trie_supervision.py`:

```python
def stage2_trie_span_score_record_to_json(
    record: Stage2TrieSpanScoreRecord,
) -> dict[str, object]:
    """Convert a Stage-2 trie span-score record into JSON-safe data."""

    return {
        "sample_id": record.sample_id,
        "rollout_index": record.rollout_index,
        "candidate_source": record.candidate_source,
        "span_role": record.span_role,
        "object_role": record.object_role,
        "token_start": record.token_start,
        "token_end": record.token_end,
        "mean_token_logprob": record.mean_token_logprob,
        "min_token_logprob": record.min_token_logprob,
        "object_iou": record.object_iou,
        "support_count": record.support_count,
        "loss_weight": record.loss_weight,
    }
```

- [ ] **Step 3: Add trainer dump hook**

In `src/trainers/stage2_two_channel.py`, after Channel-B loss metrics are aggregated for the step, collect metadata entries:

```python
records: list[dict[str, object]] = []
for item in batch_metadata:
    for record in item.get("stage2_trie_span_scores", []):
        records.append(record)
```

Write them to:

```python
monitor_dumps/stage2_trie_span_scores/step_<global_step>.jsonl
```

Use the trainer's existing monitor-dump root helper if present. If there is already a JSONL dump helper for rollout artifacts, reuse it.

- [ ] **Step 4: Run span diagnostic tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest tests/test_stage2_ab_training.py -k "stage2_trie_span_score" -q
```

Expected: JSON conversion and trainer collection tests pass.

## Task 9: Add Stage-2 Trie Smoke Configs

**Files:**

- Create: `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_zero_fp.yaml`
- Create: `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml`
- Create: `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_tail_append_zero_fp.yaml`
- Create: `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml`

- [ ] **Step 1: Identify the closest existing config base**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
ls configs/stage2_two_channel/smoke/*overfit_train8*64steps*.yaml
ls configs/stage2_two_channel/smoke/*view_train64*128steps*.yaml
```

Expected: command shows the existing single-path CE ablation configs. Use the matching `tail_insert` or `fn_slot_shuffle` config as source shape.

- [ ] **Step 2: Create train8 tail-append zero-FP config**

Copy the nearest current train8 tail config into:

```text
configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_zero_fp.yaml
```

Set these YAML values:

```yaml
custom:
  stage2_ab:
    channel_b:
      insertion_order: tail_append
      fallback_loss_weight: 0.25
      triage_posterior:
        num_rollouts: 4
      fp_policy:
        mode: zero_loss_context
        weak_positive_weight: 0.05
        require_explorer_support: true
        min_support_count: 1
        require_token_score: false
    pipeline:
      objective:
        - name: token_ce
          enabled: true
          weight: 1.0
          channels: [A]
          application:
            preset: anchor_text_only
        - name: stage2_trie_ce
          enabled: true
          weight: 1.0
          channels: [B]
          application:
            preset: rollout_trie_hard_ce
          config:
            support_weight: 1.0
            balance_weight: 1.0
            struct_weight: 1.0
            desc_weight: 1.0
            coord_hard_ce_weight: 1.0
            eos_weight: 1.0
            normalization: token_mean
```

Keep dataset, checkpoint, optimizer, no-eval, and max-step values identical to the current train8 config.

- [ ] **Step 3: Create train8 fn-slot-shuffle zero-FP config**

Create:

```text
configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml
```

Use the same objective block as Step 2, but set:

```yaml
custom:
  stage2_ab:
    channel_b:
      insertion_order: fn_slot_shuffle
```

- [ ] **Step 4: Create train64 tail-append zero-FP config**

Create:

```text
configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_tail_append_zero_fp.yaml
```

Use the same objective and policy block as Step 2, with the train64 dataset and step settings from the existing train64 no-eval config.

- [ ] **Step 5: Create train64 fn-slot-shuffle zero-FP config**

Create:

```text
configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml
```

Use the same objective block as Step 4, but set:

```yaml
custom:
  stage2_ab:
    channel_b:
      insertion_order: fn_slot_shuffle
```

- [ ] **Step 6: Run config parse checks**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python src/sft.py --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_zero_fp.yaml --cfg-only
conda run -n ms python src/sft.py --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml --cfg-only
conda run -n ms python src/sft.py --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_tail_append_zero_fp.yaml --cfg-only
conda run -n ms python src/sft.py --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml --cfg-only
```

Expected: all four configs load, resolve `stage2_trie_ce`, and do not enable Channel-B `token_ce`.

## Task 10: Run Narrow Verification

**Files:**

- No source edits expected.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m pytest \
  tests/test_stage2_trie_supervision.py \
  tests/test_stage2_trie_ce_module.py \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_ab_training.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run import and syntax checks**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python -m py_compile \
  src/trainers/stage2_two_channel/trie_supervision.py \
  src/trainers/teacher_forcing/modules/stage2_trie_ce.py \
  src/trainers/stage2_two_channel/target_builder.py \
  src/trainers/stage2_two_channel.py \
  src/trainers/teacher_forcing/module_registry.py \
  src/trainers/teacher_forcing/objective_pipeline.py \
  src/config/schema.py
```

Expected: py-compile passes.

- [ ] **Step 3: Verify no Channel-B double supervision in configs**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
rg -n "stage2_trie_ce|token_ce|channels: \\[A, B\\]|channels: \\[B\\]" configs/stage2_two_channel/smoke src tests
```

Expected: trie configs show `token_ce` on Channel-A and `stage2_trie_ce` on Channel-B. No trie config enables `token_ce` for Channel-B.

## Task 11: Run Tiny Train-Side Overfit Ablations

**Files:**

- No source edits expected.

- [ ] **Step 1: Run train8 tail-append zero-FP**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python src/sft.py \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_zero_fp.yaml
```

Expected:

- run starts without config or import errors;
- resolved assignment is `greedy_iou`;
- rollout decode policy is unconstrained;
- `stage2_trie/target_positions` is greater than zero;
- `loss/B/stage2_trie_ce` is logged;
- `stage2_trie/fallback_candidate_share` is logged;
- `stage2_trie/fallback_loss_share` is logged as a temporary compatibility alias for the candidate-share proxy;
- no eval is launched.

- [ ] **Step 2: Run train8 fn-slot-shuffle zero-FP**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python src/sft.py \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml
```

Expected: same launch and metric expectations as Step 1, with `stage2_trie/insertion_fn_slot_shuffle_count` greater than zero.

- [ ] **Step 3: Summarize train8 memorization evidence**

Read the produced train metrics and record:

```text
artifact root:
config:
checkpoint:
dataset scope: train8
steps:
recall:
recall_eq_1_fraction:
F1:
invalid rollout rate:
empty rollout rate:
fallback loss share:
duplicate indicators:
loss/B/stage2_trie_ce trend:
```

Success target:

- recall greater than or equal to 0.95;
- recall-equals-one fraction greater than or equal to 0.875;
- fallback loss share below 0.35;
- invalid and empty rollout rates near zero or decreasing.

- [ ] **Step 4: Run train64 only after train8 starts correctly**

Run the two train64 configs:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python src/sft.py \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_tail_append_zero_fp.yaml
conda run -n ms python src/sft.py \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train64_noeval_128steps_stage2_trie_fn_slot_shuffle_zero_fp.yaml
```

Success target:

- recall greater than or equal to 0.85;
- recall-equals-one fraction greater than or equal to 0.70;
- fallback loss share below 0.35;
- invalid and empty rollout rates not increasing.

## Task 12: Decide Weak-Positive FP Ablation After Zero-FP Evidence

**Files:**

- Create only if zero-FP train8 starts cleanly: `configs/stage2_two_channel/smoke/*stage2_trie_*_weak_fp.yaml`

- [ ] **Step 1: Create weak-positive train8 config pair**

For each train8 zero-FP config, create a weak-FP sibling with:

```yaml
custom:
  stage2_ab:
    channel_b:
      fp_policy:
        mode: weak_positive_context
        weak_positive_weight: 0.05
        require_explorer_support: true
        min_support_count: 1
        require_token_score: false
```

Keep every other value identical to the zero-FP sibling.

- [ ] **Step 2: Run weak-positive train8 pair**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
conda run -n ms python src/sft.py \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_tail_append_weak_fp.yaml
conda run -n ms python src/sft.py \
  --config configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_overfit_train8_noeval_64steps_stage2_trie_fn_slot_shuffle_weak_fp.yaml
```

Expected: weak-positive metrics are nonzero only when explorer support exists.

- [ ] **Step 3: Compare against zero-FP**

Record:

```text
policy:
insertion order:
recall:
precision:
F1:
duplicate indicators:
weak FP count:
neutral FP count:
span score distribution:
fallback loss share:
```

Proceed to train64 weak-positive only if train8 weak-positive improves recall or recall-equals-one without raising duplicate or invalid counters.

## Completion Criteria

Implementation is ready for review when:

- targeted pytest and py-compile checks pass;
- four zero-FP Stage-2 trie configs parse;
- train8 tail-append and train8 fn-slot-shuffle launch successfully;
- Stage-2 trie metrics appear in logs;
- span score sidecar JSONL is created when records exist;
- no eval is required or launched for the tiny overfit baseline;
- the run summary records whether Stage-2 trie beats the current single-path CE train-side baseline.

## Review Checklist

Before claiming completion, inspect:

- `git diff -- src/trainers/stage2_two_channel src/trainers/teacher_forcing src/config tests configs/stage2_two_channel/smoke`
- `rg -n "stage2_trie_ce|fp_policy|fallback_loss_share|stage2_trie_span_scores" src tests configs`
- the latest train8 artifact metrics and monitor dumps.

Do not update stable docs or OpenSpec until the tiny-overfit evidence is summarized and the user approves promotion from Superpowers roadmap to stable contract.
