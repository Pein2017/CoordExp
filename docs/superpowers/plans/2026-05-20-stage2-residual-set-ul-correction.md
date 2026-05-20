# Stage-2 Residual-Set UL Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage-2 Channel-B residual-set self-prefix correction with strict unlabeled-object consensus mining, shared `TeacherForcingTargetIR`, and a new explicit `residual_set_correction` objective path.

**Architecture:** Keep the stable semantics in OpenSpec and implement the first code slice in the existing Stage-2 Channel-B and teacher-forcing pipeline. Add a compact residual state/event abstraction, compile correction events into shared `TeacherForcingTargetIR` atoms, and consume those atoms through a `residual_set_correction` module that reuses the shared teacher-forcing probability decomposition. Preserve hard SFT and current Stage-2 trie baselines as explicit comparators.

**Tech Stack:** Python dataclasses, PyTorch logits math, CoordExp compact-full Stage-2 AB trainer, shared `src/training/teacher_forcing` IR/probability helpers, YAML config schema, pytest under `conda run -n ms`.

---

Date: 2026-05-20

Status: proposal for user review. Do not implement until the user explicitly approves.

Worktree root:

```text
/data/CoordExp/.worktrees/unified-training-infra-refactor
```

Stable contract:

```text
openspec/changes/add-stage2-residual-set-ul-correction/
```

Decision note:

```text
progress/explorations/2026-05-20_stage2_residual_set_self_prefix_ul_redesign.md
```

## Execution Policy

- Work only under `/data/CoordExp/.worktrees/unified-training-infra-refactor`.
- Do not edit `/data/CoordExp` root `main` while executing this plan.
- Do not edit upstream Hugging Face or Qwen3-VL model files.
- Do not reintroduce duplicate loss, coordinate regression loss, bbox regression aux loss, geometry regularizers, or continuation-margin loss as active defaults.
- Keep `stage2_trie_ce` and hard SFT available as baselines, but make `residual_set_correction` an explicit opt-in path.
- Use `target_position` as the canonical label-token position and verify `logit_position + 1 == target_position` in tests and runtime guards.
- Use `<|im_end|>` as STOP/EOS and `<|endoftext|>` as padding only.
- Treat K rollouts as K independent self-prefix samples. Do not average K losses as pseudo-labels.
- Run narrow tests after each task group before touching the next group.

## Planned File Map

Create:

- `src/trainers/stage2_two_channel/residual_set.py`  
  Owns residual object records, active candidate state, `ValidAction`, `CorrectionEvent`, deterministic roll-in, exact coordinate transition rules, and event-to-IR atom construction helpers.

- `src/trainers/stage2_two_channel/ul_consensus.py`  
  Owns K-valid rollout eligibility, per-rollout pre-dedup, strict same-description complete-link UL clustering, cluster artifact row construction, and UL metrics.

- `src/trainers/teacher_forcing/modules/residual_set_correction.py`  
  Consumes residual-set target IR sidecars from Stage-2 metadata, delegates atom loss math to `src.training.teacher_forcing.probabilities.teacher_forcing_atom_loss`, validates logits rows, and emits residual-set metrics.

- `tests/test_stage2_residual_set_correction.py`  
  Consolidated unit tests for residual state transitions, correction events, deterministic roll-in, coordinate tail behavior, STOP exclusivity, and event-to-IR invariants.

- `tests/test_stage2_residual_ul_consensus.py`  
  Unit tests for UL eligibility, clustering, pre-dedup, promotion/rejection/quarantine, local `G*_k`, weights, and artifact rows.

- `tests/test_stage2_residual_set_loss_module.py`  
  Unit tests for `residual_set_correction` loss math, logits-position alignment, coverage-strength behavior, mixed labeled/UL support weight, and metric keys.

Modify:

- `src/trainers/stage2_two_channel/types.py`  
  Add residual-set metadata sidecar keys with minimal `NotRequired[...]` fields.

- `src/trainers/stage2_two_channel/target_builder.py`  
  Add residual-set Channel-B target construction branch selected by config. Keep legacy clean-prefix/trie target construction available for comparator configs.

- `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`  
  Add a bridge from residual `CorrectionEvent` atoms into shared `TeacherForcingTargetIR`; keep existing legacy coordinate-target adapter intact for baseline paths.

- `src/trainers/stage2_two_channel/objective_runner.py`  
  Let Stage-2 metrics pass through `stage2_ab/channel_b/residual_set/*` keys and preserve old `stage2_trie/*` metrics.

- `src/trainers/stage2_two_channel.py`  
  Aggregate residual-set counters and write `ul_clusters.jsonl` under the active monitor/debug/smoke artifact root.

- `src/bootstrap/stage2_policy_provenance.py`  
  Record residual-set objective id, base seed, roll-in policy, UL thresholds,
  STOP margin weight, and artifact policy in run metadata.

- `src/trainers/teacher_forcing/module_registry.py`  
  Register `residual_set_correction` as an explicit Stage-2 objective module with strict config keys and application preset `rollout_self_prefix`.

- `src/trainers/teacher_forcing/objective_pipeline.py`  
  Route `residual_set_correction` to the new module and keep registry coverage checks strict.

- `src/config/schema.py`  
  Add strict `residual_set_correction.config` schema, mutual exclusion with legacy Channel-B target modules, removed-loss rejection, `num_rollouts` ownership, base seed `17`, UL defaults, and STOP-margin default `0.0`.

- `tests/test_stage2_ab_config_contract.py`  
  Add config acceptance/rejection tests for residual-set config and legacy conflicts.

- `tests/test_stage2_ab_training.py`  
  Add trainer wiring tests for metadata propagation, residual-set metric aggregation, and artifact gating.

- `tests/test_teacher_forcing_objective_runner.py`  
  Extend shared IR tests only if residual-set loss reveals missing generic validation; do not make Stage-2 provenance logic leak into generic objective tests.

- `configs/stage2_two_channel/smoke/*.yaml`  
  Add at most two residual-set smoke leaves after unit tests pass. Prefer editing nearby smoke templates over creating many new YAMLs.

## Shared Test Helper Contract

Several tasks below use small local helpers to keep test intent readable. Add
the helpers in the test file that first uses them; do not create a separate
test utility module.

In `tests/test_stage2_residual_set_correction.py`, define these helpers before
the first test:

```python
def coord_token(bin_value: int) -> int:
    return 100 + int(bin_value)


def make_object(
    desc: str,
    bbox: tuple[int, int, int, int],
    *,
    object_id: str = "obj",
    provenance: str = "labeled_gt",
    loss_weight: float = 1.0,
) -> ResidualObject:
    return ResidualObject(
        object_id=object_id,
        desc=desc,
        desc_token_ids=tuple(ord(ch) for ch in desc),
        coord_token_ids=tuple(coord_token(v) for v in bbox),
        provenance=provenance,
        loss_weight=loss_weight,
    )


def make_state_for_objects(
    objects: list[ResidualObject],
    *,
    active_candidate_ids: set[str] | None = None,
) -> ResidualState:
    by_id = {obj.object_id: obj for obj in objects}
    remaining = frozenset(by_id)
    active = frozenset(active_candidate_ids) if active_candidate_ids is not None else remaining
    return ResidualState(
        emitted_object_ids=frozenset(),
        remaining_object_ids=remaining,
        active_candidate_ids=active,
        objects_by_id=by_id,
    )


def only_action(actions: tuple[ValidAction, ...], token_id: int) -> ValidAction:
    matches = [action for action in actions if action.token_id == token_id]
    assert len(matches) == 1
    return matches[0]


def valid_coord_actions(state: ResidualState, *, coord_role: str) -> tuple[ValidAction, ...]:
    return tuple(action for action in enumerate_valid_actions(state, slot=coord_role))


def apply_action(state: ResidualState, action: ValidAction) -> ResidualState:
    return transition_state(state, action)


def make_role_vocab(
    *,
    text_ids: set[int] | None = None,
    coord_ids: set[int] | None = None,
    schema_ids: set[int] | None = None,
    stop_id: int = 2,
) -> RoleVocab:
    return RoleVocab(
        text_token_ids=frozenset(text_ids or {101, 201}),
        coord_token_ids=frozenset(coord_ids or set(range(100, 1100))),
        schema_token_ids=frozenset(schema_ids or {11, 22}),
        stop_token_id=stop_id,
    )
```

After `CorrectionEvent` exists, add this helper in the same file:

```python
def make_event(
    *,
    anchor_position: int = 1,
    target_position: int = 2,
    logit_position: int | None = None,
    observed_token_id: int | None = None,
    valid_token_ids: set[int] | None = None,
    selected_token_id: int | None = None,
    role: TokenRole = TokenRole.TEXT,
) -> CorrectionEvent:
    valid_ids = valid_token_ids or {selected_token_id or 101}
    actions = tuple(
        ValidAction(
            token_id=token_id,
            token_text=None,
            role=role,
            candidate_ids_before=frozenset({"a"}),
            candidate_ids_after=frozenset({"a"}),
            selected_object_id="a",
        )
        for token_id in sorted(valid_ids)
    )
    target = target_position
    logit = target_position - 1 if logit_position is None else logit_position
    return CorrectionEvent(
        kind="transition_failure",
        sample_id="sample-1",
        rollout_index=0,
        anchor_position=anchor_position,
        observed_token_id=observed_token_id,
        atom_drafts=(
            CorrectionAtomDraft(
                target_position=target,
                logit_position=logit,
                valid_actions=actions,
            ),
        ),
        state_before=make_state_for_objects([make_object("person", (1, 2, 3, 4), object_id="a")]),
    )
```

In `tests/test_stage2_residual_ul_consensus.py`, define local helpers that wrap
the public dataclasses from `ul_consensus.py`:

```python
def make_unmatched(
    desc: str,
    bbox: tuple[int, int, int, int],
    *,
    local_index: int = 0,
) -> ULMember:
    return ULMember(
        rollout_id="",
        local_index=local_index,
        desc_id=desc,
        desc_text=desc,
        bbox_norm1000=bbox,
    )


def make_valid_rollout(rollout_id: str, members: list[ULMember]) -> ULRolloutEvidence:
    return ULRolloutEvidence(
        rollout_id=rollout_id,
        is_valid=True,
        skip_reason=None,
        unmatched_members=tuple(
            replace(member, rollout_id=rollout_id) for member in members
        ),
    )


def make_invalid_rollout(rollout_id: str, *, reason: str) -> ULRolloutEvidence:
    return ULRolloutEvidence(
        rollout_id=rollout_id,
        is_valid=False,
        skip_reason=reason,
        unmatched_members=(),
    )
```

In `tests/test_stage2_residual_set_loss_module.py`, define local helpers that
construct real pipeline contracts:

```python
def make_spec(*, coverage_strength: float) -> PipelineModuleSpec:
    return PipelineModuleSpec.from_mapping(
        {
            "name": "residual_set_correction",
            "enabled": True,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": {"coverage_strength": coverage_strength},
        }
    )


def make_ir(
    *,
    valid_token_ids: set[int],
    selected_token_id: int,
    target_position: int = 1,
    logit_position: int = 0,
    coverage_target_weights: dict[int, float] | None = None,
    expected_loss_weight: float = 1.0,
    action_weights: dict[int, float] | None = None,
    support_provenance: str = "labeled_only",
) -> TeacherForcingTargetIR:
    if action_weights:
        assert set(action_weights).issubset(valid_token_ids)
        expected_loss_weight = max(float(value) for value in action_weights.values())
    atom = SupervisionAtom(
        batch_index=0,
        logit_position=logit_position,
        target_position=target_position,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset(valid_token_ids),
        selected_token_id=selected_token_id,
        latent_valid_token_ids=frozenset(valid_token_ids),
        coverage_target_weights=coverage_target_weights,
        loss_tags=frozenset({"stage2", "channel_b", "residual_set"}),
        loss_weight=expected_loss_weight,
        coord_role=None,
        provenance={
            "support_provenance": support_provenance,
            "action_weights": dict(action_weights or {}),
        },
    )
    return TeacherForcingTargetIR(schema_version=1, atoms=(atom,), metadata={})


def make_context(
    *,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    irs: tuple[TeacherForcingTargetIR, ...],
    text_ids: set[int],
) -> TeacherForcingContext:
    return TeacherForcingContext(
        channel="B",
        registry_context="rollout",
        input_ids=input_ids,
        logits=logits,
        logits_ce=logits,
        meta=({"stage2_channel": "B", "residual_set_target_ir": irs[0]},),
        coord_token_ids=tuple(range(100, 1100)),
        extra={"role_vocab": RoleVocab(text_token_ids=frozenset(text_ids), schema_token_ids=frozenset({3, 4}), coord_token_ids=frozenset(range(100, 1100)), stop_token_id=2)},
    )
```

In `tests/test_stage2_ab_training.py`, reuse existing Stage-2 fixtures when
available. If no local helper exists, define `make_gt` and `make_pred` using
`src.trainers.rollout_matching.contracts.GTObject` and `ParsedPredObject`; the
helper must construct the smallest valid object records needed by the test and
must not import production trainer internals just to create fixtures.

## Task 1: Config Contract And Registry Gate

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/trainers/teacher_forcing/module_registry.py`
- Modify: `src/trainers/teacher_forcing/objective_pipeline.py`
- Modify: `tests/test_stage2_ab_config_contract.py`

- [ ] **Step 1: Inspect current Stage-2 objective config owners**

Run:

```bash
cd /data/CoordExp/.worktrees/unified-training-infra-refactor
rg -n "stage2_trie_ce|token_ce|OBJECTIVE_MODULE_CATALOG|OBJECTIVE_CONFIG_ALLOWLIST|pipeline.objective|triage_posterior|pseudo_positive|duplicate_control|insertion_order" src/config/schema.py src/trainers/teacher_forcing tests/test_stage2_ab_config_contract.py
```

Expected: output shows `STAGE2_TRIE_CE_MODULE_NAME`, `OBJECTIVE_MODULE_CATALOG`, pipeline objective validation, and current Stage-2 config tests.

- [ ] **Step 2: Add failing acceptance test for residual-set config**

Add to `tests/test_stage2_ab_config_contract.py` near the Stage-2 objective validation tests:

```python
def test_stage2_pipeline_accepts_residual_set_correction_objective() -> None:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}
    raw["stage2_ab"]["channel_b"].pop("triage_posterior", None)

    prompts = ConfigLoader.resolve_prompts(raw)
    loaded = TrainingConfig.from_mapping(raw, prompts)

    objective = loaded.stage2_ab.pipeline.objective[1]
    assert objective.name == "residual_set_correction"
    assert objective.application["preset"] == "rollout_self_prefix"
    assert objective.config["base_seed"] == 17
    assert objective.config["num_rollouts"] == 3
    assert objective.config["lambda_ul_promoted"] == 0.5
```

Add `_residual_set_config()` beside the existing config helpers:

```python
def _residual_set_config() -> dict:
    return {
        "rollin_policy": "random_valid_branch",
        "rollin_resample_policy": "fixed_event",
        "base_seed": 17,
        "coord_span_policy": "bbox_tail_from_anchor",
        "strict_builder_invariants": True,
        "lambda_ul_promoted": 0.5,
        "lambda_continue_margin": 0.0,
        "continue_margin_m": 0.0,
        "coverage_strength": 0.0,
        "num_rollouts": 3,
        "min_ul_valid_rollouts": 3,
        "ul_consensus_ratio": 1.0,
        "ul_geometry": {
            "iou_min": 0.75,
            "center_distance_scale_max": 0.05,
            "area_ratio_max": 1.5,
            "aspect_ratio_max": 1.5,
            "consumed_overlap_iou_min": 0.75,
        },
        "artifact_policy": {"ul_clusters": "monitor_debug_smoke"},
    }
```

- [ ] **Step 3: Add failing rejection tests for residual-set conflicts**

Add:

```python
def test_residual_set_rejects_legacy_channel_b_trie_double_supervision() -> None:
    raw = _make_stage2_training_payload()
    trie_cfg = _stage2_pipeline_with_channel_b_trie_ce()["objective"][1]
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        trie_cfg,
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": _residual_set_config(),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}

    with pytest.raises(ValueError, match="residual_set_correction.*stage2_trie_ce"):
        TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))
```

Add a second rejection test that uses `pseudo_positive.enabled=true` with `residual_set_correction` and expects `ValueError` matching `pseudo_positive`.

The implementation must move the residual-set K validation to the cross-section
`Stage2ABConfig.from_mapping` level: non-residual configs keep the current
legacy rule that `pseudo_positive.enabled=false` implies
`stage2_ab.channel_b.triage_posterior.num_rollouts == 2`; residual-set configs
derive K from `residual_set_correction.config.num_rollouts` and may omit legacy
`triage_posterior`.

- [ ] **Step 4: Implement config and registry changes**

Add constants in `src/config/schema.py`:

```python
STAGE2_RESIDUAL_SET_MODULE_NAME = "residual_set_correction"
STAGE2_RESIDUAL_SET_APPLICATION_PRESETS: set[str] = {"rollout_self_prefix"}
STAGE2_RESIDUAL_SET_CONFIG_KEYS: set[str] = {
    "rollin_policy",
    "rollin_resample_policy",
    "base_seed",
    "coord_span_policy",
    "strict_builder_invariants",
    "lambda_ul_promoted",
    "lambda_continue_margin",
    "continue_margin_m",
    "coverage_strength",
    "num_rollouts",
    "min_ul_valid_rollouts",
    "ul_consensus_ratio",
    "ul_geometry",
    "artifact_policy",
}
```

Register in `src/trainers/teacher_forcing/module_registry.py`:

```python
"residual_set_correction": ObjectiveModuleDefinition(
    family="text",
    semantic_role="residual_set_correction",
    config_keys=frozenset({
        "rollin_policy",
        "rollin_resample_policy",
        "base_seed",
        "coord_span_policy",
        "strict_builder_invariants",
        "lambda_ul_promoted",
        "lambda_continue_margin",
        "continue_margin_m",
        "coverage_strength",
        "num_rollouts",
        "min_ul_valid_rollouts",
        "ul_consensus_ratio",
        "ul_geometry",
        "artifact_policy",
    }),
    application_presets=frozenset({"rollout_self_prefix"}),
    projected_atoms=(
        ObjectiveLossAtomDefinition(
            atom_name="residual_set",
            state_key="residual_set_correction_contrib",
        ),
    ),
    emission_group="text",
),
```

Route in `src/trainers/teacher_forcing/objective_pipeline.py`:

```python
from .modules import (
    run_residual_set_correction_module,
    run_stage2_trie_ce_module,
    run_token_ce_module,
)

objective_registry = {
    "token_ce": lambda spec: run_token_ce_module(context=context, spec=spec),
    "hard_sft": lambda spec: run_token_ce_module(context=context, spec=spec),
    "stage2_trie_ce": lambda spec: run_stage2_trie_ce_module(context=context, spec=spec),
    "residual_set_correction": lambda spec: run_residual_set_correction_module(
        context=context,
        spec=spec,
    ),
}
```

- [ ] **Step 5: Run config tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py -q
```

Expected: the new tests and existing Stage-2 config tests pass.

## Task 2: Residual State, ValidAction, And CorrectionEvent

**Files:**

- Create: `src/trainers/stage2_two_channel/residual_set.py`
- Modify: `src/trainers/stage2_two_channel/types.py`
- Test: `tests/test_stage2_residual_set_correction.py`

- [ ] **Step 1: Write failing tests for valid-action coalescing and STOP exclusivity**

Add:

```python
def test_shared_desc_prefix_coalesces_valid_action_by_token_id() -> None:
    state = make_state_for_objects(
        [
            make_object("person_left", (10, 20, 30, 40), object_id="a"),
            make_object("person_right", (50, 20, 70, 40), object_id="b"),
        ]
    )

    actions = state.valid_actions_at_boundary()

    person_actions = [a for a in actions if a.token_text == "person"]
    assert len(person_actions) == 1
    assert person_actions[0].candidate_ids_after == frozenset({"a", "b"})
    assert person_actions[0].selected_object_id is None


def test_stop_is_valid_only_when_residual_set_is_empty() -> None:
    nonempty = make_state_for_objects([make_object("person", (10, 20, 30, 40))])
    empty = make_state_for_objects([])

    assert all(action.role != TokenRole.STOP for action in nonempty.valid_actions_at_boundary())
    assert [action.role for action in empty.valid_actions_at_boundary()] == [TokenRole.STOP]
```

- [ ] **Step 2: Implement residual dataclasses and pure state transitions**

Create `src/trainers/stage2_two_channel/residual_set.py` with the public core:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from src.training.teacher_forcing.roles import TokenRole

CoordRole = Literal["x1", "y1", "x2", "y2"]
CorrectionKind = Literal[
    "transition_failure",
    "premature_stop",
    "fp_boundary",
    "repeated_object_boundary",
    "matched_object_repair",
]


@dataclass(frozen=True, slots=True)
class ResidualObject:
    object_id: str
    desc: str
    desc_token_ids: tuple[int, ...]
    coord_token_ids: tuple[int, int, int, int]
    provenance: Literal["labeled_gt", "ul_promoted_local"]
    loss_weight: float = 1.0
    source_index: int | None = None


@dataclass(frozen=True, slots=True)
class ValidAction:
    token_id: int
    token_text: str | None
    role: TokenRole
    candidate_ids_before: frozenset[str]
    candidate_ids_after: frozenset[str]
    selected_object_id: str | None = None
    coord_role: CoordRole | None = None
    loss_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ResidualState:
    emitted_object_ids: frozenset[str]
    remaining_object_ids: frozenset[str]
    active_candidate_ids: frozenset[str]
    objects_by_id: Mapping[str, ResidualObject]

    def valid_actions_at_boundary(self) -> tuple[ValidAction, ...]:
        return tuple(_valid_boundary_actions(self))


@dataclass(frozen=True, slots=True)
class CorrectionAtomDraft:
    target_position: int
    logit_position: int
    valid_actions: tuple[ValidAction, ...]
    state_before_slot: ResidualState | None = None
    coord_role: CoordRole | None = None


@dataclass(frozen=True, slots=True)
class CorrectionEvent:
    kind: CorrectionKind
    sample_id: str
    rollout_index: int
    anchor_position: int
    observed_token_id: int | None
    atom_drafts: tuple[CorrectionAtomDraft, ...]
    state_before: ResidualState
    provenance: Mapping[str, Any] = field(default_factory=dict)
```

Each `CorrectionEvent` may produce one or more atom drafts. Text/schema
corrections usually have one draft; `bbox_tail_from_anchor` coordinate repair
MUST emit one draft per supervised coordinate slot so `x1/y1/x2/y2` can each
carry its own valid set, selected token, commitment state, loss weight, and
position. Keep helper functions private in this module.

- [ ] **Step 3: Add transition tests for coordinate onset and bbox tail**

Add:

```python
def test_x1_ambiguity_filters_candidates_by_exact_coord_token() -> None:
    state = make_state_for_objects(
        [
            make_object("person", (120, 20, 300, 400), object_id="a"),
            make_object("person", (640, 22, 820, 410), object_id="b"),
            make_object("person", (850, 25, 940, 420), object_id="c"),
        ],
        active_candidate_ids={"a", "b", "c"},
    )

    actions = valid_coord_actions(state, coord_role="x1")

    assert {a.token_id for a in actions} == {coord_token(120), coord_token(640), coord_token(850)}
    chosen = apply_action(state, only_action(actions, coord_token(640)))
    assert chosen.active_candidate_ids == frozenset({"b"})


def test_shared_x1_keeps_bbox_tail_ambiguous_until_y1() -> None:
    state = make_state_for_objects(
        [
            make_object("person", (120, 20, 300, 400), object_id="a"),
            make_object("person", (120, 80, 310, 430), object_id="b"),
        ],
        active_candidate_ids={"a", "b"},
    )

    after_x1 = apply_action(state, only_action(valid_coord_actions(state, coord_role="x1"), coord_token(120)))
    assert after_x1.active_candidate_ids == frozenset({"a", "b"})

    y1_actions = valid_coord_actions(after_x1, coord_role="y1")
    assert {a.token_id for a in y1_actions} == {coord_token(20), coord_token(80)}
```

- [ ] **Step 4: Run residual state tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_set_correction.py -q
```

Expected: the residual state/action tests pass.

## Task 3: Event-To-IR Adapter And Logits Alignment Guard

**Files:**

- Modify: `src/trainers/stage2_two_channel/residual_set.py`
- Modify: `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`
- Test: `tests/test_stage2_residual_set_correction.py`
- Test: `tests/test_stage2_teacher_forcing_adapter_contract.py`

- [ ] **Step 1: Add failing test for event-to-IR alignment**

Add:

```python
def test_correction_event_to_ir_uses_next_token_logit_row() -> None:
    input_ids = torch.tensor([[11, 22, 101, 102, 103]])
    event = make_event(
        anchor_position=1,
        target_position=2,
        observed_token_id=999,
        valid_token_ids={101, 201},
        selected_token_id=101,
        role=TokenRole.TEXT,
    )

    ir = build_residual_set_target_ir(
        input_ids=input_ids,
        batch_index=0,
        events=(event,),
        role_vocab=make_role_vocab(text_ids={101, 201}),
    )

    atom = ir.atoms[0]
    assert atom.target_position == 2
    assert atom.logit_position == 1
    assert atom.selected_token_id == 101
    assert atom.valid_token_ids == frozenset({101, 201})
    assert atom.provenance["observed_token_id"] == 999
```

- [ ] **Step 2: Implement adapter function with hard validation**

Add `build_residual_set_target_ir(...)` in
`src/trainers/stage2_two_channel/teacher_forcing_adapter.py`. Keep
`residual_set.py` focused on state/events; the existing adapter owns Stage-2 to
shared-IR conversion.

```python
def build_residual_set_target_ir(
    *,
    input_ids: torch.Tensor,
    batch_index: int,
    events: Sequence[CorrectionEvent],
    role_vocab: RoleVocab,
) -> TeacherForcingTargetIR:
    atoms: list[SupervisionAtom] = []
    for event in events:
        for draft_index, draft in enumerate(event.atom_drafts):
            if draft.target_position != draft.logit_position + 1:
                raise ValueError("CorrectionAtomDraft target_position must equal logit_position + 1")
            if not draft.valid_actions:
                raise ValueError("CorrectionAtomDraft valid_actions must be nonempty")
            live_token_id = int(input_ids[batch_index, draft.target_position].item())
            valid_ids = frozenset(action.token_id for action in draft.valid_actions)
            if live_token_id not in valid_ids:
                raise ValueError("corrected roll-in selected token must be in valid actions")
            selected_action = next(action for action in draft.valid_actions if action.token_id == live_token_id)
            atom_weight = max(float(action.loss_weight) for action in draft.valid_actions)
            atoms.append(
                SupervisionAtom(
                    batch_index=batch_index,
                    logit_position=int(draft.logit_position),
                    target_position=int(draft.target_position),
                    allowed_token_roles=frozenset({selected_action.role}),
                    selected_token_role=selected_action.role,
                    valid_token_ids=valid_ids,
                    selected_token_id=live_token_id,
                    latent_valid_token_ids=valid_ids,
                    coverage_target_weights=None,
                    loss_tags=frozenset({"stage2", "channel_b", "residual_set"}),
                    loss_weight=atom_weight,
                    coord_role=selected_action.coord_role,
                    provenance=dict(event.provenance) | {
                        "stage": "stage2",
                        "channel": "B",
                        "correction_kind": event.kind,
                        "draft_index": draft_index,
                        "observed_token_id": event.observed_token_id,
                    },
                )
            )
    return TeacherForcingTargetIR(schema_version=1, atoms=tuple(atoms), metadata={"stage": "stage2", "stage2_channel": "B", "objective": "residual_set_correction"})
```

Use the project’s current `TEACHER_FORCING_TARGET_IR_SCHEMA_VERSION` constant instead of a literal `1` in implementation.

- [ ] **Step 3: Add tests for wrong live token and wrong shift**

Add:

```python
def test_event_to_ir_rejects_wrong_shift() -> None:
    event = make_event(anchor_position=1, target_position=3, logit_position=1)
    with pytest.raises(ValueError, match="target_position.*logit_position"):
        build_residual_set_target_ir(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(text_ids={4}),
        )


def test_event_to_ir_rejects_selected_token_outside_valid_actions() -> None:
    event = make_event(target_position=2, logit_position=1, valid_token_ids={7}, selected_token_id=7)
    with pytest.raises(ValueError, match="selected token.*valid actions"):
        build_residual_set_target_ir(
            input_ids=torch.tensor([[1, 2, 9]]),
            batch_index=0,
            events=(event,),
            role_vocab=make_role_vocab(text_ids={7, 9}),
        )
```

Add a coordinate-tail test that builds one event with four `atom_drafts` and
asserts that `build_residual_set_target_ir(...)` emits four atoms with
coordinate roles `x1/y1/x2/y2` and adjacent `logit_position + 1 ==
target_position` for every slot.

- [ ] **Step 4: Run adapter tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_set_correction.py tests/test_stage2_teacher_forcing_adapter_contract.py -q
```

Expected: all adapter and residual-set tests pass.

## Task 4: UL Consensus Mining And Artifacts

**Files:**

- Create: `src/trainers/stage2_two_channel/ul_consensus.py`
- Test: `tests/test_stage2_residual_ul_consensus.py`
- Modify: `src/trainers/stage2_two_channel/types.py`

- [ ] **Step 1: Add failing tests for K-valid consensus and per-rollout pre-dedup**

Add:

```python
def test_ul_consensus_uses_k_valid_denominator_and_promotes_ratio_one() -> None:
    rollouts = [
        make_valid_rollout("r0", [make_unmatched("person", (10, 10, 30, 30))]),
        make_valid_rollout("r1", [make_unmatched("person", (11, 10, 31, 30))]),
        make_valid_rollout("r2", [make_unmatched("person", (10, 11, 30, 31))]),
        make_invalid_rollout("r3", reason="parse_error"),
    ]

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=ULGeometryConfig(iou_min=0.75, center_distance_scale_max=0.05, area_ratio_max=1.5, aspect_ratio_max=1.5),
    )

    assert result.k_valid == 3
    assert result.skip_reasons["parse_error"] == 1
    assert len(result.promoted_clusters) == 1
    assert result.promoted_clusters[0].support_ratio == 1.0


def test_same_rollout_near_duplicates_contribute_one_vote() -> None:
    rollout = make_valid_rollout(
        "r0",
        [
            make_unmatched("person", (10, 10, 30, 30), local_index=0),
            make_unmatched("person", (11, 10, 31, 30), local_index=1),
        ],
    )

    result = mine_ul_consensus(
        [rollout, make_valid_rollout("r1", [make_unmatched("person", (12, 10, 32, 30))]), make_valid_rollout("r2", [make_unmatched("person", (10, 12, 30, 32))])],
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=ULGeometryConfig(iou_min=0.75, center_distance_scale_max=0.05, area_ratio_max=1.5, aspect_ratio_max=1.5),
    )

    promoted = result.promoted_clusters[0]
    assert promoted.support_rollout_ids == ("r0", "r1", "r2")
    assert promoted.members_by_rollout["r0"].local_index == 0
    assert result.duplicate_like_suppressed_count == 1


def test_cross_rollout_duplicate_burst_is_quarantined_by_consumed_overlap() -> None:
    rollouts = [
        make_valid_rollout("r0", [make_unmatched("person", (10, 10, 30, 30))]),
        make_valid_rollout("r1", [make_unmatched("person", (11, 10, 31, 30))]),
        make_valid_rollout("r2", [make_unmatched("person", (10, 11, 30, 31))]),
    ]
    consumed = [make_unmatched("person", (10, 10, 30, 30))]

    result = mine_ul_consensus(
        rollouts,
        min_ul_valid_rollouts=3,
        consensus_ratio=1.0,
        geometry=ULGeometryConfig(iou_min=0.75, center_distance_scale_max=0.05, area_ratio_max=1.5, aspect_ratio_max=1.5, consumed_overlap_iou_min=0.75),
        consumed_members=consumed,
    )

    assert result.promoted_clusters == ()
    assert len(result.quarantined_clusters) == 1
    assert result.quarantined_clusters[0].reason == "consumed_target_overlap"
```

- [ ] **Step 2: Implement UL clustering dataclasses and complete-link gate**

Use this public shape in `src/trainers/stage2_two_channel/ul_consensus.py`:

```python
@dataclass(frozen=True, slots=True)
class ULGeometryConfig:
    iou_min: float
    center_distance_scale_max: float
    area_ratio_max: float
    aspect_ratio_max: float
    consumed_overlap_iou_min: float


@dataclass(frozen=True, slots=True)
class ULConsensusCluster:
    desc_id: str
    desc_text: str
    support_rollout_ids: tuple[str, ...]
    support_ratio: float
    decision: Literal["promoted", "rejected", "quarantined"]
    reason: str
    members_by_rollout: Mapping[str, ULMember]
    pairwise_geometry: Mapping[str, float]
    consumed_overlap: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ULConsensusResult:
    k_valid: int
    skip_reasons: Mapping[str, int]
    promoted_clusters: tuple[ULConsensusCluster, ...]
    rejected_clusters: tuple[ULConsensusCluster, ...]
    quarantined_clusters: tuple[ULConsensusCluster, ...]
    duplicate_like_suppressed_count: int
```

Promotion rules:

- same canonical description id;
- one vote per rollout after local pre-dedup;
- `K_valid >= min_ul_valid_rollouts`;
- `support_rollouts == K_valid`;
- `support_ratio == 1.0`; the first implementation rejects authored
  `consensus_ratio != 1.0`;
- all-pairs complete-link geometry passes every configured threshold;
- same-description high-overlap with consumed labeled/UL/emitted objects
  quarantines or rejects the cluster before promotion.

- [ ] **Step 3: Add artifact row tests**

Add:

```python
def test_ul_cluster_artifact_rows_include_rejected_and_quarantined() -> None:
    result = make_consensus_result_with_all_decisions()

    rows = ul_cluster_artifact_rows(result, image_id="image-1")

    decisions = {row["decision"] for row in rows}
    assert decisions == {"promoted", "rejected", "quarantined"}
    assert all(row["image_id"] == "image-1" for row in rows)
    assert all("member_boxes" in row for row in rows)
    assert all("pairwise_geometry" in row for row in rows)
    assert all("consumed_overlap" in row for row in rows)
    assert all("support_ratio" in row for row in rows)
```

- [ ] **Step 4: Run UL tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_ul_consensus.py -q
```

Expected: all UL consensus and artifact-row tests pass.

## Task 5: Residual-Set Loss Module

**Files:**

- Create: `src/trainers/teacher_forcing/modules/residual_set_correction.py`
- Modify: `src/trainers/teacher_forcing/modules/__init__.py`
- Test: `tests/test_stage2_residual_set_loss_module.py`

- [ ] **Step 1: Add failing loss tests for valid-set marginal and coverage strength**

Add:

```python
def test_residual_set_module_uses_valid_set_marginal_not_selected_only() -> None:
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    logits = torch.full((1, 2, 8), -10.0)
    logits[0, 0, 1] = 0.0
    logits[0, 0, 2] = 0.0
    ir = make_ir(valid_token_ids={1, 2}, selected_token_id=1, target_position=1, logit_position=0)

    out = run_residual_set_correction_module(
        context=make_context(input_ids=input_ids, logits=logits, irs=(ir,), text_ids={1, 2}),
        spec=make_spec(coverage_strength=0.0),
    )

    assert float(out.loss) < 0.01
    assert out.metrics["stage2_ab/channel_b/residual_set/atom_count"] == 1.0


def test_residual_set_module_coverage_strength_zero_disables_coverage() -> None:
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    logits = torch.full((1, 2, 8), -10.0)
    logits[0, 0, 1] = 4.0
    logits[0, 0, 2] = 0.0
    ir = make_ir(valid_token_ids={1, 2}, selected_token_id=1, coverage_target_weights={1: 0.5, 2: 0.5})

    no_coverage = run_residual_set_correction_module(
        context=make_context(input_ids=input_ids, logits=logits, irs=(ir,), text_ids={1, 2}),
        spec=make_spec(coverage_strength=0.0),
    )
    with_coverage = run_residual_set_correction_module(
        context=make_context(input_ids=input_ids, logits=logits, irs=(ir,), text_ids={1, 2}),
        spec=make_spec(coverage_strength=1.0),
    )

    assert float(with_coverage.loss) > float(no_coverage.loss)
```

- [ ] **Step 2: Implement module by delegating atom math**

Use this execution contract:

```python
def run_residual_set_correction_module(
    *,
    context: TeacherForcingContext,
    spec: PipelineModuleSpec,
) -> ModuleResult:
    if str(context.channel).upper() != "B":
        return ModuleResult(loss=context.logits.float().sum() * 0.0, metrics={})
    config = build_residual_set_correction_config(spec.config)
    role_vocab = _role_vocab_from_context(context)
    loss_terms: list[torch.Tensor] = []
    component_totals: dict[str, torch.Tensor] = {}
    atom_count = 0
    for batch_index, segment_start, segment_end, segment_meta in iter_segment_views(input_ids=context.input_ids, meta=context.meta):
        target_ir = segment_meta.get("residual_set_target_ir")
        if target_ir is None:
            continue
        for atom in target_ir.atoms:
            _validate_residual_atom_positions(atom, segment_start=segment_start, segment_end=segment_end, batch_index=batch_index, input_ids=context.input_ids)
            row_logits = context.logits[atom.batch_index, atom.logit_position]
            atom_loss = teacher_forcing_atom_loss(
                row_logits,
                atom=atom,
                role_vocab=role_vocab,
                coverage_strength=config.coverage_strength,
            )
            loss_terms.append(atom_loss.total * float(atom.loss_weight))
            atom_count += 1
            _add(component_totals, "type", atom_loss.type)
            _add(component_totals, "valid", atom_loss.valid)
            _add(component_totals, "coverage", atom_loss.coverage)
    loss = _mean_or_zero(loss_terms, context.logits)
    return ModuleResult(
        loss=loss,
        metrics=_residual_metrics(loss=loss, atom_count=atom_count, component_totals=component_totals),
        state={"residual_set_correction_contrib": loss},
    )
```

The implementation must read full-rank `context.logits`, not `context.logits_ce[:, :-1, :]`, so row positions remain explicit and flash-attention-compatible.

- [ ] **Step 3: Add mixed labeled/UL weight test**

Add:

```python
def test_mixed_labeled_ul_support_uses_max_atom_loss_weight() -> None:
    ir = make_ir(
        valid_token_ids={1, 2},
        selected_token_id=1,
        action_weights={1: 1.0, 2: 0.5},
        support_provenance="mixed_labeled_ul",
    )

    atom = ir.atoms[0]
    assert atom.loss_weight == 1.0
    assert atom.provenance["support_provenance"] == "mixed_labeled_ul"
```

- [ ] **Step 4: Run loss module tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_set_loss_module.py tests/test_teacher_forcing_objective_runner.py -q
```

Expected: residual module tests pass and shared teacher-forcing objective tests remain green.

## Task 6: Stage-2 Target Builder Integration

**Files:**

- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel/types.py`
- Modify: `src/trainers/stage2_two_channel/objective_runner.py`
- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `src/bootstrap/stage2_policy_provenance.py`
- Test: `tests/test_stage2_ab_training.py`
- Test: `tests/test_stage2_residual_set_correction.py`

- [ ] **Step 1: Add failing integration test for residual path metadata**

Add to `tests/test_stage2_ab_training.py`:

```python
def test_channel_b_residual_set_path_attaches_target_ir_and_skips_legacy_trie() -> None:
    segment, meta, _length = build_channel_b_segment_with_objective(
        objective_name="residual_set_correction",
        rollout_objects=[make_pred("person", (10, 20, 30, 40))],
        gt_objects=[make_gt("person", (10, 20, 30, 40)), make_gt("cat", (50, 60, 70, 80))],
    )

    assert meta["stage2_channel"] == "B"
    assert "residual_set_target_ir" in meta
    assert "stage2_trie_targets" not in meta
    assert meta["residual_set_rollin_policy"] == "random_valid_branch"
    assert meta["residual_set_base_seed"] == 17
```

- [ ] **Step 2: Implement config-selected branch**

In `target_builder.py`, branch only when the active Channel-B objective list
selects `residual_set_correction`. Do not replace the current
`_build_channel_b_supervision_targets(...) -> _ChannelBSupervisionTargets`
return contract. Either extend `_ChannelBSupervisionTargets` with residual-set
fields or add a sibling residual builder that is called from the same layer that
currently creates `meta_entry`.

```python
if _uses_residual_set_correction(objective_specs):
    residual_result = build_residual_set_correction_targets(
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        gts=gts,
        rollout_group=rollout_group,
        config=residual_set_config,
    )
    supervision_targets = replace(
        supervision_targets,
        residual_set_target_ir=residual_result.target_ir,
        residual_set_event_summaries=residual_result.event_summaries,
        residual_set_metrics=residual_result.metrics,
    )
```

Then `_build_channel_b_meta_entry(...)` copies those residual fields into
`Stage2ChannelBMeta`. Keep the existing `stage2_trie_targets` construction in
the legacy branch.

- [ ] **Step 3: Add earliest-anchor tests**

Add tests covering:

- matched TP repair anchors before `object_start`, `desc_start`, or `box_start`;
- FN premature STOP anchors at the STOP target position;
- FP/duplicate boundary anchors before the next object boundary action;
- raw bad tokens are stored as provenance only and never as positive targets.

Each test must assert:

```python
for draft in event.atom_drafts:
    assert draft.logit_position + 1 == draft.target_position
    assert draft.valid_actions
    selected_token = corrected_input_ids[draft.target_position]
    assert selected_token in {action.token_id for action in draft.valid_actions}
assert event.observed_token_id == raw_bad_token_id
```

Do not assert that `observed_token_id` is outside valid actions. At object
boundaries a raw duplicate/FP token can be the same token id as a valid
remaining-object continuation; the correction is path/set based, not token-id
unlikelihood.

- [ ] **Step 4: Run integration tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_set_correction.py tests/test_stage2_ab_training.py -q
```

Expected: residual target-builder integration tests pass and existing Stage-2 trainer tests remain green.

## Task 7: Metrics, Artifacts, And Compatibility Guards

**Files:**

- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `src/trainers/stage2_two_channel/objective_runner.py`
- Test: `tests/test_stage2_ab_training.py`
- Test: `tests/test_stage2_residual_ul_consensus.py`

- [ ] **Step 1: Add failing metric prefix test**

Add:

```python
def test_residual_set_metrics_use_stable_prefixes() -> None:
    logs = build_stage2_core_loss_logs(
        channel="B",
        pipeline_metrics_ctx={
            "stage2_ab/channel_b/residual_set/atom_count": 2.0,
            "stage2_ab/channel_b/residual_set/ul/promoted_clusters": 1.0,
        },
        token_ce_module_w=0.0,
        run_a_text=False,
        token_desc_ce_weight=1.0,
        fn_desc_ce_weight=1.0,
    )

    assert logs["stage2_ab/channel_b/residual_set/atom_count"] == 2.0
    assert logs["stage2_ab/channel_b/residual_set/ul/promoted_clusters"] == 1.0


def test_residual_metric_pass_through_does_not_rewrite_legacy_trie_metrics() -> None:
    logs = build_stage2_core_loss_logs(
        channel="B",
        pipeline_metrics_ctx={
            "stage2_trie/target_positions": 3.0,
            "stage2_ab/channel_b/residual_set/atom_count": 2.0,
        },
        token_ce_module_w=0.0,
        run_a_text=False,
        token_desc_ce_weight=1.0,
        fn_desc_ce_weight=1.0,
    )

    assert logs["stage2_trie/target_positions"] == 3.0
    assert logs["stage2_ab/channel_b/residual_set/atom_count"] == 2.0
```

- [ ] **Step 2: Add artifact gate test**

Add:

```python
def test_ul_clusters_artifact_written_only_when_artifact_policy_enabled(tmp_path: Path) -> None:
    rows = [{"image_id": "image-1", "decision": "promoted", "member_boxes": []}]

    disabled = write_ul_clusters_artifact(tmp_path / "monitor_dumps" / "disabled", rows, enabled=False)
    enabled = write_ul_clusters_artifact(tmp_path / "monitor_dumps" / "enabled", rows, enabled=True)

    assert disabled is None
    assert not (tmp_path / "monitor_dumps" / "disabled" / "ul_clusters.jsonl").exists()
    assert enabled == tmp_path / "monitor_dumps" / "enabled" / "ul_clusters.jsonl"
    assert (tmp_path / "monitor_dumps" / "enabled" / "ul_clusters.jsonl").read_text().strip()
```

- [ ] **Step 3: Implement metric pass-through and artifact writer**

Rules:

- pass through keys starting with `stage2_ab/channel_b/residual_set/`;
- write `ul_clusters.jsonl` under the active Stage-2 monitor/debug/smoke artifact root only when artifact dumping is enabled. For train monitor dumps, use the existing `train_monitor_dump.out_dir` resolution and its default under `args.output_dir/monitor_dumps/...`; do not create an unrelated residual-set root;
- include promoted, rejected, and quarantined rows;
- do not count `ul_promoted_local` as labeled recall.

- [ ] **Step 4: Run metrics/artifact tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_ab_training.py tests/test_stage2_residual_ul_consensus.py -q
```

Expected: metric pass-through and artifact gating tests pass.

## Task 8: Smoke Configs And Runnable Checks

**Files:**

- Modify or create at most two files under `configs/stage2_two_channel/smoke/`
- Modify: `docs/IMPLEMENTATION_MAP.md` only after smoke path is runnable
- Modify: `docs/training/README.md` only after smoke path is runnable
- Modify: `progress/diagnostics/*.md` after smoke results exist

- [ ] **Step 1: Add compact residual smoke config by editing nearest template**

Preferred names:

```text
configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml
configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_thorough.yaml
```

The config must set:

```yaml
stage2_ab:
  schedule:
    b_ratio: 1.0
  pipeline:
    objective:
      - name: residual_set_correction
        enabled: true
        weight: 1.0
        channels: ["B"]
        application:
          preset: rollout_self_prefix
        config:
          rollin_policy: random_valid_branch
          rollin_resample_policy: fixed_event
          base_seed: 17
          coord_span_policy: bbox_tail_from_anchor
          strict_builder_invariants: true
          lambda_ul_promoted: 0.5
          lambda_continue_margin: 0.0
          continue_margin_m: 0.0
          coverage_strength: 0.0
          num_rollouts: 3
          min_ul_valid_rollouts: 3
          ul_consensus_ratio: 1.0
          ul_geometry:
            iou_min: 0.75
            center_distance_scale_max: 0.05
            area_ratio_max: 1.5
            aspect_ratio_max: 1.5
            consumed_overlap_iou_min: 0.75
          artifact_policy:
            ul_clusters: monitor_debug_smoke
```

Use the tuned A2-ET-RMP-CE checkpoint path already used by the current worktree
smoke configs, specifically the `checkpoint-3664` adapter in
`configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_1step.yaml`.
If the smoke keeps `b_ratio: 1.0`, the residual objective list may contain only
the B-side `residual_set_correction` entry. If any mixed A/B schedule is used,
the objective list must restate `token_ce` as `channels: ["A"]` plus
`residual_set_correction` as `channels: ["B"]` because YAML list inheritance is
replacement, not append.

- [ ] **Step 2: Run narrow unit suite before smoke**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_residual_set_correction.py \
  tests/test_stage2_residual_ul_consensus.py \
  tests/test_stage2_residual_set_loss_module.py \
  tests/test_stage2_ab_training.py \
  tests/test_stage2_teacher_forcing_adapter_contract.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run config parse check**

Run:

```bash
conda run -n ms python - <<'PY'
from pathlib import Path
from src.config.loader import ConfigLoader

path = "configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml"
cfg = ConfigLoader.load_materialized_training_config(path)
objective_names = [spec.name for spec in cfg.stage2_ab.pipeline.objective]
assert "residual_set_correction" in objective_names
assert all(spec.name not in {"bbox_geo", "coord_reg", "duplicate_unlikelihood"} for spec in cfg.stage2_ab.pipeline.objective)
for adapter in cfg.model.adapters:
    assert Path(adapter).exists(), adapter
print(path)
print(objective_names)
print(cfg.training.artifact_subdir)
PY
```

Expected: config resolves, checkpoint path exists, no removed-loss modules are selected, and artifact root is created only by the actual smoke run.

- [ ] **Step 4: Run Stage-2 one-step smoke**

Run the one-step smoke on the tuned checkpoint. Use up to 8 GPUs only if the existing launcher and config already support distributed Stage-2 smoke safely.

Expected evidence to record:

- config path;
- checkpoint path;
- seed `17`;
- output/artifact root;
- parse/drop counters;
- residual-set atom count;
- UL promoted/rejected/quarantined counters;
- STOP/continuation diagnostics;
- no `target_position/logit_position` invariant failures.

- [ ] **Step 5: Run thorough smoke after one-step passes**

Run the thorough config and compare against available hard SFT/current Stage-2 baseline artifacts. Treat performance as smoke evidence only unless the scope is at least val200.

Record:

- labeled recall;
- duplicate rate;
- missed object rate;
- object coherence rate;
- residual-set correction event distribution;
- UL cluster sample rows for visualization review;
- whether any performance drop correlates with residual event type, UL promotion, STOP handling, or coordinate onset.

## Task 9: Final Review Before Implementation Claim

**Files:**

- Modify: progress diagnostics note created during smoke
- Modify: implementation map/docs only after runnable evidence exists

- [ ] **Step 1: Run OpenSpec validation**

Run:

```bash
openspec validate add-stage2-residual-set-ul-correction --strict
```

Expected: `Change 'add-stage2-residual-set-ul-correction' is valid`.

- [ ] **Step 2: Run removed-mechanism guard tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_removed_training_mechanisms_absent.py tests/test_teacher_forcing_config_contract.py -q
```

Expected: removed duplicate/coord/bbox/geometry training mechanisms remain absent from active residual-set configs.

- [ ] **Step 3: Request subagent review**

Dispatch at least two read-only reviewers:

- algorithm/math reviewer: residual-set semantics, UL promotion safety, next-token alignment, coordinate commitment;
- pipeline/config reviewer: config strictness, baseline preservation, artifacts, smoke commands, docs route.

Each reviewer must return blockers first. Fix blockers before asking the user for implementation approval.

- [ ] **Step 4: Present final implementation readiness summary**

Report:

- OpenSpec validation status;
- super-power plan path;
- subagent review status and fixes;
- exact files expected to change during implementation;
- narrow tests and smoke commands that will be run first;
- unresolved risks, especially UL cluster false promotion and performance drop diagnosis.
