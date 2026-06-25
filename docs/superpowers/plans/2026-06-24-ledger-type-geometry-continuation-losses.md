# Ledger Type Geometry Continuation Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the review-selected mandatory type-family partition loss, trainable continuation-vs-stop loss, hard bbox positive-area penalty, and diagnostic span-drop salvage view to the active Stage-1 ledger hard-SFT pilot, pending explicit user implementation approval.

**Architecture:** Keep the existing `research_teacher_forcing` hard-SFT teacher-forcing objective as the owner of token-level loss math. Add explicit target-builder provenance for continuation boundaries and selected bbox geometry, pass simple term weights from `objective.terms`, and compute each auxiliary as an additive objective component with unsuffixed metric names. Keep ledger inference unchanged; the new losses train the same adapter/model path and require saved-adapter reload before benchmark interpretation.

**Tech Stack:** Python, PyTorch, pytest, CoordExp teacher-forcing target IR, CoordExp config schema, Stage-1 detection teacher-forcing smoke YAMLs, compact detection parsing/eval diagnostics.

---

## Approval Gate

This plan records the super-power roadmap for the next implementation process. It is not execution approval.

Do not edit code, edit configs, commit implementation changes, launch train128, or launch production training from this plan until explicitly assigned by the user after review convergence. Reviewer timeout, reviewer disconnection, or partial review is unresolved, not approval.

Before implementation:

- Bidirectional type gating is OpenSpec-promoted: create and validate `openspec/changes/bidirectional-type-gating-losses` for the four-family exclusive type objective, its config term, and metric semantics before code edits to that stable surface.
- Ledger mechanisms remain experiment-only: coverage ledger, continuation boundary loss, hard bbox positive-area penalty, saved-adapter train128 smoke, and diagnostic span salvage are research-only smoke implementation work, not stable contracts and not production eligible unless later promoted separately.

Worktree:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
git status --short --branch
```

Expected active branch:

```text
codex/ledger-auxiliary-loss
```

## Task 0: Review And Governance Gate

**Files:**
- Modify only if approved: `openspec/changes/bidirectional-type-gating-losses/proposal.md`
- Modify only if approved: `openspec/changes/bidirectional-type-gating-losses/specs/stage1-detection-objectives/spec.md`
- Modify only if approved: `openspec/changes/bidirectional-type-gating-losses/specs/teacher-forcing-unified-loss-registry/spec.md`
- Modify only if approved: `openspec/changes/bidirectional-type-gating-losses/specs/trainer-metrics-components/spec.md`
- Modify only if approved: `openspec/changes/bidirectional-type-gating-losses/tasks.md`
- Modify only if approved: `research/ideas/ledger-auxiliary-loss/discussion.md`

- [ ] **Step 1: Record review-convergence state**

Before any implementation task, record:

```text
roadmap review state: ready for user governance decision or hold
next decision: bidirectional type gating is OpenSpec-promoted; ledger mechanisms are experiment-only
review lanes: architecture/code-boundary, test/smoke, research/loss, docs/governance, adapter/eval-validity
unresolved P0/P1 findings: none
implementation approval: not granted until explicit user approval
train128 launch approval: not granted until explicit user approval
production approval: not granted
```

- [ ] **Step 2: Create focused OpenSpec change for type gating before code edits**

Create an OpenSpec change covering only the promoted bidirectional type-family gate:

```text
objective.terms.token_type_mass
four-family exclusive type objective over schema, coord, desc, stop
control/pad/excluded tokens outside the family denominator
type-gate raw mean and contribution metric semantics
```

Run:

```bash
openspec validate bidirectional-type-gating-losses --strict
```

Expected: the OpenSpec change validates before Tasks 1-4 begin.

Record the ledger-specific experiment-only scope in `research/ideas/ledger-auxiliary-loss/discussion.md`:

```text
stable contract path: bidirectional type-family gating loss
experiment-only path: coverage ledger, continuation, bbox positive-area penalty, saved-adapter train128 smoke, and diagnostic span salvage
production eligible: no
required follow-up before promotion: archive/sync the type-gating OpenSpec after implementation evidence; separately promote any non-type ledger mechanisms before production use
```

- [ ] **Step 3: Stop until approval**

Do not begin Task 1 until the user explicitly approves implementation after this review-convergence loop.

## Source Of Truth

- Research decisions: `research/ideas/ledger-auxiliary-loss/discussion.md`
- Existing ledger plan: `docs/superpowers/plans/2026-06-23-coverage-ledger-auxiliary-loss.md`
- Active smoke config: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`
- Comparator config: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml`
- Objective math: `src/training/teacher_forcing/probabilities.py`
- Objective runner: `src/training/objectives/teacher_forcing.py`
- Target builder: `src/detection/teacher_forcing/target_builder.py`
- Token branch metadata: `src/detection/teacher_forcing/trie.py`
- Config schema: `src/config/schema.py`
- Trainer objective bridge config handoff: `src/trainers/metrics/teacher_forcing.py`
- Runtime payload provenance: `src/sft.py`

## Review Convergence Notes

Round 1 used five independent read-only lanes:

```text
architecture/code-boundary
test/verification and train128 smoke
research/model-loss failure modes
docs/governance/reproducibility
adapter/inference compatibility and eval validity
```

Accepted P1 revisions now reflected in this plan:

```text
geometry uses explicit coord-bin-derived token IDs, not sorted token IDs
terminal STOP continuation compares against template-resolved continuation opener IDs
mandatory active v0 loss stack has a preflight launch blocker
auxiliary losses use term-specific denominators and explicit contribution metrics
train128 smoke is paired baseline-vs-ledger, not ledger-only benchmark evidence
inference uses YAML-first closed compact sorted config, not legacy flag defaults
selected train128 JSONL verifies source SHA and deterministic sample IDs
adapter checkpoint gate verifies adapter payload and token-embedding adapter metadata
compact_span_drop_salvage has a diagnostic materialization artifact
OpenSpec path or explicit experiment-only exception is required before implementation
```

Round 2 closure accepted additional P1 revisions:

```text
continuation STOP loss fails fast if boundary opener ids are missing
final optimization loss is the sum of per-term means, not all-atom averaged atom totals
research packet no longer contradicts the OpenSpec-vs-experiment-only governance gate
Task 0 closure state names the required user governance decision before implementation
baseline-vs-ledger preflight diff allowlist includes the new objective-term deltas
saved-adapter checkpoint gate calls the inference checkpoint resolver and compact token-row validator
continuation metric contract includes continue_minus_stop_margin and continue_accuracy
strict train128 evaluation scope means strict-template inference parser evidence plus raw f1ish diagnostic eval
```

No P0 findings were reported. Implementation remains unapproved until the user explicitly approves it after this convergence review and chooses the OpenSpec path or an explicit experiment-only exception.

## Non-Negotiable Contracts

- V0 scope is standard SFT-style hard teacher forcing: `objective.id: research_teacher_forcing`, `objective.profile: hard_sft`, sorted object order, `detection_template.id: compact_object_box_closed`.
- This does not target the older `objective.id: standard_ce` implementation surface.
- `token_type_mass` is mandatory in the active v0 smoke config, uses no `mode` key, and has `weight: 1.0`.
- `continuation_margin` is trainable in v0 and has `weight: 0.2`.
- `bbox_positive_area` is trainable in v0 and has `weight: 0.1`.
- Metric names do not get `_weighted` suffixes.
- The first config name should stay aligned with the existing `coverage_ledger_closed_hard_sft_128*` family. Do not create long stacked suffixes such as `coverage_ledger_closed_hard_sft_128_type_geom_cont.yaml`.
- Old configs should keep parsing during the first implementation pass unless they opt into the new active v0 loss stack.
- In this roadmap, `smoke run` means the `128`-sample overfitting trial: train on the selected 128 training examples, save an adapter checkpoint, reload that adapter through the inference engine, and evaluate on those same 128 training examples.
- The first smoke benchmark after implementation is a narrow paired comparison: baseline hard-SFT comparator versus ledger + mandatory type + continuation + geometry. Do not run the full ablation matrix.
- Any single-arm ledger run is launch-health only. Do not claim improvement, overfit quality, or production readiness without the paired baseline run on the same selected 128 rows.
- Production training is outside this smoke-focused plan. When the separate production launch config is prepared, it must reload the well-trained sorted pure-CE adapter, reduce LR settings below the previous `5.0e-5` production baseline, and train for `2` epochs instead of `4`.

## File Structure

- `src/config/schema.py`: add reusable weighted teacher-forcing term config and the `bbox_positive_area` term; allow hard-SFT to opt into the new v0 terms.
- `src/detection/teacher_forcing/trie.py`: add selected object bbox and coordinate-bin metadata to `TokenBranch` so loss metadata does not parse token strings.
- `src/detection/teacher_forcing/target_builder.py`: attach continuation-boundary, selected-bbox, and geometry valid/invalid token-id provenance to atoms.
- `src/training/teacher_forcing/probabilities.py`: compute direct valid-token NLL plus optional additive four-family type, continuation, geometry, and coverage terms with per-term denominators.
- `src/training/objectives/teacher_forcing.py`: read term weights from `ObjectiveSpec.config`, aggregate component totals, and emit metric events.
- `src/trainers/metrics/teacher_forcing.py`: pass term weights from the parsed objective config into `ObjectiveSpec.config`.
- `src/training/teacher_forcing/metrics.py`: add unsuffixed component metric helpers and required metric keys.
- `src/sft.py`: preserve term weights in runtime payload/provenance.
- `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`: turn on the approved term stack.
- `tests/test_teacher_forcing_objective_runner.py`: loss algebra, type-family, continuation, geometry tests.
- `tests/test_teacher_forcing_target_builder.py`: metadata/provenance tests.
- `tests/test_teacher_forcing_config_contract.py`: config schema and hard-SFT gating tests.
- `tests/test_teacher_forcing_metric_contract.py`: metric key and flattening tests.
- `src/detection/evaluation.py`: add the `compact_span_drop_salvage` diagnostic view after strict parsing, not inside strict parser behavior.
- `scripts/evaluation/materialize_compact_span_drop_salvage.py`: materialize diagnostic-only span-drop salvage metrics/counters from `gt_vs_pred.jsonl`.
- `tests/test_detection_template_parsing_eval.py`: salvage-view tests.

## Task 1: Config Schema And Smoke YAML

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/training/coverage_ledger/preflight.py`
- Modify: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`
- Modify: `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml`
- Modify: `tests/test_teacher_forcing_config_contract.py`

- [ ] **Step 1: Write failing config tests for weighted terms**

Add tests that load the active ledger smoke config and assert:

```python
def test_coverage_ledger_hard_sft_128_enables_v0_loss_stack() -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        "configs/stage1/detection_teacher_forcing/smoke/"
        "coverage_ledger_closed_hard_sft_128.yaml"
    )

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.terms.token_type_mass.enabled is True
    assert cfg.objective.terms.token_type_mass.weight == pytest.approx(1.0)
    assert cfg.objective.terms.continuation_margin.enabled is True
    assert cfg.objective.terms.continuation_margin.weight == pytest.approx(0.2)
    assert cfg.objective.terms.bbox_positive_area.enabled is True
    assert cfg.objective.terms.bbox_positive_area.weight == pytest.approx(0.1)

    assert cfg.training["save_strategy"] == "steps"
    assert cfg.training["save_steps"] == 128
    assert cfg.training["save_total_limit"] == 2
    assert cfg.training["save_last_epoch"] is True
    assert cfg.training["save_delay_steps"] == 0
```

Add a paired-smoke comparator test:

```python
def test_coverage_ledger_hard_sft_128_baseline_saves_adapter_but_keeps_aux_losses_disabled() -> None:
    cfg = ConfigLoader.load_materialized_training_config(
        "configs/stage1/detection_teacher_forcing/smoke/"
        "coverage_ledger_closed_hard_sft_128_baseline.yaml"
    )

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.terms.token_type_mass.enabled is False
    assert cfg.objective.terms.continuation_margin.enabled is False
    assert cfg.objective.terms.bbox_positive_area.enabled is False
    assert cfg.objective.terms.coverage_ledger.enabled is False
    assert cfg.training["save_strategy"] == "steps"
    assert cfg.training["save_steps"] == 128
    assert cfg.training["save_total_limit"] == 2
    assert cfg.training["save_last_epoch"] is True
    assert cfg.training["save_delay_steps"] == 0
```

Add an invalid-weight test:

```python
@pytest.mark.parametrize("term", ["token_type_mass", "continuation_margin", "bbox_positive_area"])
def test_teacher_forcing_weighted_terms_reject_negative_weight(term: str) -> None:
    payload = _coverage_ledger_payload()
    payload["objective"]["terms"][term] = {"enabled": True, "weight": -0.1}

    with pytest.raises(ValueError, match=rf"objective\.terms\.{term}\.weight"):
        DetectionTrainingConfig.from_mapping(payload)
```

Add an active-smoke mandatory-stack launch-blocker test. This is a config/preflight contract, not a new `mode` key:

```python
@pytest.mark.parametrize("term", ["token_type_mass", "continuation_margin", "bbox_positive_area"])
def test_active_ledger_smoke_rejects_disabled_required_v0_terms(term: str) -> None:
    payload = _coverage_ledger_payload()
    payload["objective"]["terms"][term]["enabled"] = False

    with pytest.raises(ValueError, match=rf"active ledger hard-SFT v0 requires .*{term}"):
        validate_active_ledger_hard_sft_v0_smoke(payload)
```

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py -q
```

Expected before implementation: tests fail because term configs do not expose `weight`, `bbox_positive_area` is unknown, or hard-SFT rejects enabled terms.

- [ ] **Step 2: Add a weighted term config dataclass**

In `src/config/schema.py`, add this near `TeacherForcingEnabledModuleConfig`:

```python
@dataclass(frozen=True)
class TeacherForcingWeightedModuleConfig:
    enabled: bool = False
    weight: float = 1.0

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.terms.*.enabled",
        )
        object.__setattr__(self, "weight", float(self.weight))
```

Do not emit wildcard error paths for weights. For concrete error paths, parse and validate each weighted term with a term-specific helper such as `parse_weighted_teacher_forcing_term(raw_term, path="objective.terms.token_type_mass")`, so failures name the concrete path such as `objective.terms.token_type_mass.weight`.

- [ ] **Step 3: Wire weighted terms into `TeacherForcingModulesConfig`**

Change these fields:

```python
token_type_mass: TeacherForcingWeightedModuleConfig = field(
    default_factory=TeacherForcingWeightedModuleConfig
)
continuation_margin: TeacherForcingWeightedModuleConfig = field(
    default_factory=TeacherForcingWeightedModuleConfig
)
bbox_positive_area: TeacherForcingWeightedModuleConfig = field(
    default_factory=lambda: TeacherForcingWeightedModuleConfig(weight=0.1)
)
```

In `from_mapping`, parse:

```python
bbox_positive_area = parse_dataclass_strict(
    TeacherForcingWeightedModuleConfig,
    data.pop("bbox_positive_area", {}),
    path="objective.terms.bbox_positive_area",
)
```

Return it in the dataclass constructor.

- [ ] **Step 4: Adjust hard-SFT gating**

In `TeacherForcingObjectiveConfig.__post_init__`, remove `token_type_mass.enabled` and `continuation_margin.enabled` from the legacy hard-SFT rejection list. Keep rejecting `conditional_valid_set_likelihood.enabled`, `within_valid_coverage.enabled`, and positive `coverage_strength` for hard-SFT.

Add explicit hard-SFT validation:

```python
if self.profile == "hard_sft":
    for module_key, cfg in (
        ("objective.terms.token_type_mass", self.terms.token_type_mass),
        ("objective.terms.continuation_margin", self.terms.continuation_margin),
        ("objective.terms.bbox_positive_area", self.terms.bbox_positive_area),
    ):
        if bool(cfg.enabled) and float(cfg.weight) <= 0.0:
            raise ValueError(f"{module_key}.weight must be > 0 when enabled")
```

- [ ] **Step 5: Add active v0 smoke-stack preflight validation**

In `src/training/coverage_ledger/preflight.py`, add a helper named exactly `validate_active_ledger_hard_sft_v0_smoke(config: Mapping[str, Any] | DetectionTrainingConfig) -> None`.

This helper is not a new config `mode`. It is a launch/preflight guard for the active checked-in train128 ledger smoke surface.

It must require all of these conditions when the config is the active ledger hard-SFT smoke surface:

```text
objective.profile == hard_sft
objective.terms.coverage_ledger.enabled is true
detection_template.id == compact_object_box_closed
debug.train_sample_limit == 128
objective.terms.token_type_mass.enabled is true and weight == 1.0
objective.terms.continuation_margin.enabled is true and weight == 0.2
objective.terms.bbox_positive_area.enabled is true and weight == 0.1
```

It must not reject `coverage_ledger_closed_hard_sft_128_baseline.yaml`, because that file is the explicit no-ledger/no-new-loss comparator.

Call this helper from the coverage-ledger preflight path before artifact materialization. Task 6 also runs a direct save/term contract check before launch.

Update the paired baseline-vs-ledger preflight diff allowlist so the exact intended v0 objective-term deltas are accepted and unrelated drift is still rejected:

```text
/objective/terms/token_type_mass/enabled
/objective/terms/token_type_mass/weight
/objective/terms/continuation_margin/enabled
/objective/terms/continuation_margin/weight
/objective/terms/bbox_positive_area/enabled
/objective/terms/bbox_positive_area/weight
```

Add or update preflight tests proving the active ledger-vs-baseline pair passes this allowlist and an unrelated objective/training/template change still fails.

- [ ] **Step 6: Update the active smoke YAML**

Modify `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`:

```yaml
objective:
  terms:
    token_type_mass:
      enabled: true
      weight: 1.0
    continuation_margin:
      enabled: true
      weight: 0.2
    bbox_positive_area:
      enabled: true
      weight: 0.1
training:
  save_strategy: steps
  save_steps: 128
  save_total_limit: 2
  save_last_epoch: true
  save_delay_steps: 0
```

Do not rename the file.

Modify `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml` only to enable adapter checkpoint saving for the paired smoke:

```yaml
training:
  save_strategy: steps
  save_steps: 128
  save_total_limit: 2
  save_last_epoch: true
  save_delay_steps: 0
```

Leave the baseline objective terms disabled so it remains the no-ledger/no-new-loss comparator.

- [ ] **Step 7: Run config tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py -q
```

Expected after implementation: config tests pass; older configs without `weight` still parse with defaults.

- [ ] **Step 8: Commit**

```bash
git add src/config/schema.py \
  src/training/coverage_ledger/preflight.py \
  configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
  tests/test_teacher_forcing_config_contract.py
git commit -m "feat: add weighted teacher-forcing v0 loss terms"
```

## Task 2: Target IR Metadata For Continuation And Geometry

**Files:**
- Modify: `src/detection/teacher_forcing/trie.py`
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Modify: `tests/test_teacher_forcing_target_builder.py`

- [ ] **Step 1: Write failing target-builder tests**

Add tests:

```python
def test_hard_sft_marks_continue_and_stop_boundaries() -> None:
    sample = _sample((
        _object("cat", (1, 2, 10, 20), index=0),
        _object("dog", (3, 4, 30, 40), index=1),
    ))
    result, tokenizer = _build(sample, profile="hard_sft", policy_name="sorted")

    continue_atoms = [
        atom for atom in result.target_ir.atoms
        if atom.provenance.get("continuation_boundary") is True
        and atom.provenance.get("continuation_target") == "continue"
    ]
    stop_atoms = [
        atom for atom in result.target_ir.atoms
        if atom.provenance.get("continuation_boundary") is True
        and atom.provenance.get("continuation_target") == "stop"
    ]

    assert len(continue_atoms) == 2
    assert len(stop_atoms) == 1
    assert continue_atoms[0].provenance["remaining_object_count"] == 2
    assert continue_atoms[1].provenance["remaining_object_count"] == 1
    assert stop_atoms[0].selected_token_role is TokenRole.STOP
    assert stop_atoms[0].provenance["remaining_object_count"] == 0
    assert tokenizer.token_id("<|im_end|>") == stop_atoms[0].provenance["stop_token_id"]
```

Add geometry metadata test:

```python
def test_coord_tail_atoms_carry_selected_bbox_for_geometry() -> None:
    sample = _sample((_object("cat", (1, 2, 10, 20), index=0),))
    result, _tokenizer = _build(sample, profile="hard_sft", policy_name="sorted")

    tail_atoms = [atom for atom in result.target_ir.atoms if atom.coord_role in {"x2", "y2"}]

    assert {atom.coord_role for atom in tail_atoms} == {"x2", "y2"}
    assert all(atom.provenance["selected_bbox_xyxy"] == (1, 2, 10, 20) for atom in tail_atoms)
    x2_atom = next(atom for atom in tail_atoms if atom.coord_role == "x2")
    y2_atom = next(atom for atom in tail_atoms if atom.coord_role == "y2")
    assert x2_atom.provenance["geometry_axis"] == "x"
    assert x2_atom.provenance["geometry_threshold_bin"] == 1
    assert x2_atom.provenance["bbox_positive_area_valid_token_ids"]
    assert x2_atom.provenance["bbox_positive_area_invalid_token_ids"]
    assert y2_atom.provenance["geometry_axis"] == "y"
    assert y2_atom.provenance["geometry_threshold_bin"] == 2
```

Run:

```bash
python -m pytest tests/test_teacher_forcing_target_builder.py -q
```

Expected before implementation: metadata assertions fail.

- [ ] **Step 2: Extend `TokenBranch` with bbox metadata**

In `src/detection/teacher_forcing/trie.py`, update the dataclass:

```python
@dataclass(frozen=True)
class TokenBranch:
    object_index: int
    object_instance_id: str
    token_ids: Sequence[int]
    token_roles: Sequence[TokenRole]
    coord_roles: Sequence[str | None]
    bbox_xyxy: tuple[int, int, int, int]
    coord_token_id_by_bin: Sequence[int]
```

Keep the existing length check. Add validation that `bbox_xyxy` has exactly four integers and `coord_token_id_by_bin` has exactly 1000 entries.

- [ ] **Step 3: Populate `bbox_xyxy` in `_prepare_object`**

In `src/detection/teacher_forcing/target_builder.py`, when constructing `TokenBranch`, add:

```python
bbox_xyxy=tuple(int(value) for value in obj.bbox_2d.values),
coord_token_id_by_bin=tuple(
    _single_token_id(tokenizer, f"<|coord_{index}|>") for index in range(1000)
),
```

- [ ] **Step 4: Add atom provenance**

In `_build_atoms`, before appending an object-token atom:

```python
provenance = {
    "object_index": selected_index,
    "object_instance_id": selected_branch.object_instance_id,
    "branch_position": branch_position,
    "candidate_object_indices": tuple(branch.object_index for branch in compatible),
}
coord_role = selected_branch.coord_role_at(branch_position)
if coord_role in {"x2", "y2"}:
    provenance["selected_bbox_xyxy"] = selected_branch.bbox_xyxy
    axis = "x" if coord_role == "x2" else "y"
    threshold_bin = selected_branch.bbox_xyxy[0] if coord_role == "x2" else selected_branch.bbox_xyxy[1]
    coord_ids = selected_branch.coord_token_id_by_bin
    provenance["geometry_axis"] = axis
    provenance["geometry_threshold_bin"] = int(threshold_bin)
    provenance["bbox_positive_area_valid_token_ids"] = tuple(coord_ids[int(threshold_bin) + 1 :])
    provenance["bbox_positive_area_invalid_token_ids"] = tuple(coord_ids[: int(threshold_bin) + 1])
if branch_position == 0:
    provenance.update(
        {
            "continuation_boundary": True,
            "continuation_target": "continue",
            "remaining_object_count": len(remaining),
            "continuation_token_ids": tuple(sorted(valid_token_ids)),
            "stop_token_id": int(stop_token_id),
        }
    )
```

Use `coord_role=coord_role` in the `SupervisionAtom`.

For the terminal stop atom, use:

```python
continuation_token_ids = tuple(
    sorted(next_token_ids_for_prefix(tuple(branches_by_index.values()), position=0))
)
provenance={
    "terminal": IM_END_TOKEN,
    "continuation_boundary": True,
    "continuation_target": "stop",
    "remaining_object_count": 0,
    "continuation_token_ids": continuation_token_ids,
    "stop_token_id": int(stop_token_id),
}
```

Resolve the continuation opener from the actual template/tokenized branches. Do not hard-code `<|object_ref_start|>` in loss math.

- [ ] **Step 5: Run target-builder tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_target_builder.py tests/test_teacher_forcing_ir_contract.py -q
```

Expected after implementation: metadata tests pass and IR validation still rejects unsupported mixed role sets.

- [ ] **Step 6: Commit**

```bash
git add src/detection/teacher_forcing/trie.py \
  src/detection/teacher_forcing/target_builder.py \
  tests/test_teacher_forcing_target_builder.py
git commit -m "feat: annotate teacher forcing atoms for v0 auxiliaries"
```

## Task 3: Additive Teacher-Forcing Loss Math

**Files:**
- Modify: `src/training/teacher_forcing/probabilities.py`
- Modify: `src/training/objectives/teacher_forcing.py`
- Modify: `tests/test_teacher_forcing_objective_runner.py`

- [ ] **Step 1: Extend test helper to pass weights**

In `tests/test_teacher_forcing_objective_runner.py`, extend `_run`:

```python
def _run(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    ir: TeacherForcingTargetIR,
    role_vocab: RoleVocab | None = None,
    coverage_strength: float = 0.0,
    token_type_mass_weight: float = 0.0,
    continuation_margin_weight: float = 0.0,
    bbox_positive_area_weight: float = 0.0,
    label_rows: LabelLogitRowMap | None = None,
) -> torch.Tensor:
```

Add those keys into the `ObjectiveSpec.config` dict.

- [ ] **Step 2: Write failing type-loss algebra test**

Add:

```python
def test_token_type_mass_adds_extra_family_pressure_beyond_valid_nll() -> None:
    logits = torch.tensor([[[0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                            [0.0] * 10]], dtype=torch.float32)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    role_vocab = _role_vocab(text_ids={1, 2, 3}, schema_ids={4, 5}, coord_ids={6, 7}, stop_id=9)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        token_type_mass_weight=1.0,
    )

    log_probs = torch.log_softmax(logits[0, 0], dim=-1)
    valid_nll = -log_probs[1]
    schema_mass = torch.logsumexp(log_probs.index_select(0, torch.tensor([4, 5])), dim=0)
    coord_mass = torch.logsumexp(log_probs.index_select(0, torch.tensor([6, 7])), dim=0)
    text_mass = torch.logsumexp(log_probs.index_select(0, torch.tensor([1, 2, 3])), dim=0)
    stop_mass = log_probs[9]
    type_nll = -torch.log_softmax(
        torch.stack([schema_mass, coord_mass, text_mass, stop_mass]),
        dim=0,
    )[2]
    assert actual.item() == pytest.approx((valid_nll + type_nll).item())
```

Expected before implementation: actual equals only valid NLL.

Add excluded-control regression:

```python
def test_token_type_mass_excludes_control_tokens_from_family_denominator() -> None:
    logits = torch.zeros((1, 2, 11), dtype=torch.float32)
    logits[0, 0, 1] = 2.0
    logits[0, 0, 10] = 20.0
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )

    actual = _run(
        logits=logits,
        input_ids=torch.tensor([[0, 1]], dtype=torch.long),
        ir=_ir(atom),
        role_vocab=_role_vocab(text_ids={1}, schema_ids={2}, coord_ids={3}, stop_id=9),
        token_type_mass_weight=1.0,
    )

    log_probs = torch.log_softmax(logits[0, 0], dim=-1)
    valid_nll = -log_probs[1]
    family_masses = torch.stack([log_probs[2], log_probs[3], log_probs[1], log_probs[9]])
    expected_type = -torch.log_softmax(family_masses, dim=0)[2]
    assert actual.item() == pytest.approx((valid_nll + expected_type).item())
```

- [ ] **Step 3: Write failing continuation and geometry tests**

Add continuation test:

```python
def test_continuation_margin_trains_continue_vs_stop_boundary() -> None:
    logits = torch.zeros((1, 2, 10), dtype=torch.float32)
    logits[0, 0, 4] = 1.0
    logits[0, 0, 9] = 3.0
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.SCHEMA}),
        selected_token_role=TokenRole.SCHEMA,
        valid_token_ids=frozenset({4}),
        selected_token_id=4,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "continue",
            "continuation_token_ids": (4,),
            "stop_token_id": 9,
            "remaining_object_count": 1,
        },
    )
    input_ids = torch.tensor([[0, 4]], dtype=torch.long)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=_role_vocab(schema_ids={4, 5}, stop_id=9),
        continuation_margin_weight=0.2,
    )

    valid_nll = -torch.log_softmax(logits[0, 0], dim=-1)[4]
    binary_logits = torch.stack([logits[0, 0, 4], logits[0, 0, 9]])
    continuation_nll = -torch.log_softmax(binary_logits, dim=0)[0]
    assert actual.item() == pytest.approx((valid_nll + 0.2 * continuation_nll).item())
```

Add terminal stop test:

```python
def test_continuation_margin_trains_terminal_stop_against_opener() -> None:
    logits = torch.zeros((1, 2, 10), dtype=torch.float32)
    logits[0, 0, 4] = 3.0
    logits[0, 0, 9] = 1.0
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.STOP}),
        selected_token_role=TokenRole.STOP,
        valid_token_ids=frozenset({9}),
        selected_token_id=9,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "stop",
            "continuation_token_ids": (4,),
            "stop_token_id": 9,
            "remaining_object_count": 0,
        },
    )
    input_ids = torch.tensor([[0, 9]], dtype=torch.long)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=_role_vocab(schema_ids={4, 5}, stop_id=9),
        continuation_margin_weight=0.2,
    )

    valid_nll = -torch.log_softmax(logits[0, 0], dim=-1)[9]
    binary_logits = torch.stack([logits[0, 0, 4], logits[0, 0, 9]])
    stop_nll = -torch.log_softmax(binary_logits, dim=0)[1]
    assert actual.item() == pytest.approx((valid_nll + 0.2 * stop_nll).item())
```

Add missing-opener regression:

```python
def test_continuation_margin_requires_nonempty_boundary_opener_ids() -> None:
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.STOP}),
        selected_token_role=TokenRole.STOP,
        valid_token_ids=frozenset({9}),
        selected_token_id=9,
        provenance={
            "continuation_boundary": True,
            "continuation_target": "stop",
            "continuation_token_ids": (),
            "stop_token_id": 9,
        },
    )

    with pytest.raises(ValueError, match="continuation_token_ids"):
        _run(
            logits=torch.zeros((1, 2, 10), dtype=torch.float32),
            input_ids=torch.tensor([[0, 9]], dtype=torch.long),
            ir=_ir(atom),
            role_vocab=_role_vocab(schema_ids={4}, stop_id=9),
            continuation_margin_weight=0.2,
        )
```

Add geometry test:

```python
def test_bbox_positive_area_penalizes_invalid_x2_mass() -> None:
    logits = torch.full((1, 2, 12), -5.0, dtype=torch.float32)
    logits[0, 0, 6] = 2.0
    logits[0, 0, 7] = 1.0
    logits[0, 0, 8] = 4.0
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({8}),
        selected_token_id=8,
        provenance={
            "selected_bbox_xyxy": (7, 1, 8, 9),
            "geometry_axis": "x",
            "geometry_threshold_bin": 7,
            "bbox_positive_area_valid_token_ids": (8, 9),
            "bbox_positive_area_invalid_token_ids": (6, 7),
        },
    )
    atom = replace(atom, coord_role="x2")
    input_ids = torch.tensor([[0, 8]], dtype=torch.long)
    role_vocab = _role_vocab(coord_ids={6, 7, 8, 9}, stop_id=10)

    actual = _run(
        logits=logits,
        input_ids=input_ids,
        ir=_ir(atom),
        role_vocab=role_vocab,
        bbox_positive_area_weight=0.1,
    )

    log_probs = torch.log_softmax(logits[0, 0], dim=-1)
    valid_nll = -log_probs[8]
    coord_log_probs = log_probs.index_select(0, torch.tensor([6, 7, 8, 9]))
    coord_log_probs = coord_log_probs - torch.logsumexp(coord_log_probs, dim=0)
    geometry_nll = -torch.logsumexp(coord_log_probs.index_select(0, torch.tensor([2, 3])), dim=0)
    assert actual.item() == pytest.approx((valid_nll + 0.1 * geometry_nll).item())
```

Add a nonmonotonic coord-id regression test:

```python
def test_bbox_positive_area_uses_explicit_coord_bins_not_sorted_token_ids() -> None:
    logits = torch.full((1, 2, 50), -5.0, dtype=torch.float32)
    logits[0, 0, 10] = 4.0
    logits[0, 0, 20] = 1.0
    logits[0, 0, 30] = 1.0
    logits[0, 0, 40] = 4.0
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.COORD}),
        selected_token_role=TokenRole.COORD,
        valid_token_ids=frozenset({40}),
        selected_token_id=40,
        provenance={
            "selected_bbox_xyxy": (1, 0, 3, 2),
            "geometry_axis": "x",
            "geometry_threshold_bin": 1,
            "bbox_positive_area_valid_token_ids": (40, 20),
            "bbox_positive_area_invalid_token_ids": (30, 10),
        },
    )
    atom = replace(atom, coord_role="x2")

    actual = _run(
        logits=logits,
        input_ids=torch.tensor([[0, 40]], dtype=torch.long),
        ir=_ir(atom),
        role_vocab=_role_vocab(coord_ids={10, 20, 30, 40}, stop_id=49),
        bbox_positive_area_weight=0.1,
    )

    log_probs = torch.log_softmax(logits[0, 0], dim=-1)
    valid_nll = -log_probs[40]
    coord_log_probs = log_probs.index_select(0, torch.tensor([10, 20, 30, 40]))
    coord_log_probs = coord_log_probs - torch.logsumexp(coord_log_probs, dim=0)
    expected_geometry = -torch.logsumexp(
        log_probs.index_select(0, torch.tensor([40, 20]))
        - torch.logsumexp(log_probs.index_select(0, torch.tensor([10, 20, 30, 40])), dim=0),
        dim=0,
    )
    assert actual.item() == pytest.approx((valid_nll + 0.1 * expected_geometry).item())
```

Import `replace` from `dataclasses` at the top of the test file.

Add multi-atom denominator regressions with concrete fixtures:

```text
test_continuation_margin_uses_boundary_denominator_not_all_atoms:
  fixture: three supervised atoms; exactly one atom has continuation_boundary=true
  expected: final objective contribution is continuation_margin_weight * that one boundary loss
  rejected behavior: continuation_margin_weight * that one boundary loss / 3

test_bbox_positive_area_uses_eligible_tail_denominator_not_all_atoms:
  fixture: four supervised atoms; exactly two atoms are x2/y2 geometry-eligible
  expected: final objective contribution is bbox_positive_area_weight * mean(two eligible geometry losses)
  rejected behavior: averaging the two geometry losses across all four supervised atoms
```

Add geometry partition-integrity regressions with concrete fixtures:

```text
test_bbox_positive_area_rejects_overlapping_valid_invalid_sets:
  fixture: valid set and invalid set share one coord token id
  expected: ValueError naming disjoint geometry valid/invalid ids

test_bbox_positive_area_rejects_partial_coord_partition:
  fixture: role_vocab.coord_token_ids has four ids but valid union invalid covers only three
  expected: ValueError naming coord-token partition

test_bbox_positive_area_requires_selected_tail_inside_valid_set:
  fixture: selected_token_id is in invalid ids for an x2/y2 atom
  expected: ValueError naming selected tail coord token
```

- [ ] **Step 4: Refactor atom-loss dataclass**

In `src/training/teacher_forcing/probabilities.py`, update `TeacherForcingAtomLoss`:

```python
@dataclass(frozen=True, slots=True)
class TeacherForcingAtomLoss:
    total: torch.Tensor
    valid: torch.Tensor
    token_type_mass: torch.Tensor
    continuation_margin: torch.Tensor
    bbox_positive_area: torch.Tensor
    coverage: torch.Tensor
    selected_type_probability: torch.Tensor
    valid_probability: torch.Tensor
    invalid_area_mass: torch.Tensor
```

`total` is a per-atom diagnostic/compatibility field only. The optimizer loss must be assembled in `src/training/objectives/teacher_forcing.py` from per-term means and must not average `atom_loss.total` across all atoms.

Keep compatibility only if needed by tests by assigning `type = token_type_mass` through a property:

```python
@property
def type(self) -> torch.Tensor:
    return self.token_type_mass
```

- [ ] **Step 5: Add helper functions**

Add helpers in `probabilities.py`:

```python
def _type_family_loss(
    log_probs: torch.Tensor,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
    *,
    device: torch.device,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    family_roles = (
        TokenRole.SCHEMA,
        TokenRole.COORD,
        TokenRole.TEXT,
        TokenRole.STOP,
    )
    family_log_masses = []
    selected_index = None
    for index, role in enumerate(family_roles):
        role_ids = _token_ids_tensor(
            role_vocab.token_ids_for_role(role),
            device=device,
            vocab_size=vocab_size,
            field_name=f"type_family.{role.value}",
        )
        family_log_masses.append(torch.logsumexp(log_probs.index_select(0, role_ids), dim=0))
        if role is atom.selected_token_role:
            selected_index = index
    if selected_index is None:
        raise ValueError("selected_token_role is not in the supervised type-family partition")
    family_logits = torch.stack(family_log_masses)
    family_log_probs = family_logits - torch.logsumexp(family_logits, dim=0)
    type_loss = -family_log_probs[selected_index]
    selected_probability = family_log_probs[selected_index].exp()
    return type_loss, selected_probability
```

This is a four-family partition over schema, coord, text, and stop. Control/pad/excluded tokens must not appear in the family denominator. The ordinary valid-token NLL still uses the full vocabulary log-softmax.

```python
def _continuation_loss(log_probs: torch.Tensor, atom: SupervisionAtom, *, device: torch.device, vocab_size: int) -> torch.Tensor:
    if not bool(atom.provenance.get("continuation_boundary", False)):
        return log_probs.new_tensor(0.0)
    stop_token_id = int(atom.provenance["stop_token_id"])
    stop_ids = _token_ids_tensor(frozenset({stop_token_id}), device=device, vocab_size=vocab_size, field_name="stop_token_id")
    stop_logit = log_probs.index_select(0, stop_ids)[0]
    target = atom.provenance.get("continuation_target")
    continue_token_ids = frozenset(int(i) for i in atom.provenance.get("continuation_token_ids", ()))
    if not continue_token_ids:
        raise ValueError("continuation_margin requires nonempty continuation_token_ids")
    continue_ids = _token_ids_tensor(continue_token_ids, device=device, vocab_size=vocab_size, field_name="continuation_token_ids")
    continue_logit = torch.logsumexp(log_probs.index_select(0, continue_ids), dim=0)
    if target == "stop":
        return -torch.log_softmax(torch.stack([continue_logit, stop_logit]), dim=0)[1]
    if target == "continue":
        return -torch.log_softmax(torch.stack([continue_logit, stop_logit]), dim=0)[0]
    raise ValueError("continuation boundary requires target 'continue' or 'stop'")
```

```python
def _bbox_positive_area_loss(log_probs: torch.Tensor, atom: SupervisionAtom, role_vocab: RoleVocab, *, device: torch.device, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if atom.coord_role not in {"x2", "y2"}:
        zero = log_probs.new_tensor(0.0)
        return zero, zero
    valid_ids_raw = frozenset(int(i) for i in atom.provenance.get("bbox_positive_area_valid_token_ids", ()))
    invalid_ids_raw = frozenset(int(i) for i in atom.provenance.get("bbox_positive_area_invalid_token_ids", ()))
    if not valid_ids_raw:
        raise ValueError("bbox_positive_area requires nonempty bbox_positive_area_valid_token_ids")
    if not invalid_ids_raw:
        raise ValueError("bbox_positive_area requires nonempty bbox_positive_area_invalid_token_ids")
    if not valid_ids_raw.issubset(role_vocab.coord_token_ids):
        raise ValueError("bbox_positive_area_valid_token_ids must be coord token ids")
    if not invalid_ids_raw.issubset(role_vocab.coord_token_ids):
        raise ValueError("bbox_positive_area_invalid_token_ids must be coord token ids")
    if valid_ids_raw & invalid_ids_raw:
        raise ValueError("bbox positive-area valid and invalid token ids must be disjoint")
    coord_ids_raw = frozenset(int(i) for i in role_vocab.coord_token_ids)
    if valid_ids_raw | invalid_ids_raw != coord_ids_raw:
        raise ValueError("bbox positive-area valid and invalid token ids must partition coord token ids")
    if int(atom.selected_token_id) not in valid_ids_raw:
        raise ValueError("selected tail coord token must be inside bbox_positive_area_valid_token_ids")
    coord_tensor = _token_ids_tensor(coord_ids_raw, device=device, vocab_size=vocab_size, field_name="geometry_coord_token_ids")
    valid_tensor = _token_ids_tensor(valid_ids_raw, device=device, vocab_size=vocab_size, field_name="bbox_positive_area_valid_token_ids")
    invalid_tensor = _token_ids_tensor(invalid_ids_raw, device=device, vocab_size=vocab_size, field_name="bbox_positive_area_invalid_token_ids")
    coord_log_probs = log_probs.index_select(0, coord_tensor)
    coord_denominator = torch.logsumexp(coord_log_probs, dim=0)
    valid_mass_loss = -(torch.logsumexp(log_probs.index_select(0, valid_tensor), dim=0) - coord_denominator)
    invalid_area_mass = (torch.logsumexp(log_probs.index_select(0, invalid_tensor), dim=0) - coord_denominator).exp()
    return valid_mass_loss, invalid_area_mass
```

Do not infer geometry bins from sorted token ids. The target builder owns tokenizer-aware coord-bin resolution and must provide explicit valid/invalid token-id sets.

- [ ] **Step 6: Update `teacher_forcing_atom_loss` signature**

Use:

```python
def teacher_forcing_atom_loss(
    logits_row: torch.Tensor,
    *,
    atom: SupervisionAtom,
    role_vocab: RoleVocab,
    coverage_strength: float,
) -> TeacherForcingAtomLoss:
```

Inside the fp32 block:

```python
log_probs = F.log_softmax(logits_fp32, dim=-1)
valid_log_probs = log_probs.index_select(dim=-1, index=valid_ids)
log_valid = torch.logsumexp(valid_log_probs, dim=-1)
valid_loss = -log_valid

type_loss, selected_type_probability = _type_family_loss(
    log_probs,
    atom,
    role_vocab,
    device=device,
    vocab_size=vocab_size,
)

continuation_loss = _continuation_loss(log_probs, atom, device=device, vocab_size=vocab_size)
geometry_loss, invalid_area_mass = _bbox_positive_area_loss(
    log_probs,
    atom,
    role_vocab,
    device=device,
    vocab_size=vocab_size,
)

total = valid_loss
```

Return raw component tensors. Preserve coverage behavior, but base coverage on `valid_log_probs - log_valid` as it already does. `TeacherForcingAtomLoss.total` is a per-atom diagnostic/compatibility field only; the optimizer loss must be assembled from per-term means in `src/training/objectives/teacher_forcing.py`.

- [ ] **Step 7: Pass weights through the objective**

In `src/training/objectives/teacher_forcing.py`, read config floats:

```python
token_type_mass_weight = config_float(spec.config, "token_type_mass_weight", default=0.0, minimum=0.0)
continuation_margin_weight = config_float(spec.config, "continuation_margin_weight", default=0.0, minimum=0.0)
bbox_positive_area_weight = config_float(spec.config, "bbox_positive_area_weight", default=0.0, minimum=0.0)
```

Use these weights only in the objective aggregation. Do not pass them into `teacher_forcing_atom_loss`; the atom helper returns raw component losses and metadata.

Update component totals and denominators. Do not average every auxiliary over all atoms:

```python
denominators = {
    "valid": 0,
    "token_type_mass": 0,
    "continuation_margin": 0,
    "bbox_positive_area": 0,
    "coverage": 0,
}
```

Denominator rules:

```text
valid: all supervised atoms
token_type_mass: all supervised hard-SFT atoms
continuation_margin: atoms with provenance.continuation_boundary is true
bbox_positive_area: atoms whose coord_role is x2 or y2 and that carry bbox_positive_area_valid_token_ids
coverage: atoms with coverage target when coverage_strength > 0
```

Enabled terms must fail fast when their eligible denominator is zero in the active v0 smoke surface.

Assemble the final optimizer loss from per-term means:

```python
valid_mean = component_totals["valid"] / denominators["valid"]
token_type_mass_mean = component_totals["token_type_mass"] / denominators["token_type_mass"]
continuation_margin_mean = component_totals["continuation_margin"] / denominators["continuation_margin"]
bbox_positive_area_mean = component_totals["bbox_positive_area"] / denominators["bbox_positive_area"]

loss = (
    valid_mean
    + token_type_mass_weight * token_type_mass_mean
    + continuation_margin_weight * continuation_margin_mean
    + bbox_positive_area_weight * bbox_positive_area_mean
)
if coverage_strength > 0.0:
    coverage_mean = component_totals["coverage"] / denominators["coverage"]
    loss = loss + coverage_strength * coverage_mean
```

Do not compute the optimizer loss by summing `atom_loss.total` over all atoms and dividing by the number of atoms. The per-atom `total` field is diagnostic-only for this stack.

Update raw component totals:

```python
_add_component(component_totals, "valid", atom_loss.valid)
_add_component(component_totals, "token_type_mass", atom_loss.token_type_mass)
_add_component(component_totals, "continuation_margin", atom_loss.continuation_margin)
_add_component(component_totals, "bbox_positive_area", atom_loss.bbox_positive_area)
_add_component(component_totals, "coverage", atom_loss.coverage)
```

Use raw means for `teacher_forcing/loss/{term}` and explicit contribution metrics for weighted effects:

```text
teacher_forcing/loss/token_type_mass = raw mean over token_type_mass denominator
teacher_forcing/loss/token_type_mass/contribution = token_type_mass_weight * raw mean
teacher_forcing/loss/continuation_margin = raw mean over continuation boundary denominator
teacher_forcing/loss/continuation_margin/contribution = continuation_margin_weight * raw mean
teacher_forcing/loss/bbox_positive_area = raw mean over eligible tail denominator
teacher_forcing/loss/bbox_positive_area/contribution = bbox_positive_area_weight * raw mean
```

- [ ] **Step 8: Run objective tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_objective_runner.py -q
```

Expected after implementation: existing pure valid-set tests still pass with default auxiliary weights at `0.0`; new weighted tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/training/teacher_forcing/probabilities.py \
  src/training/objectives/teacher_forcing.py \
  tests/test_teacher_forcing_objective_runner.py
git commit -m "feat: add teacher forcing type continuation geometry losses"
```

## Task 4: Trainer Config Handoff And Metrics

**Files:**
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/training/teacher_forcing/metrics.py`
- Modify: `src/sft.py`
- Modify: `tests/test_teacher_forcing_metric_contract.py`
- Modify: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 1: Write failing metric tests**

In `tests/test_teacher_forcing_metric_contract.py`, add expected keys:

```python
assert "teacher_forcing/loss/bbox_positive_area" in required
assert "teacher_forcing/loss/bbox_positive_area/contribution" in required
assert "teacher_forcing/type/schema_mass_at_schema" in required
assert "teacher_forcing/continuation/continue_minus_stop_margin" in required
assert "teacher_forcing/continuation/continue_accuracy" in required
assert "teacher_forcing/continuation/continue_mass" in required
assert "teacher_forcing/continuation/stop_mass" in required
assert "teacher_forcing/continuation/boundary_count" in required
assert "teacher_forcing/geometry/bbox_positive_area_invalid_mass" in required
assert "teacher_forcing/geometry/bbox_positive_area_eligible_count" in required
assert not any(key.endswith("_weighted") for key in required)
```

Add flattening assertions for diagnostic events:

```python
events = teacher_forcing_diagnostic_events(
    token_type_mass=0.75,
    token_type_mass_contribution=0.75,
    continuation_margin=0.25,
    continuation_margin_contribution=0.05,
    continue_minus_stop_margin=0.22,
    continue_accuracy=0.75,
    continue_mass=0.61,
    stop_mass=0.39,
    continuation_boundary_count=4,
    bbox_positive_area=0.1,
    bbox_positive_area_contribution=0.01,
    invalid_area_mass=0.04,
    eligible_tail_count=8,
)
flat = flatten_metric_events(events)
assert flat["teacher_forcing/loss/continuation_margin"] == pytest.approx(0.25)
assert flat["teacher_forcing/loss/continuation_margin/contribution"] == pytest.approx(0.05)
assert flat["teacher_forcing/loss/bbox_positive_area"] == pytest.approx(0.1)
assert flat["teacher_forcing/loss/bbox_positive_area/contribution"] == pytest.approx(0.01)
assert flat["teacher_forcing/continuation/continue_minus_stop_margin"] == pytest.approx(0.22)
assert flat["teacher_forcing/continuation/continue_accuracy"] == pytest.approx(0.75)
assert flat["teacher_forcing/continuation/continue_mass"] == pytest.approx(0.61)
assert flat["teacher_forcing/continuation/stop_mass"] == pytest.approx(0.39)
assert flat["teacher_forcing/continuation/boundary_count"] == pytest.approx(4)
assert flat["teacher_forcing/geometry/bbox_positive_area_invalid_mass"] == pytest.approx(0.04)
assert flat["teacher_forcing/geometry/bbox_positive_area_eligible_count"] == pytest.approx(8)
assert not any(key.endswith("_weighted") for key in flat)
```

- [ ] **Step 2: Pass term weights from trainer mixin**

In `src/trainers/metrics/teacher_forcing.py`, add helpers:

```python
def _term_weight(objective_cfg: Any, term_name: str) -> float:
    terms = (
        objective_cfg.get("terms")
        if isinstance(objective_cfg, Mapping)
        else getattr(objective_cfg, "terms", None)
    )
    term = terms.get(term_name) if isinstance(terms, Mapping) else getattr(terms, term_name, None)
    if term is None:
        return 0.0
    enabled = term.get("enabled") if isinstance(term, Mapping) else getattr(term, "enabled", False)
    if not bool(enabled):
        return 0.0
    weight = term.get("weight", 1.0) if isinstance(term, Mapping) else getattr(term, "weight", 1.0)
    return float(weight)
```

Add to the `ObjectiveSpec.config`:

```python
"token_type_mass_weight": _term_weight(objective_cfg, "token_type_mass"),
"continuation_margin_weight": _term_weight(objective_cfg, "continuation_margin"),
"bbox_positive_area_weight": _term_weight(objective_cfg, "bbox_positive_area"),
```

- [ ] **Step 3: Emit component metrics**

In `src/training/objectives/teacher_forcing.py`, after computing raw component totals and per-term denominators, create diagnostic events using the term-specific denominators:

```python
metric_events = (
    *teacher_forcing_loss_events(total_loss=loss, atom_count=valid_denominator),
    *teacher_forcing_component_loss_events(
        token_type_mass=token_type_mass_mean,
        token_type_mass_contribution=token_type_mass_weight * token_type_mass_mean,
        continuation_margin=continuation_margin_mean,
        continuation_margin_contribution=continuation_margin_weight * continuation_margin_mean,
        bbox_positive_area=bbox_positive_area_mean,
        bbox_positive_area_contribution=bbox_positive_area_weight * bbox_positive_area_mean,
        continue_minus_stop_margin=continue_minus_stop_margin_mean,
        continue_accuracy=continue_accuracy_mean,
        continue_mass=continue_mass_mean,
        stop_mass=stop_mass_mean,
        continuation_boundary_count=continuation_denominator,
        invalid_area_mass=invalid_area_mass_mean,
        eligible_tail_count=geometry_denominator,
    ),
)
```

Implement `teacher_forcing_component_loss_events` in `src/training/teacher_forcing/metrics.py`. Use unsuffixed metric names only:

```text
teacher_forcing/loss/token_type_mass
teacher_forcing/loss/token_type_mass/contribution
teacher_forcing/loss/continuation_margin
teacher_forcing/loss/continuation_margin/contribution
teacher_forcing/loss/bbox_positive_area
teacher_forcing/loss/bbox_positive_area/contribution
teacher_forcing/continuation/continue_minus_stop_margin
teacher_forcing/continuation/continue_accuracy
teacher_forcing/continuation/continue_mass
teacher_forcing/continuation/stop_mass
teacher_forcing/continuation/boundary_count
teacher_forcing/geometry/bbox_positive_area_invalid_mass
teacher_forcing/geometry/bbox_positive_area_eligible_count
```

- [ ] **Step 4: Preserve term weights in runtime payload**

In `src/sft.py`, extend the existing `terms` payload for:

```python
"token_type_mass": {
    "enabled": _get_section_value(token_type_mass_cfg, "enabled"),
    "weight": _get_section_value(token_type_mass_cfg, "weight"),
},
"continuation_margin": {
    "enabled": _get_section_value(continuation_margin_cfg, "enabled"),
    "weight": _get_section_value(continuation_margin_cfg, "weight"),
},
"bbox_positive_area": {
    "enabled": _get_section_value(bbox_positive_area_cfg, "enabled"),
    "weight": _get_section_value(bbox_positive_area_cfg, "weight"),
},
```

- [ ] **Step 5: Run metric and integration tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_metric_contract.py tests/test_training_runtime_sft_integration.py -q
```

Expected after implementation: metric keys flatten without `_weighted` suffixes and runtime payload includes weights.

- [ ] **Step 6: Commit**

```bash
git add src/trainers/metrics/teacher_forcing.py \
  src/training/teacher_forcing/metrics.py \
  src/sft.py \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_training_runtime_sft_integration.py
git commit -m "feat: report teacher forcing v0 auxiliary metrics"
```

## Task 5: Diagnostic Span-Drop Salvage View

**Files:**
- Modify: `src/detection/evaluation.py`
- Add: `scripts/evaluation/materialize_compact_span_drop_salvage.py`
- Modify: `tests/test_detection_template_parsing_eval.py`

- [ ] **Step 1: Locate the existing eval parser boundary**

Run:

```bash
rg -n "strict_expected|parser_mode|malformed|parse_error|compact" src/detection src/infer src/eval tests -g '*.py'
```

Use `src/detection/evaluation.py`, which owns strict template parsing helpers for metric-bearing evaluation. Do not change the strict parser to accept malformed rows.

- [ ] **Step 2: Write failing salvage tests**

In `tests/test_detection_template_parsing_eval.py`, add:

```python
def test_compact_span_drop_salvage_keeps_valid_spans_and_drops_invalid_geometry() -> None:
    text = (
        f"{OBJECT_REF_START_TOKEN}cat{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
        "<|coord_1|><|coord_2|><|coord_3|><|coord_4|>"
        f"{BOX_END_TOKEN}"
        f"{OBJECT_REF_START_TOKEN}bad{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
        "<|coord_9|><|coord_9|><|coord_9|><|coord_10|>"
        f"{BOX_END_TOKEN}"
        f"{OBJECT_REF_START_TOKEN}dog{OBJECT_REF_END_TOKEN}{BOX_START_TOKEN}"
        "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
        f"{BOX_END_TOKEN}"
    )

    with pytest.raises(ValueError):
        parse_detection_output_strict_expected(
            text,
            expected_template="compact_object_box_closed",
        )

    salvaged = parse_detection_output_compact_span_drop_salvage(
        text,
        expected_template="compact_object_box_closed",
    )

    assert salvaged["view_name"] == "compact_span_drop_salvage"
    assert salvaged["objects"] == [
        {
            "desc": "cat",
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        },
        {
            "desc": "dog",
            "bbox_2d": ["<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"],
        },
    ]
    assert salvaged["counters"]["rows_strict_malformed"] == 1
    assert salvaged["counters"]["rows_salvaged"] == 1
    assert salvaged["counters"]["objects_dropped_invalid_span"] == 1
    assert salvaged["counters"]["objects_kept_valid_span"] == 2
```

Also add:

```python
def test_compact_span_drop_salvage_records_clean_rows_without_salvage() -> None:
    salvaged = parse_detection_output_compact_span_drop_salvage(
        COMPACT_TEXT,
        expected_template="compact",
    )

    assert salvaged["counters"]["rows_strict_malformed"] == 0
    assert salvaged["counters"]["rows_salvaged"] == 0
    assert salvaged["counters"]["objects_dropped_invalid_span"] == 0
    assert salvaged["counters"]["objects_kept_valid_span"] == 1
```

Import `parse_detection_output_compact_span_drop_salvage` from `src.detection.evaluation`.

- [ ] **Step 3: Implement salvage as a named diagnostic view**

In `src/detection/evaluation.py`, add a public helper named exactly `parse_detection_output_compact_span_drop_salvage(text: str, *, expected_template: TemplateId | str, object_field_order: str = "desc_first") -> dict[str, Any]`.

Implementation rules:

- split candidate compact object spans using existing marker/span extraction helpers;
- parse each candidate object span independently with existing strict object validation;
- keep valid object spans;
- drop invalid object spans;
- keep row-level strict metrics unchanged;
- report the diagnostic counters:

```text
rows_strict_malformed
rows_salvaged
objects_dropped_invalid_span
objects_kept_valid_span
```

- [ ] **Step 4: Add a diagnostic materialization script**

Create `scripts/evaluation/materialize_compact_span_drop_salvage.py`.

Required CLI:

```bash
python scripts/evaluation/materialize_compact_span_drop_salvage.py \
  --pred-jsonl temp/infer/ledger_aux_train128/ledger/gt_vs_pred.jsonl \
  --out-json temp/infer/ledger_aux_train128/ledger/eval/compact_span_drop_salvage_metrics.json \
  --out-jsonl temp/infer/ledger_aux_train128/ledger/eval/compact_span_drop_salvage_rows.jsonl \
  --expected-template compact_object_box_closed
```

Required output JSON:

```json
{
  "view_name": "compact_span_drop_salvage",
  "diagnostic_only": true,
  "strict_metrics_replaced": false,
  "counters": {
    "rows_strict_malformed": 0,
    "rows_salvaged": 0,
    "objects_dropped_invalid_span": 0,
    "objects_kept_valid_span": 0
  }
}
```

The script must read each prediction row, apply the diagnostic helper to the prediction text, write one diagnostic JSONL row per input row, and aggregate the four counters. It must not overwrite or mutate the strict `gt_vs_pred.jsonl`, `summary.json`, or `eval/metrics.json` artifacts.

- [ ] **Step 5: Run salvage tests**

Run:

```bash
python -m pytest tests/test_detection_template_parsing_eval.py -q
```

Expected after implementation: salvage view works and strict parser behavior remains strict.

- [ ] **Step 6: Commit**

```bash
git add src/detection/evaluation.py \
  scripts/evaluation/materialize_compact_span_drop_salvage.py \
  tests/test_detection_template_parsing_eval.py
git commit -m "feat: add compact span-drop salvage diagnostics"
```

## Task 6: Train128 Overfit Smoke Verification Gate

**Files:**
- Modify only if tests reveal missing docs/provenance: `research/ideas/ledger-auxiliary-loss/discussion.md`
- No source edits expected if Tasks 1-5 pass.

- [ ] **Step 1: Run narrow unit tests**

Run:

```bash
python -m pytest \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_teacher_forcing_target_builder.py \
  tests/test_teacher_forcing_objective_runner.py \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_training_runtime_sft_integration.py \
  -q
```

Expected: all pass.

- [ ] **Step 2: Run config/preflight checks for the 128 overfit smoke**

Run the ledger preflight against the exact 128-sample smoke pair:

```bash
source /root/miniconda3/bin/activate ms
python scripts/training/coverage_ledger_preflight.py \
  --config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  --baseline-config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
  --output-root temp/coverage_ledger_preflight_smoke
```

Expected preflight artifacts:

```text
temp/coverage_ledger_preflight_smoke/ledger/selected_samples.json
temp/coverage_ledger_preflight_smoke/ledger/alignment_debug.jsonl
temp/coverage_ledger_preflight_smoke/ledger/overlays/index.json
```

Run the supporting contract tests:

```bash
source /root/miniconda3/bin/activate ms
python -m pytest \
  tests/test_coverage_ledger_smoke_configs.py \
  tests/test_coverage_ledger_preflight_artifacts.py \
  tests/test_train_batch_contract.py \
  -q
```

Expected: preflight exits `0`, `alignment_debug.jsonl` has 128 lines, every preflight failure status is `"ok"`, 16 overlays are indexed, preflight-selected rows match the training dataset rows, config/preflight tests pass, and no batch contract regression appears.

- [ ] **Step 3: Verify the 128 smoke config will save an adapter checkpoint**

Run:

```bash
source /root/miniconda3/bin/activate ms
python - <<'PY'
from src.config.loader import ConfigLoader

configs = {
    "baseline": (
        "configs/stage1/detection_teacher_forcing/smoke/"
        "coverage_ledger_closed_hard_sft_128_baseline.yaml",
        "coverage_ledger_closed_hard_sft_128_baseline",
    ),
    "ledger": (
        "configs/stage1/detection_teacher_forcing/smoke/"
        "coverage_ledger_closed_hard_sft_128.yaml",
        "coverage_ledger_closed_hard_sft_128",
    ),
}
for arm, (path, artifact_subdir) in configs.items():
    cfg = ConfigLoader.load_materialized_training_config(path)
    training = cfg.training
    assert training["max_steps"] == 256, arm
    assert training["save_strategy"] == "steps", arm
    assert training["save_steps"] == 128, arm
    assert training["save_total_limit"] == 2, arm
    assert training["save_last_epoch"] is True, arm
    assert training["save_delay_steps"] == 0, arm
    assert training["artifact_subdir"] == artifact_subdir, arm
print("paired_train128_save_contract_ok")
PY
```

Expected: prints `paired_train128_save_contract_ok`. Do not launch the smoke if this check fails, because adapter reload would not be testable.

- [ ] **Step 4: Saved-adapter reload tests remain mandatory**

Before interpreting train128 metrics, verify that the saved adapter loads through the same inference checkpoint-resolution path:

```bash
source /root/miniconda3/bin/activate ms
python -m pytest tests/test_infer_checkpoint_resolution.py tests/test_infer_batch_decoding.py -q
```

Expected: adapter checkpoint resolution and batch decoding tests pass.

- [ ] **Step 5: Launch paired train128 overfit smokes only after user approval**

The first smoke benchmark is:

```text
baseline = hard-SFT comparator with no ledger/no new losses
ledger = mandatory token_type_mass + continuation_margin + bbox_positive_area + coverage ledger
train split = same selected 128 training samples
inference/eval split = same selected 128 training rows materialized from preflight selected_samples.json
checkpoint saving = enabled for both adapters
```

Run the two arms sequentially so each can use all 8 visible GPUs. Do not launch them concurrently on the same GPUs.

```bash
tmux new-session -d -s coordexp-baseline-train128 \
  'cd /data/CoordExp/.worktrees/ledger-auxiliary-loss && \
   source /root/miniconda3/bin/activate ms && \
   config=configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
   gpus=0,1,2,3,4,5,6,7 \
   train_log_dir=temp/train_logs/ledger_aux_train128/baseline \
   bash scripts/train.sh'
```

After the baseline tmux session exits successfully, launch the ledger arm:

```bash
tmux new-session -d -s coordexp-ledger-train128 \
  'cd /data/CoordExp/.worktrees/ledger-auxiliary-loss && \
   source /root/miniconda3/bin/activate ms && \
   config=configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
   gpus=0,1,2,3,4,5,6,7 \
   train_log_dir=temp/train_logs/ledger_aux_train128/ledger \
   bash scripts/train.sh'
```

Do not run the full baseline/ledger/no-loss/loss matrix. This paired baseline-vs-ledger smoke is the minimum comparator needed for train128 interpretation.

Expected training artifacts under both `temp/detection_teacher_forcing/output/coverage_ledger_closed_hard_sft_128_baseline/` and `temp/detection_teacher_forcing/output/coverage_ledger_closed_hard_sft_128/`:

```text
resolved_config.json
runtime_env.json
effective_runtime.json
pipeline_manifest.json
experiment_manifest.json
run_metadata.json
train_data_provenance.json
checkpoint-128/
checkpoint-256/
```

- [ ] **Step 6: Resolve saved adapter checkpoints produced by both arms**

Run:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/bin/activate ms
python - <<'PY'
import json
from pathlib import Path

from src.infer.checkpoints import (
    prepare_adapter_checkpoint_for_inference,
    resolve_inference_checkpoint,
    validate_compact_token_embeddings_adapter_contract,
)

outputs = {}
for arm, subdir in {
    "baseline": "coverage_ledger_closed_hard_sft_128_baseline",
    "ledger": "coverage_ledger_closed_hard_sft_128",
}.items():
    root = Path("temp/detection_teacher_forcing/output") / subdir
    checkpoints = sorted(root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not checkpoints:
        raise SystemExit(f"{arm}: no checkpoint-* under {root}")
    ckpt = checkpoints[-1]
    if not (ckpt / "adapter_config.json").is_file():
        raise SystemExit(f"{arm}: missing adapter_config.json")
    if not (ckpt / "adapter_model.safetensors").is_file():
        raise SystemExit(f"{arm}: adapter_model.safetensors is required for compact token-row adapter validation")

    cfg = json.loads((ckpt / "adapter_config.json").read_text(encoding="utf-8"))
    modules = set(cfg.get("modules_to_save") or [])
    if "token_embeddings_adapter" not in modules:
        raise SystemExit(f"{arm}: adapter_config.json missing token_embeddings_adapter in modules_to_save")

    resolved = resolve_inference_checkpoint(model_checkpoint=str(ckpt))
    validate_compact_token_embeddings_adapter_contract(
        resolved,
        detection_template_id="compact_object_box_closed",
    )
    view = prepare_adapter_checkpoint_for_inference(
        str(ckpt),
        cache_root="temp/infer/ledger_aux_train128/adapter_views",
    )
    outputs[arm] = {
        "checkpoint": str(ckpt),
        "resolved_base_model_checkpoint": resolved.resolved_base_model_checkpoint,
        "resolved_adapter_checkpoint": resolved.resolved_adapter_checkpoint,
        "inference_adapter_view": view.path,
        "dropped_modules_to_save": list(view.dropped_modules_to_save),
        "dropped_tensor_keys": list(view.dropped_tensor_keys),
    }
    print(f"{arm.upper()}_TRAIN128_ADAPTER={ckpt}")
    print(f"{arm.upper()}_TRAIN128_INFER_ADAPTER_VIEW={view.path}")

Path("temp/infer/ledger_aux_train128").mkdir(parents=True, exist_ok=True)
Path("temp/infer/ledger_aux_train128/adapter_checkpoint_gate.json").write_text(
    json.dumps(outputs, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("paired_train128_adapter_checkpoint_gate_ok")
PY
```

Expected: prints `BASELINE_TRAIN128_ADAPTER=<checkpoint-path>`, `LEDGER_TRAIN128_ADAPTER=<checkpoint-path>`, and `paired_train128_adapter_checkpoint_gate_ok`; writes `temp/infer/ledger_aux_train128/adapter_checkpoint_gate.json`; confirms each checkpoint has adapter config, `adapter_model.safetensors`, saved token-embedding adapter metadata, compact closed-template token rows, and an inference-prepared adapter view. This is the source-of-truth gate for stripping the training-only `coverage_ledger_head` from the inference-prepared adapter view.

- [ ] **Step 7: Materialize the exact selected train128 JSONL for overfit inference**

Run:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/bin/activate ms
python - <<'PY'
import json
import hashlib
import zlib
from pathlib import Path

selected_path = Path("temp/coverage_ledger_preflight_smoke/ledger/selected_samples.json")
out_path = Path("temp/infer/ledger_aux_train128/train128.selected.coord.jsonl")
payload = json.loads(selected_path.read_text())
source_path = Path(payload["source_jsonl_resolved_path"])
expected_sha = str(payload["source_jsonl_sha256"])
actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
assert actual_sha == expected_sha, (actual_sha, expected_sha)
selected_indices = tuple(int(i) for i in payload["selected_row_indices"])
selected = set(selected_indices)
expected_sample_ids = [str(value) for value in payload["selected_sample_ids"]]
dataset_namespace = zlib.crc32(str(payload["dataset_id"]).encode("utf-8")) & 0xFFFF
expected_from_indices = [
    str((dataset_namespace << 32) | (int(row_index) & 0xFFFFFFFF))
    for row_index in selected_indices
]
assert expected_sample_ids == expected_from_indices

out_path.parent.mkdir(parents=True, exist_ok=True)
written = 0
written_indices = []
with source_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
    for row_index, line in enumerate(src):
        if row_index in selected:
            dst.write(line)
            written += 1
            written_indices.append(row_index)

assert written == 128, written
assert tuple(written_indices) == selected_indices
print(f"selected_train128_jsonl_ok={out_path}")
PY
```

Expected: prints `selected_train128_jsonl_ok=temp/infer/ledger_aux_train128/train128.selected.coord.jsonl`, proves the source JSONL SHA-256 matches the preflight manifest, verifies deterministic sample IDs for the selected row indices, and writes exactly 128 selected rows in manifest order.

- [ ] **Step 8: Run train128 inference for both saved adapters with `rp=1.10`**

Run:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/bin/activate ms
for arm in baseline ledger; do
  case "${arm}" in
    baseline) subdir=coverage_ledger_closed_hard_sft_128_baseline ;;
    ledger) subdir=coverage_ledger_closed_hard_sft_128 ;;
  esac
  latest_ckpt="$(
    find "temp/detection_teacher_forcing/output/${subdir}" \
      -maxdepth 2 -type d -name 'checkpoint-*' \
      | sort -V \
      | tail -1
  )"
  ARM="${arm}" LATEST_CKPT="${latest_ckpt}" python - <<'PY'
import os
from pathlib import Path
import yaml

arm = os.environ["ARM"]
latest_ckpt = os.environ["LATEST_CKPT"]
cfg_path = Path(f"temp/infer/ledger_aux_train128/{arm}/infer.closed_compact.sorted.rp110.yaml")
cfg_path.parent.mkdir(parents=True, exist_ok=True)
cfg = {
    "run": {
        "name": f"ledger_aux_train128_{arm}_closed_compact_sorted_rp110",
        "output_dir": "temp/infer/ledger_aux_train128",
    },
    "stages": {"infer": True, "eval": False, "vis": False},
    "detection_template": {"id": "compact_object_box_closed"},
    "infer": {
        "gt_jsonl": "temp/infer/ledger_aux_train128/train128.selected.coord.jsonl",
        "model_checkpoint": latest_ckpt,
        "prompt_variant": "coco_80",
        "bbox_format": "xyxy",
        "object_field_order": "desc_first",
        "object_ordering": "sorted",
        "mode": "coord",
        "pred_coord_mode": "auto",
        "backend": {
            "type": "hf",
            "attn_implementation": "flash_attention_2",
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 0.9,
            "max_new_tokens": 3084,
            "repetition_penalty": 1.10,
            "batch_size": 8,
            "seed": 42,
        },
        "device": "cuda:0",
        "limit": 128,
        "detect_samples": 128,
    },
    "artifacts": {
        "gt_vs_pred_jsonl": f"temp/infer/ledger_aux_train128/{arm}/gt_vs_pred.jsonl",
        "summary_json": f"temp/infer/ledger_aux_train128/{arm}/summary.json",
    },
}
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
print(cfg_path)
PY
  python scripts/run_infer.py --config "temp/infer/ledger_aux_train128/${arm}/infer.closed_compact.sorted.rp110.yaml"
done
```

Expected inference artifacts:

```text
temp/infer/ledger_aux_train128/baseline/infer.closed_compact.sorted.rp110.yaml
temp/infer/ledger_aux_train128/baseline/gt_vs_pred.jsonl
temp/infer/ledger_aux_train128/baseline/summary.json
temp/infer/ledger_aux_train128/ledger/infer.closed_compact.sorted.rp110.yaml
temp/infer/ledger_aux_train128/ledger/gt_vs_pred.jsonl
temp/infer/ledger_aux_train128/ledger/summary.json
```

- [ ] **Step 9: Evaluate both train128 inference artifacts**

Run:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/bin/activate ms
for arm in baseline ledger; do
  python scripts/evaluate_detection.py \
    --pred_jsonl "temp/infer/ledger_aux_train128/${arm}/gt_vs_pred.jsonl" \
    --out_dir "temp/infer/ledger_aux_train128/${arm}/eval" \
    --metrics f1ish \
    --num-workers 0 \
    --semantic-model model_cache/all-MiniLM-L6-v2-local \
    --semantic-threshold 0.5 \
    --semantic-device cuda:0 \
    --semantic-batch-size 64 \
    --no-segm
done
```

Expected evaluation artifacts:

```text
temp/infer/ledger_aux_train128/baseline/eval/metrics.json
temp/infer/ledger_aux_train128/baseline/eval/per_image.json
temp/infer/ledger_aux_train128/baseline/eval/matches.jsonl
temp/infer/ledger_aux_train128/ledger/eval/metrics.json
temp/infer/ledger_aux_train128/ledger/eval/per_image.json
temp/infer/ledger_aux_train128/ledger/eval/matches.jsonl
```

These F1-ish artifacts are raw train128 diagnostic metrics, not official COCO/LVIS benchmark evidence. Strict closed-template evidence for this smoke comes from Step 11 checks against each inference `summary.json` and `resolved_config.json`: `parser_mode == strict_expected`, `detection_template_id == compact_object_box_closed`, sorted object ordering, `desc_first` field order, and `repetition_penalty == 1.10`.

- [ ] **Step 10: Materialize diagnostic span-drop salvage artifacts for both arms**

Run:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/bin/activate ms
for arm in baseline ledger; do
  python scripts/evaluation/materialize_compact_span_drop_salvage.py \
    --pred-jsonl "temp/infer/ledger_aux_train128/${arm}/gt_vs_pred.jsonl" \
    --out-json "temp/infer/ledger_aux_train128/${arm}/eval/compact_span_drop_salvage_metrics.json" \
    --out-jsonl "temp/infer/ledger_aux_train128/${arm}/eval/compact_span_drop_salvage_rows.jsonl" \
    --expected-template compact_object_box_closed
done
```

Expected diagnostic artifacts:

```text
temp/infer/ledger_aux_train128/baseline/eval/compact_span_drop_salvage_metrics.json
temp/infer/ledger_aux_train128/baseline/eval/compact_span_drop_salvage_rows.jsonl
temp/infer/ledger_aux_train128/ledger/eval/compact_span_drop_salvage_metrics.json
temp/infer/ledger_aux_train128/ledger/eval/compact_span_drop_salvage_rows.jsonl
```

These artifacts are diagnostic-only and must not replace strict parser metrics.

- [ ] **Step 11: Verify train128 metric and artifact handles before interpretation**

Run:

```bash
cd /data/CoordExp/.worktrees/ledger-auxiliary-loss
source /root/miniconda3/bin/activate ms
python - <<'PY'
import hashlib
import json
from pathlib import Path
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

required_train_metric_keys = {
    "teacher_forcing/loss/token_type_mass",
    "teacher_forcing/loss/continuation_margin",
    "teacher_forcing/loss/bbox_positive_area",
    "teacher_forcing/geometry/bbox_positive_area_invalid_mass",
    "teacher_forcing/continuation/continue_minus_stop_margin",
    "teacher_forcing/continuation/continue_accuracy",
}

strict_handles = {}
for arm in ("baseline", "ledger"):
    infer_root = Path(f"temp/infer/ledger_aux_train128/{arm}")
    infer_cfg_path = infer_root / "infer.closed_compact.sorted.rp110.yaml"
    assert infer_cfg_path.is_file(), arm
    infer_cfg = yaml.safe_load(infer_cfg_path.read_text(encoding="utf-8"))
    assert infer_cfg["detection_template"]["id"] == "compact_object_box_closed", arm
    assert infer_cfg["infer"]["object_field_order"] == "desc_first", arm
    assert infer_cfg["infer"]["object_ordering"] == "sorted", arm
    assert float(infer_cfg["infer"]["generation"]["repetition_penalty"]) == 1.10, arm

    summary_path = infer_root / "summary.json"
    assert summary_path.is_file(), arm
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_infer = summary["infer"]
    assert summary_infer["detection_template_id"] == "compact_object_box_closed", arm
    assert summary_infer["object_field_order"] == "desc_first", arm
    assert summary_infer["object_ordering"] == "sorted", arm
    assert summary_infer["parsing"]["mode"] == "strict_expected", arm
    assert float(summary["generation"]["repetition_penalty"]) == 1.10, arm
    provenance = summary.get("inference_provenance") or {}
    for key in ("prompt_policy_fingerprint", "decode_policy_fingerprint", "model_identity_fingerprint"):
        assert provenance.get(key), (arm, key)

    resolved_pointer = infer_root / "resolved_config.path"
    assert resolved_pointer.is_file(), arm
    resolved_path = Path(resolved_pointer.read_text(encoding="utf-8").strip())
    assert resolved_path.is_file(), arm
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved_infer = resolved["infer"]
    assert resolved_infer["detection_template_id"] == "compact_object_box_closed", arm
    assert resolved_infer["object_field_order"] == "desc_first", arm
    assert resolved_infer["object_ordering"] == "sorted", arm
    assert resolved_infer["parsing"]["mode"] == "strict_expected", arm
    strict_handles[arm] = {
        "infer_config": str(infer_cfg_path),
        "infer_config_sha256": sha256(infer_cfg_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": sha256(resolved_path),
        "prompt_template_hash": summary_infer["prompt_template_hash"],
        "prompt_policy_fingerprint": provenance["prompt_policy_fingerprint"],
        "decode_policy_fingerprint": provenance["decode_policy_fingerprint"],
        "model_identity_fingerprint": provenance["model_identity_fingerprint"],
    }

    assert (infer_root / "gt_vs_pred.jsonl").is_file(), arm
    eval_root = infer_root / "eval"
    metrics_path = eval_root / "metrics.json"
    assert metrics_path.is_file(), arm
    metrics_payload = json.loads(metrics_path.read_text())
    assert metrics_payload.get("comparison_scope") in {None, "raw_f1ish", "diagnostic_f1ish"}, arm
    assert "metrics" in metrics_payload, arm
    assert (eval_root / "per_image.json").is_file(), arm
    assert (eval_root / "matches.jsonl").is_file(), arm

    salvage_path = eval_root / "compact_span_drop_salvage_metrics.json"
    salvage = json.loads(salvage_path.read_text())
    assert salvage["view_name"] == "compact_span_drop_salvage", arm
    assert salvage["diagnostic_only"] is True, arm
    assert salvage["strict_metrics_replaced"] is False, arm
    counters = salvage["counters"]
    for key in (
        "rows_strict_malformed",
        "rows_salvaged",
        "objects_dropped_invalid_span",
        "objects_kept_valid_span",
    ):
        assert isinstance(counters[key], int), (arm, key)

ledger_log_dir = Path("temp/train_logs/ledger_aux_train128/ledger")
baseline_log_dir = Path("temp/train_logs/ledger_aux_train128/baseline")
assert ledger_log_dir.exists()
assert baseline_log_dir.exists()
ledger_log_text = "\n".join(
    path.read_text(encoding="utf-8", errors="replace")
    for path in sorted(ledger_log_dir.glob("*.log"))
)
for key in required_train_metric_keys:
    assert key in ledger_log_text, key
assert "teacher_forcing/ledger/auc_batch" in ledger_log_text

Path("temp/infer/ledger_aux_train128/strict_infer_artifact_gate.json").write_text(
    json.dumps(strict_handles, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("strict_train128_infer_artifact_gate_ok")
print("paired_train128_artifact_gate_ok")
PY
```

Expected: prints `strict_train128_infer_artifact_gate_ok` and `paired_train128_artifact_gate_ok`, and writes `temp/infer/ledger_aux_train128/strict_infer_artifact_gate.json`.

Required tiny-run handles:

```text
baseline saved adapter reload path
ledger saved adapter reload path
baseline strict-template inference summary plus raw f1ish metrics
ledger strict-template inference summary plus raw f1ish metrics
baseline compact_span_drop_salvage metrics
ledger compact_span_drop_salvage metrics
teacher_forcing/loss/token_type_mass
teacher_forcing/loss/continuation_margin
teacher_forcing/loss/bbox_positive_area
teacher_forcing/geometry/bbox_positive_area_invalid_mass
teacher_forcing/continuation/continue_minus_stop_margin
teacher_forcing/continuation/continue_accuracy
teacher_forcing/ledger/auc_batch
temp/infer/ledger_aux_train128/baseline/summary.json
temp/infer/ledger_aux_train128/baseline/eval/metrics.json
temp/infer/ledger_aux_train128/ledger/summary.json
temp/infer/ledger_aux_train128/ledger/eval/metrics.json
```

Claim scope:

```text
train128 overfit smoke evidence only
not full validation
not production training readiness until the smoke artifacts and metrics above are present
```

- [ ] **Step 12: Record smoke outcome before production planning**

Append a short result entry to `research/ideas/ledger-auxiliary-loss/discussion.md` with:

```text
scope: train128 overfit smoke
baseline config: configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml
ledger config: configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml
baseline checkpoint: exact BASELINE_TRAIN128_ADAPTER path printed by Step 6
ledger checkpoint: exact LEDGER_TRAIN128_ADAPTER path printed by Step 6
adapter checkpoint gate: temp/infer/ledger_aux_train128/adapter_checkpoint_gate.json plus sha256
rp: 1.10
preflight root: temp/coverage_ledger_preflight_smoke/ledger
selected_samples.json sha256: sha256 of temp/coverage_ledger_preflight_smoke/ledger/selected_samples.json
source_jsonl_sha256: value copied from selected_samples.json
selected row count: 128
alignment_debug.jsonl line count and sha256: line count plus sha256
overlay index path and sha256: temp/coverage_ledger_preflight_smoke/ledger/overlays/index.json plus sha256
train log handles: temp/train_logs/ledger_aux_train128/baseline and temp/train_logs/ledger_aux_train128/ledger
baseline train artifacts: resolved_config.json, effective_runtime.json, experiment_manifest.json, run_metadata.json
ledger train artifacts: resolved_config.json, effective_runtime.json, experiment_manifest.json, run_metadata.json
baseline infer config path and sha256: temp/infer/ledger_aux_train128/baseline/infer.closed_compact.sorted.rp110.yaml plus sha256
ledger infer config path and sha256: temp/infer/ledger_aux_train128/ledger/infer.closed_compact.sorted.rp110.yaml plus sha256
baseline resolved infer config path and sha256: value from baseline resolved_config.path plus sha256
ledger resolved infer config path and sha256: value from ledger resolved_config.path plus sha256
baseline prompt/decode/model fingerprints: prompt_template_hash, prompt_policy_fingerprint, decode_policy_fingerprint, model_identity_fingerprint
ledger prompt/decode/model fingerprints: prompt_template_hash, prompt_policy_fingerprint, decode_policy_fingerprint, model_identity_fingerprint
baseline inference artifacts: temp/infer/ledger_aux_train128/baseline/summary.json, gt_vs_pred.jsonl
ledger inference artifacts: temp/infer/ledger_aux_train128/ledger/summary.json, gt_vs_pred.jsonl
baseline eval artifacts: temp/infer/ledger_aux_train128/baseline/eval/metrics.json, per_image.json, matches.jsonl
ledger eval artifacts: temp/infer/ledger_aux_train128/ledger/eval/metrics.json, per_image.json, matches.jsonl
strict inference artifact gate: temp/infer/ledger_aux_train128/strict_infer_artifact_gate.json plus sha256
baseline strict parse counters: numeric parse counters copied from baseline summary
ledger strict parse counters: numeric parse counters copied from ledger summary
baseline salvage counters: numeric counters copied from baseline compact_span_drop_salvage_metrics.json
ledger salvage counters: numeric counters copied from ledger compact_span_drop_salvage_metrics.json
teacher forcing losses: token_type_mass, continuation_margin, bbox_positive_area
continuation metrics: continue_minus_stop_margin and continue_accuracy
ledger metrics: auc_batch and accuracy if logged
paired train128 deltas: ledger minus baseline for raw f1ish and diagnostic salvage f1ish when available
interpretation: train128 overfit smoke only, not full validation
production implication: lower LR than 5.0e-5 and 2 epochs, exact production LR chosen after smoke comparison
```

- [ ] **Step 13: Final commit**

After verification and before launch, commit any uncommitted implementation changes from this plan:

```bash
git status --short
git add src/config/schema.py \
  src/detection/teacher_forcing/trie.py \
  src/detection/teacher_forcing/target_builder.py \
  src/training/teacher_forcing/probabilities.py \
  src/training/objectives/teacher_forcing.py \
  src/trainers/metrics/teacher_forcing.py \
  src/training/teacher_forcing/metrics.py \
  src/training/coverage_ledger/preflight.py \
  src/sft.py \
  src/detection/evaluation.py \
  scripts/evaluation/materialize_compact_span_drop_salvage.py \
  configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_teacher_forcing_target_builder.py \
  tests/test_teacher_forcing_objective_runner.py \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_detection_template_parsing_eval.py
git commit -m "feat: wire ledger hard-sft auxiliary loss stack"
```

If the per-task commits already committed all implementation files, this final commit step should report a clean tree and create no commit.

## Self-Review

- Spec coverage: the plan covers mandatory type partition, continuation, hard geometry, unsuffixed metrics, concise config naming, span-drop salvage, adapter reload, and train128 launch scope.
- Placeholder scan: no angle-bracket placeholders remain. Commands name the expected implementation files directly.
- Type consistency: the plan consistently uses `token_type_mass_weight`, `continuation_margin_weight`, `bbox_positive_area_weight`, `selected_bbox_xyxy`, `continuation_boundary`, `continuation_target`, `continuation_token_ids`, and `stop_token_id`.
