# Recursive Detection Token-Type Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate recursive detection supervision into hard structural CE, hard-plus-trie description CE, global token-type gating, and coordinate-vocabulary SoftCE.

**Architecture:** Reuse the existing compact token-type groups and `TokenTarget.type_gate_*` fields, but wire them into full-sequence `random_permutation_et_rmp_ce`. Restrict trie multi-positive targets to free-text description positions, keep structural/coordinate positions hard-CE targets, and move `instance_trie_gaussian` coordinate SoftCE to a coord-vocab softmax because the type gate owns full-vocab coordinate exclusivity.

**Tech Stack:** Python dataclasses, PyTorch loss functions, latest detection config schema, CoordExp compact detection target builder, `pytest` via `conda run -n ms`.

---

## File Map

- Modify `src/detection/objective.py`
  - Generalize type-gate attachment from prefix-rollin to full recursive targets.
  - Restrict trie multi-positive target creation to `TokenRole.DESC`.
  - Keep coordinate candidate sidecars attached to coord targets even when the target kind is hard CE.
- Modify `src/detection/loss.py`
  - Add desc hard CE on top of trie support/balance loss.
  - Keep type-gate loss additive and outside role-specific main loss weighting.
- Modify `src/detection/coord_soft_targets.py`
  - Add coord-vocab SoftCE for `instance_trie_gaussian`.
  - Preserve legacy IoU/CIoU full-vocab support/balance behavior unless explicitly migrated later.
- Modify `src/detection/runtime.py`
  - Resolve and pass `objective.type_gate` for `random_permutation_et_rmp_ce`.
- Modify `src/detection/dataset.py`
  - Pass type-gate config into full-sequence preparation.
- Modify `src/config/schema.py`
  - Allow `objective.type_gate` for `random_permutation_et_rmp_ce`.
  - Keep rejection for SFT variants.
- Modify instance-trie Gaussian configs under `configs/stage1/recursive_detection_ce_latest/`
  - Enable conservative `objective.type_gate` for A5/A6/A7 instance-trie Gaussian configs.
- Modify tests:
  - `tests/test_recursive_detection_ce_target_builder.py`
  - `tests/test_recursive_detection_ce_loss_adapter.py`
  - `tests/test_instance_trie_gaussian_coord_softce.py`
  - `tests/test_latest_training_config_contract.py`
  - `tests/test_compact_type_gate.py`

This worktree already has unrelated pending feature-branch edits. Do not revert them. Do not commit unless the user explicitly asks.

## Task 1: Full-Sequence Type-Gate Target Metadata

**Files:**
- Modify: `src/detection/objective.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/detection/dataset.py`
- Test: `tests/test_recursive_detection_ce_target_builder.py`

- [ ] **Step 1: Write the failing target-builder test**

Add a test near the existing recursive target-builder tests:

```python
def test_random_permutation_type_gate_attaches_allowed_type_ids() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-1201:src-7",
            desc="cat",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-1202:src-3",
            desc="dog",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
        type_gate_config={
            "enabled": True,
            "weights": {
                "struct": 1.0,
                "coord": 1.0,
                "desc": 1.0,
                "eos": 0.5,
            },
        },
    )
    first_entry = prepared.tokenized.object_entries[0]
    targets = _target_map(prepared)

    object_ref_target = targets[first_entry.object_ref_start_span.start]
    desc_target = targets[first_entry.desc_span.start]
    coord_target = targets[first_entry.coord_spans[0].start]

    assert object_ref_target.type_gate_weight == pytest.approx(1.0)
    assert desc_target.type_gate_weight == pytest.approx(1.0)
    assert coord_target.type_gate_weight == pytest.approx(1.0)
    assert object_ref_target.teacher_token_id in object_ref_target.type_gate_token_ids
    assert desc_target.teacher_token_id in desc_target.type_gate_token_ids
    assert coord_target.teacher_token_id in coord_target.type_gate_token_ids
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_target_builder.py::test_random_permutation_type_gate_attaches_allowed_type_ids -q
```

Expected: fail because `prepare_detection_training_example` does not accept or attach `type_gate_config` for random-permutation examples.

- [ ] **Step 3: Generalize type-gate attachment**

In `src/detection/objective.py`, rename `_apply_prefix_rollin_type_gate` to `_apply_compact_type_gate` and keep its implementation shape:

```python
def _apply_compact_type_gate(
    token_targets: Sequence[TokenTarget],
    *,
    tokenizer: TokenizerWithOffsets,
    type_gate_config: Any | None,
) -> tuple[TokenTarget, ...]:
    if not bool(_cfg_value(type_gate_config, "enabled", False)):
        return tuple(token_targets)
    groups = build_compact_token_type_groups(tokenizer)
    weights_cfg = _cfg_value(type_gate_config, "weights")

    def _weight(name: str) -> float:
        value = _cfg_value(weights_cfg, name, 0.0)
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"type_gate.weights.{name} must be finite and >= 0")
        return weight

    weights_by_group = {
        "struct": _weight("struct"),
        "coord": _weight("coord"),
        "desc": _weight("desc"),
        "eos": _weight("eos"),
    }

    out: list[TokenTarget] = []
    for target in token_targets:
        allowed_ids = allowed_type_token_ids_for_target(target, groups)
        group_weights: list[float] = []
        if any(token_id in groups.struct for token_id in allowed_ids):
            group_weights.append(weights_by_group["struct"])
        if any(token_id in groups.coord for token_id in allowed_ids):
            group_weights.append(weights_by_group["coord"])
        if any(token_id in groups.desc for token_id in allowed_ids):
            group_weights.append(weights_by_group["desc"])
        if any(token_id in groups.eos for token_id in allowed_ids):
            group_weights.append(weights_by_group["eos"])
        out.append(
            replace(
                target,
                type_gate_token_ids=tuple(sorted(allowed_ids)),
                type_gate_weight=max(group_weights) if group_weights else 0.0,
            )
        )
    return tuple(out)
```

Update both prefix-rollin and full-sequence paths to call this helper.

Add `type_gate_config: Any | None = None` to `prepare_detection_training_example(...)`, and after `build_recursive_detection_targets(...)` call:

```python
recursive_detection_targets = build_recursive_detection_targets(...)
gated_targets = _apply_compact_type_gate(
    recursive_detection_targets.token_targets,
    tokenizer=tokenizer,
    type_gate_config=type_gate_config,
)
if gated_targets is not recursive_detection_targets.token_targets:
    loss_atoms = _build_loss_atoms(
        tokenized=tokenized,
        token_targets=gated_targets,
    )
    recursive_detection_targets = RecursiveDetectionTargets(
        token_targets=_assign_loss_atoms(
            token_targets=gated_targets,
            loss_atoms=loss_atoms,
        ),
        state_weighting=recursive_detection_targets.state_weighting,
        normalization=recursive_detection_targets.normalization,
        loss_atoms=loss_atoms,
        state_weighting_diagnostics=recursive_detection_targets.state_weighting_diagnostics,
    )
```

Use equality of config enabled state rather than object identity if needed; the important behavior is to rebuild atoms after replacing targets.

- [ ] **Step 4: Wire runtime and dataset**

In `src/detection/runtime.py`, set `type_gate_config` for both recursive variants:

```python
type_gate_config = None
if training_config.objective.variant in {
    "random_permutation_et_rmp_ce",
    "prefix_rollin_et_rmp_ce",
}:
    type_gate_config = training_config.objective.type_gate
```

In `src/detection/dataset.py`, pass `self.config.type_gate_config` into `prepare_detection_training_example(...)` in the non-prefix branch:

```python
prepared = prepare_detection_training_example(
    normalized,
    template=detection_template,
    tokenizer=self.tokenizer,
    mode=self.config.mode,
    state_weighting=self._state_weighting_for_prepare(),
    normalization=self._normalization_for_prepare(),
    type_gate_config=self.config.type_gate_config,
    messages=messages,
)
```

- [ ] **Step 5: Run the test**

Run:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_target_builder.py::test_random_permutation_type_gate_attaches_allowed_type_ids -q
```

Expected: pass.

## Task 2: Keep Structural Boundary Tokens Hard CE

**Files:**
- Modify: `src/detection/objective.py`
- Test: `tests/test_recursive_detection_ce_target_builder.py`

- [ ] **Step 1: Write the failing shared-prefix boundary test**

Update the existing `test_compact_prefix_description_can_branch_into_box_start_or_desc_continuation` expectation, or add this new test:

```python
def test_structural_box_start_stays_hard_ce_when_description_prefix_branches() -> None:
    tokenizer = SpecialTokenAwareTokenizer()
    sample = _sample(
        _object(
            normalized_index=0,
            source_index=7,
            instance_id="img-9:ann-1301:src-7",
            desc="car",
            coords=("<|coord_10|>", "<|coord_20|>", "<|coord_30|>", "<|coord_40|>"),
        ),
        _object(
            normalized_index=1,
            source_index=3,
            instance_id="img-9:ann-1302:src-3",
            desc="cart",
            coords=("<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"),
        ),
    )

    prepared = prepare_detection_training_example(
        sample,
        template=CompactFullTemplate(),
        tokenizer=tokenizer,
        mode="random_permutation_et_rmp_ce",
    )
    first_entry = prepared.tokenized.object_entries[0]
    targets = _target_map(prepared)

    bbox_start_target = targets[first_entry.bbox_start_span.start]
    assert bbox_start_target.kind == "hard_ce"
    assert bbox_start_target.token_role is TokenRole.BBOX_START
    assert _token_texts(tokenizer, bbox_start_target.valid_token_ids) == (BOX_START_TOKEN,)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_target_builder.py::test_structural_box_start_stays_hard_ce_when_description_prefix_branches -q
```

Expected: fail because current trie construction can make the `<|box_start|>` row multi-positive with a desc-continuation child.

- [ ] **Step 3: Restrict trie multi-positive target kind to desc tokens**

In `_append_recursive_entry_targets(...)`, replace the target-kind decision with role-aware logic:

```python
token_role = tokenized.token_roles[position]
kind: TrieTargetKind = (
    "trie_multi_positive"
    if len(trie_branch_targets) > 1 and token_role is TokenRole.DESC
    else "hard_ce"
)
```

Use `token_role=token_role` when constructing `TokenTarget` so the role is read once.

Keep `coord_soft_targets` and `coord_instance_candidates` attached for coordinate positions even when `kind == "hard_ce"`.

- [ ] **Step 4: Run target-builder tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_recursive_detection_ce_target_builder.py::test_structural_box_start_stays_hard_ce_when_description_prefix_branches \
  tests/test_recursive_detection_ce_target_builder.py::test_compact_first_desc_token_uses_remaining_object_trie \
  tests/test_recursive_detection_ce_target_builder.py::test_compact_same_desc_diverges_at_first_coordinate_token \
  -q
```

Expected: new structural test passes; the same-desc coordinate test will need its expectation updated so the coordinate target is hard CE metadata with coordinate sidecars rather than trie multi-positive.

- [ ] **Step 5: Update coordinate target metadata expectation**

In `test_compact_same_desc_diverges_at_first_coordinate_token`, change:

```python
assert first_coord_target.kind == "trie_multi_positive"
```

to:

```python
assert first_coord_target.kind == "hard_ce"
```

Keep assertions for `coord_soft_targets` because coordinate SoftCE still needs candidate sidecars.

## Task 3: Description Trie Loss Adds Hard CE

**Files:**
- Modify: `src/detection/loss.py`
- Test: `tests/test_recursive_detection_ce_loss_adapter.py`

- [ ] **Step 1: Write the failing loss test**

Add this test near the trie loss tests:

```python
def test_description_trie_loss_adds_teacher_hard_ce_anchor() -> None:
    target = _branch_target(
        position=1,
        teacher_token_id=0,
        branches=((0, 1), (1, 1)),
        semantic_role=SemanticRole.DESC_IDENTITY,
    )
    target = replace(target, token_role=TokenRole.DESC)
    logits = torch.tensor([[0.2, 1.4, -0.5]], dtype=torch.float32)

    result = compute_recursive_detection_ce_batch_loss(
        logits,
        _targets(token_targets=(target,)),
        weights=RecursiveDetectionLossWeights(support_weight=2.0, balance_weight=1.0),
    )

    step_log_probs = torch.log_softmax(logits[0], dim=-1)
    hard_ce = -step_log_probs[0]
    support_balance = loss_module.support_balance_loss(
        logits[0],
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        support_weight=2.0,
        balance_weight=1.0,
    )
    assert result.loss.item() == pytest.approx((hard_ce + support_balance).item())
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py::test_description_trie_loss_adds_teacher_hard_ce_anchor -q
```

Expected: fail because current trie positions use only support/balance loss.

- [ ] **Step 3: Add hard CE only for desc trie targets**

In `_compute_sample_loss(...)`, after computing `position_loss = support_balance_loss(...)`, add:

```python
if target.token_role is TokenRole.DESC:
    position_loss = position_loss + (-step_log_probs[target.teacher_token_id])
```

If importing `TokenRole` into `loss.py` would introduce an undesirable dependency, compare `str(getattr(target, "token_role", "")) == "TokenRole.DESC"` is not acceptable. Import `TokenRole` from `src.detection.tokenization` and use the enum directly.

Mirror the same diagnostic-only logic in `_recursive_objective_diagnostic_events(...)` if it computes per-position trie CE metrics independently; otherwise aggregate diagnostics will not match training loss.

- [ ] **Step 4: Run focused loss tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_recursive_detection_ce_loss_adapter.py::test_description_trie_loss_adds_teacher_hard_ce_anchor \
  tests/test_recursive_detection_ce_loss_adapter.py::test_type_gate_allowed_mass_is_added_to_position_loss \
  tests/test_recursive_detection_ce_loss_adapter.py::test_eos_loss_weight_scales_main_ce_but_not_type_gate \
  -q
```

Expected: pass.

## Task 4: Coord-Vocab SoftCE For Instance-Trie Gaussian

**Files:**
- Modify: `src/detection/coord_soft_targets.py`
- Modify: `src/detection/loss.py`
- Test: `tests/test_instance_trie_gaussian_coord_softce.py`
- Test: `tests/test_recursive_detection_ce_loss_adapter.py`

- [ ] **Step 1: Write the failing coord-only normalization test**

Add to `tests/test_instance_trie_gaussian_coord_softce.py`:

```python
def test_instance_trie_gaussian_softce_ignores_non_coord_logits() -> None:
    cfg = _cfg()
    candidate = _candidate("box", "x1", (100, 100, 160, 220))
    logits = torch.zeros(1200, dtype=torch.float32)
    logits[10:1010] = torch.linspace(-1.0, 1.0, steps=1000)
    baseline = full_vocab_coord_soft_ce(
        logits,
        (candidate,),
        cfg,
        current_slot="x1",
        teacher_coord_value=100,
    )

    changed = logits.clone()
    changed[1100] = 50.0
    with_non_coord_spike = full_vocab_coord_soft_ce(
        changed,
        (candidate,),
        cfg,
        current_slot="x1",
        teacher_coord_value=100,
    )

    assert with_non_coord_spike.weighted_loss.item() == pytest.approx(
        baseline.weighted_loss.item()
    )
```

This test intentionally keeps the existing function name to minimize public API churn. The implementation can add a helper internally.

- [ ] **Step 2: Run the failing test**

Run:

```bash
conda run -n ms python -m pytest tests/test_instance_trie_gaussian_coord_softce.py::test_instance_trie_gaussian_softce_ignores_non_coord_logits -q
```

Expected: fail because full-vocab normalization currently changes when a non-coordinate logit spikes.

- [ ] **Step 3: Implement coord-vocab normalization for instance-trie Gaussian**

In `full_vocab_coord_soft_ce(...)`, keep legacy IoU/CIoU routing unchanged. For `instance_trie_gaussian`, replace:

```python
log_probs = F.log_softmax(logits.float(), dim=-1)
coord_log_probs = log_probs.index_select(0, dist.token_ids)
```

with:

```python
coord_logits = logits.float().index_select(0, dist.token_ids)
coord_log_probs = F.log_softmax(coord_logits, dim=-1)
```

Keep `target_probs` aligned with `dist.token_ids`.

Update `support_mass` for this path to represent coordinate-vocab mass, not full-vocab mass:

```python
support_mass = coord_log_probs.exp().sum()
outside_support_mass = 1.0 - support_mass
```

For coord-vocab softmax, `support_mass` should be 1.0 up to floating-point error. If this metric name is misleading, keep it for compatibility and add a new numeric diagnostic in the returned dataclass later.

- [ ] **Step 4: Add a loss-adapter test for type gate owning non-coord leakage**

Add to `tests/test_recursive_detection_ce_loss_adapter.py`:

```python
def test_coord_type_gate_penalizes_non_coord_leakage_separately_from_softce() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )
    target = _hard_target(
        position=1,
        teacher_token_id=110,
        semantic_role=SemanticRole.BBOX_COORD,
        token_role=TokenRole.COORD,
        coord_instance_candidates=(
            CoordInstanceCandidateSpec("obj-0", (100, 100, 160, 220)),
        ),
        coord_slot_name="x1",
    )
    gated = replace(
        target,
        type_gate_token_ids=tuple(range(10, 1010)),
        type_gate_weight=0.5,
    )
    logits = torch.zeros(1, 1200, dtype=torch.float32)
    logits[0, 10:1010] = torch.linspace(-1.0, 1.0, steps=1000)
    baseline = compute_recursive_detection_ce_batch_loss(
        logits,
        _targets(token_targets=(gated,)),
        weights=RecursiveDetectionLossWeights(coord_soft_ce=cfg),
    )

    leaked_logits = logits.clone()
    leaked_logits[0, 1100] = 50.0
    leaked = compute_recursive_detection_ce_batch_loss(
        leaked_logits,
        _targets(token_targets=(gated,)),
        weights=RecursiveDetectionLossWeights(coord_soft_ce=cfg),
    )

    assert leaked.loss.item() > baseline.loss.item()
```

- [ ] **Step 5: Run coord SoftCE tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_instance_trie_gaussian_coord_softce.py::test_instance_trie_gaussian_softce_ignores_non_coord_logits \
  tests/test_recursive_detection_ce_loss_adapter.py::test_coord_type_gate_penalizes_non_coord_leakage_separately_from_softce \
  -q
```

Expected: pass.

## Task 5: Config Schema And Production Config Wiring

**Files:**
- Modify: `src/config/schema.py`
- Modify: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_instance_trie_gaussian_softce_a5.yaml`
- Modify: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ce_gaussian_mix0p2_a6.yaml`
- Modify: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ce_gaussian_mix0p5_a7.yaml`
- Modify matching smoke configs for A5/A6/A7
- Test: `tests/test_latest_training_config_contract.py`
- Test: `tests/test_instance_trie_gaussian_config_diff.py`

- [ ] **Step 1: Write the failing schema acceptance test**

Add to `tests/test_latest_training_config_contract.py`:

```python
def test_random_permutation_et_rmp_accepts_type_gate_section() -> None:
    payload = _base_latest_detection_config()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "type_gate": {
            "enabled": True,
            "mode": "allowed_type_mass",
            "weights": {
                "struct": 1.0,
                "coord": 1.0,
                "desc": 1.0,
                "eos": 0.5,
            },
        },
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.type_gate is not None
    assert cfg.objective.type_gate.enabled is True
    assert cfg.objective.type_gate.weights.coord == pytest.approx(1.0)
```

- [ ] **Step 2: Run the failing schema test**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_random_permutation_et_rmp_accepts_type_gate_section -q
```

Expected: fail because objectized objective sections are currently accepted only for prefix-rollin.

- [ ] **Step 3: Update schema validation**

In `DetectionObjectiveConfig.__post_init__`, allow `type_gate` when:

```python
self.variant in {"random_permutation_et_rmp_ce", "prefix_rollin_et_rmp_ce"}
```

Keep `rollin`, `target`, `boundary`, and `eos` restricted to `prefix_rollin_et_rmp_ce`.

For non-prefix variants, replace the current unexpected-section check with:

```python
unexpected = [
    name
    for name in ("rollin", "target", "boundary", "eos")
    if getattr(self, name) is not None
]
if self.variant != "prefix_rollin_et_rmp_ce" and unexpected:
    raise ValueError(...)
if self.variant not in {"random_permutation_et_rmp_ce", "prefix_rollin_et_rmp_ce"} and self.type_gate is not None:
    raise ValueError(...)
```

- [ ] **Step 4: Add config sections**

Add this section to each instance-trie Gaussian production and smoke config:

```yaml
objective:
  type_gate:
    enabled: true
    mode: allowed_type_mass
    weights:
      struct: 1.0
      desc: 1.0
      coord: 1.0
      eos: 0.5
```

Preserve existing `objective.coord_soft_ce` fields in the same `objective` mapping.

- [ ] **Step 5: Update config-diff whitelist**

In `tests/test_instance_trie_gaussian_config_diff.py`, add allowed paths:

```python
"/objective/type_gate/enabled",
"/objective/type_gate/mode",
"/objective/type_gate/weights/struct",
"/objective/type_gate/weights/desc",
"/objective/type_gate/weights/coord",
"/objective/type_gate/weights/eos",
```

Assert all A5/A6/A7 configs resolve with type gate enabled and the expected weights.

- [ ] **Step 6: Run config tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_training_config_contract.py::test_random_permutation_et_rmp_accepts_type_gate_section \
  tests/test_instance_trie_gaussian_config_diff.py \
  -q
```

Expected: pass.

## Task 6: Metrics And Regression Sweep

**Files:**
- Modify: `src/trainers/metrics/recursive_detection.py`
- Test: `tests/test_latest_training_config_contract.py`
- Test: `tests/test_recursive_detection_ce_loss_adapter.py`

- [ ] **Step 1: Add metric expectation test**

Extend existing metric tests to assert type-gate config fields are visible when enabled:

```python
assert logged["recursive_detection_ce/type_gate/config_enabled"] == pytest.approx(1.0)
assert logged["recursive_detection_ce/type_gate/struct_weight"] == pytest.approx(1.0)
assert logged["recursive_detection_ce/type_gate/coord_weight"] == pytest.approx(1.0)
assert logged["recursive_detection_ce/type_gate/desc_weight"] == pytest.approx(1.0)
assert logged["recursive_detection_ce/type_gate/eos_weight"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the failing metric test**

Run the specific metric test that was extended:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_instance_trie_gaussian_metrics_do_not_require_coord_softce_tau -q
```

Expected: fail until metrics expose type-gate config.

- [ ] **Step 3: Emit type-gate config metrics**

In `src/trainers/metrics/recursive_detection.py`, when `cfg` has a `type_gate` field, add numeric metrics:

```python
"recursive_detection_ce/type_gate/config_enabled": 1.0 if type_gate_enabled else 0.0
"recursive_detection_ce/type_gate/struct_weight": float(weights.struct)
"recursive_detection_ce/type_gate/coord_weight": float(weights.coord)
"recursive_detection_ce/type_gate/desc_weight": float(weights.desc)
"recursive_detection_ce/type_gate/eos_weight": float(weights.eos)
```

If `type_gate` is absent, emit `config_enabled: 0.0` and omit weights to avoid inventing defaults.

- [ ] **Step 4: Run focused regression tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_compact_type_gate.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_instance_trie_gaussian_coord_softce.py \
  tests/test_latest_training_config_contract.py \
  tests/test_instance_trie_gaussian_config_diff.py \
  -q
```

Expected: pass.

## Task 7: Documentation Update

**Files:**
- Modify: `docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`

- [ ] **Step 1: Update instance-trie draft**

Add a short section to `docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md`:

```markdown
## Orthogonal Token-Type Supervision

The active feature-branch objective separates token-type validity from
coordinate smoothness. Schema/control/boundary tokens use hard CE plus the
struct/eos type gate. Free-text description tokens use hard CE and, at
description trie branch positions, support/balance trie CE plus the desc type
gate. Coordinate tokens use the coord type gate for coordinate-token
exclusivity and `instance_trie_gaussian` SoftCE over the 1000 coordinate-token
vocabulary for smooth coordinate supervision.

Coordinate SoftCE should not be interpreted as a full-vocabulary gate. Its
softmax scope is the coordinate-token vocabulary; full-vocabulary leakage is
owned by `objective.type_gate`.
```

- [ ] **Step 2: Update Stage-1 objective summary**

In `docs/training/STAGE1_OBJECTIVE.md`, update the instance-trie Gaussian bullet
to mention:

```text
type-gated schema/desc/coord/eos positions
structural boundary hard CE
description hard CE + trie support/balance
coord-vocab-only Gaussian SoftCE
```

- [ ] **Step 3: Read docs snippets**

Run:

```bash
sed -n '40,120p' docs/training/INSTANCE_TRIE_GAUSSIAN_SOFTCE_DRAFT.md
sed -n '120,150p' docs/training/STAGE1_OBJECTIVE.md
```

Expected: snippets describe the separated roles without saying coordinate SoftCE performs token-type gating.

## Final Verification

- [ ] **Step 1: Run focused tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_compact_type_gate.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_instance_trie_gaussian_coord_softce.py \
  tests/test_latest_training_config_contract.py \
  tests/test_instance_trie_gaussian_config_diff.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run config parse smoke**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_latest_compact_sft_smoke_configs_parse_with_hard_ce_objectives -q
```

Expected: pass, proving unrelated compact SFT configs still parse.

- [ ] **Step 3: Inspect diff**

Run:

```bash
rtk git diff --stat
rtk git diff --name-only
```

Expected: changed files are limited to objective/loss/runtime/config/tests/docs and the intended A5/A6/A7 configs.

## Self-Review Checklist

- Spec coverage: tasks cover structural hard CE, desc hard-plus-trie CE, global type-gate wiring, coord-vocab SoftCE, config wiring, metrics, docs, and focused verification.
- Placeholder scan: this plan contains no placeholder implementation steps.
- Type consistency: all named functions and files exist in the current worktree except the new behavior-specific tests and helper rename planned above.
- Commit policy: this plan intentionally omits commit steps because the repo instructions say to use commits when requested; ask the user before staging or committing.
