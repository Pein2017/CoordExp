# Coord-Repel V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative Stage-1 teacher-forced coord-repel auxiliary loss, exact packed teacher-forcing remap support, and the independent compact-box-end template ablation without changing inference-time decoding or attention behavior.

**Architecture:** Keep the teacher-forcing target IR as a lightweight transport object. Add small typed helpers for packed physical layout, teacher-forcing atom remapping, and coord-repel slot/loss computation. Integrate coord-repel inside the existing `teacher_forcing` objective path so training still performs one model forward and uses the current ms-swift, Transformers, and FlashAttention v2 runtime surface.

**Tech Stack:** Python dataclasses, PyTorch fp32 loss math, existing CoordExp teacher-forcing objective runner, ms-swift static packing/padding-free batches, Transformers `flash_attention_2`, pytest.

---

## Current State And Launch Gate

Status: implementation approved and partially implemented. The source/config
work through coord-repel core loss, objective integration, `compact_box_end`,
packed teacher-forcing remap, and packed coord-repel config leaves has passed
targeted unit/config verification. Production eligibility is still blocked on
real-GPU smoke, longer loss-numerics stability, and task-health checks.

Launch gate: do not start 8-GPU 4-epoch production training unless the packed
smoke and stability gates pass with sane coord-repel metrics, no NaNs, no
sidecar remap errors, and a decreasing/controlled total-loss trend.

Primary source records:
- `docs/superpowers/specs/2026-06-12-coord-repel-v1-design.md`
- `progress/explorations/2026-06-12_coord_repel_stage1_sft_design_decisions.md`

Current implemented code facts:
- `src/training/teacher_forcing/coord_repel.py` owns slot extraction, B+/B-
  construction, fp32 coord-repel loss math, and diagnostics.
- `src/training/objectives/teacher_forcing.py` adds coord-repel as an internal
  auxiliary term while keeping main teacher-forcing loss semantics intact.
- `src/data_collators/packed_layout.py` derives physical packed-row layout and
  validates active length against source lengths.
- `src/training/teacher_forcing/packing.py` remaps per-source atoms into one
  merged `TeacherForcingTargetIR` per physical packed row.
- `src/data_collators/enrichers.py::TeacherForcingTargetIREnricher` now
  supports exact packed teacher-forcing sidecar remap instead of rejecting it.
- `src/detection/runtime.py::assert_detection_runtime_supported` allows packed
  teacher forcing only under exact static-packing guardrails and keeps encoded
  cache rejected.
- Config leaves live under
  `configs/stage1/detection_teacher_forcing/coord_repel/`.

Recent targeted verification:
- `145 passed, 3 xfailed` for batch extras, sidecar bridge, teacher-forcing
  config contract, packing fingerprints, and coord-repel objective integration.
- Config materialization/runtime guard validation passed for the packed
  coord-repel production, smoke, base-2B, and `compact_box_end` leaves.

## Design Decisions Locked For V1

- Stage-1 teacher-forced SFT only.
- Forward-pass eval uses the same teacher-forcing path as training; no rollout dependency.
- No inference-time changes, no KV-cache changes, no attention-score changes, no decode constraints.
- Coord-repel roles are `x1` and `y1` only.
- `B+` is selected GT coord bin +/- 4 inside the coordinate-token vocabulary.
- `B-` is `top_k_wrong outside B+` intersected with prior same-sample same-role bands from `negative_source`.
- Default `negative_source` is `last_and_same_desc`.
- `all_prior` is diagnostic/stress mode only.
- Balanced main config uses `weight: 0.05`, `margin: 0.25`, `top_k: 32`.
- Positive-band anchor loss is out of scope for V1.
- Static packing support is required before claiming coord-repel supports the intended training speedup.
- V1 does not add a new public `packing.remap_contract` key. It uses the existing `objective.target_ir.exact_packing_mapping.enabled` key as the explicit packed teacher-forcing guard: unpacked teacher-forcing may keep the default false value, but packed teacher-forcing requires `true` and records the effective remap contract as `teacher_forcing_atoms_v1`.
- Encoded-sample cache remains disabled for latest teacher-forcing V1 unless a separate cache-replay proof is added.
- Ordinary static packing uses `training.packing=true` and
  `packing.static_packing=true`; `packing.padding_free_packed=true` remains
  rejected for this surface.
- Production candidate data is the COCO `rescale_32_1024_bbox_len12000` dataset,
  not the legacy `max60` dataset.
- Production candidate training uses LLM-tower-only LoRA (`freeze_vit=true`,
  `freeze_aligner=true`, `train_type=lora`) with effective packed batch size
  32. On 8 GPUs and per-device packed batch size 1, this implies gradient
  accumulation 4.

## Remaining Launch Tasks

- [ ] Run packed smoke on
  `configs/stage1/detection_teacher_forcing/coord_repel/smoke/packed_len12000_2b_coordexp_smoke.yaml`.
- [ ] Run base-2B packed smoke on
  `configs/stage1/detection_teacher_forcing/coord_repel/smoke/packed_len12000_2b_base_smoke.yaml`.
- [ ] Run longer stability probe on
  `configs/stage1/detection_teacher_forcing/coord_repel/smoke/packed_len12000_2b_coordexp_stability.yaml`.
- [ ] Inspect loss numerics: no NaNs/Infs, total loss not exploding,
  `teacher_forcing/coord_repel/weighted_loss_to_main_tf_loss_ratio` initially
  in a controlled range, active-slot metrics nonzero enough to verify the
  module is not asleep.
- [ ] Inspect packed-remap health: no selected-token mismatch, no left-padding
  preflight failure, no source-length vs attention-mask mismatch.
- [ ] Inspect static-plan coverage: require `raw_missing_sample_count == 0`,
  `underfill_dropped_sample_count == 0`, and acceptable DDP repeat padding in
  `plan_ws8_drop0_align32.json` before interpreting production as full COCO
  len12000 training. For the 8-GPU production leaf, `N_aligned_packs` must be
  divisible by `32` so every optimizer step has a full 32 packed-sequence window.
- [ ] If the smoke and stability gates pass, launch production 8-GPU 4-epoch
  run with
  `configs/stage1/detection_teacher_forcing/coord_repel/prod/packed_len12000_2b_coordexp.yaml`.
- [ ] Keep `compact_box_end` as an independent ablation axis via
  `configs/stage1/detection_teacher_forcing/coord_repel/ablation/compact_box_end_packed_len12000_2b_coordexp.yaml`.

## Open Design Questions For User Approval

1. Main profile for first ablation:
   - Recommended: keep the current `pure_valid_set_marginal` teacher-forcing profile for continuity with the active Stage-1 route.
   - Risk: the main teacher-forcing denominator can be misread as singleton CE, while `pure_valid_set_marginal` trains `-log P(valid next-token set)` when multiple valid tokens exist.
   - Mitigation in this plan: coord-repel is compatible with both `hard_sft` and `pure_valid_set_marginal`; metrics must log `teacher_forcing/profile` and main-loss semantics so ablations are not mislabeled. Ratio metrics must refer to `main_tf_loss`, not CE, unless the profile is explicitly `hard_sft`.

2. `same_desc` key definition:
   - Recommended: use normalized `category_name` when present, falling back to normalized `desc`.
   - Rationale: COCO-style data has category names; free-form desc strings can include tokenization or wording drift.
   - Implementation detail: store both `category_id` and `desc_key`/`category_key` in atom provenance, then have coord-repel use a single `duplicate_group_key`.

3. `<|box_end|>` token source:
   - Recommended: require a tokenized single special token in the model/tokenizer before enabling `compact_box_end`.
   - Risk: adding a new token changes vocab, token-row config, save/load behavior, and reviewer comparison shape.
   - Implementation detail: implement as a separate milestone after core coord-repel math and packed remap tests pass.

## File Structure

Create:
- `src/data_collators/packed_layout.py`: generic physical packed-row layout and validation.
- `src/training/teacher_forcing/packing.py`: teacher-forcing-specific atom offset remap and merged packed-row IR construction.
- `src/training/teacher_forcing/coord_repel.py`: config normalization, slot extraction, band construction, fp32 loss, and diagnostics.
- `tests/test_packed_batch_layout.py`: raw packed batch to physical layout tests.
- `tests/test_teacher_forcing_packing_remap.py`: packed target-IR remap and no cross-sample leakage tests.
- `tests/test_coord_repel_loss.py`: deterministic formula, band, activation, and negative-source tests.
- `tests/test_coord_repel_objective_integration.py`: objective-level weighted contribution and metric tests.
- `tests/test_coord_repel_eval_forward_contract.py`: train/eval packed forward contract and metric-prefix tests.
- `tests/test_compact_box_end_template.py`: template render/parse/token tests.

Modify:
- `src/config/schema.py`: add strict `coord_repel` module config and template ID/config support for `compact_box_end` if approved.
- `src/detection/teacher_forcing/trie.py`: add duplicate-group metadata to `TokenBranch`.
- `src/detection/teacher_forcing/target_builder.py`: propagate duplicate-group metadata into atom provenance and support the selected compact template entry renderer.
- `src/data_collators/enrichers.py`: replace packed teacher-forcing rejection with layout/remap integration.
- `src/data_collators/batch_extras_collator.py`: pass packed layout to enrichers when needed.
- `src/trainers/metrics/teacher_forcing.py`: pass coord-repel config and profile metadata into `ObjectiveSpec`.
- `src/training/objectives/teacher_forcing.py`: add coord-repel auxiliary term to the existing `teacher_forcing` objective result.
- `src/training/teacher_forcing/metrics.py`: add coord-repel metric event helpers.
- `src/detection/runtime.py`: relax teacher-forcing packing guards only for the validated static packed contract; keep encoded cache disabled.
- `src/detection/packing.py`: replace teacher-forcing static-packing rejection with the validated `teacher_forcing_atoms_v1` packing identity and fingerprint fields.
- `src/detection/evaluation.py`: support `compact_box_end` parsing/eval routing if the template milestone is approved.
- `src/sft.py`: include coord-repel and packed-remap/template identity in runtime metadata and static-packing fingerprint.
- `src/common/detection_compact_rows.py`: define `BOX_END_TOKEN` only if the template milestone is approved.
- `src/detection/template.py`: add `compact_box_end` template support only if the template milestone is approved.
- `configs/stage1/detection_teacher_forcing/`: add smoke/prodlike coord-repel ablation leaves after schema and runtime tests exist.
- `docs/data/PACKING.md`: update teacher-forcing packing policy after tests pass.
- `docs/training/STAGE1_OBJECTIVE.md`: document coord-repel V1 semantics and limitations after tests pass.
- `docs/AGENT_INDEX.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/training/README.md`, and `docs/catalog.yaml`: update routing only if the branch promotes coord-repel or `compact_box_end` beyond experiment-only opt-in.
- `openspec/changes/`: create a stable-contract change only if final approval scopes V1 as stable public behavior rather than experiment-only opt-in.

## Implementation Tasks

### Task 1: Add Strict Coord-Repel Config Contract

**Files:**
- Modify: `src/config/schema.py`
- Test: `tests/test_teacher_forcing_config_contract.py`

- [ ] **Step 1: Write config tests first**

Add tests that prove:
- `objective.modules.coord_repel` is accepted with only `enabled`, `weight`, `margin`, `top_k`, and `negative_source`.
- unknown keys under `coord_repel` fail fast.
- `negative_source` accepts only `last_only`, `same_desc`, `last_and_same_desc`, and `all_prior`.
- defaults are disabled and V1-balanced values are available when explicitly enabled.
- `coord_repel` is allowed under `hard_sft` and `pure_valid_set_marginal`.
- `objective.target_ir.exact_packing_mapping.enabled=true` is accepted by schema for teacher-forcing configs but is only required by runtime when packing is enabled.

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py -q
```

Expected before implementation: failures mentioning missing `coord_repel` parsing, unknown `objective.modules.coord_repel`, or rejected `exact_packing_mapping.enabled=true`.

- [ ] **Step 2: Implement schema**

Add a frozen dataclass near `TeacherForcingModulesConfig`:

```python
@dataclass(frozen=True)
class TeacherForcingCoordRepelConfig:
    enabled: bool = False
    weight: float = 0.05
    margin: float = 0.25
    top_k: int = 32
    negative_source: Literal[
        "last_only",
        "same_desc",
        "last_and_same_desc",
        "all_prior",
    ] = "last_and_same_desc"
```

Validation rules:
- `enabled` must be a plain bool.
- `weight` must be finite and non-negative.
- `margin` must be finite and non-negative.
- `top_k` must be a positive int.
- `negative_source` must be one of the four V1 literals.

Extend `TeacherForcingModulesConfig.from_mapping()` to pop `coord_repel` strictly and store it on the modules dataclass.

Also update `TeacherForcingTargetIRConfig` so `exact_packing_mapping.enabled=true` can parse. This key is not a new public surface; it is an existing dormant guard that becomes meaningful once packed teacher-forcing remap is implemented.

- [ ] **Step 3: Preserve profile behavior**

Update `TeacherForcingObjectiveConfig.__post_init__()` so `hard_sft` still rejects valid-set-only modules, but does not reject `coord_repel.enabled`.

Reason: coord-repel uses selected coordinate atoms and provenance. It does not require multi-positive valid-set runtime semantics.

- [ ] **Step 4: Rerun config tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py -q
```

Expected after implementation: pass.

### Task 2: Promote Duplicate-Group Provenance For Same-Desc Negatives

**Files:**
- Modify: `src/detection/teacher_forcing/trie.py`
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Test: `tests/test_teacher_forcing_target_builder.py`

- [ ] **Step 1: Write failing provenance tests**

Add tests that build a sample with two objects sharing `category_name` or `desc` and verify coordinate atoms carry:

```python
atom.provenance["category_id"]
atom.provenance["category_key"]
atom.provenance["desc_key"]
atom.provenance["duplicate_group_key"]
```

Expected behavior:
- `duplicate_group_key` prefers normalized `category_name` when present.
- fallback is normalized `desc` when `category_name` is absent or normalizes to an empty string.
- whitespace normalization makes `"Fire  hydrant"` and `"fire hydrant"` share the same key.
- the terminal stop atom has no duplicate-group key.

Run:

```bash
python -m pytest tests/test_teacher_forcing_target_builder.py -q
```

Expected before implementation: missing provenance key failure.

- [ ] **Step 2: Extend `TokenBranch` metadata**

Add immutable fields:

```python
category_id: int
category_key: str
desc_key: str
duplicate_group_key: str
```

Normalize keys with a local helper that strips leading/trailing whitespace, lowercases, and collapses internal whitespace to one ASCII space. Treat an empty normalized `category_key` as absent and fall back to `desc_key`; if both normalize to empty, fail fast in the target builder because `same_desc` negatives would be undefined.

- [ ] **Step 3: Populate atom provenance**

In `_prepare_object()`, compute keys from `NormalizedDetectionObject`. In `_build_atoms()`, add the four fields to every object token atom provenance.

- [ ] **Step 4: Rerun target-builder tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_target_builder.py tests/test_teacher_forcing_ir_contract.py -q
```

Expected after implementation: pass.

### Task 3: Build Coord-Repel Core Loss Module

**Files:**
- Create: `src/training/teacher_forcing/coord_repel.py`
- Test: `tests/test_coord_repel_loss.py`

- [ ] **Step 1: Write deterministic loss tests**

Cover:
- high `B+` mass gives lower loss than high `B-` mass.
- `B+` uses selected coord bin +/- 4 clipped to valid coord bins.
- `B-` excludes `B+`.
- empty `B-` produces zero weighted loss and increments `empty_bminus_slots`.
- missing `B+` support fails fast.
- only `x1` and `y1` are eligible.
- `x2` and `y2` are ignored.
- future objects are not negative candidates.
- `last_only`, `same_desc`, `last_and_same_desc`, and `all_prior` select different prior sets as specified.

Run:

```bash
python -m pytest tests/test_coord_repel_loss.py -q
```

Expected before implementation: import failure for `src.training.teacher_forcing.coord_repel`.

- [ ] **Step 2: Implement public internal API**

Use these internal dataclasses and functions:

```python
@dataclass(frozen=True, slots=True)
class CoordRepelRuntimeConfig:
    enabled: bool
    weight: float
    margin: float
    top_k: int
    negative_source: str
    positive_radius: int = 4
    negative_radius: int = 4
    roles: tuple[str, ...] = ("x1", "y1")

@dataclass(frozen=True, slots=True)
class CoordRepelVocab:
    coord_token_ids: tuple[int, ...]
    token_id_to_bin: Mapping[int, int]

@dataclass(frozen=True, slots=True)
class CoordRepelSlot:
    source_sample_id: str
    physical_batch_index: int
    object_index: int
    object_order_index: int
    object_instance_id: str
    duplicate_group_key: str
    coord_role: str
    coord_bin: int
    selected_token_id: int
    logit_position: int
    target_position: int

@dataclass(frozen=True, slots=True)
class CoordRepelResult:
    loss_sum: torch.Tensor
    active_slot_count: int
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    metrics: Mapping[str, float | int | str]
```

Required callable signatures:

```python
CoordRepelSlotExtractor = Callable[
    [TeacherForcingTargetIR],
    tuple[CoordRepelSlot, ...],
]
CoordRepelLossFn = Callable[
    [torch.Tensor, TeacherForcingTargetIR, RoleVocab, CoordRepelVocab, CoordRepelRuntimeConfig],
    CoordRepelResult,
]
```

The implementation must expose concrete functions named `extract_coord_repel_slots()`, `build_coord_repel_vocab()`, and `compute_coord_repel_loss()`. `compute_coord_repel_loss()` takes `logits_by_atom`, `target_ir`, `role_vocab`, `coord_vocab`, and `config`; `logits_by_atom` is the already-resolved tensor shaped `[atom_count, vocab]` for one target IR span. Coord-repel must not index raw batch offsets itself.

- [ ] **Step 3: Implement fp32 coordinate-vocab probability math**

Inside `compute_coord_repel_loss()`:
- disable autocast on CPU/CUDA.
- convert selected rows to fp32.
- compute `coord_log_probs = log_softmax(logits_row[coord_vocab.coord_token_ids])`.
- map global coord token IDs to coordinate bins through `coord_vocab.token_id_to_bin`, not by sorting `RoleVocab.coord_token_ids`.
- compute `log_p_pos` and `log_p_neg` with `torch.logsumexp`.
- compute `slot_loss = softplus(margin + log_p_neg - log_p_pos)`.

`build_coord_repel_vocab()` must consume an ordered 1000-length coord-token-id tuple resolved from the tokenizer via existing coord-token utilities such as `get_coord_token_ids(..., validate=True)`. It must fail fast if the ordered tuple does not match `role_vocab.coord_token_ids` as a set.

- [ ] **Step 4: Implement metrics**

At minimum return:
- `coord_repel/raw_loss`
- `coord_repel/weighted_loss`
- `coord_repel/active_slots`
- `coord_repel/eligible_x1_slots`
- `coord_repel/eligible_y1_slots`
- `coord_repel/empty_bminus_slots`
- `coord_repel/p_pos_mean`
- `coord_repel/p_neg_mean`
- `coord_repel/log_margin_mean`
- `coord_repel/topk_intersection_count`
- `coord_repel/wrong_top1_in_bneg_rate`
- `coord_repel/b_seen_coverage_fraction`
- `coord_repel/weighted_loss_to_main_tf_loss_ratio`
- `coord_repel/negative_source_mode`

- [ ] **Step 5: Rerun core tests**

Run:

```bash
python -m pytest tests/test_coord_repel_loss.py -q
```

Expected after implementation: pass.

### Task 4: Integrate Coord-Repel Into The Existing Teacher-Forcing Objective

**Files:**
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/training/objectives/teacher_forcing.py`
- Modify: `src/training/teacher_forcing/metrics.py`
- Test: `tests/test_coord_repel_objective_integration.py`
- Test: `tests/test_teacher_forcing_objective_runner.py`

- [ ] **Step 1: Write integration tests**

Cover:
- disabled coord-repel leaves current teacher-forcing loss unchanged.
- enabled coord-repel adds `coord_repel.weighted_loss` to `ObjectiveResult.weighted_loss`.
- `teacher_forcing/loss/total` remains the main teacher-forcing objective loss.
- metric events include coord-repel metrics.
- `coord_repel/weighted_loss_to_main_tf_loss_ratio` uses main teacher-forcing loss as denominator and is finite when the main loss is nonzero.
- the resolved main profile is logged as `teacher_forcing/profile` or included in resolved runtime metadata so `hard_sft` and `pure_valid_set_marginal` runs cannot be confused.
- `teacher_forcing/main_loss_semantics` or equivalent metadata distinguishes `hard_sft_singleton_ce` from `valid_set_marginal`.
- a deterministic two-valid-token test proves the ratio is not named as CE under `pure_valid_set_marginal`.
- multiple spans reduce by global active-slot count, not average of span means; the test must use uneven active-slot counts such as 1 active slot in one span and 9 in another.

Run:

```bash
python -m pytest tests/test_coord_repel_objective_integration.py tests/test_teacher_forcing_objective_runner.py -q
```

Expected before implementation: missing coord-repel integration failure.

- [ ] **Step 2: Pass config through the trainer mixin**

In `TeacherForcingObjectiveMixin.compute_loss()`, include this in the `ObjectiveSpec("teacher_forcing").config` payload:

```python
"coord_repel": getattr(getattr(objective_cfg, "modules", None), "coord_repel", None),
"coord_repel_vocab": resolved_ordered_coord_repel_vocab,
"teacher_forcing_profile": getattr(objective_cfg, "profile", None),
```

- [ ] **Step 3: Add objective-local auxiliary term**

In `TeacherForcingObjective.run()`:
- compute the existing main teacher-forcing loss exactly as before.
- if `coord_repel.enabled`, call `compute_coord_repel_loss()` per resolved span.
- aggregate numerator as sum of per-span `CoordRepelResult.loss_sum` and denominator as global active slots.
- derive final coord-repel `raw_loss` only after summing all span numerators and denominators.
- set `weighted_loss = main_loss * spec.weight + coord_repel_weighted_loss`.
- keep `loss` as the main teacher-forcing raw loss so existing `teacher_forcing/loss/total` semantics do not silently change.

- [ ] **Step 4: Add metric event helper**

Add a small helper in `src/training/teacher_forcing/metrics.py` that converts coord-repel metric scalars into `MetricEvent`s with stable names.

- [ ] **Step 5: Rerun integration tests**

Run:

```bash
python -m pytest tests/test_coord_repel_objective_integration.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py -q
```

Expected after implementation: pass.

### Task 5: Add Generic Packed Layout And Teacher-Forcing Remap

**Files:**
- Create: `src/data_collators/packed_layout.py`
- Create: `src/training/teacher_forcing/packing.py`
- Modify: `src/data_collators/enrichers.py`
- Test: `tests/test_packed_batch_layout.py`
- Test: `tests/test_teacher_forcing_packing_remap.py`
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`

- [ ] **Step 1: Write packed layout tests**

Cover:
- unpacked batch yields one source segment per physical row.
- packed raw batch `[[A, B], [C]]` yields two physical rows and three source segments.
- source offsets are cumulative within a physical row.
- packed row active length equals sum of source lengths.
- padded physical width may be larger than active length.
- malformed missing source length fails fast when teacher-forcing sidecars are present.

Run:

```bash
python -m pytest tests/test_packed_batch_layout.py -q
```

Expected before implementation: import failure.

- [ ] **Step 2: Implement generic layout**

Use these dataclasses:

```python
@dataclass(frozen=True, slots=True)
class PackedSourceSegment:
    physical_batch_index: int
    source_index_in_pack: int
    source_sample_id: str
    source_length: int
    target_offset: int

@dataclass(frozen=True, slots=True)
class PackedPhysicalRow:
    physical_batch_index: int
    physical_sample_id: str
    active_length: int
    segments: tuple[PackedSourceSegment, ...]

@dataclass(frozen=True, slots=True)
class PackedBatchLayout:
    rows: tuple[PackedPhysicalRow, ...]
```

Core builder signature:

```python
PackedLayoutBuilder = Callable[
    [Sequence[Any], Mapping[str, Any], bool],
    PackedBatchLayout,
]
```

The implementation must expose a concrete function named `build_packed_batch_layout(raw_batch, collated, packed)`.

- [ ] **Step 3: Write teacher-forcing remap tests**

Cover:
- `A+B` packed row offsets atom target/logit positions by A length for B atoms.
- remapped atom `batch_index` equals physical row index.
- merged IR metadata records source sample IDs and segment ranges.
- future source samples inside the same physical row do not become prior objects for coord-repel.
- different packed source samples do not share coord-repel negative candidates.

Run:

```bash
python -m pytest tests/test_teacher_forcing_packing_remap.py -q
```

Expected before implementation: import failure.

- [ ] **Step 4: Implement teacher-forcing remap**

Use this callable shape:

```python
TeacherForcingRemapper = Callable[
    [Sequence[TeacherForcingTargetIR], PackedBatchLayout],
    tuple[TeacherForcingTargetIR, ...],
]
```

The implementation must expose a concrete function named `remap_teacher_forcing_target_irs(per_source_irs, layout)`.

Rules:
- input count must equal total source segments with teacher-forcing sidecars.
- each source atom position is shifted by the segment `target_offset`.
- each source atom `batch_index` becomes the physical packed row index.
- provenance keeps original source fields and adds `source_sample_id`, `source_index_in_pack`, and `physical_sample_id`.
- metadata records `packed_source_sample_ids`, `packed_source_lengths`, and `packed_source_offsets`.

- [ ] **Step 5: Integrate into `TeacherForcingTargetIREnricher`**

Replace the current packed rejection with:
- build layout from raw batch and collated tensors.
- collect per-source IRs and sample IDs in the same order as layout segments.
- call `remap_teacher_forcing_target_irs()`.
- set `collated["teacher_forcing_target_ir"]` to one merged IR per physical row.
- set `collated["sample_id"]` to physical row IDs.

- [ ] **Step 6: Rerun sidecar tests**

Run:

```bash
python -m pytest tests/test_packed_batch_layout.py tests/test_teacher_forcing_packing_remap.py tests/test_teacher_forcing_sidecar_bridge.py -q
```

Expected after implementation: pass.

### Task 6: Relax Teacher-Forcing Packing Runtime Guards Conservatively

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/detection/packing.py`
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/sft.py`
- Test: `tests/test_stage1_static_packing_runtime_config.py`
- Test: `tests/test_teacher_forcing_config_contract.py`
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`
- Test: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`

- [ ] **Step 1: Write runtime guard tests**

Cover:
- teacher-forcing static packing is accepted only when:
  - `training.packing: true`
  - `training.packing_mode: static`
  - `packing.static_packing: true`
  - `packing.padding_free_packed: true`
  - `objective.target_ir.exact_packing_mapping.enabled: true`
  - `training.per_device_train_batch_size: 1`
  - `model.attn_impl: flash_attention_2`
  - `training.encoded_sample_cache.enabled: false`
- dynamic packing remains rejected.
- encoded sample cache remains rejected.
- missing `padding_free_packed` remains rejected.
- missing or false `exact_packing_mapping.enabled` remains rejected for packed teacher-forcing.
- non-FA2 attention remains rejected for packed teacher-forcing.
- static packing fingerprints differ between teacher-forcing and `sorted_sft`, differ between `hard_sft` and `pure_valid_set_marginal`, and record `teacher_forcing_atoms_v1`.
- packed teacher-forcing constructs `TrainerLossBridgeSettings(packing_enabled=True)` and therefore validates Qwen packed metadata before model forward.

Run:

```bash
python -m pytest tests/test_stage1_static_packing_runtime_config.py tests/test_teacher_forcing_config_contract.py -q
```

Expected before implementation: existing tests still reject all teacher-forcing packing.

- [ ] **Step 2: Implement guard relaxation**

In `assert_detection_runtime_supported()`, replace the blanket teacher-forcing packing rejection with validated branches:
- if no packing keys are enabled, retain current accepted no-packing behavior.
- if any packing key is enabled, require the full validated packed contract above.
- keep encoded-sample cache rejected.

In `src/config/schema.py`, update the teacher-forcing packing validators so the same full contract parses and partial combinations fail early. In `src/detection/packing.py`, replace teacher-forcing static-packing rejection with an explicit `teacher_forcing_atoms_v1` identity check. Do not let teacher-forcing packing be fingerprinted as `sorted_sft`.

In `TeacherForcingObjectiveMixin.compute_loss()`, pass `TrainerLossBridgeSettings(packing_enabled=True)` only for validated packed teacher-forcing batches. Keep unpacked behavior unchanged.

- [ ] **Step 3: Include remap contract in metadata/fingerprint**

In `src/sft.py`, when objective id is `teacher_forcing`, include:

```python
"coord_repel": {
    "enabled": bool(coord_repel_cfg.enabled),
    "weight": float(coord_repel_cfg.weight),
    "margin": float(coord_repel_cfg.margin),
    "top_k": int(coord_repel_cfg.top_k),
    "negative_source": str(coord_repel_cfg.negative_source),
},
"effective_packing_remap_contract": "teacher_forcing_atoms_v1" if static_teacher_forcing_packing_enabled else None,
"teacher_forcing_profile": str(training_config.objective.profile) if objective_id == "teacher_forcing" else None,
"teacher_forcing_main_loss_semantics": "hard_sft_singleton_ce" if teacher_forcing_profile == "hard_sft" else "valid_set_marginal",
```

- [ ] **Step 4: Rerun runtime tests**

Run:

```bash
python -m pytest tests/test_stage1_static_packing_runtime_config.py tests/test_training_config_strict_unknown_keys.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected after implementation: pass.

### Task 7: Add Stage-1 Eval-Forward Coord-Repel Contract

**Files:**
- Create: `tests/test_coord_repel_eval_forward_contract.py`
- Modify: `src/sft.py` if eval packing or eval metric prefix wiring is missing
- Modify: `src/trainers/metrics/teacher_forcing.py` if train/eval metric ownership differs

- [ ] **Step 1: Write eval-forward contract test**

Cover:
- eval-style teacher-forcing batch uses the same packed layout and remapped `TeacherForcingTargetIR` code as training.
- coord-repel metrics are emitted in eval with the repository's actual eval prefix convention.
- eval coord-repel uses the same ordered coord vocab and negative-source logic as training.
- packed eval rows do not leak `B-` across source samples.

Run:

```bash
python -m pytest tests/test_coord_repel_eval_forward_contract.py -q
```

Expected before implementation: missing test target or missing eval coord-repel metrics.

- [ ] **Step 2: Implement missing eval-forward wiring**

If the test exposes a train/eval split, route eval through the same packed layout, teacher-forcing remap, and objective metric helpers. Do not add rollout assumptions.

- [ ] **Step 3: Rerun eval-forward test**

Run:

```bash
python -m pytest tests/test_coord_repel_eval_forward_contract.py tests/test_coord_repel_objective_integration.py -q
```

Expected after implementation: pass.

### Task 8: Add Compact Box-End Template Ablation

**Files:**
- Modify: `src/common/detection_compact_rows.py`
- Modify: `src/detection/template.py`
- Modify: `src/detection/evaluation.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Modify: `src/config/schema.py`
- Modify: `src/sft.py`
- Modify: `configs/stage1/detection_teacher_forcing/`
- Test: `tests/test_compact_box_end_template.py`
- Test: `tests/test_detection_template_registry.py`
- Test: `tests/test_detection_compact_full_template.py`
- Test: `tests/test_teacher_forcing_target_builder.py`

- [ ] **Step 1: Write template tests**

Cover:
- `compact_box_end` renders exactly `<|box_end|>` after each `y2`.
- no newline appears.
- no `<object_ref_end>` appears.
- parser accepts only the box-end form for `compact_box_end`.
- `compact_full` remains byte-for-byte unchanged.
- rendered teacher-forcing target IR text matches encoded assistant text.
- tokenizer encodes `<|box_end|>` as exactly one token before a `compact_box_end` config is accepted.
- `detection_template.id=compact_box_end` and `evaluation.expected_template=compact_box_end` both materialize.
- `resolve_detection_runtime_support(...).teacher_forcing_target_ir_required` is true for `compact_box_end`.
- a dataset smoke constructs a `teacher_forcing_target_ir` for `compact_box_end` and verifies rendered assistant text equals target-builder text.
- static-packing fingerprints differ between `compact_full` and `compact_box_end`.

Run:

```bash
python -m pytest tests/test_compact_box_end_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_registry.py -q
```

Expected before implementation: unsupported template id or missing token failures.

- [ ] **Step 2: Add template ID and token constant**

Add:

```python
BOX_END_TOKEN = "<|box_end|>"
TemplateId = Literal["stage1_json_pretty", "compact_full", "compact_box_end"]
```

Add `CompactBoxEndTemplate` as a sibling of `CompactFullTemplate`, not as a mode hidden inside `compact_full`.

- [ ] **Step 3: Thread template into teacher-forcing target builder**

Change `TeacherForcingTargetBuilder` to accept a compact template object or `template_id`. `_prepare_object()` must render entries with the selected template so target IR token IDs and dataset rendered text stay identical.

- [ ] **Step 4: Add token-row config**

Add `<|box_end|>` to a structural token-row group in the `compact_box_end` ablation config. The config must include an expected ID after the tokenizer/model is expanded or verified. Do not guess this ID.

Update token-row validation so the accepted structural-row count is template-specific:
- `compact_full`: `<|object_ref_start|>`, `<|box_start|>`.
- `compact_box_end`: `<|object_ref_start|>`, `<|box_start|>`, `<|box_end|>`.

Define a shared helper or literal set for compact teacher-forcing templates and use it consistently in schema, runtime support, dataset checks, prompt validation, target builder, and template factory.

Update `src/detection/evaluation.py` so strict eval parsing routes through the `compact_box_end` template when `evaluation.expected_template=compact_box_end`.

- [ ] **Step 5: Rerun template and target-builder tests**

Run:

```bash
python -m pytest tests/test_compact_box_end_template.py tests/test_detection_template_registry.py tests/test_teacher_forcing_target_builder.py tests/test_chat_template_regression.py -q
```

Expected after implementation: pass.

### Task 9: Add Config Leaves And Documentation

**Files:**
- Modify: `configs/stage1/detection_teacher_forcing/README.md`
- Modify: `configs/stage1/detection_teacher_forcing/smoke/`
- Modify: `configs/stage1/detection_teacher_forcing/prod/`
- Modify: `docs/data/PACKING.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/superpowers/specs/2026-06-12-coord-repel-v1-design.md`
- Optional stable promotion: `openspec/changes/<change-id>/` if final approval scopes this as stable public behavior.

- [ ] **Step 1: Add smoke configs**

Create or update smoke leaves for:
- `compact + no coord-repel`
- `compact + coord-repel`
- `compact_box_end + no coord-repel`
- `compact_box_end + coord-repel`
- at least one static-packed teacher-forcing leaf with the full packed contract, including `objective.target_ir.exact_packing_mapping.enabled: true`

Use `weight: 0.05` for the balanced coord-repel config. Keep `weight: 0.1` only as an explicitly named stress-smoke override.

- [ ] **Step 2: Add docs**

Update docs with:
- training-time-only loss semantics.
- no inference/decode/attention/KV changes.
- coordinate-vocab conditional probability space.
- default `negative_source: last_and_same_desc`.
- packed remap contract and no cross-source leakage guarantee.
- active metrics and interpretation boundaries.
- box-end ablation matrix.
- production gate statement.
- experiment-only versus stable-public status.
- if stable-public status is approved, an OpenSpec change that covers schema, packing, loss semantics, template ID, and metric semantics.

- [ ] **Step 3: Verify docs and configs**

Run:

```bash
python - <<'PY'
import yaml
from pathlib import Path
for path in ["docs/catalog.yaml", "progress/index.yaml"]:
    if Path(path).exists():
        yaml.safe_load(Path(path).read_text())
        print(f"{path}: ok")
PY
git diff --check
python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_stage1_static_packing_runtime_config.py tests/test_compact_box_end_template.py tests/test_coord_repel_eval_forward_contract.py -q
```

Expected after implementation: YAML parse output for existing indexes, no whitespace errors, targeted tests pass.

### Task 10: Run Real-GPU Launch Gates Before Production

**Files:**
- Modify: `configs/stage1/detection_teacher_forcing/smoke/`
- Modify: `configs/stage1/detection_teacher_forcing/prod/`
- Modify: `progress/diagnostics/` or `progress/experiments/` after runs produce artifacts
- Optional stable promotion: update launch docs/OpenSpec only after production status is approved.

- [ ] **Step 1: Add real-GPU smoke launch configs**

Create launch configs for:
- the selected non-base candidate checkpoint, if a candidate checkpoint is still in scope.
- the base 2B model as a required control arm.

Both arms must use:
- COCO Stage-1 detection data with `global_max_length: 12000`.
- no `max_objects: 60` cap as a substitute for launch evidence.
- LLM-tower-only training; vision tower and non-LLM parameter groups remain frozen unless separately approved.
- coord-repel V1 balanced settings: `weight: 0.05`, `margin: 0.25`, `top_k: 32`, `negative_source: last_and_same_desc`.
- static packing when testing the packed path: padding-free packed batches, `flash_attention_2`, per-device train batch size 1, encoded cache disabled, and `objective.target_ir.exact_packing_mapping.enabled: true`.

The smoke config must record trainable parameter groups, objective version,
template identity, packing/remap contract, dataset identity, max sequence length,
gradient accumulation, world size, precision, and resolved coord-repel settings.

- [ ] **Step 2: Launch short real-GPU loss/metric correctness smokes**

Run short real-GPU smokes for each required arm before any longer training.

Required evidence:
- authored config and resolved/runtime config.
- run directory, log path, artifact root, and exact checkpoint/model root.
- one real target example through target creation.
- one collated batch with shapes, dtype, valid counts, mask density, packed source segments, and sidecar counts.
- finite total loss, main teacher-forcing loss, raw coord-repel loss, weighted coord-repel loss, and gradients.
- nonzero `coord_repel/eligible_x1_slots` or `coord_repel/eligible_y1_slots`.
- nonzero `coord_repel/active_slots` on a duplicate-sensitive fixture or canonical-failure probe.
- finite `coord_repel/weighted_loss_to_main_tf_loss_ratio`.
- stable metric emission on the expected train and eval namespaces.
- no packed cross-source negative leakage.

Do not promote the branch if coord-repel is enabled but inactive, detached,
zero-weighted, missing from logs, or reduced with an unexplained denominator.

- [ ] **Step 3: Launch longer real-GPU stability runs with `loss-numerics-sanity`**

After short smokes pass, run longer stability launches for the same required
arms, including the base 2B control arm.

Recommended minimum: enough optimizer steps to move past warmup and observe at
least 200 post-warmup optimizer steps; if launch cost forces a smaller window,
record the reduced scope explicitly and do not treat it as production clearance.

Required trend checks:
- total loss finite and generally decreasing over the observed window.
- main teacher-forcing loss finite and generally decreasing over the observed window.
- coord-repel raw/weighted losses finite and interpretable.
- `coord_repel/weighted_loss_to_main_tf_loss_ratio` stays in a plausible range and does not dominate the main objective without explanation.
- gradient norm does not repeatedly explode, collapse to zero, or shift sharply after enabling coord-repel.
- learning-rate schedule, gradient accumulation, and distributed world size match the resolved config.
- active-slot counts and denominators are stable enough to interpret under packing.
- no NaN/Inf in loss, logits/probabilities used by coord-repel, gradients, or optimizer state.

Use the `loss-numerics-sanity` reporting shape for the stability note:

```text
Verdict
Scope
Evidence
Loss/Metric Trend
Numerics And Scaling
Masking/Target Contract
Distributed/Precision Notes
Warnings
Next Gate
Confidence
```

- [ ] **Step 4: Require task-health artifacts before production**

Loss stability alone is not enough. Before production launch, verify:
- parser/template validity does not regress.
- eval-forward coord-repel metrics appear under the expected namespace.
- packed and unpacked teacher-forcing produce equivalent atom positions on synthetic samples.
- packed samples do not share coord-repel prior bands across original source samples.
- trainable parameter reports show LLM tower/token embeddings only.
- COCO length-12000 evidence was used; no max-60-object capped substitute was used for launch acceptance.
- base 2B control-arm smoke and stability runs pass the same gates as any candidate checkpoint arm.

- [ ] **Step 5: Launch production training only after both gates pass**

Only after unit/config tests, real-GPU smoke, longer stability, and task-health
gates pass, launch production training. If a non-base candidate checkpoint is
launched for production, include the base 2B model as a matched
production/control arm unless the production scope is explicitly narrowed later.

Each production arm must use:
- 8 GPUs.
- 4 epochs.
- static packing enabled.
- padding-free packed batches.
- `flash_attention_2`.
- LLM-tower-only training.
- COCO Stage-1 length-12000 data.
- exact teacher-forcing remap contract enabled.

Record the production launch config, resolved config, run directory, artifact
root, checkpoint schedule, first-step telemetry, first-eval telemetry, and the
decision note linking back to the smoke/stability evidence.

## Required Verification Before Training

Do not start production training from this branch until all commands below pass:

```bash
python -m pytest tests/test_coord_repel_loss.py -q
python -m pytest tests/test_coord_repel_objective_integration.py -q
python -m pytest tests/test_packed_batch_layout.py tests/test_teacher_forcing_packing_remap.py -q
python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_stage1_static_packing_runtime_config.py -q
python -m pytest tests/test_compact_box_end_template.py tests/test_detection_template_registry.py -q
python -m pytest tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_sidecar_bridge.py -q
python -m pytest tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
python -m pytest tests/test_teacher_forcing_target_builder.py tests/test_teacher_forcing_ir_contract.py tests/test_teacher_forcing_metric_contract.py tests/test_chat_template_regression.py tests/test_coord_repel_eval_forward_contract.py -q
git diff --check
```

Before any full run, add a tiny or smoke launch that proves:
- coord-repel metrics appear when enabled.
- `coord_repel/active_slots` is nonzero on a known duplicate-sensitive fixture or canonical-failure probe.
- `coord_repel/weighted_loss_to_main_tf_loss_ratio` is in an interpretable range and the run logs main-loss semantics.
- packed and unpacked teacher-forcing produce equivalent atom positions on the same synthetic samples.
- packed samples do not share coord-repel prior bands across original source samples.
- the real-GPU base 2B control arm passes the same smoke and longer stability gates.
- COCO length-12000 data was used for launch acceptance, without substituting a `max_objects: 60` capped dataset.
- trainable parameter reporting confirms LLM-tower-only training.
- both `loss-numerics-sanity` numeric gates and task-health/artifact gates pass before the 8-GPU, 4-epoch packed production run.

## Review Convergence Log

Round 1 state: independent review lanes completed; no P0 findings; P1/P2 findings triaged into this plan.

Accepted findings:
- Use an ordered coord-bin vocabulary; do not derive coord-bin neighborhoods from unordered `RoleVocab.coord_token_ids`.
- Make existing `objective.target_ir.exact_packing_mapping.enabled` legal and required for packed teacher-forcing.
- Include schema validators, `src/detection/packing.py`, and bridge `packing_enabled` wiring in the packed teacher-forcing implementation scope.
- Rename the strength ratio from CE wording to `coord_repel/weighted_loss_to_main_tf_loss_ratio` and log main-loss semantics.
- Add an explicit Stage-1 eval-forward contract test instead of deferring eval validity to a launch smoke.
- Treat `compact_box_end` as a broader template/config/runtime/eval/cache identity change, not only render/parse code.
- Expand final verification to include target-builder, IR, metric, chat-template, and eval-forward contract tests.
- Add a governance decision: experiment-only opt-in can remain in local docs/configs; stable public behavior requires an OpenSpec change and router docs.
- Require a static-packed config leaf/materialization proof before claiming packing-speed support.
- Require real-GPU smoke, COCO length-12000 stability, base 2B control-arm evidence, LLM-tower-only trainability proof, and both numerics and task-health gates before production launch.

Rejected findings:
- none.

User-decision items:
- main first ablation profile: current `pure_valid_set_marginal` continuity versus adding a hard-SFT paired arm.
- `duplicate_group_key` source: category-name preferred fallback versus desc-only.
- whether `<|box_end|>` implementation should be approved in the same implementation batch or held as a second batch after core coord-repel.
- whether coord-repel V1 is experiment-only opt-in or should be promoted as stable public Stage-1 behavior with OpenSpec coverage.
- exact eval metric namespace should follow existing trainer prefix behavior unless the implementation reveals ambiguity.

## Stop State

Current stop state: ready for user approval, not ready for implementation.

Required next state before coding: explicit user approval to implement after reviewing this plan.
