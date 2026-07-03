# Loss-Only Instance Enumeration V1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the V1a training-only instance-enumeration auxiliary loss for Stage-1 Qwen3-VL detection teacher forcing, preserving standard CE/decode semantics and comparing only random versus sorted object order.

**Architecture:** Add a detection-local `instance_enumeration` module family that builds CPU sidecars from the existing teacher-forcing IR, captures Qwen final hidden states and main post-projector image embeddings without editing upstream model files, computes `L_pos` and probability-mass-margin `L_neg`, and adds the weighted aux loss after ordinary CE in `TrainerLossBridge`. PEFT probe insertion follows the existing coord-offset `modules_to_save` pattern; visual keys are detached; active configs are unpacked-only and mutually exclusive with coverage painters.

**Tech Stack:** Python dataclasses, PyTorch, Hugging Face Qwen3-VL wrappers/hooks only, ms-swift SFT/PEFT, CoordExp detection YAML schema, existing teacher-forcing sidecars, pytest.

---

Date: 2026-06-08

Status: roadmap for user approval. Do not implement code until the user explicitly approves implementation.

Primary design note:
`docs/superpowers/specs/2026-06-08-loss-only-instance-enumeration-design.md`

## Scope Lock

Implement exactly one V1a mechanism:

```text
standard CE + L_pos + L_neg
```

The only experiment axis is:

```text
random_order_sft vs sorted_sft
```

Scalar hyperparameters stay configurable:

```text
probe_lr
positive_weight
negative_weight
negative_margin
score_scale
support_floor
```

Everything else is a fixed V1a invariant, strict parser rejection, or runtime validation:

- full-sequence unpacked Stage-1 teacher forcing only;
- no row-expanded dataset wrapper;
- no packed auxiliary sidecars;
- no attention-logit bias;
- no feature painting;
- no Q/K/V edits;
- no upstream Qwen/HF file edits;
- no `output_attentions`;
- no decode or KV-cache changes;
- no DeepStack visual keys in the aux loss;
- cosine probe logits use configurable `score_scale` with default `1.0`;
- one shared linear q/k probe with `hidden_dim=256`;
- `detach_visual_keys=true`;
- global LR scheduler and warmup only;
- `weight_decay=0.0` for every optimizer group in V1 instance-enumeration runs.

## Planned File Map

Create:

- `src/detection/instance_enumeration/__init__.py`
- `src/detection/instance_enumeration/config.py`
- `src/detection/instance_enumeration/types.py`
- `src/detection/instance_enumeration/state.py`
- `src/detection/instance_enumeration/visual_regions.py`
- `src/detection/instance_enumeration/capture.py`
- `src/detection/instance_enumeration/probe.py`
- `src/detection/instance_enumeration/loss.py`
- `tests/detection/instance_enumeration/test_config.py`
- `tests/detection/instance_enumeration/test_state.py`
- `tests/detection/instance_enumeration/test_visual_regions.py`
- `tests/detection/instance_enumeration/test_capture.py`
- `tests/detection/instance_enumeration/test_probe_peft.py`
- `tests/detection/instance_enumeration/test_loss.py`
- `tests/detection/instance_enumeration/test_sft_peft_wiring.py`
- `tests/detection/instance_enumeration/test_adapter_artifacts.py`
- `tests/detection/instance_enumeration/test_sorted_random_ablation_contract.py`
- `configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml`
- `configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml`

Modify:

- `src/config/schema.py`
- `src/detection/dataset.py`
- `src/detection/runtime.py`
- `src/detection/teacher_forcing/rollin.py`
- `src/detection/teacher_forcing/target_builder.py`
- `src/data_collators/enrichers.py`
- `src/data_collators/batch_extras_collator.py`
- `src/infer/checkpoints.py`
- `src/trainers/batch_extras.py`
- `src/training/encoding/model_inputs.py`
- `src/training/bridge/loss_bridge.py`
- `src/trainers/metrics/teacher_forcing.py`
- `src/optim/coord_offset_optimizer.py`
- `src/optim/__init__.py`
- `src/sft.py`
- `tests/test_batch_extras_contract.py`
- `tests/test_model_input_bundle_contract.py`
- `tests/test_teacher_forcing_sidecar_bridge.py`
- `tests/test_trainer_loss_bridge_qwen3vl_contract.py`
- `tests/coord_tokens/test_offset_optimizer.py`
- `tests/test_detection_training_config_contract.py`
- `tests/detection/coverage/test_config.py`
- `tests/detection/coverage/test_pixel_config.py`
- `tests/test_infer_checkpoint_resolution.py`

Do not modify:

- `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- `src/detection/coverage/forward.py`
- `src/detection/coverage/painting.py`
- `src/detection/coverage/pixel_painting.py`
- row re-prefill rollout paths

## Review-Convergence Consolidation

Accepted design-review constraints now represented as tasks:

- Tiny public config, not a structural control panel: Task 1.
- Sidecar stores indices and boxes once; row views are derived: Task 2.
- Unpacked-only sidecar transport and packed aux rejection: Task 3.
- Neutral visual-region helper separated from painter code: Task 4.
- Qwen capture is scoped and non-mutating: Task 5.
- Probe participates through active PEFT `modules_to_save` copy and optimizer bucket: Task 6.
- Probability-mass `L_pos` and `L_neg` with current-overlap-negative exclusion: Task 7.
- Bridge/trainer integration preserves ordinary CE and standard decode: Task 8.
- Artifact-mode gates, comprehensive health metrics, and active wiring tests: Task 9.
- Only random versus sorted smoke configs: Task 10.

Rejected from V1a:

- CE-only objective-control matrix;
- L-pos-only matrix;
- margin/support-floor/score-scale sweeps;
- auxiliary warmup or separate scheduler;
- packed aux support;
- row-state strict mode;
- combined coverage-painter runs;
- attention-bias runs.

Review-loop resolutions accepted before implementation:

- `sorted_sft` means actual sorted rendered target-IR order:
  `TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]` must
  be sorted for the sorted lane. Do not label a run sorted if the target builder
  still uses `random_permutation` roll-in.
- V1a supports two adapter artifact modes only: `training_aux_adapter` may carry
  the probe for resume, while inference/export must reject a training-only probe
  unless a later strip tool explicitly removes it. V1a does not claim export
  readiness.
- PEFT/SFT wiring tests must live in unskipped instance-enumeration test files,
  not in the globally skipped legacy recursive-detection wiring module.

## Execution Policy

Use small commits after each task during implementation. Do not launch production training during implementation. GPU work is limited to explicit `tiny` smokes after CPU/unit gates pass.

This roadmap is the last docs-only artifact before code. Stop here until the user approves implementation.

---

### Task 1: Add Compact Config Schema And Runtime Guardrails

**Files:**

- Create: `src/detection/instance_enumeration/config.py`
- Modify: `src/config/schema.py`
- Modify: `src/detection/teacher_forcing/rollin.py`
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Modify: `tests/detection/coverage/test_config.py`
- Modify: `tests/detection/coverage/test_pixel_config.py`
- Modify: `src/detection/runtime.py`
- Create: `tests/detection/instance_enumeration/test_config.py`

- [ ] **Step 1.1: Write config parser tests**

Add tests that assert the schema default is disabled, V1a accepts only scalar hyperparameters, unknown keys fail, and coverage painters are mutually exclusive with the new aux section.

```python
import pytest

from src.config.schema import DetectionTrainingConfig
from src.detection.instance_enumeration.config import InstanceEnumerationAuxConfig


def test_instance_enum_config_default_disabled() -> None:
    cfg = InstanceEnumerationAuxConfig.from_mapping(None)
    assert cfg.enabled is False
    assert cfg.version == "v1"
    assert cfg.probe_lr is None
    assert cfg.positive_weight == pytest.approx(0.05)
    assert cfg.negative_weight == pytest.approx(0.02)
    assert cfg.negative_margin == pytest.approx(0.5)
    assert cfg.score_scale == pytest.approx(1.0)
    assert cfg.support_floor == pytest.approx(0.10)


def test_instance_enum_enabled_requires_explicit_probe_lr(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["instance_enumeration_aux"] = {"enabled": True, "version": "v1"}
    with pytest.raises(ValueError, match="instance_enumeration_aux.probe_lr"):
        DetectionTrainingConfig.from_mapping(payload)


def test_instance_enum_accepts_compact_v1_surface(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["instance_enumeration_aux"] = {
        "enabled": True,
        "version": "v1",
        "probe_lr": 5.0e-5,
        "positive_weight": 0.05,
        "negative_weight": 0.02,
        "negative_margin": 0.5,
        "score_scale": 1.0,
        "support_floor": 0.10,
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)
    assert cfg.instance_enumeration_aux.enabled is True
    assert cfg.instance_enumeration_aux.probe_lr == pytest.approx(5.0e-5)


def test_instance_enum_rejects_structural_knobs(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["instance_enumeration_aux"] = {
        "enabled": True,
        "version": "v1",
        "probe_lr": 5.0e-5,
        "sequence_mode": "row_state",
    }
    with pytest.raises(ValueError, match="instance_enumeration_aux.sequence_mode"):
        DetectionTrainingConfig.from_mapping(payload)


def test_instance_enum_mutually_exclusive_with_row_coverage(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["instance_enumeration_aux"] = {
        "enabled": True,
        "version": "v1",
        "probe_lr": 5.0e-5,
    }
    payload["row_conditioned_visual_coverage"] = {"enabled": True, "version": "v1"}
    with pytest.raises(ValueError, match="mutually exclusive"):
        DetectionTrainingConfig.from_mapping(payload)


def test_instance_enum_sorted_lane_accepts_sorted_target_ir_rollin(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["data"]["object_ordering"] = "sorted"
    payload["objective"]["target_ir"]["rollin_policy"]["name"] = "sorted"
    payload["instance_enumeration_aux"] = {
        "enabled": True,
        "version": "v1",
        "probe_lr": 5.0e-5,
    }
    payload["row_conditioned_visual_coverage"] = {"enabled": False}
    payload["pixel_painted_row_coverage"] = {"enabled": False}
    cfg = DetectionTrainingConfig.from_mapping(payload)
    assert cfg.data.object_ordering == "sorted"
    assert cfg.objective.target_ir.rollin_policy.name == "sorted"
    assert cfg.instance_enumeration_aux.enabled is True


def test_instance_enum_sorted_lane_rejects_random_target_ir_rollin(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["data"]["object_ordering"] = "sorted"
    payload["objective"]["target_ir"]["rollin_policy"]["name"] = "random_permutation"
    payload["instance_enumeration_aux"] = {
        "enabled": True,
        "version": "v1",
        "probe_lr": 5.0e-5,
    }
    payload["row_conditioned_visual_coverage"] = {"enabled": False}
    payload["pixel_painted_row_coverage"] = {"enabled": False}
    with pytest.raises(ValueError, match="data.object_ordering"):
        DetectionTrainingConfig.from_mapping(payload)


def test_instance_enum_rejects_nonzero_training_weight_decay(base_detection_payload) -> None:
    payload = base_detection_payload()
    payload["training"]["weight_decay"] = 0.1
    payload["instance_enumeration_aux"] = {
        "enabled": True,
        "version": "v1",
        "probe_lr": 5.0e-5,
    }
    with pytest.raises(ValueError, match="weight_decay"):
        DetectionTrainingConfig.from_mapping(payload)
```

If `base_detection_payload` is not available in the new test file, move the minimal helper from `tests/detection/coverage/test_config.py` into a local helper inside this file.

- [ ] **Step 1.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_config.py -q
```

Expected: fail because `src.detection.instance_enumeration.config` and `DetectionTrainingConfig.instance_enumeration_aux` do not exist yet.

- [ ] **Step 1.3: Implement config dataclass and schema integration**

Implement:

```python
@dataclass(frozen=True)
class InstanceEnumerationAuxConfig:
    enabled: bool = False
    version: Literal["v1"] = "v1"
    probe_lr: float | None = None
    positive_weight: float = 0.05
    negative_weight: float = 0.02
    negative_margin: float = 0.5
    score_scale: float = 1.0
    support_floor: float = 0.10
```

Parser behavior:

- `None` means disabled.
- Enabled requires `probe_lr > 0`.
- `positive_weight >= 0`.
- `negative_weight >= 0`.
- `negative_margin` is finite.
- `score_scale > 0`.
- `0 < support_floor <= 1`.
- Unknown keys reject through strict parsing.

Wire this into `DetectionTrainingConfig.from_mapping` as a typed top-level optional section named `instance_enumeration_aux`.

Add runtime checks:

- enabled aux requires `objective.id=teacher_forcing`;
- enabled aux rejects `training.packing=true`;
- enabled aux rejects `training.eval_packing=true`;
- enabled aux rejects `packing.static_packing=true`;
- enabled aux rejects `packing.padding_free_packed=true`;
- enabled aux rejects `row_conditioned_visual_coverage.enabled=true`;
- enabled aux rejects `pixel_painted_row_coverage.enabled=true`;
- allow `objective.target_ir.rollin_policy.name="sorted"` as the only V1a
  sorted lane, and implement it in the target builder as the identity order
  over normalized object indices;
- enabled aux does not add a coverage-style exception that allows
  `data.object_ordering="sorted"` with `rollin_policy.name="random_permutation"`;
- random and sorted aux-enabled configs must have matching `data.object_ordering`
  and `objective.target_ir.rollin_policy.name`;
- enabled aux rejects nonzero optimizer weight decay in V1 instance-enumeration runs.

Also propagate the parsed config through the detection runtime shim:

- add `instance_enumeration_aux` to the detection top-level optional sections;
- add `instance_enumeration_aux` to `DetectionTrainingConfig`;
- add `instance_enumeration_aux` to `DetectionTrainingConfig.to_mapping()`;
- add a field on `DetectionDatasetRuntimeConfig` or equivalent dataset runtime settings so `DetectionTrainingDataset` can know whether to attach `instance_enumeration_state`;
- update `build_detection_runtime_custom_shim` in `src/detection/runtime.py` so the dataset receives the resolved aux config, not a stale `custom` bucket.

Also keep the existing coverage-painter order exception scoped to coverage
painters only. Do not reuse it for `instance_enumeration_aux`.

- [ ] **Step 1.4: Run config tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_config.py tests/detection/coverage/test_config.py tests/detection/coverage/test_pixel_config.py -q
```

Expected: pass.

- [ ] **Step 1.5: Commit Task 1**

Run:

```bash
git add src/detection/instance_enumeration/config.py src/detection/instance_enumeration/__init__.py src/config/schema.py src/detection/runtime.py src/detection/teacher_forcing/rollin.py src/detection/teacher_forcing/target_builder.py tests/detection/instance_enumeration/test_config.py tests/detection/coverage/test_config.py tests/detection/coverage/test_pixel_config.py
git commit -m "feat: add instance-enumeration aux config"
```

### Task 2: Add Lean Instance Enumeration Sidecar

**Files:**

- Create: `src/detection/instance_enumeration/types.py`
- Create: `src/detection/instance_enumeration/state.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/detection/runtime.py`
- Create: `tests/detection/instance_enumeration/test_state.py`
- Modify: `tests/test_latest_detection_view_metadata.py`
- Modify: `tests/test_teacher_forcing_config_contract.py`

- [ ] **Step 2.1: Write sidecar type and row-view tests**

Add tests:

```python
import pytest

from src.detection.instance_enumeration.state import iter_instance_rows
from src.detection.instance_enumeration.types import InstanceEnumerationState


def test_iter_instance_rows_uses_rendered_order_not_numeric_id() -> None:
    state = InstanceEnumerationState(
        sample_id="sample-7",
        base_idx=7,
        ordered_object_indices=(2, 0, 1),
        object_boxes_norm1000_xyxy_by_index={
            0: (100, 100, 200, 200),
            1: (300, 300, 400, 400),
            2: (500, 500, 600, 600),
        },
        image_width=640,
        image_height=480,
        ordering_strategy="random_permutation",
        ordering_seed=17,
    )
    rows = tuple(iter_instance_rows(state))
    assert [(r.target_object_index, r.previous_object_indices) for r in rows] == [
        (2, ()),
        (0, (2,)),
        (1, (2, 0)),
    ]
    assert rows[0].future_object_indices == (0, 1)


def test_missing_box_for_derived_row_fails() -> None:
    state = InstanceEnumerationState(
        sample_id="sample-7",
        base_idx=7,
        ordered_object_indices=(2,),
        object_boxes_norm1000_xyxy_by_index={},
        image_width=640,
        image_height=480,
        ordering_strategy="random_permutation",
        ordering_seed=17,
    )
    with pytest.raises(ValueError, match="missing norm1000 box"):
        tuple(iter_instance_rows(state))
```

Also add anchor-validation tests in this same file before the Task 2 commit:

- a valid aligned `TeacherForcingTargetIR` resolves each row anchor from
  `branch_position == 0` and returns `atom.target_position`;
- missing anchors, duplicate anchors, unexpected discovered object anchors,
  wrong `branch_position`, schema-role-only matches, and out-of-order anchor
  coverage fail with sample id, base index, rendered order, and offending object
  indices in the error.

Add a dataset-side registry/leak test in `tests/test_latest_detection_view_metadata.py`:

- `REGISTERED_DETECTION_SIDECAR_KEYS` contains `instance_enumeration_state`;
- `strip_non_model_detection_sidecars` removes `instance_enumeration_state`
  from model inputs while preserving the trainer-only sidecar path.

- [ ] **Step 2.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_state.py -q
```

Expected: fail because the module is not implemented.

- [ ] **Step 2.3: Implement frozen sidecar and derived row view**

Implement `Norm1000XYXYBox = tuple[int, int, int, int]`, frozen `InstanceEnumerationState`, and frozen derived `InstanceEnumerationRowView`. `InstanceEnumerationState` stores only:

```text
sample_id
base_idx
ordered_object_indices
object_boxes_norm1000_xyxy_by_index
image_width
image_height
ordering_strategy
ordering_seed
```

`iter_instance_rows(state)` derives `target_object_index`, `previous_object_indices`, and `future_object_indices` from rendered order. It validates every referenced index has one box.

- [ ] **Step 2.4: Attach sidecar in `DetectionTrainingDataset.__getitem__`**

After aligned `TeacherForcingTargetIR` construction, add `instance_enumeration_state` only when `DetectionDatasetRuntimeConfig.instance_enumeration_aux.enabled=true`. Derive object order from:

```text
TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]
```

Use dataset scene original image width/height and norm1000 xyxy boxes. Do not store labels, descriptions, tensors, masks, token positions, target boxes, or previous boxes in the sidecar.

Add the new sidecar key to `REGISTERED_DETECTION_SIDECAR_KEYS` so dataset-side filtering keeps it with other trainer-only sidecars.

- [ ] **Step 2.5: Add anchor contract validation**

Add a state helper that validates every rendered object has exactly one aligned atom where:

```text
atom.provenance["object_index"] == object_index
atom.provenance["branch_position"] == 0
```

Use `atom.target_position` later for loss anchoring. Missing, duplicate, or out-of-order anchors are hard errors with sample id, base index, rendered order, and offending object indices.

- [ ] **Step 2.6: Run sidecar tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_state.py tests/test_latest_detection_view_metadata.py tests/test_teacher_forcing_config_contract.py -q
```

Expected: pass.

- [ ] **Step 2.7: Commit Task 2**

Run:

```bash
git add src/detection/instance_enumeration/types.py src/detection/instance_enumeration/state.py src/detection/dataset.py src/detection/runtime.py tests/detection/instance_enumeration/test_state.py tests/test_latest_detection_view_metadata.py tests/test_teacher_forcing_config_contract.py
git commit -m "feat: add instance-enumeration sidecars"
```

### Task 3: Transport Sidecars Through Collator And Batch Extras

**Files:**

- Modify: `src/trainers/batch_extras.py`
- Modify: `src/data_collators/enrichers.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Modify: `src/training/encoding/model_inputs.py`
- Modify: `tests/test_batch_extras_contract.py`
- Modify: `tests/test_model_input_bundle_contract.py`
- Modify: `tests/test_teacher_forcing_sidecar_bridge.py`

- [ ] **Step 3.1: Write sidecar transport tests**

Add tests that mirror the existing `teacher_forcing_target_ir` and `row_coverage_state` contracts:

```python
from src.trainers.batch_extras import INSTANCE_ENUMERATION_STATE_KEY, pop_batch_extras


def test_instance_enumeration_state_is_batch_extra_not_model_input() -> None:
    inputs = {INSTANCE_ENUMERATION_STATE_KEY: ("state-a",), "input_ids": [1, 2]}
    extras = pop_batch_extras(inputs)
    assert extras.instance_enumeration_state == ("state-a",)
    assert INSTANCE_ENUMERATION_STATE_KEY not in inputs


def test_instance_enumeration_collator_requires_all_unpacked_sidecars(batch_extras_collator):
    with pytest.raises(ValueError, match="instance_enumeration_state"):
        batch_extras_collator([
            {"input_ids": [1], INSTANCE_ENUMERATION_STATE_KEY: "state-a"},
            {"input_ids": [2]},
        ])


def test_instance_enumeration_collator_rejects_packed_sidecars(batch_extras_collator):
    with pytest.raises(ValueError, match="incompatible with packing"):
        batch_extras_collator([
            [{"input_ids": [1], INSTANCE_ENUMERATION_STATE_KEY: "state-a"}],
        ])
```

Use existing batch-collator helpers in `tests/test_batch_extras_contract.py`; if their names differ, create small local fake collators in the modified tests.

- [ ] **Step 3.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/test_batch_extras_contract.py tests/test_model_input_bundle_contract.py tests/test_teacher_forcing_sidecar_bridge.py -q
```

Expected: fail on missing `INSTANCE_ENUMERATION_STATE_KEY` and enricher behavior.

- [ ] **Step 3.3: Add key, dataclass field, and sidecar-only registry**

Add:

```python
INSTANCE_ENUMERATION_STATE_KEY = "instance_enumeration_state"
```

Include it in:

- `BATCH_EXTRAS_KEYS`;
- `BatchExtras.instance_enumeration_state`;
- `pop_batch_extras`;
- `SIDECAR_ONLY_KEYS` in `src/training/encoding/model_inputs.py`.

- [ ] **Step 3.4: Add `InstanceEnumerationStateEnricher`**

Implement a collator enricher like `TeacherForcingTargetIREnricher`:

- packed + any sidecar raises;
- unpacked + partial sidecar raises;
- unpacked + all sidecars adds tuple in batch order;
- absent sidecars do nothing.

Register it in `build_batch_extras_collator` after `TeacherForcingTargetIREnricher`.

- [ ] **Step 3.5: Run transport tests**

Run:

```bash
python -m pytest tests/test_batch_extras_contract.py tests/test_model_input_bundle_contract.py tests/test_teacher_forcing_sidecar_bridge.py -q
```

Expected: pass.

- [ ] **Step 3.6: Commit Task 3**

Run:

```bash
git add src/trainers/batch_extras.py src/data_collators/enrichers.py src/data_collators/batch_extras_collator.py src/training/encoding/model_inputs.py tests/test_batch_extras_contract.py tests/test_model_input_bundle_contract.py tests/test_teacher_forcing_sidecar_bridge.py
git commit -m "feat: transport instance-enumeration sidecars"
```

### Task 4: Implement Visual Token Region Masks

**Files:**

- Create: `src/detection/instance_enumeration/visual_regions.py`
- Create: `tests/detection/instance_enumeration/test_visual_regions.py`

- [ ] **Step 4.1: Write visual region tests**

Add tests:

```python
import torch
import pytest

from src.detection.instance_enumeration.visual_regions import build_region_masks


def test_region_masks_follow_qwen_post_merge_grid() -> None:
    masks = build_region_masks(
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        spatial_merge_size=2,
        image_width=1000,
        image_height=1000,
        current_box=(0, 0, 500, 500),
        previous_boxes=(),
        support_floor=0.10,
        expected_visual_token_count=4,
        image_token_count=4,
    )
    assert masks.current.shape == (4,)
    assert masks.previous.shape == (4,)
    assert masks.negative.shape == (4,)
    assert masks.current.sum().item() > 0
    assert masks.previous.sum().item() == 0


def test_negative_mask_excludes_current_overlap() -> None:
    masks = build_region_masks(
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        spatial_merge_size=2,
        image_width=1000,
        image_height=1000,
        current_box=(0, 0, 500, 500),
        previous_boxes=((0, 0, 500, 500), (500, 500, 999, 999)),
        support_floor=0.10,
        expected_visual_token_count=4,
        image_token_count=4,
    )
    assert torch.all(masks.negative <= (1.0 - masks.current).clamp(0, 1))
    assert masks.negative.sum().item() > 0


def test_empty_current_mask_is_geometry_error() -> None:
    with pytest.raises(ValueError, match="empty current support mask"):
        build_region_masks(
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            spatial_merge_size=2,
            image_width=1000,
            image_height=1000,
            current_box=(1000, 1000, 1000, 1000),
            previous_boxes=(),
            support_floor=0.10,
            expected_visual_token_count=4,
            image_token_count=4,
        )
```

Add exact-mask geometry tests beyond the smoke cases above:

- non-square original image and asymmetric `image_grid_thw`, with expected
  support vectors that prove the flatten order;
- an edge-touching box that exercises the norm1000 `999` endpoint convention;
- overlap case where current-region support wins over previous-region negativity;
- hard failures for `t != 1`, more than one image grid for a sample, video grids,
  `h` or `w` not divisible by `spatial_merge_size`, captured embedding count
  mismatch, and image placeholder count mismatch.

- [ ] **Step 4.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_visual_regions.py -q
```

Expected: fail because `visual_regions.py` is missing.

- [ ] **Step 4.3: Implement mask builder**

Implement:

- lattice source is `image_grid_thw`;
- require one image grid and `t == 1`;
- require `h % spatial_merge_size == 0` and `w % spatial_merge_size == 0`;
- visual length equals `t * (h // spatial_merge_size) * (w // spatial_merge_size)`;
- visual length matches captured image embedding count;
- visual length matches `input_ids == image_token_id` count;
- norm1000 boxes convert through original dataset image width/height;
- raw masks are area fraction per post-merge visual-token footprint;
- previous masks are max-unioned;
- support masks use `clamp(raw / support_floor, 0, 1)`;
- `m_neg = clamp(m_prev * (1 - m_cur), 0, 1)`;
- invalid/degenerate boxes are excluded before state construction and counted by metrics;
- empty current support mask raises a geometry error.

- [ ] **Step 4.4: Run visual region tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_visual_regions.py -q
```

Expected: pass.

- [ ] **Step 4.5: Commit Task 4**

Run:

```bash
git add src/detection/instance_enumeration/visual_regions.py tests/detection/instance_enumeration/test_visual_regions.py
git commit -m "feat: build instance-enumeration visual masks"
```

### Task 5: Capture Qwen Hidden States And Image Embeddings

**Files:**

- Create: `src/detection/instance_enumeration/capture.py`
- Create: `tests/detection/instance_enumeration/test_capture.py`
- Modify: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`

- [ ] **Step 5.1: Write capture tests**

Add fake-Qwen tests that prove one forward call, unchanged outputs,
hidden-state shape validation, image-embed split capture, count validation, and
hook restoration after exceptions.

```python
def test_forward_capture_returns_outputs_unchanged_and_restores_hooks(fake_qwen_model):
    with InstanceEnumerationForwardCapture(fake_qwen_model) as capture:
        outputs = fake_qwen_model(input_ids=fake_qwen_model.input_ids)
    assert outputs is fake_qwen_model.last_outputs
    assert capture.hidden_states_at_lm_head_input.shape[:2] == fake_qwen_model.input_ids.shape[:2]
    assert len(capture.image_embeds_by_sample) == 1
    assert fake_qwen_model.lm_head_hook_restored is True
    assert fake_qwen_model.get_image_features_restored is True


def test_forward_capture_rejects_sliced_hidden_states(fake_qwen_model):
    fake_qwen_model.return_sliced_hidden = True
    with pytest.raises(ValueError, match="full sequence hidden"):
        with InstanceEnumerationForwardCapture(fake_qwen_model):
            fake_qwen_model(input_ids=fake_qwen_model.input_ids)
```

Define `fake_qwen_model` in the test file with a tiny `torch.nn.Module` exposing `model.get_image_features`, `lm_head`, and Qwen-like output attributes.

Also include a PEFT-like wrapper-stack fake with `.module`, `.base_model`, and
`.model` layers. The capture resolver must find:

- the callable forward model used by the trainer;
- the owner that actually calls `lm_head`;
- the Qwen visual owner whose `get_image_features` is invoked.

The wrapper-stack test must prove the hidden/image hooks are exercised exactly
once during forward and are restored after both success and exception paths.

- [ ] **Step 5.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_capture.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: fail on missing capture module.

- [ ] **Step 5.3: Implement `InstanceEnumerationForwardCapture`**

Implement a context manager that:

- attaches a temporary pre-forward hook or wrapper around `lm_head` to capture its input hidden states;
- resolves wrapper stacks by mirroring or reusing the coverage traversal pattern
  over `.module`, `.base_model`, and `.model` instead of assuming one fixed
  owner path;
- wraps the invoked Qwen `get_image_features` owner to capture main
  `image_embeds` split by sample;
- does not modify hidden/image tensors;
- returns model outputs unchanged;
- restores hooks/wrappers in `__exit__`, including exception paths;
- validates captured hidden states match input time dimension in loss integration;
- never requests `output_attentions`.

- [ ] **Step 5.4: Run capture tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_capture.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: pass.

- [ ] **Step 5.5: Commit Task 5**

Run:

```bash
git add src/detection/instance_enumeration/capture.py tests/detection/instance_enumeration/test_capture.py tests/test_trainer_loss_bridge_qwen3vl_contract.py
git commit -m "feat: capture qwen tensors for instance-enumeration loss"
```

### Task 6: Install Probe Through PEFT And Optimizer

**Files:**

- Create: `src/detection/instance_enumeration/probe.py`
- Modify: `src/sft.py`
- Modify: `src/optim/coord_offset_optimizer.py`
- Modify: `src/optim/__init__.py`
- Create: `tests/detection/instance_enumeration/test_probe_peft.py`
- Create: `tests/detection/instance_enumeration/test_adapter_artifacts.py`
- Create: `tests/detection/instance_enumeration/test_sft_peft_wiring.py`
- Modify: `tests/coord_tokens/test_offset_optimizer.py`

- [ ] **Step 6.1: Write probe and optimizer tests**

Add tests:

```python
import torch

from src.detection.instance_enumeration.probe import InstanceEnumerationProbe
from src.optim.coord_offset_optimizer import create_multimodal_coord_offset_optimizer


def test_probe_projects_and_normalizes_qk() -> None:
    probe = InstanceEnumerationProbe(hidden_size=8, probe_dim=4)
    anchor = torch.randn(2, 8)
    visual = torch.randn(2, 5, 8)
    q, k = probe(anchor, visual)
    assert q.shape == (2, 4)
    assert k.shape == (2, 5, 4)
    assert torch.allclose(q.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_optimizer_groups_instance_enum_probe_once(tiny_peft_like_model, optimizer_args):
    optimizer_args.instance_enumeration_aux_config = SimpleNamespace(
        enabled=True,
        probe_lr=5.0e-5,
    )
    optimizer, _ = create_multimodal_coord_offset_optimizer(
        optimizer_args,
        tiny_peft_like_model,
        dataset=None,
    )
    probe_params = {id(p) for n, p in tiny_peft_like_model.named_parameters() if "instance_enumeration_probe" in n}
    groups_with_probe = [
        group for group in optimizer.param_groups
        if any(id(p) in probe_params for p in group["params"])
    ]
    assert len(groups_with_probe) == 1
    assert groups_with_probe[0]["lr"] == pytest.approx(5.0e-5)
    assert groups_with_probe[0]["weight_decay"] == 0.0
```

Reuse or copy small fake optimizer/model helpers from `tests/coord_tokens/test_offset_optimizer.py`.

Also add active PEFT-copy and artifact-mode tests before this task commits:

- a toy PEFT/modules-to-save wrapper resolves the active copied
  `instance_enumeration_probe`, not the frozen original;
- PEFT adapter state contains probe `q_proj` and `k_proj` tensors in
  `training_aux_adapter` mode;
- aux-enabled resume fails if the requested training adapter is missing probe
  tensors that should be present;
- inference checkpoint resolution rejects an adapter config/state dict that still
  declares or contains `instance_enumeration_probe`, with a clear
  `training-only probe present` style error;
- every trainable parameter appears in exactly one optimizer group;
- with aux enabled, every optimizer group has `weight_decay == 0.0`, including
  fallback LLM/LoRA groups;
- no active probe parameter appears in the fallback group.

- [ ] **Step 6.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_probe_peft.py tests/detection/instance_enumeration/test_adapter_artifacts.py tests/detection/instance_enumeration/test_sft_peft_wiring.py tests/coord_tokens/test_offset_optimizer.py tests/test_infer_checkpoint_resolution.py -q -rs
```

Expected: fail on missing probe and optimizer grouping.

- [ ] **Step 6.3: Implement `InstanceEnumerationProbe`**

Implement one shared linear q/k probe:

- module name: `instance_enumeration_probe`;
- `probe_dim=256`;
- separate `q_proj` and `k_proj`;
- q/k L2 normalization;
- accepts native dtype tensors but returns projected q/k for fp32 scoring in `loss.py`;
- no MLP, no per-layer, no per-head, no row-conditioned branches.

- [ ] **Step 6.4: Attach before PEFT and resolve active wrapped copy**

In `src/sft.py`, follow the coord-offset pattern:

- install `instance_enumeration_probe` before `sft.prepare_model`;
- append `"instance_enumeration_probe"` to `modules_to_save`;
- attach aux config to `train_args` and nested training args as `instance_enumeration_aux_config`;
- after PEFT wrapping, resolve the active `ModulesToSaveWrapper.modules_to_save[active]` probe copy;
- fail if the active trainable copy cannot be found;
- do not use the frozen original module for the loss.
- keep training-time probe ownership in the model/PEFT stack, not in
  `TrainerLossBridge`.

- [ ] **Step 6.5: Extend optimizer grouping**

Extend `create_multimodal_coord_offset_optimizer` or rename internally while preserving public imports to include one explicit instance-enumeration probe bucket when enabled:

- group active probe parameters exactly once;
- `lr = instance_enumeration_aux_config.probe_lr`;
- `weight_decay = 0.0`;
- all V1 instance-enumeration optimizer groups use weight decay `0.0`.
- do not return early to the plain multimodal optimizer when coord-offset is
  disabled but `instance_enumeration_aux_config.enabled=true`;
- reject or filter fallback parameter groups so coord-offset params and active
  probe params cannot appear twice.

- [ ] **Step 6.6: Gate training and inference adapter artifact modes**

Define these modes explicitly:

```text
training_aux_adapter:
  may contain instance_enumeration_probe in modules_to_save and safetensors
  required for aux-enabled resume

inference_adapter/export:
  must not contain instance_enumeration_probe config or tensors
  V1a rejects such adapters; strip/export tooling is out of scope until approved
```

Implement only the conservative V1a behavior:

- aux-enabled resume must restore the active probe state or fail;
- inference checkpoint resolution must reject an unstripped training aux adapter
  before generation setup;
- do not silently drop probe tensors during training resume.

- [ ] **Step 6.7: Run probe/optimizer/artifact tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_probe_peft.py tests/detection/instance_enumeration/test_adapter_artifacts.py tests/detection/instance_enumeration/test_sft_peft_wiring.py tests/coord_tokens/test_offset_optimizer.py tests/test_infer_checkpoint_resolution.py -q -rs
```

Expected: pass.

- [ ] **Step 6.8: Commit Task 6**

Run:

```bash
git add src/detection/instance_enumeration/probe.py src/sft.py src/optim/coord_offset_optimizer.py src/optim/__init__.py src/infer/checkpoints.py tests/detection/instance_enumeration/test_probe_peft.py tests/detection/instance_enumeration/test_adapter_artifacts.py tests/detection/instance_enumeration/test_sft_peft_wiring.py tests/coord_tokens/test_offset_optimizer.py tests/test_infer_checkpoint_resolution.py
git commit -m "feat: wire instance-enumeration probe through peft"
```

### Task 7: Implement Probability-Mass Auxiliary Loss

**Files:**

- Create: `src/detection/instance_enumeration/loss.py`
- Create: `tests/detection/instance_enumeration/test_loss.py`

- [ ] **Step 7.1: Write loss formula tests**

Add tests:

```python
import torch

from src.detection.instance_enumeration.loss import instance_enumeration_loss_from_scores


def test_high_current_score_lowers_positive_loss() -> None:
    scores_good = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)
    scores_bad = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32)
    current = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    previous = torch.zeros_like(current)
    good = instance_enumeration_loss_from_scores(scores_good, current, previous, margin=0.5)
    bad = instance_enumeration_loss_from_scores(scores_bad, current, previous, margin=0.5)
    assert good.loss_pos < bad.loss_pos


def test_previous_mass_raises_negative_loss() -> None:
    scores = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32)
    current = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    previous = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    result = instance_enumeration_loss_from_scores(scores, current, previous, margin=0.5)
    assert result.valid_neg_row_count == 1
    assert result.loss_neg.item() > 0
    assert result.margin_satisfied_rate.item() == 0


def test_empty_negative_mask_preserves_positive_loss_and_zeroes_negative() -> None:
    scores = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)
    current = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    previous = torch.zeros_like(current)
    result = instance_enumeration_loss_from_scores(scores, current, previous, margin=0.5)
    assert result.valid_pos_row_count == 1
    assert result.valid_neg_row_count == 0
    assert result.loss_neg.item() == 0
```

Also add one high-level aux API test before bridge integration. The API should
live in `src/detection/instance_enumeration/loss.py` and be named
`compute_instance_enumeration_aux_loss(...)` or similarly explicit. It consumes:

```text
instance_enumeration_state tuple
teacher_forcing_target_ir tuple
captured hidden states at lm_head input
captured image embeddings by sample
image_grid_thw
input_ids
active InstanceEnumerationProbe
InstanceEnumerationAuxConfig
```

It returns weighted/unweighted losses plus `instance_enum/*` metrics. The tests
must prove:

- nonmonotonic rendered order uses previous rows from `ordered_object_indices`;
- anchors are looked up by `branch_position == 0` and `target_position`;
- future boxes never enter previous/negative masks;
- image placeholder count, captured image-embedding count, and `image_grid_thw`
  post-merge count must agree;
- one valid sample emits the default monitoring metrics before any bridge code
  calls the API.

- [ ] **Step 7.2: Verify tests fail before implementation**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_loss.py -q
```

Expected: fail because loss module is missing.

- [ ] **Step 7.3: Implement stable loss math**

Implement fp32 math:

```python
log_p = torch.log_softmax(score_scale * scores.float(), dim=-1)
log_mass_cur = torch.logsumexp(log_p + masked_log(m_cur.float()), dim=-1)
log_mass_neg = torch.logsumexp(log_p + masked_log(m_neg.float()), dim=-1)
loss_pos = -log_mass_cur
loss_neg = torch.nn.functional.softplus(margin + log_mass_neg - log_mass_cur)
```

Rules:

- valid current rows require nonempty current support mask;
- `m_neg = previous_mask * (1 - current_mask)` after max-union support conversion;
- empty or tiny negative rows materialize `L_neg=0` for bookkeeping and are excluded from negative reducers;
- no future boxes enter previous masks;
- aggregate optimization loss by sample mean, not row sum;
- row-weighted outputs are diagnostics only;
- metrics include default health keys from the design spec.

- [ ] **Step 7.4: Implement high-level aux API**

Keep `TrainerLossBridge` thin by implementing row iteration, anchor lookup, mask
construction, probe scoring, weighted aggregation, and metrics in
`compute_instance_enumeration_aux_loss(...)`. The bridge should only collect
capture artifacts, resolve the active probe/config, call CE, call this API, and
add the weighted loss.

- [ ] **Step 7.5: Run loss tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration/test_loss.py -q
```

Expected: pass.

- [ ] **Step 7.6: Commit Task 7**

Run:

```bash
git add src/detection/instance_enumeration/loss.py tests/detection/instance_enumeration/test_loss.py
git commit -m "feat: compute instance-enumeration auxiliary loss"
```

### Task 8: Integrate Aux Loss Into TrainerLossBridge

**Files:**

- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `tests/test_teacher_forcing_sidecar_bridge.py`
- Modify: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`

- [ ] **Step 8.1: Write bridge integration tests**

Add tests that prove:

- ordinary CE-only path unchanged when aux disabled or sidecar absent;
- aux-enabled sidecar runs exactly one model forward inside capture;
- result loss equals CE plus weighted aux loss;
- row coverage metrics stay isolated;
- full logits/hidden time dimension is required;
- visual keys are detached.
- the bridge resolves the active post-PEFT probe/config installed by `src/sft.py`
  and never instantiates a fresh probe.

Example assertion pattern:

```python
def test_bridge_adds_instance_enum_loss_after_ce(fake_qwen_bridge_model, valid_instance_enum_batch):
    result = TrainerLossBridge().compute_loss(
        model=fake_qwen_bridge_model,
        raw_batch=valid_instance_enum_batch.raw_batch,
        batch_extras=valid_instance_enum_batch.batch_extras,
        supervision=valid_instance_enum_batch.supervision,
        objectives=valid_instance_enum_batch.objectives,
        sample_id_to_batch_index=valid_instance_enum_batch.sample_id_to_batch_index,
    )
    assert result.loss.requires_grad
    assert result.instance_enumeration_metrics["instance_enum/loss"] > 0
    assert fake_qwen_bridge_model.forward_call_count == 1
```

Define fake batch/model helpers locally if no existing bridge harness covers Qwen image features.

- [ ] **Step 8.2: Verify tests fail before integration**

Run:

```bash
python -m pytest tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: fail on missing bridge metrics/result fields.

- [ ] **Step 8.3: Extend bridge result and compute path**

Add `instance_enumeration_metrics: Mapping[str, float]` to `TrainerLossBridgeResult`.

When aux is disabled:

- run the existing path unchanged.

When aux is enabled:

- require `BatchExtras.instance_enumeration_state`;
- require `teacher_forcing_target_ir`;
- reject packed metadata and `logits_to_keep`;
- run Qwen once inside `InstanceEnumerationForwardCapture`;
- validate full hidden/logit shape;
- compute ordinary objective result first or preserve equivalent CE semantics;
- resolve the active post-PEFT probe handle and aux config from the training
  context installed in Task 6;
- call `compute_instance_enumeration_aux_loss(...)` with captured hidden states,
  image embeddings, sidecars, target IR, `image_grid_thw`, `input_ids`, active
  probe, and config;
- return `objective_result.loss + weighted_aux`.

Do not call row-coverage forward context or feature painter for instance-enumeration aux.

- [ ] **Step 8.4: Stash metrics into trainer logs**

In `src/trainers/metrics/teacher_forcing.py`, mirror the row-coverage metric stash/consume pattern with `instance_enum/` keys. Log only default metrics every step; debug-only metrics can be routed through an internal debug flag or omitted in V1a.

- [ ] **Step 8.5: Run bridge tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
```

Expected: pass.

- [ ] **Step 8.6: Commit Task 8**

Run:

```bash
git add src/training/bridge/loss_bridge.py src/trainers/metrics/teacher_forcing.py tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py
git commit -m "feat: add instance-enumeration loss to bridge"
```

### Task 9: Add Wiring And Health Gate Tests

**Files:**

- Create or modify: `tests/detection/instance_enumeration/test_probe_peft.py`
- Create or modify: `tests/detection/instance_enumeration/test_sft_peft_wiring.py`
- Create or modify: `tests/detection/instance_enumeration/test_adapter_artifacts.py`
- Modify: `tests/test_teacher_forcing_sidecar_bridge.py`
- Modify: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`

- [ ] **Step 9.1: Add wiring gate test**

Add one tiny aux-only backward test that proves:

```text
finite nonzero aux loss
active modules_to_save probe copy resolved
active probe grad nonzero
at least one adapter/LoRA grad nonzero and attributable to aux path
visual key aux grad absent when detach_visual_keys=true
image placeholder count == captured image embed count == image_grid_thw post-merge count
row-conditioned visual coverage disabled
```

Test skeleton:

```python
def test_instance_enum_aux_backward_reaches_active_probe_and_adapter(tiny_aux_qwen_peft_model, tiny_valid_batch):
    result = run_aux_only_backward(tiny_aux_qwen_peft_model, tiny_valid_batch)
    assert result.loss.isfinite()
    assert result.loss.item() > 0
    assert result.active_probe_grad_norm > 0
    assert result.adapter_grad_norm > 0
    assert result.visual_key_grad_norm == 0
    assert result.image_token_count == result.image_embed_count == result.grid_token_count
```

Define `run_aux_only_backward` and fake model helpers in the test module if a real Qwen smoke is too heavy for CPU.

- [ ] **Step 9.2: Add adapter artifact gate assertions**

Assert the V1a artifact split:

- training aux adapter checkpoints include `instance_enumeration_probe` tensors
  and restore them for resume;
- partial adapter loading cannot silently continue with a freshly initialized
  probe when aux resume requested probe tensors;
- inference adapter resolution rejects adapters whose `adapter_config.json` or
  `adapter_model.safetensors` still contains `instance_enumeration_probe`;
- V1a does not claim export stripping; a future strip/export task must add its
  own tests before deployable adapter claims.

- [ ] **Step 9.3: Add tiny-training metric gate assertions**

Add a metric schema test that asserts these default keys exist when aux runs:

```text
instance_enum/loss
instance_enum/loss_pos
instance_enum/loss_neg
instance_enum/valid_pos_row_count
instance_enum/valid_neg_row_count
instance_enum/skipped_neg_empty_rate
instance_enum/current_mask_geometry_error_count
instance_enum/current_mask_sum
instance_enum/negative_mask_sum
instance_enum/current_mass
instance_enum/previous_mass
instance_enum/mass_ratio_cur_to_neg
instance_enum/margin_satisfied_rate
instance_enum/probe_entropy
instance_enum/probe_effective_tokens
instance_enum/probe_top1_in_current
instance_enum/probe_top5_current_mass
instance_enum/probe_grad_norm
instance_enum/adapter_grad_norm
instance_enum/adapter_nonzero_grad_param_count
instance_enum/aux_to_ce_loss_ratio
instance_enum/weighted_aux_to_total_loss_ratio
instance_enum/lr_probe
```

- [ ] **Step 9.4: Run gate tests**

Run:

```bash
python -m pytest tests/detection/instance_enumeration tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q -rs
```

Expected: pass.

- [ ] **Step 9.5: Commit Task 9**

Run:

```bash
git add tests/detection/instance_enumeration tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py
git commit -m "test: gate instance-enumeration aux wiring"
```

### Task 10: Add Random And Sorted Smoke Configs

**Files:**

- Create: `configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml`
- Create: `configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml`
- Modify: `tests/test_detection_training_config_contract.py`
- Create: `tests/detection/instance_enumeration/test_sorted_random_ablation_contract.py`

- [ ] **Step 10.1: Write config parse tests**

Add assertions:

```python
@pytest.mark.parametrize(
    "path,expected_order",
    [
        ("configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml", "random_permutation"),
        ("configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml", "sorted"),
    ],
)
def test_instance_enum_smoke_configs_parse(path, expected_order):
    cfg = ConfigLoader.load_training_config(path)[1]
    assert cfg.instance_enumeration_aux.enabled is True
    assert cfg.data.object_ordering == expected_order
    assert cfg.objective.target_ir.rollin_policy.name == expected_order
    assert cfg.row_conditioned_visual_coverage.enabled is False
    assert cfg.pixel_painted_row_coverage.enabled is False
    assert cfg.training.get("packing", False) is False
    assert cfg.training.get("weight_decay") == 0.0
```

Add a dataset/target-IR materialization test with a deterministic three-object
sample:

- random smoke config produces a deterministic `random_permutation` metadata
  order for the configured seed;
- sorted smoke config produces exactly sorted/rendered
  `selected_normalized_object_indices`;
- the `InstanceEnumerationState.ordered_object_indices` copied by the dataset
  matches the target IR metadata in both lanes.

- [ ] **Step 10.2: Verify tests fail before configs exist**

Run:

```bash
python -m pytest tests/test_detection_training_config_contract.py tests/detection/instance_enumeration/test_sorted_random_ablation_contract.py -q
```

Expected: fail on missing smoke configs or missing parser fields.

- [ ] **Step 10.3: Add smoke configs**

Create two configs by mimicking nearby Stage-1 teacher-forcing smoke leaves. Both must use:

```yaml
training:
  weight_decay: 0.0

instance_enumeration_aux:
  enabled: true
  version: v1
  probe_lr: 5.0e-5
  positive_weight: 0.05
  negative_weight: 0.02
  negative_margin: 0.5
  support_floor: 0.10
row_conditioned_visual_coverage:
  enabled: false
pixel_painted_row_coverage:
  enabled: false
```

Only `data.object_ordering` differs:

```text
random config: random_permutation
sorted config: sorted
```

The target-IR roll-in policy must also differ in the same way:

```yaml
objective:
  target_ir:
    rollin_policy:
      name: random_permutation  # random config
      base_seed: 17
```

and:

```yaml
objective:
  target_ir:
    rollin_policy:
      name: sorted  # sorted config
      base_seed: 17
```

- [ ] **Step 10.4: Run config parse tests**

Run:

```bash
python -m pytest tests/test_detection_training_config_contract.py tests/detection/instance_enumeration/test_config.py tests/detection/instance_enumeration/test_sorted_random_ablation_contract.py -q
```

Expected: pass.

- [ ] **Step 10.5: Commit Task 10**

Run:

```bash
git add configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml tests/test_detection_training_config_contract.py tests/detection/instance_enumeration/test_sorted_random_ablation_contract.py
git commit -m "config: add instance-enumeration smoke configs"
```

### Task 11: Run Final CPU Verification And Prepare Tiny Smoke Gate

**Files:**

- Modify only if failures expose V1a implementation bugs.

- [ ] **Step 11.1: Run narrow unit suite**

Run:

```bash
python -m pytest tests/detection/instance_enumeration -q
python -m pytest tests/test_batch_extras_contract.py tests/test_model_input_bundle_contract.py tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py -q
python -m pytest tests/test_detection_training_config_contract.py tests/detection/coverage/test_config.py tests/detection/coverage/test_pixel_config.py -q
python -m pytest tests/coord_tokens/test_offset_optimizer.py tests/test_infer_checkpoint_resolution.py -q
```

Expected: pass.

- [ ] **Step 11.2: Materialize resolved random and sorted configs**

Run the repo’s existing config loader or dry-run config path for:

```text
configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml
configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml
```

Expected checks:

- `instance_enumeration_aux.enabled=true`;
- `row_conditioned_visual_coverage.enabled=false`;
- `pixel_painted_row_coverage.enabled=false`;
- `training.packing=false`;
- `packing.static_packing=false`;
- `packing.padding_free_packed=false`;
- `training.weight_decay=0.0`;
- object order differs only as random versus sorted;
- `objective.target_ir.rollin_policy.name` matches `data.object_ordering`;
- materialized `selected_normalized_object_indices` is sorted for the sorted
  lane.

- [ ] **Step 11.3: Run one tiny CPU or single-GPU wiring smoke**

Use the committed tiny smoke config, which inherits
`configs/stage1/detection_teacher_forcing/smoke/compact_full_tiny.yaml` and
therefore limits training to one sample and one step. The smoke must report
evidence scope as `tiny`.

Config-only dry runs:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml --cfg-only
python -m src.sft --config configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml --cfg-only
```

One real tiny wiring smoke after CPU/unit gates pass:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml --debug
```

Inspect the run's `logging.jsonl` under the resolved smoke output directory
(`temp/detection_teacher_forcing/output/...`) and require the metric keys below
on the first logged training step. The predeclared smoke tolerance is:

```text
0.0 < instance_enum/weighted_aux_to_total_loss_ratio <= 0.25
```

Required pass conditions:

```text
finite nonzero instance_enum/loss
valid_pos_row_count > 0
current_mask_geometry_error_count == 0
probe_grad_norm finite and > 0
adapter_grad_norm finite and > 0
weighted_aux_to_total_loss_ratio within predeclared smoke tolerance
```

- [ ] **Step 11.4: Stop before production interpretation**

Do not claim training success until rollout/eval reports:

```text
duplicate box rate
same-label same-region repeat rate
max-row hit rate
recall
precision
early EOS rate
invalid/parse drop counters
overlap-object recall/precision slice
```

- [ ] **Step 11.5: Commit final implementation stabilization**

Run:

```bash
git status --short
git add src/detection/instance_enumeration src/config/schema.py src/detection/dataset.py src/detection/runtime.py src/detection/teacher_forcing/rollin.py src/detection/teacher_forcing/target_builder.py src/data_collators/enrichers.py src/data_collators/batch_extras_collator.py src/infer/checkpoints.py src/trainers/batch_extras.py src/training/encoding/model_inputs.py src/training/bridge/loss_bridge.py src/trainers/metrics/teacher_forcing.py src/optim/coord_offset_optimizer.py src/optim/__init__.py src/sft.py configs/stage1/detection_teacher_forcing/ablation/instance_enum_random_sft_smoke.yaml configs/stage1/detection_teacher_forcing/ablation/instance_enum_sorted_sft_smoke.yaml tests/detection/instance_enumeration tests/test_batch_extras_contract.py tests/test_model_input_bundle_contract.py tests/test_teacher_forcing_sidecar_bridge.py tests/test_trainer_loss_bridge_qwen3vl_contract.py tests/coord_tokens/test_offset_optimizer.py tests/test_detection_training_config_contract.py tests/test_infer_checkpoint_resolution.py tests/detection/coverage/test_config.py tests/detection/coverage/test_pixel_config.py
git commit -m "test: verify instance-enumeration v1a smoke gates"
```

## Implementation Stop State

After Task 11, the branch may be considered ready for tiny rollout/eval interpretation only if:

- all P0/P1 design constraints in the primary design note are satisfied;
- the only active ablation axis is random versus sorted;
- no implementation modifies Q/K/V, attention logits, raw pixels, visual features, upstream Qwen files, or decode/KV-cache behavior;
- aux metrics show nonzero active-probe and adapter participation;
- sidecar, anchor, geometry, and packed-aux rejection tests pass.
- unstripped training aux adapters are rejected for inference/export claims, and
  no deployable stripped adapter claim is made in V1a.

Before implementation starts, this roadmap itself must be committed and the user must provide explicit approval to proceed.
