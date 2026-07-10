# Row-Conditioned Visual Coverage Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first faithful row-conditioned visual coverage prototype for Stage-1 compact detection teacher forcing and row-boundary re-prefill rollout, while preserving standard SFT/CE and stopping short of cache-efficient decoding.

**Architecture:** Add an experimental row-coverage surface around the existing detection teacher-forcing path. The dataset expands one image into row-state examples with a coverage sidecar; the trainer consumes that sidecar, paints coverage on the actual Qwen visual-token lattice, applies a scalar-gated additive residual to image features through a local wrapper, and computes ordinary CE through the existing teacher-forcing objective. Rollout uses the same coverage painter and tuning helper with row-boundary re-prefill.

**Tech Stack:** Python, PyTorch, Hugging Face Qwen3-VL through local wrappers only, existing CoordExp detection scene/runtime modules, detection-config-native experimental section, JSONL/YAML artifacts, pytest.

---

## Scope Lock

This plan implements:

- `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-design.md`
- `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-protocol.md`
- `progress/directions/2026-06-05_row_conditioned_visual_coverage.md`

Implementation must preserve these constraints:

- no upstream Hugging Face model file edits;
- no attention-bias routing, attention-mask routing, logit biasing, EOS forcing,
  fallback decoding, or other non-AR/custom decode mechanism anywhere in v1;
- no residual-set, trie/set, duplicate-unlikelihood, or fallback objective;
- no packing in the first training path;
- no cache-efficient rollout claim in v1;
- `random_sft` and `sorted_sft` remain separate ablation modes.

## 2026-06-07 Validation Refinement Addendum

Treat the prototype as a **Direct Feature Painting Oracle**. The purpose is to
prove or falsify the faithful feature-painting mechanism, not to optimize
KV-cache efficiency. Any append-coverage-token or other cache-compatible decode
path is a separate ablation and must not be mixed into the primary claim path.

Before interpreting a loss curve, rollout metric, or eval score, future work
must verify and record:

- **Residual norm:** alpha caps are not enough. Log
  `||delta_F_i|| / ||F_i||`, max/mean token ratios, and boundary/interior
  embedding norms. If residuals dominate original visual features, hold
  interpretation until conservative initialization, tighter caps, embedding
  norm control, or regularization is tested.
- **Union memory:** v1 accumulates all prior boxes into shared
  boundary/interior channels. It does not carry instance id, object count,
  class/text identity, recency, or confidence.
- **BBox boundary only:** the boundary channel marks bbox extent on the
  post-merge visual-token lattice, not a true object contour.
- **Interior is not suppression:** interior coverage means already enumerated.
  It must not zero, hide, or directly suppress visual features.
- **Faithful re-prefill:** training/decode equivalence is claimed only for
  row-boundary re-prefill with freshly painted features for the current
  committed prefix.
- **Coordinate alignment:** visually validate
  `norm1000 bbox -> original image coordinates -> processor grid metadata ->
  patch grid -> post-merge visual-token lattice -> painted cell overlay`.
- **Loss masking:** prefix rows are context only; the current target row and
  terminal stop row receive CE; prefix rows are not re-supervised unless an
  explicit ablation says so.
- **Order bias:** keep fixed-order and random/mixed-order results separate and
  plan order ablations before broad claims.
- **Teacher/rollout exposure gap:** compare clean GT coverage with corrupted
  prefixes such as bbox jitter, dropped/duplicated previous objects,
  enlarged/shrunk boxes, wrong-label/right-box, right-label/shifted-box, and
  optional confidence-weighted coverage.

Minimal ablations for the first interpretation pass:

- baseline SFT with no coverage;
- boundary-only painting;
- interior-only painting;
- boundary plus weak interior painting;
- boundary plus weak interior plus count or log-count channel;
- clean GT coverage versus corrupted-prefix coverage;
- fixed order versus random/mixed order;
- faithful re-prefill painting decode versus a later cache-compatible
  coverage-token variant.

Prioritize hypothesis diagnostics over global metrics:

- duplicate burst rate;
- repeated same-instance emission;
- premature assistant-stop rate;
- crowded-scene recall;
- overlapping and nested-object recall;
- row-wise recovery after prefix corruption;
- residual norm ratio;
- coordinate-alignment overlays.

mAP, recall, and precision are secondary global metrics for this mechanism
gate. They are necessary but not sufficient to prove that coverage memory is
the active cause of a result.

Test snippets below define required assertions and API contracts. When a snippet
uses helper names such as `two_object_scene`, `row_coverage_state`,
`make_loss_bridge`, or `fake_rollout_model`, the implementer must either reuse
an existing local helper in that test file or define a small fixture/helper in
the same patch; those names are not assumed to exist globally.

## File Structure

Create a focused coverage package:

- Create: `src/detection/coverage/__init__.py`
- Create: `src/detection/coverage/config.py`
- Create: `src/detection/coverage/types.py`
- Create: `src/detection/coverage/geometry.py`
- Create: `src/detection/coverage/painting.py`
- Create: `src/detection/coverage/prefix_rendering.py`
- Create: `src/detection/coverage/row_state_dataset.py`
- Create: `src/detection/coverage/forward.py`
- Create: `src/detection/coverage/rollout.py`
- Create: `src/detection/coverage/artifacts.py`

Modify existing boundaries narrowly:

- Modify: `src/datasets/geometry.py`
- Modify: `src/config/schema.py`
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/trainers/batch_extras.py`
- Modify: `src/data_collators/enrichers.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Modify: `src/training/encoding/model_inputs.py`
- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/sft.py`

Add experiment configs:

- Create: `configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml`
- Create: `configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml`
- Create: `configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml`

Add tests:

- Create: `tests/detection/coverage/test_config.py`
- Create: `tests/detection/coverage/test_geometry.py`
- Create: `tests/detection/coverage/test_painting.py`
- Create: `tests/detection/coverage/test_row_state_dataset.py`
- Create: `tests/detection/coverage/test_forward.py`
- Create: `tests/detection/coverage/test_rollout.py`
- Modify: `tests/test_teacher_forcing_target_builder.py`
- Modify: `tests/test_teacher_forcing_config_contract.py`
- Modify: `tests/test_teacher_forcing_sidecar_bridge.py`

## Execution Boundary

Use small commits after each task. Do not run production training during this
plan. GPU smoke is optional and must be explicitly labeled `tiny`.

---

### Task 1: Parse Experimental Coverage Config

**Files:**
- Modify: `src/config/schema.py`
- Create: `src/detection/coverage/config.py`
- Create: `src/detection/coverage/__init__.py`
- Test: `tests/detection/coverage/test_config.py`

- [ ] **Step 1: Write config tests**

Add:

```python
import pytest

from src.detection.coverage.config import RowCoverageConfig
from src.config.schema import DetectionTrainingConfig


def test_disabled_when_section_missing():
    cfg = RowCoverageConfig.from_mapping(None)
    assert cfg.enabled is False
    assert cfg.version == "v1"


def test_v1_config_accepts_protocol_shape():
    cfg = RowCoverageConfig.from_mapping(
        {
            "enabled": True,
            "version": "v1",
            "row_state_policy": "prefix_expansion",
            "tune_operator": "additive_residual",
            "descriptor": {
                "boundary_channel": True,
                "interior_channel": True,
                "boundary_band_width_tokens": 1.0,
                "accumulation": "clamp_0_1",
            },
            "residual": {
                "boundary_alpha_init": 0.0,
                "interior_alpha_init": 0.0,
                "boundary_alpha_max": 0.25,
                "interior_alpha_max": 0.08,
            },
            "training": {
                "unpacked_only": True,
                "include_terminal_state": True,
                "ordering_modes": ["random_sft", "sorted_sft"],
            },
            "rollout": {
                "strategy": "row_boundary_reprefill",
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.row_state_policy == "prefix_expansion"
    assert cfg.tune_operator == "additive_residual"
    assert cfg.descriptor.boundary_band_width_tokens == 1.0
    assert cfg.training.include_terminal_state is True
    assert cfg.rollout.strategy == "row_boundary_reprefill"


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="row_conditioned_visual_coverage.bad_key"):
        RowCoverageConfig.from_mapping({"enabled": True, "bad_key": 1})


def test_config_rejects_non_v1_operator():
    with pytest.raises(ValueError, match="tune_operator"):
        RowCoverageConfig.from_mapping(
            {
                "enabled": True,
                "tune_operator": "attention_bias",
            }
        )


def test_config_rejects_disabled_descriptor_channels():
    with pytest.raises(ValueError, match="descriptor.boundary_channel"):
        RowCoverageConfig.from_mapping(
            {
                "enabled": True,
                "descriptor": {"boundary_channel": False},
            }
        )


def test_config_rejects_inverted_boundary_interior_strength():
    with pytest.raises(ValueError, match="boundary_alpha_max"):
        RowCoverageConfig.from_mapping(
            {
                "enabled": True,
                "residual": {
                    "boundary_alpha_max": 0.05,
                    "interior_alpha_max": 0.08,
                }
            }
        )


def test_detection_config_accepts_native_row_coverage_section():
    # Reuse or locally copy the existing _detection_payload helper from
    # tests/test_detection_training_config_contract.py.
    payload = _detection_payload()
    payload["row_conditioned_visual_coverage"] = {
        "enabled": True,
        "version": "v1",
        "row_state_policy": "prefix_expansion",
        "tune_operator": "additive_residual",
        "training": {
            "unpacked_only": True,
            "include_terminal_state": True,
            "ordering_modes": ["random_sft", "sorted_sft"],
        },
    }
    cfg = DetectionTrainingConfig.from_mapping(payload)
    assert cfg.row_conditioned_visual_coverage.enabled is True
```

- [ ] **Step 2: Run config tests to verify RED**

Run:

```bash
python -m pytest tests/detection/coverage/test_config.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError` for
`src.detection.coverage`, missing `RowCoverageConfig`, or unknown detection
config top-level key `row_conditioned_visual_coverage`.

- [ ] **Step 3: Implement strict dataclasses**

Add dataclasses with defaults matching the protocol:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CoverageDescriptorConfig:
    boundary_channel: bool = True
    interior_channel: bool = True
    boundary_band_width_tokens: float = 1.0
    accumulation: str = "clamp_0_1"


@dataclass(frozen=True)
class CoverageResidualConfig:
    boundary_alpha_init: float = 0.0
    interior_alpha_init: float = 0.0
    boundary_alpha_max: float = 0.25
    interior_alpha_max: float = 0.08


@dataclass(frozen=True)
class CoverageTrainingConfig:
    unpacked_only: bool = True
    include_terminal_state: bool = True
    ordering_modes: tuple[str, ...] = ("random_sft", "sorted_sft")


@dataclass(frozen=True)
class CoverageRolloutConfig:
    strategy: str = "row_boundary_reprefill"


@dataclass(frozen=True)
class RowCoverageConfig:
    enabled: bool = False
    version: str = "v1"
    row_state_policy: str = "prefix_expansion"
    tune_operator: str = "additive_residual"
    descriptor: CoverageDescriptorConfig = CoverageDescriptorConfig()
    residual: CoverageResidualConfig = CoverageResidualConfig()
    training: CoverageTrainingConfig = CoverageTrainingConfig()
    rollout: CoverageRolloutConfig = CoverageRolloutConfig()
```

`RowCoverageConfig.from_mapping(None)` must return disabled defaults.

The parser must:

- accept missing section as disabled;
- reject non-mapping sections;
- reject unknown keys at every nested level with dotted paths;
- reject `version != "v1"`;
- reject `row_state_policy != "prefix_expansion"`;
- reject `tune_operator != "additive_residual"`;
- reject `descriptor.accumulation != "clamp_0_1"`;
- reject `descriptor.boundary_channel is not True`;
- reject `descriptor.interior_channel is not True`;
- reject `training.unpacked_only is not True`;
- reject `training.include_terminal_state is not True`;
- reject `tuple(training.ordering_modes) != ("random_sft", "sorted_sft")`;
- reject `rollout.strategy != "row_boundary_reprefill"`;
- reject non-finite or negative alpha limits.
- reject `residual.boundary_alpha_max <= residual.interior_alpha_max`.

- [ ] **Step 4: Add native detection config section**

In `src/config/schema.py`:

- import `RowCoverageConfig` from `src.detection.coverage.config`;
- add `row_conditioned_visual_coverage: RowCoverageConfig` to
  `DetectionTrainingConfig`;
- include `"row_conditioned_visual_coverage"` in the detection optional
  top-level set;
- parse it with `RowCoverageConfig.from_mapping(payload.get("row_conditioned_visual_coverage"))`;
- parse it before `_detection_validate_order_matches_objective`;
- update `_detection_validate_order_matches_objective` or its caller so
  `data.object_ordering: sorted` is allowed only when
  `row_conditioned_visual_coverage.enabled` is true and
  `objective.target_ir.rollin_policy.name` remains `random_permutation`;
- when coverage is enabled, reject `training.group_by_length: true`,
  `training.packing: true`, and `packing.static_packing: true` at schema or
  preflight time, before runtime dataset construction;
- serialize it in `to_mapping`;
- keep `custom` rejected for detection configs.

- [ ] **Step 5: Run the config tests**

Run:

```bash
python -m pytest tests/detection/coverage/test_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/config/schema.py src/detection/coverage/__init__.py src/detection/coverage/config.py tests/detection/coverage/test_config.py
git commit -m "Add experimental row coverage config parser"
```

---

### Task 2: Implement Bbox And Visual-Lattice Painting

**Files:**
- Create: `src/detection/coverage/types.py`
- Create: `src/detection/coverage/geometry.py`
- Create: `src/detection/coverage/painting.py`
- Modify: `src/datasets/geometry.py`
- Test: `tests/detection/coverage/test_geometry.py`
- Test: `tests/detection/coverage/test_painting.py`

- [ ] **Step 1: Write geometry tests**

Add:

```python
from src.datasets.geometry import norm1000_xyxy_to_pixel_xyxy
from src.detection.coverage.geometry import rect_intersection_area


def test_norm1000_xyxy_to_pixel_xyxy_uses_image_extent():
    assert norm1000_xyxy_to_pixel_xyxy((0, 0, 999, 999), width=100, height=50) == (
        0.0,
        0.0,
        99.0,
        49.0,
    )


def test_rect_intersection_area():
    assert rect_intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25.0
    assert rect_intersection_area((0, 0, 2, 2), (3, 3, 4, 4)) == 0.0
```

- [ ] **Step 2: Write painting tests**

Add:

```python
import pytest
import torch

from src.detection.coverage.painting import (
    paint_batched_coverage_descriptors,
    paint_coverage_descriptor,
)
from src.detection.coverage.types import CoverageBoxNorm1000, CoverageState


def _state(boxes):
    return CoverageState(
        sample_id="sample-0",
        base_idx=0,
        row_state_k=len(boxes),
        target_kind="object_row",
        target_object_index=len(boxes),
        prefix_object_indices=tuple(range(len(boxes))),
        coverage_object_indices=tuple(range(len(boxes))),
        rendered_teacher_forcing_indices=tuple(range(len(boxes) + 1)),
        supervised_object_indices=(len(boxes),),
        image_width=100,
        image_height=100,
        coverage_boxes=tuple(boxes),
        ordering_strategy="sorted",
        ordering_seed=None,
    )


def test_empty_coverage_is_zero_descriptor():
    desc = paint_coverage_descriptor(
        _state(()),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    assert desc.boundary_mass.shape == (4,)
    assert desc.interior_mass.shape == (4,)
    assert torch.count_nonzero(desc.boundary_mass) == 0
    assert torch.count_nonzero(desc.interior_mass) == 0


def test_center_box_paints_boundary_and_interior_without_global_normalization():
    box = CoverageBoxNorm1000(250, 250, 749, 749)
    one = paint_coverage_descriptor(
        _state((box,)),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    two = paint_coverage_descriptor(
        _state((box, CoverageBoxNorm1000(0, 0, 100, 100))),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    assert float(one.boundary_mass.max()) <= 1.0
    assert float(one.interior_mass.max()) <= 1.0
    assert float(one.boundary_mass.sum()) > 0.0
    assert float(one.interior_mass.sum()) > 0.0
    assert torch.all(two.boundary_mass >= one.boundary_mass)
    assert torch.all(two.interior_mass >= one.interior_mass)


def test_token_aligned_box_paints_expected_cells():
    desc = paint_coverage_descriptor(
        _state((CoverageBoxNorm1000(0, 0, 499, 499),)),
        image_grid_thw=torch.tensor([1, 2, 2]),
        merge_size=1,
        band_width_tokens=1.0,
    )
    assert float(desc.interior_mass[0]) > 0.0
    assert float(desc.interior_mass[3]) == 0.0


def test_tiny_box_has_visible_bounded_boundary_signal():
    desc = paint_coverage_descriptor(
        _state((CoverageBoxNorm1000(490, 490, 500, 500),)),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    assert 0.0 < float(desc.boundary_mass.max()) <= 1.0
    assert float(desc.boundary_mass.sum()) >= float(desc.interior_mass.sum())


def test_adjacent_boxes_do_not_change_unrelated_far_tokens():
    first = paint_coverage_descriptor(
        _state((CoverageBoxNorm1000(0, 0, 100, 100),)),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=1,
        band_width_tokens=1.0,
    )
    second = paint_coverage_descriptor(
        _state((CoverageBoxNorm1000(0, 0, 100, 100), CoverageBoxNorm1000(120, 0, 220, 100))),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=1,
        band_width_tokens=1.0,
    )
    assert torch.equal(second.boundary_mass[10:], first.boundary_mass[10:])


def test_overlapping_boxes_saturate_only_local_cells():
    desc = paint_coverage_descriptor(
        _state((CoverageBoxNorm1000(0, 0, 700, 700), CoverageBoxNorm1000(250, 250, 999, 999))),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    assert float(desc.boundary_mass.max()) <= 1.0
    assert float(desc.interior_mass.max()) <= 1.0
    assert float(desc.interior_mass[0]) > 0.0
    assert float(desc.interior_mass[-1]) > 0.0


def test_clipped_and_degenerate_boxes_follow_geometry_policy():
    clipped = paint_coverage_descriptor(
        _state((CoverageBoxNorm1000(-100, -100, 100, 100),)),
        image_grid_thw=torch.tensor([1, 4, 4]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    assert float(clipped.boundary_mass.sum()) > 0.0
    with pytest.raises(ValueError, match="degenerate"):
        paint_coverage_descriptor(
            _state((CoverageBoxNorm1000(100, 100, 100, 200),)),
            image_grid_thw=torch.tensor([1, 4, 4]),
            merge_size=2,
            band_width_tokens=1.0,
        )


def test_rejects_grid_not_divisible_by_merge_size():
    with pytest.raises(ValueError, match="divisible by merge_size"):
        paint_coverage_descriptor(
            _state(()),
            image_grid_thw=torch.tensor([1, 5, 4]),
            merge_size=2,
            band_width_tokens=1.0,
        )


def test_batched_descriptor_order_matches_qwen_split_order():
    descriptors = paint_batched_coverage_descriptors(
        (_state((CoverageBoxNorm1000(0, 0, 499, 499),)), _state(())),
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 2]]),
        merge_size=2,
        band_width_tokens=1.0,
    )
    assert descriptors.boundary_mass.shape == (5,)
    assert float(descriptors.boundary_mass[:4].sum()) > 0.0
    assert torch.count_nonzero(descriptors.boundary_mass[4:]) == 0
```

- [ ] **Step 3: Run geometry and painting tests to verify RED**

Run:

```bash
python -m pytest tests/detection/coverage/test_geometry.py tests/detection/coverage/test_painting.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError` for
`src.detection.coverage` or missing `norm1000_xyxy_to_pixel_xyxy`.

- [ ] **Step 4: Add the shared geometry helper**

In `src/datasets/geometry.py`, add:

```python
def norm1000_xyxy_to_pixel_xyxy(
    bbox: Sequence[int | float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four xyxy coordinates")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    scale_x = (float(width) - 1.0) / 999.0
    scale_y = (float(height) - 1.0) / 999.0
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
```

- [ ] **Step 5: Add coverage types**

In `src/detection/coverage/types.py`, add frozen dataclasses:

```python
from dataclasses import dataclass
from typing import Literal, Sequence

import torch


TargetKind = Literal["object_row", "assistant_stop"]


@dataclass(frozen=True)
class CoverageBoxNorm1000:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class CoverageState:
    sample_id: str
    base_idx: int
    row_state_k: int
    target_kind: TargetKind
    target_object_index: int | None
    prefix_object_indices: Sequence[int]
    coverage_object_indices: Sequence[int]
    rendered_teacher_forcing_indices: Sequence[int]
    supervised_object_indices: Sequence[int]
    image_width: int
    image_height: int
    coverage_boxes: Sequence[CoverageBoxNorm1000]
    ordering_strategy: str
    ordering_seed: int | None


@dataclass(frozen=True)
class CoverageDescriptor:
    boundary_mass: torch.Tensor
    interior_mass: torch.Tensor
```

- [ ] **Step 6: Implement lattice painter**

In `src/detection/coverage/geometry.py` and `painting.py`, implement:

```python
def rect_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
```

`paint_coverage_descriptor` must:

- accept `image_grid_thw` as a tensor or sequence of three ints;
- reject `t != 1`;
- accept explicit `merge_size`;
- reject `h % merge_size != 0` or `w % merge_size != 0`;
- use row-major flattened token order on the post-merge lattice
  `(h // merge_size, w // merge_size)`;
- compute merged-token rectangles covering the full image extent;
- accumulate boundary and interior masses with `torch.clamp(sum, 0.0, 1.0)`;
- return tensors on the same device as `image_grid_thw` when it is a tensor.
- compute boundary mass with the protocol outer-minus-inner band:

```text
outer = expand(box, one token footprint)
inner = shrink(box, one token footprint)
boundary_area = area(token intersect outer) - area(token intersect inner)
boundary_mass = clamp(boundary_area / area(token), 0, 1)
```

- [ ] **Step 7: Implement batched descriptor concatenation**

Add `paint_batched_coverage_descriptors` that accepts one `CoverageState` per
image and an `image_grid_thw` tensor shaped `(batch, 3)`. It must concatenate
per-image descriptors in the same order Qwen uses when it splits final image
embeddings by:

```text
image_grid_thw.prod(-1) // merge_size**2
```

Reject descriptor length mismatches before returning.

- [ ] **Step 8: Run geometry and painting tests**

Run:

```bash
python -m pytest tests/detection/coverage/test_geometry.py tests/detection/coverage/test_painting.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/datasets/geometry.py src/detection/coverage/types.py src/detection/coverage/geometry.py src/detection/coverage/painting.py tests/detection/coverage/test_geometry.py tests/detection/coverage/test_painting.py
git commit -m "Add visual coverage lattice painter"
```

---

### Task 3: Build Explicit Row-State Teacher-Forcing Examples

**Files:**
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Modify: `src/detection/dataset.py`
- Create: `src/detection/coverage/prefix_rendering.py`
- Create: `src/detection/coverage/row_state_dataset.py`
- Test: `tests/test_teacher_forcing_target_builder.py`
- Test: `tests/detection/coverage/test_row_state_dataset.py`

- [ ] **Step 1: Extend target-builder tests for explicit row states**

Add tests that prove:

```python
def test_explicit_row_state_supervises_only_next_object_tokens(tokenizer, two_object_scene):
    result = build_teacher_forcing_target(
        two_object_scene,
        tokenizer=tokenizer,
        profile="hard_sft",
        epoch=0,
        stable_sample_id="row-state",
        explicit_rollin_indices=(0, 1),
        coverage_object_indices=(0,),
        supervised_object_indices=(1,),
        supervise_stop=False,
    )
    assert result.ok
    assert result.target_ir.metadata["row_state_mode"] == "explicit"
    assert result.target_ir.metadata["supervised_normalized_object_indices"] == (1,)
    assert result.target_ir.metadata["rendered_teacher_forcing_indices"] == (0, 1)
    assert result.target_ir.metadata["coverage_object_indices"] == (0,)
    assert result.target_ir.metadata["target_kind"] == "object_row"
    target_atoms = result.target_ir.metadata["target_object_indices_by_atom"]
    assert set(target_atoms.values()) == {1}
    first_target_offset = result.target_ir.metadata["first_supervised_token_offset"]
    object_atoms = [
        atom
        for atom in result.target_ir.atoms
        if "object_index" in atom.provenance
    ]
    assert object_atoms
    assert {atom.provenance["object_index"] for atom in object_atoms} == {1}
    assert all(atom.target_position >= first_target_offset for atom in object_atoms)


def test_explicit_terminal_state_supervises_stop_only(tokenizer, two_object_scene):
    result = build_teacher_forcing_target(
        two_object_scene,
        tokenizer=tokenizer,
        profile="hard_sft",
        epoch=0,
        stable_sample_id="terminal-state",
        explicit_rollin_indices=(0, 1),
        coverage_object_indices=(0, 1),
        supervised_object_indices=(),
        supervise_stop=True,
    )
    assert result.ok
    assert result.target_ir.metadata["target_kind"] == "assistant_stop"
    assert result.target_ir.metadata["supervised_normalized_object_indices"] == ()
    assert result.target_ir.atoms
    assert result.target_ir.metadata["target_object_indices_by_atom"] == {}
    assert result.target_ir.metadata["stop_token_supervised"] is True
    assert all("object_index" not in atom.provenance for atom in result.target_ir.atoms)
    assert {atom.provenance.get("target_kind") for atom in result.target_ir.atoms} == {"assistant_stop"}


def test_explicit_empty_row_state_has_empty_prefix_and_first_target(tokenizer, two_object_scene):
    result = build_teacher_forcing_target(
        two_object_scene,
        tokenizer=tokenizer,
        profile="hard_sft",
        epoch=0,
        stable_sample_id="row-state-0",
        explicit_rollin_indices=(0,),
        coverage_object_indices=(),
        supervised_object_indices=(0,),
        supervise_stop=False,
    )
    assert result.ok
    assert result.target_ir.metadata["rendered_teacher_forcing_indices"] == (0,)
    assert result.target_ir.metadata["coverage_object_indices"] == ()
    assert result.target_ir.metadata["supervised_normalized_object_indices"] == (0,)


def test_shared_prefix_renderer_matches_teacher_forcing_prefix_tokens(tokenizer, two_object_scene):
    from src.detection.coverage.prefix_rendering import render_row_state_prefix

    result = build_teacher_forcing_target(
        two_object_scene,
        tokenizer=tokenizer,
        profile="hard_sft",
        epoch=0,
        stable_sample_id="row-state",
        explicit_rollin_indices=(0, 1),
        coverage_object_indices=(0,),
        supervised_object_indices=(1,),
        supervise_stop=False,
    )
    prefix = render_row_state_prefix(
        two_object_scene,
        prefix_object_indices=(0,),
        tokenizer=tokenizer,
        input_prefix_token_id=result.input_ids[0],
    )
    first_target_token_offset = result.target_ir.metadata["first_supervised_token_offset"]
    assert tuple(prefix.input_ids) == tuple(result.input_ids[:first_target_token_offset])
```

- [ ] **Step 2: Run target-builder tests to verify RED**

Run:

```bash
python -m pytest tests/test_teacher_forcing_target_builder.py::test_explicit_row_state_supervises_only_next_object_tokens tests/test_teacher_forcing_target_builder.py::test_explicit_terminal_state_supervises_stop_only tests/test_teacher_forcing_target_builder.py::test_explicit_empty_row_state_has_empty_prefix_and_first_target tests/test_teacher_forcing_target_builder.py::test_shared_prefix_renderer_matches_teacher_forcing_prefix_tokens -q
```

Expected before implementation: FAIL with `TypeError` for unexpected keyword
argument `explicit_rollin_indices`, or with the leakage assertion if explicit
mode already exists but includes the target in the prefix.

- [ ] **Step 3: Add optional explicit targeting parameters**

Extend `build_teacher_forcing_target` and `TeacherForcingTargetBuilder.build`
with keyword-only optional parameters:

```python
explicit_rollin_indices: Sequence[int] | None = None
coverage_object_indices: Sequence[int] | None = None
supervised_object_indices: Sequence[int] | None = None
supervise_stop: bool | None = None
```

Default `None` preserves current random-permutation behavior exactly.

When explicit mode is used:

- validate indices are in range;
- use `explicit_rollin_indices` for rendered teacher-forcing text order;
- use `coverage_object_indices` only for row-coverage metadata and visual
  descriptor painting;
- build atoms only for `supervised_object_indices`;
- include the stop atom only when `supervise_stop is True`;
- reject empty `supervised_object_indices` unless `supervise_stop is True`;
- for object-row states, permit the supervised target to appear in
  `explicit_rollin_indices` so shifted CE labels have real token spans, but
  reject the supervised target in `coverage_object_indices`;
- set metadata:

```python
"row_state_mode": "explicit"
"target_kind": "assistant_stop" or "object_row"
"supervised_normalized_object_indices": tuple(supervised_object_indices)
"rendered_teacher_forcing_indices": tuple(explicit_rollin_indices)
"coverage_object_indices": tuple(coverage_object_indices)
"first_supervised_token_offset": int
"target_object_indices_by_atom": dict[int, int]
"stop_token_supervised": bool
```

Extract shared row-state serialization into
`src/detection/coverage/prefix_rendering.py`. It must call the same
object-row rendering path as `TeacherForcingTargetBuilder`, not duplicate a
free-form string template. It should expose:

```python
def render_row_state_prefix(
    sample: DetectionScene | NormalizedDetectionSample | Mapping[str, Any],
    *,
    prefix_object_indices: Sequence[int],
    tokenizer: Any,
    input_prefix_token_id: int | None = None,
) -> RowStatePrefixRenderResult:
    ...
```

`TeacherForcingTargetBuilder` must use this shared helper for explicit
row-state mode, and rollout must use the same helper for row-boundary re-prefill.
The parity invariant is: for a fixed `S_k`, rollout prefill tokens equal the
teacher-forcing input prefix up to the first supervised target token, and both
paths paint identical descriptors from `coverage_object_indices`.

- [ ] **Step 4: Write row-state dataset tests**

Add:

```python
from src.detection.coverage.row_state_dataset import RowCoverageTrainingDataset


def test_row_coverage_dataset_expands_n_objects_to_n_plus_one_states(base_detection_dataset):
    dataset = RowCoverageTrainingDataset(base_detection_dataset)
    scene = base_detection_dataset.scene_for_row(0)
    assert len(dataset) == len(scene.objects) + 1
    first = dataset[0]
    last = dataset[len(scene.objects)]
    assert first["row_coverage_state"].row_state_k == 0
    assert first["row_coverage_state"].prefix_object_indices == ()
    assert first["row_coverage_state"].coverage_object_indices == ()
    assert first["row_coverage_state"].rendered_teacher_forcing_indices == (0,)
    assert first["row_coverage_state"].supervised_object_indices == (0,)
    assert first["row_coverage_state"].target_kind == "object_row"
    assert last["row_coverage_state"].row_state_k == len(scene.objects)
    assert last["row_coverage_state"].target_kind == "assistant_stop"
    assert last["row_coverage_state"].supervised_object_indices == ()


def test_row_coverage_state_does_not_include_target_box(base_detection_dataset):
    dataset = RowCoverageTrainingDataset(base_detection_dataset)
    sample = dataset[1]
    state = sample["row_coverage_state"]
    assert state.target_object_index == 1
    assert state.prefix_object_indices == (0,)
    assert state.rendered_teacher_forcing_indices == (0, 1)
    assert state.coverage_object_indices == (0,)
    assert state.supervised_object_indices == (1,)
    assert len(state.coverage_boxes) == 1
    assert state.target_object_index not in state.prefix_object_indices
    assert state.target_object_index not in state.coverage_object_indices
    assert state.target_object_index in state.rendered_teacher_forcing_indices


def test_row_coverage_dataset_records_ordering_strategy_and_seed(base_detection_dataset):
    dataset = RowCoverageTrainingDataset(base_detection_dataset)
    sample = dataset[1]
    state = sample["row_coverage_state"]
    assert state.ordering_strategy in {"random_permutation", "sorted"}
    assert "row_state" in sample["sample_id"]
```

- [ ] **Step 5: Run row-state dataset tests to verify RED**

Run:

```bash
python -m pytest tests/detection/coverage/test_row_state_dataset.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError` for
`src.detection.coverage.row_state_dataset`.

- [ ] **Step 6: Add row-state dataset wrapper**

Implement `RowCoverageTrainingDataset` so it:

- wraps a `DetectionTrainingDataset`;
- builds a flat index of `(base_idx, row_state_k)` where `k in [0, object_count]`;
- forwards `set_epoch(epoch)` to the base dataset and rebuilds the flat index;
- calls a new base dataset method that encodes an explicit row-state sample;
- adds `row_coverage_state` to the returned encoded row;
- preserves `sample_id`, `dataset`, `base_idx`, and detection metadata.

Add `DetectionTrainingDataset.encode_teacher_forcing_row_state` with signature:

```python
def encode_teacher_forcing_row_state(
    self,
    *,
    base_idx: int,
    row_state_k: int,
) -> dict[str, Any]:
    return self._encode_teacher_forcing_row_state_impl(
        base_idx=base_idx,
        row_state_k=row_state_k,
    )
```

It must keep two different notions separate:

```text
prefix_object_indices = tuple(range(row_state_k)) for object-row states
coverage_object_indices = tuple(range(row_state_k)) for object-row states
rendered_teacher_forcing_indices = tuple(range(row_state_k + 1)) for object-row states
supervised_object_indices = (row_state_k,) for object-row states

prefix_object_indices = tuple(range(object_count)) for terminal state
coverage_object_indices = tuple(range(object_count)) for terminal state
rendered_teacher_forcing_indices = tuple(range(object_count)) for terminal state
supervised_object_indices = () for terminal state
supervise_stop = row_state_k == object_count
```

The current target object must not appear in `prefix_object_indices` or
`coverage_object_indices` for an object-row state. It may appear in
`rendered_teacher_forcing_indices`, because the current target-builder path
supervises shifted CE positions inside rendered assistant text. The terminal
state is the only state whose prefix and coverage contain all objects.

- [ ] **Step 7: Run target-builder and row-state tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_target_builder.py tests/detection/coverage/test_row_state_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/detection/teacher_forcing/target_builder.py src/detection/dataset.py src/detection/coverage/prefix_rendering.py src/detection/coverage/row_state_dataset.py tests/test_teacher_forcing_target_builder.py tests/detection/coverage/test_row_state_dataset.py
git commit -m "Add row-state teacher-forcing dataset"
```

---

### Task 4: Carry Coverage State Through Collation And Input Boundaries

**Files:**
- Modify: `src/trainers/batch_extras.py`
- Modify: `src/data_collators/enrichers.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Modify: `src/detection/dataset.py`
- Modify: `src/training/encoding/model_inputs.py`
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`

- [ ] **Step 1: Write sidecar tests**

Add:

```python
import torch

from src.data_collators.enrichers import RowCoverageStateEnricher
from src.training.encoding.model_inputs import classify_backend_key


def test_row_coverage_state_is_bridge_consumed_key():
    assert classify_backend_key("row_coverage_state") == "bridge_consumed"


def test_row_coverage_state_enricher_rejects_packing(row_coverage_state):
    collated = {}
    batch = [[{"row_coverage_state": row_coverage_state}]]
    enricher = RowCoverageStateEnricher()
    with pytest.raises(ValueError, match="row_coverage_state sidecars are incompatible with packing"):
        enricher(collated=collated, raw_batch=batch, packed=True)


def test_row_coverage_state_enricher_attaches_unpacked_tuple(row_coverage_state):
    collated = {}
    batch = [{"row_coverage_state": row_coverage_state}]
    enricher = RowCoverageStateEnricher()
    enricher(collated=collated, raw_batch=batch, packed=False)
    assert collated["row_coverage_state"] == (row_coverage_state,)


def test_row_coverage_state_survives_pop_batch_extras(row_coverage_state):
    from src.trainers.batch_extras import pop_batch_extras

    inputs = {"row_coverage_state": (row_coverage_state,), "input_ids": object()}
    extras = pop_batch_extras(inputs)
    assert extras.row_coverage_state == (row_coverage_state,)
    assert "row_coverage_state" not in inputs
    assert "input_ids" in inputs


def test_row_coverage_state_not_forwarded_to_model(row_coverage_state):
    from src.training.encoding.model_inputs import ModelInputBundle

    bundle = ModelInputBundle.from_mapping(
        {"input_ids": object(), "row_coverage_state": (row_coverage_state,)},
        runner_owns_loss=True,
    )
    assert "row_coverage_state" in bundle.bridge_auxiliaries()
    assert "row_coverage_state" not in bundle.forwarded_inputs()


def test_bridge_accepts_row_coverage_state_from_batch_extras(row_coverage_state):
    extras = BatchExtras(row_coverage_state=(row_coverage_state,))
    bridge = make_loss_bridge(batch_extras=extras)
    assert bridge.resolve_row_coverage_state({}) == (row_coverage_state,)


def test_bridge_rejects_conflicting_row_coverage_sidecars(row_coverage_state, other_row_coverage_state):
    extras = BatchExtras(row_coverage_state=(row_coverage_state,))
    bridge = make_loss_bridge(batch_extras=extras)
    with pytest.raises(ValueError, match="conflicting row_coverage_state"):
        bridge.resolve_row_coverage_state({"row_coverage_state": (other_row_coverage_state,)})


def test_bridge_deduplicates_identical_row_coverage_sidecars(row_coverage_state):
    extras = BatchExtras(row_coverage_state=(row_coverage_state,))
    bridge = make_loss_bridge(batch_extras=extras)
    assert bridge.resolve_row_coverage_state(
        {"row_coverage_state": (row_coverage_state,)}
    ) == (row_coverage_state,)


def test_teacher_forcing_mixin_delivers_row_coverage_state_to_bridge(spy_teacher_forcing_trainer, row_coverage_state):
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": torch.tensor([[1, 2]]),
        "row_coverage_state": (row_coverage_state,),
    }
    spy_teacher_forcing_trainer.compute_loss(model=spy_teacher_forcing_trainer.model, inputs=batch)
    assert spy_teacher_forcing_trainer.bridge.seen_row_coverage_state == (row_coverage_state,)
```

- [ ] **Step 2: Run sidecar tests to verify RED**

Run:

```bash
python -m pytest tests/test_teacher_forcing_sidecar_bridge.py -k row_coverage_state -q
```

Expected before implementation: FAIL because `RowCoverageStateEnricher`,
`row_coverage_state` classification, or `BatchExtras.row_coverage_state` is
missing.

- [ ] **Step 3: Add batch-extra field**

In `src/trainers/batch_extras.py`, add:

```python
ROW_COVERAGE_STATE_KEY = "row_coverage_state"
```

and add `row_coverage_state: Any = None` to `BatchExtras`.

Also add `ROW_COVERAGE_STATE_KEY` to `BATCH_EXTRAS_KEYS`, ensure
`pop_batch_extras` removes and stores it, and ensure stashed extras preserve it
for every mixin that calls `maybe_pop_and_stash_batch_extras`.

- [ ] **Step 4: Add bridge-consumed key**

In `src/training/encoding/model_inputs.py`, add:

```python
"row_coverage_state",
```

to `BRIDGE_CONSUMED_MODEL_INPUT_KEYS`, not to forwarded keys. This keeps the
raw sidecar available to `TrainerLossBridge` while preventing accidental
forwarding into HF Qwen.

- [ ] **Step 5: Add collator enricher**

In `src/data_collators/enrichers.py`, add `RowCoverageStateEnricher` following
the unpacked-only pattern used by `TeacherForcingTargetIREnricher`.

In `src/data_collators/batch_extras_collator.py`, instantiate it and call it
after `TeacherForcingTargetIREnricher`.

Canonical ownership rule: after collation, `BatchExtras.row_coverage_state` is
the canonical source. `TrainerLossBridge` may accept an identical duplicate from
bridge auxiliaries for compatibility, must reject conflicting duplicates, and
must treat a missing state as the ordinary non-coverage path.

Add `row_coverage_state` to the sidecar stripping allowlist in
`src/detection/dataset.py` so it is never forwarded to HF model code. The
teacher-forcing mixin integration test must prove the sidecar survives
`maybe_pop_and_stash_batch_extras`, reaches `TrainerLossBridge`, and is absent
from model-forward kwargs.

- [ ] **Step 6: Run sidecar tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_sidecar_bridge.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/trainers/batch_extras.py src/data_collators/enrichers.py src/data_collators/batch_extras_collator.py src/detection/dataset.py src/training/encoding/model_inputs.py tests/test_teacher_forcing_sidecar_bridge.py
git commit -m "Carry row coverage state through trainer sidecars"
```

---

### Task 5: Add Scalar-Gated Visual Feature Tuning Helper

**Files:**
- Create: `src/detection/coverage/forward.py`
- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/sft.py`
- Test: `tests/detection/coverage/test_forward.py`
- Test: `tests/test_teacher_forcing_sidecar_bridge.py`

- [ ] **Step 1: Write forward-helper tests with a fake Qwen model**

Add:

```python
import pytest
import torch

from src.detection.coverage.forward import (
    RowCoverageFeatureTuner,
    RowCoverageForwardContext,
    resolve_qwen_visual_owner,
    resolve_visual_merge_size,
    run_row_coverage_forward,
)
from src.detection.coverage.types import CoverageDescriptor


class FakeLanguageModel:
    def __init__(self):
        self.calls = []

    def __call__(self, *, image_embeds, deepstack_visual_embeds=None, **kwargs):
        self.calls.append(
            {
                "image_embeds": image_embeds,
                "deepstack_visual_embeds": deepstack_visual_embeds,
                "kwargs": dict(kwargs),
            }
        )
        logits = image_embeds.sum().reshape(1, 1, 1)
        if deepstack_visual_embeds:
            logits = logits + sum(item.sum() for item in deepstack_visual_embeds).reshape(1, 1, 1)
        return type("Output", (), {"logits": logits})()


class FakeInnerQwen:
    def __init__(self, *, deepstack_active: bool = False):
        self.called = 0
        self.forward_kwargs = []
        self.visual = type("Visual", (), {"spatial_merge_size": 2})()
        self.language_model = FakeLanguageModel()
        self.deepstack_active = deepstack_active
        self.last_split_sizes = []

    def get_image_features(self, pixel_values, image_grid_thw):
        self.called += 1
        split_sizes = (
            image_grid_thw.prod(dim=-1) // (self.visual.spatial_merge_size**2)
        ).tolist()
        self.last_split_sizes = [int(size) for size in split_sizes]
        total = int(sum(split_sizes))
        image_embeds = torch.ones((total, 8), dtype=torch.float32)
        image_splits = tuple(torch.split(image_embeds, split_sizes, dim=0))
        if not self.deepstack_active:
            return image_splits, []
        deepstack = torch.full((total, 8), 2.0, dtype=torch.float32)
        deepstack_splits = [tuple(torch.split(deepstack, split_sizes, dim=0))]
        return image_splits, deepstack_splits

    def __call__(self, **kwargs):
        self.forward_kwargs.append(dict(kwargs))
        image_embeds, deepstack = self.get_image_features(kwargs["pixel_values"], kwargs["image_grid_thw"])
        image_embeds = torch.cat(image_embeds, dim=0)
        if deepstack:
            deepstack = [torch.cat(level, dim=0) for level in deepstack]
        return self.language_model(
            image_embeds=image_embeds,
            deepstack_visual_embeds=deepstack,
            **kwargs,
        )


class FakeQwenForConditionalGeneration:
    def __init__(self, *, deepstack_active: bool = False):
        self.config = type("Config", (), {"model_type": "qwen3_vl"})()
        self.model = FakeInnerQwen(deepstack_active=deepstack_active)

    def __call__(self, **kwargs):
        return self.model(**kwargs)


def test_empty_zero_alpha_tuning_is_identity():
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.0, interior_alpha_init=0.0)
    features = torch.randn(4, 8)
    desc = CoverageDescriptor(boundary_mass=torch.zeros(4), interior_mass=torch.zeros(4))
    tuned = tuner(features, desc)
    assert torch.equal(tuned, features)


def test_forward_helper_replaces_image_feature_path_with_tuned_features():
    model = FakeQwenForConditionalGeneration()
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    with torch.no_grad():
        tuner.boundary_embedding.fill_(3.0)
    desc = CoverageDescriptor(boundary_mass=torch.ones(4), interior_mass=torch.zeros(4))
    outputs = run_row_coverage_forward(
        model=model,
        inputs_for_model={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "image_grid_thw": torch.tensor([1, 4, 4]),
        },
        tuner=tuner,
        descriptor=desc,
    )
    assert model.model.called == 1
    call = model.model.language_model.calls[-1]
    assert torch.all(call["image_embeds"] > 1.0)
    assert float(outputs.logits.item()) > 32.0
    assert outputs.logits.shape == (1, 1, 1)


def test_resolves_inner_qwen_visual_owner():
    model = FakeQwenForConditionalGeneration()
    assert resolve_qwen_visual_owner(model) is model.model


def test_resolve_visual_merge_size_prefers_active_qwen_owner():
    model = FakeQwenForConditionalGeneration()
    processor = type("Processor", (), {"image_processor": type("ImageProcessor", (), {"merge_size": 1})()})()
    assert resolve_visual_merge_size(model, processor=processor) == 2


def test_forward_helper_preserves_qwen_image_feature_return_contract():
    model = FakeQwenForConditionalGeneration()
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    desc = CoverageDescriptor(boundary_mass=torch.ones(4), interior_mass=torch.zeros(4))
    outputs = run_row_coverage_forward(
        model=model,
        inputs_for_model={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "image_grid_thw": torch.tensor([1, 4, 4]),
        },
        tuner=tuner,
        descriptor=desc,
    )
    call = model.model.language_model.calls[-1]
    assert call["image_embeds"].shape == (4, 8)
    assert call["deepstack_visual_embeds"] == []
    assert outputs.logits.shape == (1, 1, 1)


def test_forward_helper_preserves_qwen_multi_image_split_contract():
    model = FakeQwenForConditionalGeneration()
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    with torch.no_grad():
        tuner.boundary_embedding.fill_(3.0)
    desc = CoverageDescriptor(boundary_mass=torch.ones(5), interior_mass=torch.zeros(5))
    outputs = run_row_coverage_forward(
        model=model,
        inputs_for_model={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((2, 3, 16, 16)),
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]]),
        },
        tuner=tuner,
        descriptor=desc,
    )
    call = model.model.language_model.calls[-1]
    assert model.model.last_split_sizes == [4, 1]
    assert call["image_embeds"].shape == (5, 8)
    assert torch.all(call["image_embeds"][:4] > 1.0)
    assert torch.all(call["image_embeds"][4:] > 1.0)
    assert outputs.logits.shape == (1, 1, 1)


def test_forward_helper_rejects_multi_image_descriptor_length_mismatch():
    model = FakeQwenForConditionalGeneration()
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    desc = CoverageDescriptor(boundary_mass=torch.ones(4), interior_mass=torch.zeros(4))
    with pytest.raises(ValueError, match="descriptor length"):
        run_row_coverage_forward(
            model=model,
            inputs_for_model={
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.zeros((2, 3, 16, 16)),
                "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]]),
            },
            tuner=tuner,
            descriptor=desc,
        )


def test_deepstack_policy_tunes_active_deepstack():
    model = FakeQwenForConditionalGeneration(deepstack_active=True)
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    with torch.no_grad():
        tuner.boundary_embedding.fill_(3.0)
    desc = CoverageDescriptor(boundary_mass=torch.ones(4), interior_mass=torch.zeros(4))
    result = run_row_coverage_forward(
        model=model,
        inputs_for_model={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "image_grid_thw": torch.tensor([1, 4, 4]),
        },
        tuner=tuner,
        descriptor=desc,
        deepstack_policy="tune_or_raise",
    )
    call = model.model.language_model.calls[-1]
    assert result.logits.shape == (1, 1, 1)
    assert call["deepstack_visual_embeds"][0].shape == (4, 8)
    assert not torch.equal(
        call["deepstack_visual_embeds"][0],
        torch.full((4, 8), 2.0, dtype=torch.float32),
    )


def test_forward_helper_restores_visual_method_after_exception():
    model = FakeQwenForConditionalGeneration()
    original = model.model.get_image_features
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    desc = CoverageDescriptor(boundary_mass=torch.ones(3), interior_mass=torch.zeros(3))
    with pytest.raises(ValueError, match="descriptor length"):
        run_row_coverage_forward(
            model=model,
            inputs_for_model={
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.zeros((1, 3, 16, 16)),
                "image_grid_thw": torch.tensor([1, 4, 4]),
            },
            tuner=tuner,
            descriptor=desc,
        )
    assert model.model.get_image_features is original


def test_coverage_forward_removes_stale_cache_and_disables_prefill_cache():
    model = FakeQwenForConditionalGeneration()
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.0, interior_alpha_init=0.0)
    desc = CoverageDescriptor(boundary_mass=torch.zeros(4), interior_mass=torch.zeros(4))
    outputs = run_row_coverage_forward(
        model=model,
        inputs_for_model={
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "image_grid_thw": torch.tensor([1, 4, 4]),
            "past_key_values": object(),
            "use_cache": True,
        },
        tuner=tuner,
        descriptor=desc,
    )
    assert getattr(outputs, "logits") is not None
    assert model.model.forward_kwargs
    call_kwargs = model.model.forward_kwargs[-1]
    assert "past_key_values" not in call_kwargs
    assert call_kwargs["use_cache"] is False


def test_forward_context_records_tuner_state_hash_and_source():
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    assert context.context_source == "attached_model"
    assert context.tuner_state_hash
    assert context.tuner_state_hash == context.compute_tuner_state_hash()


def test_row_coverage_context_is_registered_before_trainer_parameters(fake_train_model, row_coverage_config):
    from src.sft import install_row_coverage_forward_context

    context = install_row_coverage_forward_context(fake_train_model, config=row_coverage_config)
    named = dict(fake_train_model.named_parameters())
    assert context.context_source == "attached_model"
    assert any(name.startswith("_coordexp_row_coverage_forward_context.tuner") for name in named)


def test_row_coverage_tuner_checkpoint_roundtrip_restores_hash(tmp_path):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    with torch.no_grad():
        context.tuner.boundary_embedding.fill_(7.0)
    expected_hash = context.compute_tuner_state_hash()
    path = tmp_path / "row_coverage_tuner.pt"
    context.save_tuner_state(path)

    loaded = RowCoverageForwardContext.load_tuner_state(
        path,
        hidden_size=8,
        expected_hash=expected_hash,
    )
    assert loaded.context_source == "loaded_checkpoint"
    assert loaded.compute_tuner_state_hash() == expected_hash


def test_tuner_reports_large_delta_norm_when_embeddings_are_large():
    tuner = RowCoverageFeatureTuner(hidden_size=8, boundary_alpha_init=0.1, interior_alpha_init=0.0)
    with torch.no_grad():
        tuner.boundary_embedding.fill_(1000.0)
    features = torch.ones((4, 8))
    desc = CoverageDescriptor(boundary_mass=torch.ones(4), interior_mass=torch.zeros(4))
    tuned, stats = tuner(features, desc, return_stats=True)
    assert tuned.shape == features.shape
    assert stats["delta_norm"] > stats["feature_norm"]
```

- [ ] **Step 2: Run forward-helper tests to verify RED**

Run:

```bash
python -m pytest tests/detection/coverage/test_forward.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError` for
`src.detection.coverage.forward` or missing `RowCoverageForwardContext`.

- [ ] **Step 3: Implement additive residual tuner and forward context**

`RowCoverageFeatureTuner` must:

- subclass `torch.nn.Module`;
- create `boundary_embedding` and `interior_embedding` with `hidden_size`;
- create trainable scalar parameters for alpha gates;
- bound gates into `[0, max_alpha]`;
- require descriptor length equals feature length;
- return `features + delta`;
- optionally return diagnostic stats with `feature_norm`, `delta_norm`,
  `delta_to_feature_norm_ratio`, `boundary_alpha`, and `interior_alpha`.

Use:

```python
bounded = max_alpha * torch.sigmoid(raw_alpha)
```

Set `raw_alpha` so an init of `0.0` produces an effective alpha of exactly
`0.0` by special-casing zero init with a stored disabled scalar until training
sets a nonzero init, or use direct clamped parameters:

```python
alpha = torch.clamp(self.boundary_alpha, min=0.0, max=self.boundary_alpha_max)
```

The direct clamp is sufficient for v1 and preserves exact identity at init.

Add `RowCoverageForwardContext` to own:

```text
coverage config
painter
merge-size resolver
tuner module
tuner state hash
context source: attached_model | loaded_checkpoint | fresh_test_only
no-cache policy
deepstack policy
```

It must expose one tuner object reused by training and rollout, or a state hash
when rollout loads the trained tuner from a checkpoint.

Do not claim that scalar alpha bounds imply a fully bounded residual norm. The
v1 guarantee is scalar-gated additive tuning. The helper must report delta norms
so capacity-control reports can distinguish coverage structure from an overly
large learned visual-token residual.

- [ ] **Step 4: Implement local forward wrapper**

`run_row_coverage_forward` must:

- compute or accept original image features;
- tune features with the same `RowCoverageFeatureTuner`;
- resolve the actual Qwen visual owner, such as `model.model`, before patching;
- resolve the active visual merge size with `resolve_visual_merge_size(model,
  processor=None)`, preferring the resolved owner's
  `visual.spatial_merge_size`, then `model.visual.spatial_merge_size`, then
  `processor.image_processor.merge_size`;
- use a context-managed replacement of the resolved owner's
  `get_image_features` so HF forward receives tuned features without editing
  upstream files;
- preserve Qwen-style `get_image_features` return contracts, including
  per-image split tuples from `torch.split(image_embeds, split_sizes)`, where
  `split_sizes = image_grid_thw.prod(-1) // spatial_merge_size**2`;
- tune descriptors in Qwen split order, then return split tuned features with
  the same tuple boundaries so the real Qwen forward can `torch.cat` them;
- restore the original method in `finally`;
- reject models without `get_image_features`;
- reject feature/descriptor length mismatches.
- remove `past_key_values` and set `use_cache=False` for coverage-active
  prefill/teacher-forcing forwards.
- tune active Qwen deepstack visual embeddings with the same coverage descriptor
  when their length matches the post-merge lattice; reject only when their shape
  cannot be matched. Silent mismatched deepstack state is forbidden.

- [ ] **Step 5: Install the trainable tuner before trainer construction**

In `src/sft.py`, after model preparation and adapter wrapping are complete but
before `SFTTrainer` or optimizer construction, install a
`RowCoverageForwardContext` on the trainable model when
`training_config.row_conditioned_visual_coverage.enabled` is true:

```python
model._coordexp_row_coverage_forward_context = RowCoverageForwardContext.from_model(
    model,
    config=training_config.row_conditioned_visual_coverage,
)
```

The context owns the trainable `RowCoverageFeatureTuner`. Do not create trainable
parameters lazily inside `TrainerLossBridge.compute_loss`, because such
parameters may be absent from optimizer, DDP, and checkpoint discovery.

Add a pre-trainer assertion/test that at least one
`_coordexp_row_coverage_forward_context` tuner parameter appears in
`dict(model.named_parameters())` before trainer construction.

The installed context or its tuner must be a registered `torch.nn.Module`.
Training must either attach it before PEFT wrapping and include it in
`modules_to_save`, or write a CoordExp row-coverage checkpoint artifact with
save/load helpers. Add a round-trip test that mutates tuner parameters, saves
them, reloads them for rollout, and verifies the recorded tuner hash matches.

- [ ] **Step 6: Integrate with TrainerLossBridge**

In `TrainerLossBridge.compute_loss`:

- read `row_coverage_state` from bridge auxiliaries or batch extras;
- when absent, keep current path exactly unchanged;
- when present, paint descriptors from sidecars and `image_grid_thw`;
- resolve the already-attached `RowCoverageForwardContext` from the core model
  under the stable local name `_coordexp_row_coverage_forward_context`;
- fail fast if coverage state is present but the context has not been attached;
- call `run_row_coverage_forward` through that context instead of
  `core_model(**inputs_for_model)`;
- validate full logits as before.

- [ ] **Step 7: Pass sidecar through teacher-forcing mixin**

In `TeacherForcingObjectiveMixin.compute_loss`, do not strip
`row_coverage_state` before `TrainerLossBridge` consumes it. Keep
`strip_non_model_detection_sidecars` after `maybe_pop_and_stash_batch_extras`,
and rely on `ModelInputBundle` to classify the bridge key.

- [ ] **Step 8: Run forward and bridge tests**

Run:

```bash
python -m pytest tests/detection/coverage/test_forward.py tests/test_teacher_forcing_sidecar_bridge.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/detection/coverage/forward.py src/training/bridge/loss_bridge.py src/trainers/metrics/teacher_forcing.py src/sft.py tests/detection/coverage/test_forward.py tests/test_teacher_forcing_sidecar_bridge.py
git commit -m "Apply row coverage residuals in model forward"
```

---

### Task 6: Enable The Training Surface And Smoke Configs

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: `src/sft.py`
- Create: `configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml`
- Create: `configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml`
- Test: `tests/test_teacher_forcing_config_contract.py`

- [ ] **Step 1: Write config contract tests**

Add tests that resolve the two smoke configs and assert:

```python
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig


def _resolve_detection_config(path: str) -> DetectionTrainingConfig:
    return DetectionTrainingConfig.from_mapping(ConfigLoader.load_yaml_with_extends(path))


def test_row_coverage_smoke_configs_are_unpacked_and_teacher_forcing():
    expected_orderings = {
        "configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml": "random_permutation",
        "configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml": "sorted",
    }
    for path, ordering in expected_orderings.items():
        resolved_config = _resolve_detection_config(path)
        assert resolved_config.row_conditioned_visual_coverage.enabled is True
        assert resolved_config.row_conditioned_visual_coverage.training.ordering_modes == ("random_sft", "sorted_sft")
        assert resolved_config.objective.id == "teacher_forcing"
        assert resolved_config.objective.profile == "hard_sft"
        assert resolved_config.training["packing"] is False
        assert resolved_config.training["group_by_length"] is False
        assert resolved_config.packing.static_packing is False
        assert resolved_config.detection_template.id == "compact_full"
        assert resolved_config.data.object_ordering == ordering
```

Also assert:

```python
def test_row_coverage_rejects_packing_enabled():
    with pytest.raises(ValueError, match="row-conditioned visual coverage requires training.packing=false"):
        raw = ConfigLoader.load_yaml_with_extends(
            "configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml"
        )
        raw["training"]["packing"] = True
        DetectionTrainingConfig.from_mapping(raw)


def test_row_coverage_rejects_group_by_length_enabled():
    with pytest.raises(ValueError, match="training.group_by_length=false"):
        raw = ConfigLoader.load_yaml_with_extends(
            "configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml"
        )
        raw["training"]["group_by_length"] = True
        DetectionTrainingConfig.from_mapping(raw)


def test_row_coverage_runtime_wraps_detection_dataset(tmp_path, fake_swift_template):
    from src.detection.coverage.row_state_dataset import RowCoverageTrainingDataset
    from src.detection.runtime import (
        build_detection_dataset,
        build_detection_runtime_custom_shim,
        resolve_detection_prompts,
    )

    raw = ConfigLoader.load_yaml_with_extends(
        "configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml"
    )
    jsonl_path = write_tiny_detection_jsonl(tmp_path)
    raw["data"]["train_jsonl"] = str(jsonl_path)
    raw["data"]["image_root"] = str(tmp_path)
    cfg = DetectionTrainingConfig.from_mapping(raw)
    system_prompt, _ = resolve_detection_prompts(cfg)
    custom_config = build_detection_runtime_custom_shim(cfg)
    dataset = build_detection_dataset(
        jsonl_path,
        swift_template=fake_swift_template,
        training_config=cfg,
        custom_config=custom_config,
        system_prompt=system_prompt,
        seed=0,
        sample_limit=None,
        dataset_name="train",
    )
    assert isinstance(dataset, RowCoverageTrainingDataset)
    assert "row_coverage_state" in dataset[0]


def test_row_coverage_length_bucketing_preflight_stays_disabled():
    raw = ConfigLoader.load_yaml_with_extends(
        "configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml"
    )
    cfg = DetectionTrainingConfig.from_mapping(raw)
    assert cfg.row_conditioned_visual_coverage.enabled is True
    assert cfg.training["group_by_length"] is False
```

- [ ] **Step 2: Run config contract tests to verify RED**

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py::test_row_coverage_smoke_configs_are_unpacked_and_teacher_forcing tests/test_teacher_forcing_config_contract.py::test_row_coverage_rejects_packing_enabled tests/test_teacher_forcing_config_contract.py::test_row_coverage_rejects_group_by_length_enabled tests/test_teacher_forcing_config_contract.py::test_row_coverage_runtime_wraps_detection_dataset tests/test_teacher_forcing_config_contract.py::test_row_coverage_effective_runtime_disables_length_bucketing -q
```

Expected before implementation: FAIL because the smoke config files or row
coverage validation are missing.

- [ ] **Step 3: Wrap detection dataset when coverage is enabled**

In the dataset construction route:

- read `training_config.row_conditioned_visual_coverage`;
- if disabled, return existing dataset unchanged;
- if enabled, require `objective.id == "teacher_forcing"`;
- require `training["packing"] is False`;
- require `training["group_by_length"] is False` for the v1 prototype, because
  the row-state wrapper is not yet exposed to the current length-bucketing
  `DetectionTrainingDataset` `isinstance` path;
- require `packing.static_packing is False`;
- wrap the base `DetectionTrainingDataset` in `RowCoverageTrainingDataset`.

Use a narrow helper:

```python
def maybe_wrap_row_coverage_dataset(
    dataset: DetectionTrainingDataset,
    *,
    custom_config: Any,
    training_config: DetectionTrainingConfig,
) -> Any:
    cfg = training_config.row_conditioned_visual_coverage
    if not cfg.enabled:
        return dataset
    if getattr(training_config.objective, "id", None) != "teacher_forcing":
        raise ValueError("row-conditioned visual coverage requires objective.id=teacher_forcing")
    if bool(training_config.training.get("packing", False)):
        raise ValueError("row-conditioned visual coverage requires training.packing=false")
    if bool(training_config.training.get("group_by_length", False)):
        raise ValueError("row-conditioned visual coverage v1 requires training.group_by_length=false")
    if training_config.packing.static_packing:
        raise ValueError("row-conditioned visual coverage requires packing.static_packing=false")
    return RowCoverageTrainingDataset(dataset)
```

Also update `src/detection/runtime.py` so teacher-forcing mode supports both
`data.object_ordering: random_permutation` and `data.object_ordering: sorted`
when row coverage is enabled. Keep the existing target-IR roll-in policy
validation explicit: the rendered target-builder order is controlled by the
row-state dataset, while `data.object_ordering` selects the row-state ablation
strategy.

- [ ] **Step 4: Add smoke configs**

Both configs must inherit:

```yaml
extends: ../smoke/compact_full_tiny.yaml
```

The random smoke config must set:

```yaml
data:
  object_ordering: random_permutation
row_conditioned_visual_coverage:
  enabled: true
  version: v1
  row_state_policy: prefix_expansion
  tune_operator: additive_residual
  training:
    unpacked_only: true
    include_terminal_state: true
    ordering_modes: [random_sft, sorted_sft]
training:
  packing: false
  group_by_length: false
```

The sorted smoke config must set:

```yaml
data:
  object_ordering: sorted
row_conditioned_visual_coverage:
  enabled: true
  version: v1
  row_state_policy: prefix_expansion
  tune_operator: additive_residual
  training:
    unpacked_only: true
    include_terminal_state: true
    ordering_modes: [random_sft, sorted_sft]
training:
  packing: false
  group_by_length: false
```

- [ ] **Step 5: Run config contract tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run a tiny config materialization check**

Run:

```bash
PYTHONPATH=. python - <<'PY'
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig

for path in [
    "configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml",
    "configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml",
]:
    raw = ConfigLoader.load_yaml_with_extends(path)
    cfg = DetectionTrainingConfig.from_mapping(raw)
    assert cfg.row_conditioned_visual_coverage.enabled is True
    assert cfg.training["packing"] is False
    assert cfg.training["group_by_length"] is False
    print(path, cfg.data.object_ordering)
PY
```

Expected stdout includes both config paths and their distinct orderings:

```text
row_coverage_random_sft_smoke.yaml random_permutation
row_coverage_sorted_sft_smoke.yaml sorted
```

- [ ] **Step 7: Commit**

```bash
git add src/config/schema.py src/detection/runtime.py src/sft.py configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml tests/test_teacher_forcing_config_contract.py
git commit -m "Enable row coverage teacher-forcing surface"
```

---

### Task 7: Add Faithful Row-Boundary Re-Prefill Rollout

**Files:**
- Create: `src/detection/coverage/rollout.py`
- Create: `src/detection/coverage/artifacts.py`
- Create: `configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml`
- Test: `tests/detection/coverage/test_rollout.py`

- [ ] **Step 1: Write rollout tests with fake model and parser**

Add:

```python
import pytest

from src.detection.coverage.forward import RowCoverageForwardContext
from src.detection.coverage.rollout import run_row_boundary_reprefill


def test_rollout_updates_coverage_after_valid_row(fake_rollout_model, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        max_rows=3,
    )
    assert len(result.rows) == 2
    assert result.coverage_states[0].coverage_object_indices == ()
    assert result.coverage_states[1].coverage_object_indices == (0,)
    assert result.finish_reason == "assistant_stop"


def test_rollout_records_invalid_row_without_committing(fake_rollout_model_invalid, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model_invalid,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        max_rows=3,
    )
    assert result.rows == ()
    assert result.finish_reason == "parse_invalid"


def test_valid_then_invalid_row_preserves_prior_commits(fake_rollout_model_valid_then_invalid, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model_valid_then_invalid,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        max_rows=3,
    )
    assert len(result.rows) == 1
    assert result.finish_reason == "parse_invalid"


def test_no_coverage_reprefill_uses_zero_descriptor_and_same_commit_logic(fake_rollout_model, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        coverage_enabled=False,
        max_rows=3,
    )
    assert len(result.rows) == 2
    assert all(row["coverage_enabled"] is False for row in result.artifact_rows)
    assert all(row["descriptor_boundary_sum"] == 0.0 for row in result.artifact_rows)


def test_rollout_does_not_reuse_cross_row_cache(fake_rollout_model_cache_guard, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model_cache_guard,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        max_rows=3,
    )
    assert result.cache_reused_across_rows is False
    assert all(row["cache_reused_across_rows"] is False for row in result.artifact_rows)
    assert fake_rollout_model_cache_guard.row_prefill_calls
    assert all("past_key_values" not in call.kwargs for call in fake_rollout_model_cache_guard.row_prefill_calls)
    assert all(call.kwargs.get("use_cache") is False for call in fake_rollout_model_cache_guard.row_prefill_calls)


def test_rollout_records_encoder_call_policy(fake_rollout_model, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        max_rows=3,
    )
    assert result.image_encoder_calls == 1
    assert result.uses_precomputed_image_features is True
    assert result.reencode_policy == "precompute_once"
    assert result.claim_eligible is True
    assert result.context_source == "attached_model"
    assert result.tuner_state_hash == context.tuner_state_hash


def test_rollout_rejects_fresh_context_when_coverage_enabled(fake_rollout_model, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="fresh_test_only")
    with pytest.raises(ValueError, match="coverage rollout requires trained or loaded tuner"):
        run_row_boundary_reprefill(
            model=fake_rollout_model,
            scene=fake_scene,
            parser=fake_parser,
            forward_context=context,
            coverage_enabled=True,
            max_rows=3,
        )


def test_primary_rollout_downgrades_multi_encode_smoke(fake_rollout_model_multi_encode, fake_parser, fake_scene):
    context = RowCoverageForwardContext(hidden_size=8, context_source="attached_model")
    result = run_row_boundary_reprefill(
        model=fake_rollout_model_multi_encode,
        scene=fake_scene,
        parser=fake_parser,
        forward_context=context,
        control_mode="primary_coverage",
        evidence_scope="smoke-only runtime limitation",
        max_rows=3,
    )
    assert result.image_encoder_calls > 1
    assert result.claim_eligible is False
    assert result.evidence_scope == "smoke-only runtime limitation"
```

- [ ] **Step 2: Run rollout tests to verify RED**

Run:

```bash
python -m pytest tests/detection/coverage/test_rollout.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError` for
`src.detection.coverage.rollout` or missing `run_row_boundary_reprefill`.

- [ ] **Step 3: Implement rollout state dataclasses**

Add:

```python
@dataclass(frozen=True)
class RowCoverageRolloutResult:
    rows: Sequence[Any]
    coverage_states: Sequence[CoverageState]
    finish_reason: str
    decode_steps: int
    artifact_rows: Sequence[Mapping[str, Any]]
    cache_reused_across_rows: bool
    image_encoder_calls: int
    uses_precomputed_image_features: bool
    reencode_policy: str
    evidence_scope: str
    claim_eligible: bool
    tuner_state_hash: str
    context_source: str
```

- [ ] **Step 4: Implement re-prefill loop**

The loop must:

- require a `RowCoverageForwardContext` whose `context_source` is
  `attached_model` or `loaded_checkpoint` when `coverage_enabled=True`; reject
  `fresh_test_only` contexts outside explicit no-coverage tests;
- compute base image features once when the helper can use precomputed features;
- build coverage state from committed valid rows only;
- call the same coverage painter and forward tuner as training;
- re-prefill prompt plus committed row text at every row boundary;
- remove cross-row `past_key_values` before every row-boundary prefill;
- decode one next row or assistant stop;
- commit only parse-valid rows;
- stop on assistant stop, parse-invalid strict failure, max rows, or max tokens;
- record `coverage_object_indices`, descriptor stats, finish reason, and parse
  status for every attempted row.
- support `coverage_enabled=False` by running the same row-boundary re-prefill
  loop with zero descriptors and the same commit/parser logic;
- record `cache_reused_across_rows=false`, `image_encoder_calls`,
  `uses_precomputed_image_features`, `reencode_policy`, `evidence_scope`,
  `claim_eligible`, `tuner_state_hash`, and `context_source`.
- count image encoder calls at the resolved Qwen visual owner. For
  `control_mode=primary_coverage`, set `claim_eligible=false` when
  `image_encoder_calls > 1` unless the evidence scope is explicitly
  `tiny` or `smoke-only runtime limitation`; do not present such runs as primary
  mechanism evidence.

- [ ] **Step 5: Add artifact writer**

Write JSONL/JSON helpers for:

```text
coverage_state_rows.jsonl
coverage_descriptor_stats.json
rollout_rows.jsonl
rollout_summary.json
```

Use `ensure_ascii=True` and `sort_keys=True` for JSON artifacts.

- [ ] **Step 6: Run rollout tests**

Run:

```bash
python -m pytest tests/detection/coverage/test_rollout.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/detection/coverage/rollout.py src/detection/coverage/artifacts.py configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml tests/detection/coverage/test_rollout.py
git commit -m "Add row-boundary coverage rollout helper"
```

---

### Task 8: Add Mechanism Probe And Guardrail Reporting

**Files:**
- Create: `scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py`
- Create: `scripts/analysis/row_conditioned_visual_coverage/run_rollout.py`
- Create: `scripts/analysis/row_conditioned_visual_coverage/report.py`
- Test: `tests/detection/coverage/test_rollout.py`

- [ ] **Step 1: Define report rows**

Mechanism probe rows must include:

```text
sample_id
image_id
object_ordering
row_state_k
target_kind
coverage_enabled
control_mode
x1_target_rank
y1_target_rank
x2_target_rank
y2_target_rank
x1_target_neighborhood_mass
y1_target_neighborhood_mass
x2_target_neighborhood_mass
y2_target_neighborhood_mass
same_desc_competitor_rank
residual_vs_eos_margin
residual_vs_emitted_margin
duplicate_local_mass
descriptor_boundary_sum
descriptor_interior_sum
```

Rollout guardrail rows must include:

```text
sample_id
image_id
coverage_enabled
raw_pred_count
parse_valid_count
parse_invalid_count
assistant_stop_seen
finish_reason
same_desc_iou_gt_0p95_duplicate_count
matched_gt_count
missed_gt_count
other_extra_count
cache_reused_across_rows
image_encoder_calls
uses_precomputed_image_features
reencode_policy
evidence_scope
claim_eligible
tuner_state_hash
context_source
```

- [ ] **Step 2: Define required mechanism controls**

Mechanism and report code must support these `control_mode` labels:

```text
primary_coverage
no_coverage_reprefill
zero_mass_descriptor
boundary_only
interior_only
shuffled_same_image_coverage
wrong_image_coverage
stale_prefix_coverage
```

Only `primary_coverage` is the v1 mechanism claim path. All other modes are
controls and must be labeled as controls in artifacts and reports.

- [ ] **Step 3: Add report tests to verify RED**

Add a fake report test that includes at least one row for
`primary_coverage`, one for `no_coverage_reprefill`, and one control row. It
must assert:

```python
report = write_row_coverage_report(fake_rows, output_root)
mechanism_summary = json.loads((output_root / "mechanism_summary.json").read_text())
guardrail_metrics = json.loads((output_root / "guardrail_metrics.json").read_text())

assert set(mechanism_summary["by_object_ordering"]) == {"random_sft", "sorted_sft"}
assert set(mechanism_summary["by_control_mode"]) >= {
    "primary_coverage",
    "no_coverage_reprefill",
    "boundary_only",
}
assert mechanism_summary["by_control_mode"]["primary_coverage"]["row_count"] == 1
assert guardrail_metrics["by_control_mode"]["no_coverage_reprefill"]["parse_invalid_count"] == 0
assert "## random_sft" in report.report_text
assert "## sorted_sft" in report.report_text
assert report.report_text.count("primary_coverage") == 1
```

Run:

```bash
python -m pytest tests/detection/coverage/test_rollout.py::test_report_keeps_baseline_and_controls_separate -q
```

Expected before implementation: FAIL because the report writer or required
fields do not exist.

- [ ] **Step 4: Add script CLIs**

Each script must accept:

```text
--config
--output-root
--limit
--coverage-enabled true|false
--control-mode primary_coverage|no_coverage_reprefill|zero_mass_descriptor|boundary_only|interior_only|shuffled_same_image_coverage|wrong_image_coverage|stale_prefix_coverage
```

No stable production CLI is added. These are research analysis scripts.

- [ ] **Step 5: Add report writer**

`report.py` should write:

```text
report.md
mechanism_summary.json
guardrail_metrics.json
```

The report must keep `random_sft` and `sorted_sft` in separate sections.
It must also keep `primary_coverage`, `no_coverage_reprefill`, and every
control mode in separate subsections.

- [ ] **Step 6: Add smoke checks**

Add a test that feeds tiny fake mechanism and rollout rows to the report writer
and parses the emitted summaries:

```python
result = write_row_coverage_report(fake_mechanism_rows + fake_rollout_rows, output_root)
mechanism_summary = json.loads((output_root / "mechanism_summary.json").read_text())
guardrail_metrics = json.loads((output_root / "guardrail_metrics.json").read_text())

assert mechanism_summary["by_object_ordering"]["random_sft"]["row_count"] == 1
assert mechanism_summary["by_object_ordering"]["sorted_sft"]["row_count"] == 1
assert guardrail_metrics["by_control_mode"]["primary_coverage"]["same_desc_iou_gt_0p95_duplicate_count"] == 2
assert guardrail_metrics["by_control_mode"]["no_coverage_reprefill"]["parse_invalid_count"] == 1
assert result.report_text.count("## random_sft") == 1
assert result.report_text.count("## sorted_sft") == 1
```

- [ ] **Step 7: Run report tests**

Run:

```bash
python -m pytest tests/detection/coverage/test_rollout.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py scripts/analysis/row_conditioned_visual_coverage/run_rollout.py scripts/analysis/row_conditioned_visual_coverage/report.py tests/detection/coverage/test_rollout.py
git commit -m "Add row coverage mechanism reporting"
```

---

### Task 9: Final Verification And Documentation Sync

**Files:**
- Modify: `docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-protocol.md` if implementation changes the protocol.
- Modify: `docs/superpowers/plans/2026-06-05-row-conditioned-visual-coverage-prototype.md` only if execution discovers a necessary plan correction.

- [ ] **Step 1: Run focused unit suite**

Run:

```bash
python -m pytest tests/detection/coverage tests/test_teacher_forcing_sidecar_bridge.py tests/test_teacher_forcing_target_builder.py tests/test_teacher_forcing_config_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run broad relevant training checks**

Run:

```bash
python -m pytest tests/test_training_surface_resolver.py tests/test_objective_profile_resolution.py tests/test_compact_full_encoding_contract.py tests/test_training_architecture_golden_thread.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run smoke config parse**

Run:

```bash
PYTHONPATH=. python - <<'PY'
from src.config.loader import ConfigLoader
from src.config.schema import DetectionTrainingConfig

for path in [
    "configs/stage1/detection_teacher_forcing/ablation/row_coverage_random_sft_smoke.yaml",
    "configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml",
]:
    cfg = DetectionTrainingConfig.from_mapping(ConfigLoader.load_yaml_with_extends(path))
    assert cfg.row_conditioned_visual_coverage.enabled is True
    assert cfg.training["packing"] is False
    assert cfg.training["group_by_length"] is False
    print(path, cfg.data.object_ordering)
PY
```

Expected stdout lists both config paths and their distinct object orderings.

- [ ] **Step 4: Run baseline/coverage report smoke**

Run:

```bash
python -m py_compile \
  src/detection/coverage/config.py \
  src/detection/coverage/types.py \
  src/detection/coverage/geometry.py \
  src/detection/coverage/painting.py \
  src/detection/coverage/prefix_rendering.py \
  src/detection/coverage/row_state_dataset.py \
  src/detection/coverage/forward.py \
  src/detection/coverage/rollout.py \
  src/detection/coverage/artifacts.py \
  scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py \
  scripts/analysis/row_conditioned_visual_coverage/run_rollout.py \
  scripts/analysis/row_conditioned_visual_coverage/report.py

PYTHONPATH=. python scripts/analysis/row_conditioned_visual_coverage/run_mechanism_probe.py --help
PYTHONPATH=. python scripts/analysis/row_conditioned_visual_coverage/run_rollout.py --help
PYTHONPATH=. python scripts/analysis/row_conditioned_visual_coverage/report.py --help

PYTHONPATH=. python scripts/analysis/row_conditioned_visual_coverage/run_rollout.py \
  --config configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml \
  --output-root temp/row_coverage_smoke/no_coverage \
  --limit 2 \
  --coverage-enabled false \
  --control-mode no_coverage_reprefill

PYTHONPATH=. python scripts/analysis/row_conditioned_visual_coverage/run_rollout.py \
  --config configs/analysis/row_conditioned_visual_coverage/rollout_smoke.yaml \
  --output-root temp/row_coverage_smoke/primary_coverage \
  --limit 2 \
  --coverage-enabled true \
  --control-mode primary_coverage

PYTHONPATH=. python scripts/analysis/row_conditioned_visual_coverage/report.py \
  --input-root temp/row_coverage_smoke \
  --output-root temp/row_coverage_smoke/report
```

Expected: commands exit 0, the rollout commands write separate
`rollout_summary.json` artifacts, and the report command writes `report.md`.
If the scripts are fake-data-only at this point, label the artifact scope as
`tiny` or `smoke`.

- [ ] **Step 5: Verify stable route boundaries stayed untouched**

Run:

```bash
BASE=$(git merge-base HEAD origin/main)
git diff --exit-code "$BASE"...HEAD -- docs/AGENT_INDEX.md docs/catalog.yaml docs/training openspec
```

Expected: no output and exit 0.

Run:

```bash
BASE=$(git merge-base HEAD origin/main)
git diff --name-only "$BASE"...HEAD -- docs/superpowers progress src tests configs scripts
```

Expected: changed files are limited to the row-coverage implementation,
research docs, tests, scripts, and smoke configs described by this plan.

- [ ] **Step 6: Run optional tiny forward smoke**

Only run this when a GPU is intentionally available:

```bash
config=configs/stage1/detection_teacher_forcing/ablation/row_coverage_sorted_sft_smoke.yaml gpus=0 bash scripts/train.sh
```

Expected: one training step completes and logs row-coverage config metadata.

- [ ] **Step 7: Update progress evidence**

Append a short section to:

```text
progress/directions/2026-06-05_row_conditioned_visual_coverage.md
```

Record:

```text
scope: implementation smoke or docs-only
configs checked
tests run
artifact root if any
known limitations
```

- [ ] **Step 8: Commit final docs sync**

```bash
git add docs/superpowers/specs/2026-06-05-row-conditioned-visual-coverage-protocol.md docs/superpowers/plans/2026-06-05-row-conditioned-visual-coverage-prototype.md progress/directions/2026-06-05_row_conditioned_visual_coverage.md
git commit -m "Document row coverage prototype verification"
```

---

## Interpretation Rules

Do not claim success from duplicate reduction alone. A positive result requires:

```text
coverage changes intended row-conditioned binding readouts
and
rollout does not improve by lowering recall, truncating rows, increasing invalid parses, or suppressing overlap/crowded cases
```

Do not promote `row_conditioned_visual_coverage` into OpenSpec
until the first prototype establishes a stable config and artifact contract.

## Execution Choice

Plan complete. Two execution options:

1. Subagent-Driven: dispatch a fresh subagent per task and review between tasks.
2. Inline Execution: execute tasks in this session with checkpoints.

The recommended implementation path is Subagent-Driven because the dataset,
forward wrapper, rollout, and reporting tasks have distinct failure surfaces.
