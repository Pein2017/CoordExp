# IoU-Gibbs Coordinate SoftCE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add A5/A6 production-scale ablation candidates that replace hard coordinate CE with data-calibrated continuous coordinate soft targets while preserving the A2 compact-full training setup and recursive support/balance coefficients.

**Architecture:** Latest recursive detection owns the feature. `objective.coord_soft_ce` selects a continuous geometry-aware target distribution, recursive target building attaches bbox/slot metadata, and the recursive CE loss replaces coordinate positions with full-vocab dense-support soft-target CE using the A2 support/balance coefficients. A5 uses `iou_gibbs_v0`; A6 uses the same support/balance machinery with `ciou_gibbs_v0` as the geometry energy. The old fixed Gaussian `sigma`/`truncate` shape is deprecated for latest recursive detection and must not be used by A5/A6.

**Tech Stack:** Python dataclasses, PyTorch tensor losses, strict latest-schema YAML configs, compact-full detection sidecars, pytest under `conda run -n ms`.

---

Date: 2026-05-11

Spec: `docs/superpowers/specs/2026-05-11-iou-gibbs-coordinate-softce-design.md`

Status: implemented, smoke-preflighted, and production-launched after user approval. Core helper, schema/runtime wiring, compact-full trie-coordinate replacement, A5/A6 configs, calibration artifacts, related unit/config tests, tiny smokes, DDP4 preflights, and post-launch smoke/regression checks have passed. Remaining work is nonintrusive production monitoring and final measured-result analysis when the runs finish.

## Hard Guardrails

| Guardrail | Requirement |
|---|---|
| Training comparison | A5 must extend A2/support2 and keep model, data, optimizer, epoch count, LoRA, batch semantics, prompt, template, token rows, cache/packing, eval cadence, and non-coordinate recursive behavior unchanged. |
| Template | Use `compact_full`, `coord_token`, `xyxy`, same dataset config. |
| Loss owner | Recursive CE owns this loss. Do not use `custom.coord_soft_ce_w1` or `CoordSoftCEW1LossMixin`. |
| Target family | Use `iou_gibbs_v0` for A5 and `ciou_gibbs_v0` for A6. Both start with `tau=0.0090909091`, derived from train one-token IoU median, unless a separate calibration artifact proves a different CIoU tau. |
| Weighting | Preserve A2 recursive support/balance coefficients for coordinate positions: support `2.0`, balance `1.0`. The support set itself becomes the dense geometry-valid coordinate-token set. |
| No extra objectives | Do not add W1, SmoothL1, decoded CIoU, or optimized hard coordinate CE in A5. |
| Deprecated Gaussian | Latest `objective.coord_soft_ce` must reject fixed Gaussian knobs such as `sigma`, `truncate`, `target_sigma`, and `target_truncate`. |
| Invalid candidate boxes | Candidate coord bins that violate `x1 < x2` or `y1 < y2` get zero target mass. Do not canonicalize by swapping edges. |
| Numerical failures | Do not sanitize optimized loss with `nan_to_num`; raise on non-finite loss or invalid targets. |
| Python navigation | Use Serena MCP for Python symbol exploration and edits after narrowing with `rg`. |
| Repo safety | Do not launch production training from this implementation plan. Do not revert unrelated dirty work. Smoke/preflight runs are allowed only after implementation and risk audit gates. |
| GPU split | Final production intent is two concurrent 4-GPU experiments: A5 on one 4-GPU slice and A6 on the other, not one 8-GPU experiment. |

## Planned File Map

| Path | Role |
|---|---|
| `scripts/analysis/compute_iou_gibbs_coord_stats.py` | Reproduce tau calibration and no-training target-shape audit over JSONL. |
| `progress/diagnostics/coord_softce_iou_gibbs_tau_v0.md` | Checked A5 IoU-Gibbs summary artifact from the calibration/audit script. |
| `progress/diagnostics/coord_softce_ciou_gibbs_tau_v0.md` | Checked A6 CIoU-Gibbs target-shape audit artifact using the same train JSONL and tau source. |
| `src/config/schema.py` | Add strict `CoordSoftCEConfig` under latest `objective.coord_soft_ce`; reject old Gaussian knobs in the new surface. |
| `src/detection/coord_soft_targets.py` | New focused owner for coordinate candidate dataclasses, IoU-Gibbs target distribution, and full-vocab support/balance soft-target CE. |
| `src/detection/objective.py` | Add coordinate slot metadata to recursive token targets and build singleton/support-mixture bbox candidates with same-slot validation. |
| `src/detection/runtime.py` | Resolve `coord_soft_ce` into recursive CE runtime config using `token_rows.groups.coord_geometry`. |
| `src/detection/loss.py` | Replace coordinate hard/support CE with IoU-Gibbs support/balance soft-target CE when metadata is present; emit diagnostics. |
| `src/detection/__init__.py` | Export new public dataclasses/helpers needed by tests and downstream modules. |
| `src/trainers/metrics/recursive_detection.py` | Pass runtime `coord_soft_ce` config into recursive CE and expose flat reporter aliases. |
| `src/tokens/coord/soft_ce_w1.py` | Add legacy/deprecated module note only; do not route latest recursive detection through it. |
| `src/trainers/losses/coord_soft_ce_w1.py` | Add legacy/deprecated module note only; do not change old runtime behavior unless tests prove compatibility. |
| `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml` | A5 production-scale ablation candidate extending A2/support2. |
| `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml` | A6 production-scale ablation candidate extending A2/support2. |
| `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_tiny.yaml` | One-step A5 tiny smoke config for trainer plumbing. |
| `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_tiny.yaml` | One-step A6 tiny smoke config for trainer plumbing. |
| `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight.yaml` | Short 4-GPU A5 preflight config before production launch. |
| `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_ddp4_preflight.yaml` | Short 4-GPU A6 preflight config before production launch. |
| `tests/test_iou_gibbs_coord_softce.py` | Unit tests for target distribution and support/balance soft-target CE math. |
| `tests/test_iou_gibbs_coord_stats.py` | Tiny-fixture tests for tau calibration and target-shape audit math. |
| `tests/test_recursive_detection_ce_loss_adapter.py` | Recursive CE tests for coordinate replacement, metrics, and fail-fast behavior. |
| `tests/test_latest_training_config_contract.py` | Schema/runtime/config tests for `objective.coord_soft_ce`, A5 inheritance, and deprecated knob rejection. |
| `tests/test_recursive_detection_ce_target_builder.py` | Target metadata tests over real compact-full recursive targets, including trie-coordinate softCE replacement. |
| `docs/training/STAGE1_OBJECTIVE.md` | Document A5 as an implemented or planned ablation surface after tests pass. |
| `docs/training/README.md` | Route readers to the A5 config and smoke/preflight gates. |
| `docs/catalog.yaml` | Register the A5/A6 configs using the existing `config_surfaces.training` shape. |

## Target API Shape

Use this public shape unless code inspection reveals a better local owner name.

```python
# src/config/schema.py
@dataclass(frozen=True)
class CoordSoftCEConfig:
    enabled: bool
    target_distribution: Literal["iou_gibbs_v0", "ciou_gibbs_v0"]
    tau: float
    tau_source: Literal["train_one_token_iou_median_v0"]
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    replace_coord_hard_ce: bool = True
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"
```

```python
# src/detection/coord_soft_targets.py
CoordSlotName = Literal["x1", "y1", "x2", "y2"]

@dataclass(frozen=True)
class CoordSoftTargetCandidate:
    object_instance_id: str
    slot_name: CoordSlotName
    bbox_xyxy: tuple[int, int, int, int]
    probability: float
```

```python
# src/detection/coord_soft_targets.py
@dataclass(frozen=True)
class CoordSoftTargetRuntimeConfig:
    target_distribution: Literal["iou_gibbs_v0", "ciou_gibbs_v0"]
    tau: float
    coord_token_start: int
    coord_token_end: int
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"
```

Disabled runtime is represented as `coord_soft_ce is None`. The active runtime
dataclass must not also carry an `enabled` flag.

Extend recursive targets with:

```python
coord_slot_name: CoordSlotName | None = None
coord_soft_targets: tuple[CoordSoftTargetCandidate, ...] = ()
```

Coordinate-softCE eligibility is `target.coord_slot_name is not None` and
`target.coord_soft_targets` is non-empty. Do not gate on
`SemanticRole.BBOX_COORD` alone because trie divergence coordinate positions may
retain `ENTRY_TRIE_DECISION` for normalization.

## Task 0: Preflight Snapshot

**Files:**

- Read: `docs/superpowers/specs/2026-05-11-iou-gibbs-coordinate-softce-design.md`
- Read: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`
- Read: `src/config/schema.py`
- Read: `src/detection/objective.py`
- Read: `src/detection/loss.py`
- Read: `src/detection/runtime.py`
- Read: `src/trainers/metrics/recursive_detection.py`
- Read: `src/tokens/coord/soft_ce_w1.py`
- Read: `src/trainers/losses/coord_soft_ce_w1.py`

- [ ] **Step 1: Check worktree status**

Run:

```bash
git status --short
```

Expected: unrelated dirty files may exist. Do not revert unrelated changes.

- [ ] **Step 2: Narrow current code owners**

Run:

```bash
rg -n "DetectionObjectiveConfig|RecursiveDetectionCERuntimeConfig|TokenTarget|_TrieObjectInstance|compute_recursive_detection_ce_batch_loss|coord_soft_ce_w1|target_sigma|target_truncate" src tests configs docs
```

Expected: hits in schema, runtime, objective, loss, trainer metrics, legacy coord softCE files, and tests.

- [ ] **Step 3: Use Serena for Python symbol inspection**

Inspect these symbols before editing:

```text
src/config/schema.py::DetectionObjectiveConfig
src/detection/runtime.py::RecursiveDetectionCERuntimeConfig
src/detection/runtime.py::resolve_recursive_detection_ce_runtime_cfg
src/detection/objective.py::TokenTarget
src/detection/objective.py::_TrieObjectInstance
src/detection/objective.py::_append_recursive_entry_targets
src/detection/loss.py::_compute_sample_loss
src/trainers/metrics/recursive_detection.py::RecursiveDetectionCEMixin/compute_loss
```

Expected: identify exact edit sites and keep changes within latest recursive detection owners.

## Task 1: Calibration And Target-Shape Audit

**Files:**

- Create: `scripts/analysis/compute_iou_gibbs_coord_stats.py`
- Create: `tests/test_iou_gibbs_coord_stats.py`
- Create or update after running: `progress/diagnostics/coord_softce_iou_gibbs_tau_v0.md`

- [ ] **Step 1: Write failing tiny-fixture calibration tests**

Create `tests/test_iou_gibbs_coord_stats.py`:

```python
from __future__ import annotations

import pytest

from scripts.analysis.compute_iou_gibbs_coord_stats import (
    one_token_iou_losses,
    summarize_losses,
)


def test_one_token_iou_losses_skip_invalid_edge_moves() -> None:
    losses = one_token_iou_losses((0, 0, 10, 10))

    # x1 - 1 and y1 - 1 are invalid at the boundary; all other one-token
    # moves preserve xyxy order and remain in range.
    assert len(losses) == 6
    assert all(loss > 0.0 for loss in losses)


def test_summarize_losses_reports_median() -> None:
    summary = summarize_losses([0.1, 0.3, 0.2])

    assert summary["count"] == 3
    assert summary["median"] == pytest.approx(0.2)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_iou_gibbs_coord_stats.py -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement calibration script**

Create `scripts/analysis/compute_iou_gibbs_coord_stats.py` with public helpers:

```python
def one_token_iou_losses(bbox_xyxy: tuple[int, int, int, int]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    losses: list[float] = []
    for slot_index in range(4):
        for delta in (-1, 1):
            candidate = [x1, y1, x2, y2]
            candidate[slot_index] += delta
            cx1, cy1, cx2, cy2 = candidate
            if not (0 <= cx1 < cx2 <= 999 and 0 <= cy1 < cy2 <= 999):
                continue
            losses.append(1.0 - iou_xyxy((x1, y1, x2, y2), (cx1, cy1, cx2, cy2)))
    return losses
```

Add a CLI that accepts:

```text
--jsonl public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
--output progress/diagnostics/coord_softce_iou_gibbs_tau_v0.md
--target-audit-sample 30000
```

The CLI must record:

```text
train_jsonl
sha256
row_count
object_count
included_perturbations
skipped_invalid_perturbations
median
mean
p75
p90
p95
p99
target entropy/perplexity/peak_prob/std, candidate_count, support_bin_count, and effective_support_size by min-side decile, area decile, slot, boundary flag, min_side<=32, and min_side<=50
```

Use structured JSON inside the markdown artifact or write a sibling `.json`.

- [ ] **Step 4: Run tiny tests and verify pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_iou_gibbs_coord_stats.py -q
```

Expected: PASS.

- [ ] **Step 5: Run calibration/audit artifact generation**

Run:

```bash
conda run -n ms python scripts/analysis/compute_iou_gibbs_coord_stats.py --jsonl public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl --output progress/diagnostics/coord_softce_iou_gibbs_tau_v0.md --target-audit-sample 30000
```

Expected: output artifact reports `median: 0.0090909091` or a clearly explained mismatch requiring spec review before implementation continues.

- [ ] **Step 6: Commit calibration artifact**

Run:

```bash
git add scripts/analysis/compute_iou_gibbs_coord_stats.py tests/test_iou_gibbs_coord_stats.py progress/diagnostics/coord_softce_iou_gibbs_tau_v0.md
git commit -m "analysis: calibrate iou gibbs coord softce tau"
```

Expected: commit succeeds with only calibration/audit files staged.

## Task 2: Schema Contract And Gaussian Deprecation

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/tokens/coord/soft_ce_w1.py`
- Modify: `src/trainers/losses/coord_soft_ce_w1.py`
- Test: `tests/test_latest_training_config_contract.py`

- [ ] **Step 1: Write failing schema tests**

Add tests to `tests/test_latest_training_config_contract.py`:

```python
def test_latest_random_permutation_accepts_iou_gibbs_coord_softce() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "iou_gibbs_v0",
            "tau": 0.0090909091,
            "tau_source": "train_one_token_iou_median_v0",
            "weighting": "preserve_recursive_support_balance",
            "replace_coord_hard_ce": True,
            "apply_to_multi_positive": "support_mixture",
        },
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.enabled is True
    assert cfg.objective.coord_soft_ce.tau == pytest.approx(0.0090909091)
    assert cfg.objective.coord_soft_ce.target_distribution == "iou_gibbs_v0"
    assert cfg.objective.coord_soft_ce.weighting == "preserve_recursive_support_balance"
```

```python
def test_latest_random_permutation_accepts_ciou_gibbs_coord_softce() -> None:
    payload = _latest_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "ciou_gibbs_v0",
            "tau": 0.0090909091,
            "tau_source": "train_one_token_iou_median_v0",
            "weighting": "preserve_recursive_support_balance",
            "replace_coord_hard_ce": True,
            "apply_to_multi_positive": "support_mixture",
        },
    }

    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.coord_soft_ce is not None
    assert cfg.objective.coord_soft_ce.target_distribution == "ciou_gibbs_v0"
```

```python
@pytest.mark.parametrize(
    "deprecated_key",
    ["sigma", "truncate", "target_sigma", "target_truncate", "window", "radius"],
)
def test_coord_softce_rejects_fixed_gaussian_knobs(deprecated_key: str) -> None:
    payload = _latest_payload()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "iou_gibbs_v0",
        "tau": 0.0090909091,
        "tau_source": "train_one_token_iou_median_v0",
        deprecated_key: 2.0,
    }

    with pytest.raises(ValueError, match=rf"objective\.coord_soft_ce\.{deprecated_key}"):
        LatestDetectionTrainingConfig.from_mapping(payload)
```

```python
def test_coord_softce_requires_positive_data_derived_tau() -> None:
    payload = _latest_payload()
    payload["objective"]["coord_soft_ce"] = {
        "enabled": True,
        "target_distribution": "iou_gibbs_v0",
        "tau": 0.0,
        "tau_source": "train_one_token_iou_median_v0",
    }

    with pytest.raises(ValueError, match=r"objective\.coord_soft_ce\.tau.*> 0"):
        LatestDetectionTrainingConfig.from_mapping(payload)
```

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_latest_random_permutation_accepts_iou_gibbs_coord_softce tests/test_latest_training_config_contract.py::test_latest_random_permutation_accepts_ciou_gibbs_coord_softce tests/test_latest_training_config_contract.py::test_coord_softce_rejects_fixed_gaussian_knobs tests/test_latest_training_config_contract.py::test_coord_softce_requires_positive_data_derived_tau -q
```

Expected: FAIL because `coord_soft_ce` is not yet a known latest objective field.

- [ ] **Step 3: Implement strict schema dataclass**

Add `CoordSoftCEConfig` before `DetectionObjectiveConfig` in `src/config/schema.py`:

```python
@dataclass(frozen=True)
class CoordSoftCEConfig:
    enabled: bool
    target_distribution: Literal["iou_gibbs_v0", "ciou_gibbs_v0"]
    tau: float
    tau_source: Literal["train_one_token_iou_median_v0"]
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    replace_coord_hard_ce: bool = True
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"

    def __post_init__(self) -> None:
        _latest_detection_validate_bool(self.enabled, path="objective.coord_soft_ce.enabled")
        _latest_detection_validate_choice(
            self.target_distribution,
            path="objective.coord_soft_ce.target_distribution",
            allowed={"iou_gibbs_v0", "ciou_gibbs_v0"},
        )
        if not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool):
            raise TypeError("objective.coord_soft_ce.tau must be numeric")
        if not math.isfinite(float(self.tau)) or float(self.tau) <= 0.0:
            raise ValueError("objective.coord_soft_ce.tau must be finite and > 0")
        _latest_detection_validate_choice(
            self.tau_source,
            path="objective.coord_soft_ce.tau_source",
            allowed={"train_one_token_iou_median_v0"},
        )
        _latest_detection_validate_choice(
            self.weighting,
            path="objective.coord_soft_ce.weighting",
            allowed={"preserve_recursive_support_balance"},
        )
        _latest_detection_validate_bool(
            self.replace_coord_hard_ce,
            path="objective.coord_soft_ce.replace_coord_hard_ce",
        )
        if not self.replace_coord_hard_ce:
            raise ValueError(
                "objective.coord_soft_ce.replace_coord_hard_ce=false is unsupported"
            )
        _latest_detection_validate_choice(
            self.apply_to_multi_positive,
            path="objective.coord_soft_ce.apply_to_multi_positive",
            allowed={"support_mixture"},
        )
```

Add `coord_soft_ce: Optional[CoordSoftCEConfig] = None` to
`DetectionObjectiveConfig`, permit it only for recursive ET-RMP variants, and
update the unexpected-key check so fixed Gaussian keys produce explicit errors:

```python
for deprecated_key in ("sigma", "truncate", "target_sigma", "target_truncate", "window", "radius"):
    if deprecated_key in raw_coord_soft_ce:
        raise ValueError(f"objective.coord_soft_ce.{deprecated_key} is deprecated; use iou_gibbs_v0 or ciou_gibbs_v0")
```

- [ ] **Step 4: Add legacy/deprecated notes without behavior changes**

At the top-level module docstring or near the public entrypoint in
`src/tokens/coord/soft_ce_w1.py` and `src/trainers/losses/coord_soft_ce_w1.py`,
add a short note:

```python
# Legacy fixed Gaussian coordinate softCE/W1 surface. Latest recursive
# detection uses objective.coord_soft_ce with continuous Gibbs targets instead.
```

Do not add runtime warnings unless the existing tests are updated to expect
them; warnings may destabilize old supported surfaces.

- [ ] **Step 5: Run schema tests and verify pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_latest_random_permutation_accepts_iou_gibbs_coord_softce tests/test_latest_training_config_contract.py::test_latest_random_permutation_accepts_ciou_gibbs_coord_softce tests/test_latest_training_config_contract.py::test_coord_softce_rejects_fixed_gaussian_knobs tests/test_latest_training_config_contract.py::test_coord_softce_requires_positive_data_derived_tau -q
```

Expected: PASS.

- [ ] **Step 6: Commit schema/deprecation contract**

Run:

```bash
git add src/config/schema.py src/tokens/coord/soft_ce_w1.py src/trainers/losses/coord_soft_ce_w1.py tests/test_latest_training_config_contract.py
git commit -m "feat: add continuous coord softce schema"
```

Expected: commit succeeds with only schema, legacy notes, and tests staged.

## Task 3: IoU-Gibbs Target And Support/Balance SoftCE Helper

**Files:**

- Create: `src/detection/coord_soft_targets.py`
- Modify: `src/detection/__init__.py`
- Test: `tests/test_iou_gibbs_coord_softce.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_iou_gibbs_coord_softce.py`:

```python
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    build_iou_gibbs_coord_target,
    full_vocab_coord_support_balance_ce,
)


def _cfg() -> CoordSoftTargetRuntimeConfig:
    return CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=0.0090909091,
        coord_token_start=10,
        coord_token_end=1009,
    )


def test_iou_gibbs_target_is_normalized_and_peaks_at_gt_coord() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (100, 100, 200, 200), 1.0)

    target = build_iou_gibbs_coord_target((candidate,), _cfg())

    assert target.token_ids.shape == (1000,)
    assert target.probs.sum().item() == pytest.approx(1.0)
    peak_index = int(target.probs.argmax().item())
    assert int(target.token_ids[peak_index].item()) == 110
    assert target.entropy.item() > 0.0


def test_boundary_invalid_candidates_get_zero_mass() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (0, 100, 5, 200), 1.0)

    target = build_iou_gibbs_coord_target((candidate,), _cfg())

    invalid = target.token_ids >= 15
    assert torch.all(target.probs[invalid] == 0)
    assert target.probs.sum().item() == pytest.approx(1.0)


def test_large_box_has_broader_target_than_tiny_box() -> None:
    tiny = CoordSoftTargetCandidate("tiny", "x1", (100, 100, 110, 200), 1.0)
    large = CoordSoftTargetCandidate("large", "x1", (100, 100, 700, 200), 1.0)

    tiny_target = build_iou_gibbs_coord_target((tiny,), _cfg())
    large_target = build_iou_gibbs_coord_target((large,), _cfg())

    assert large_target.std.item() > tiny_target.std.item()
    assert large_target.entropy.item() > tiny_target.entropy.item()


def test_support2_balance1_differs_from_pure_softce_but_preserves_full_vocab_pressure() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (3, 1, 8, 9), 1.0)
    logits = torch.zeros((1020,), dtype=torch.float32)
    logits[0] = 5.0

    result = full_vocab_coord_support_balance_ce(
        logits,
        (candidate,),
        _cfg(),
        support_weight=2.0,
        balance_weight=1.0,
    )

    assert result.weighted_loss.item() > result.pure_soft_ce_equiv.item()
    assert result.pure_soft_ce_equiv.item() > 5.0


def test_support_balance_11_equals_manual_full_vocab_softce() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (100, 100, 200, 200), 1.0)
    logits = torch.linspace(-2.0, 2.0, steps=1020, dtype=torch.float32)
    cfg = _cfg()
    dist = build_iou_gibbs_coord_target((candidate,), cfg)

    result = full_vocab_coord_support_balance_ce(
        logits,
        (candidate,),
        cfg,
        support_weight=1.0,
        balance_weight=1.0,
    )
    manual = -(dist.probs * F.log_softmax(logits, dim=-1).index_select(0, dist.token_ids)).sum()

    assert result.weighted_loss.item() == pytest.approx(manual.item())


def test_geometry_valid_support_mask_does_not_depend_on_probability_underflow() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (100, 100, 900, 900), 1.0)
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=1e-6,
        coord_token_start=10,
        coord_token_end=1009,
    )

    dist = build_iou_gibbs_coord_target((candidate,), cfg)

    assert int(dist.support_mask.sum().item()) == 900
    assert int((dist.probs > 0).sum().item()) <= int(dist.support_mask.sum().item())


def test_iou_gibbs_target_rejects_mixed_candidate_slots() -> None:
    candidates = (
        CoordSoftTargetCandidate("box-x", "x1", (100, 100, 200, 200), 0.5),
        CoordSoftTargetCandidate("box-y", "y1", (100, 100, 200, 200), 0.5),
    )

    with pytest.raises(ValueError, match="same coordinate slot"):
        build_iou_gibbs_coord_target(candidates, _cfg())


def test_nonfinite_logits_raise_instead_of_nan_to_num() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (3, 1, 8, 9), 1.0)
    logits = torch.zeros((1020,), dtype=torch.float32)
    logits[11] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        full_vocab_coord_support_balance_ce(
            logits,
            (candidate,),
            _cfg(),
            support_weight=2.0,
            balance_weight=1.0,
        )
```

- [ ] **Step 2: Run helper tests and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_iou_gibbs_coord_softce.py -q
```

Expected: FAIL because `src.detection.coord_soft_targets` does not exist.

- [ ] **Step 3: Implement helper module**

Create `src/detection/coord_soft_targets.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

CoordSlotName = Literal["x1", "y1", "x2", "y2"]
_COORD_SLOT_NAMES: tuple[str, ...] = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class CoordSoftTargetCandidate:
    object_instance_id: str
    slot_name: CoordSlotName
    bbox_xyxy: tuple[int, int, int, int]
    probability: float

    def __post_init__(self) -> None:
        if self.slot_name not in _COORD_SLOT_NAMES:
            raise ValueError(f"unsupported coordinate slot {self.slot_name!r}")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four coord-token bins")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in self.bbox_xyxy):
            raise TypeError("bbox_xyxy must contain integer coord-token bins")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
            raise ValueError(f"bbox_xyxy must be valid token-space xyxy; got {self.bbox_xyxy}")
        if not torch.isfinite(torch.tensor(float(self.probability))) or float(self.probability) <= 0.0:
            raise ValueError("coord soft target probability must be finite and > 0")
```

Add runtime/result dataclasses:

```python
@dataclass(frozen=True)
class CoordSoftTargetRuntimeConfig:
    target_distribution: Literal["iou_gibbs_v0", "ciou_gibbs_v0"]
    tau: float
    coord_token_start: int
    coord_token_end: int
    weighting: Literal["preserve_recursive_support_balance"] = (
        "preserve_recursive_support_balance"
    )
    apply_to_multi_positive: Literal["support_mixture"] = "support_mixture"


@dataclass(frozen=True)
class CoordSoftTargetDistribution:
    token_ids: torch.Tensor
    probs: torch.Tensor
    support_mask: torch.Tensor
    entropy: torch.Tensor
    peak_prob: torch.Tensor
    perplexity: torch.Tensor
    effective_support_size: torch.Tensor
    std: torch.Tensor
    candidate_count: torch.Tensor
    support_bin_count: torch.Tensor


@dataclass(frozen=True)
class CoordSoftCELoss:
    weighted_loss: torch.Tensor
    support_loss: torch.Tensor
    support_mass: torch.Tensor
    outside_support_mass: torch.Tensor
    balance_loss: torch.Tensor
    pure_soft_ce_equiv: torch.Tensor
    target_entropy: torch.Tensor
    kl_like: torch.Tensor
    peak_prob: torch.Tensor
    perplexity: torch.Tensor
    effective_support_size: torch.Tensor
    target_std: torch.Tensor
    candidate_count: torch.Tensor
    support_bin_count: torch.Tensor
    support_mixture: bool
```

Implement:

```python
def full_vocab_coord_support_balance_ce(
    logits: torch.Tensor,
    candidates: Sequence[CoordSoftTargetCandidate],
    cfg: CoordSoftTargetRuntimeConfig,
    *,
    support_weight: float,
    balance_weight: float,
) -> CoordSoftCELoss:
    if not torch.isfinite(logits.float()).all():
        raise ValueError("coord softCE received non-finite logits")
    dist = build_iou_gibbs_coord_target(candidates, cfg, device=logits.device)
    if int(dist.token_ids.max().item()) >= int(logits.shape[-1]):
        raise ValueError("coord token id exceeds logits vocab size")

    log_probs = F.log_softmax(logits.float(), dim=-1)
    coord_log_probs = log_probs.index_select(0, dist.token_ids)
    support_mask = dist.support_mask
    log_m = torch.logsumexp(coord_log_probs[support_mask], dim=0)
    support_loss = -log_m
    support_mass = log_m.exp()
    outside_support_mass = 1.0 - support_mass
    balance_loss = -(dist.probs * (coord_log_probs - log_m)).sum()
    pure_soft_ce_equiv = support_loss + balance_loss
    weighted_loss = float(support_weight) * support_loss + float(balance_weight) * balance_loss
    kl_like = pure_soft_ce_equiv - dist.entropy

    for name, value in (
        ("weighted_loss", weighted_loss),
        ("support_loss", support_loss),
        ("balance_loss", balance_loss),
        ("kl_like", kl_like),
    ):
        if not torch.isfinite(value):
            raise ValueError(f"coord softCE produced non-finite {name}")

    return CoordSoftCELoss(
        weighted_loss=weighted_loss,
        support_loss=support_loss,
        support_mass=support_mass,
        outside_support_mass=outside_support_mass,
        balance_loss=balance_loss,
        pure_soft_ce_equiv=pure_soft_ce_equiv,
        target_entropy=dist.entropy,
        kl_like=kl_like,
        peak_prob=dist.peak_prob,
        perplexity=dist.perplexity,
        effective_support_size=dist.effective_support_size,
        target_std=dist.std,
        candidate_count=dist.candidate_count,
        support_bin_count=dist.support_bin_count,
        support_mixture=len(candidates) > 1,
    )
```

`build_iou_gibbs_coord_target` must:

- require exactly 1000 coord bins;
- build `bins = 0..999`, not token IDs;
- require all candidates in a mixture to share the same `slot_name`;
- mask invalid candidate boxes to zero mass;
- return an explicit geometry-valid `support_mask` independent of probability
  underflow; never compute support from `dist.probs > 0`;
- normalize the candidate mixture after applying object probabilities;
- compute target probabilities in float64/log-space before returning tensors;
- return finite entropy, perplexity, effective support size, peak probability,
  valid candidate count, and std.

- [ ] **Step 4: Export helper objects**

Update `src/detection/__init__.py` to export:

```python
CoordSlotName
CoordSoftTargetCandidate
CoordSoftTargetRuntimeConfig
CoordSoftCELoss
build_iou_gibbs_coord_target
full_vocab_coord_support_balance_ce
```

- [ ] **Step 5: Run helper tests and verify pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_iou_gibbs_coord_softce.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit helper module**

Run:

```bash
git add src/detection/coord_soft_targets.py src/detection/__init__.py tests/test_iou_gibbs_coord_softce.py
git commit -m "feat: add iou gibbs coord soft target helper"
```

Expected: commit succeeds with helper module and tests staged.

## Task 4: Coordinate Geometry Metadata In Recursive Targets

**Files:**

- Modify: `src/detection/objective.py`
- Modify: `src/detection/__init__.py`
- Test: `tests/test_prefix_rollin_dataset_alignment.py`
- Test: `tests/test_recursive_detection_ce_loss_adapter.py`

- [ ] **Step 1: Write failing target-metadata test**

Add a real compact-full recursive target test:

```python
def test_recursive_coord_targets_carry_bbox_slot_soft_targets() -> None:
    example = _example(k=0)
    coord_targets = [
        target
        for target in example.recursive_detection_targets.token_targets
        if getattr(target, "coord_slot_name", None) is not None
    ]

    assert coord_targets
    for target in coord_targets:
        assert target.coord_slot_name in {"x1", "y1", "x2", "y2"}
        assert target.coord_soft_targets
        for candidate in target.coord_soft_targets:
            assert candidate.slot_name == target.coord_slot_name
            x1, y1, x2, y2 = candidate.bbox_xyxy
            assert 0 <= x1 < x2 <= 999
            assert 0 <= y1 < y2 <= 999
            assert candidate.probability > 0.0
```

Add a synthetic same-slot fail-fast test in the nearest objective/loss adapter
test file:

```python
def test_coord_soft_targets_reject_mixed_slots_at_same_trie_offset() -> None:
    with pytest.raises(ValueError, match="same coordinate slot"):
        _build_mixed_slot_recursive_target_fixture()
```

The helper fixture may be minimal; it should create two active trie instances
whose next child tokens at the same offset correspond to different bbox slots.

- [ ] **Step 2: Run metadata tests and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_prefix_rollin_dataset_alignment.py::test_recursive_coord_targets_carry_bbox_slot_soft_targets tests/test_recursive_detection_ce_loss_adapter.py::test_coord_soft_targets_reject_mixed_slots_at_same_trie_offset -q
```

Expected: FAIL because `TokenTarget.coord_slot_name` and `coord_soft_targets` do not exist.

- [ ] **Step 3: Extend target and trie metadata**

In `src/detection/objective.py`, import `CoordSlotName` and
`CoordSoftTargetCandidate` from `src.detection.coord_soft_targets`.

Extend `TokenTarget`:

```python
coord_slot_name: CoordSlotName | None = None
coord_soft_targets: tuple[CoordSoftTargetCandidate, ...] = ()
```

Extend `_TrieObjectInstance`:

```python
bbox_xyxy: tuple[int, int, int, int]
coord_slot_by_trie_offset: dict[int, CoordSlotName]
coord_token_by_slot: dict[CoordSlotName, int]
```

Build `bbox_xyxy` and `coord_token_by_slot` from `obj.bbox_2d.tokens`, which is
the source that carries the serialized `<|coord_*|>` strings. Use
`entry.coord_spans`, `entry.trie_eligible_span`, and `tokenized.input_ids` to
cross-check slot offsets and token-id alignment; do not assume
`TokenizedObjectEntry` itself contains token strings. Parse coord-token strings
with a strict helper and require already valid `xyxy`:

```python
def _coord_token_to_bin(token: str) -> int:
    match = re.fullmatch(r"<\|coord_(\d{1,3})\|>", token)
    if match is None:
        raise ValueError(f"expected coord token, got {token!r}")
    value = int(match.group(1))
    if value < 0 or value > 999:
        raise ValueError(f"coord token bin must be in [0, 999], got {value}")
    return value
```

- [ ] **Step 4: Track active trie instances**

Extend `_EntryTrieNode`:

```python
instances: list[_TrieObjectInstance] = field(default_factory=list)
```

Update trie construction:

```python
root.instances.append(instance)
for token_id in instance.token_ids:
    node = node.children.setdefault(token_id, _EntryTrieNode())
    node.instances.append(instance)
```

- [ ] **Step 5: Attach same-slot support candidates**

Before creating `coord_soft_targets` at a trie offset:

```python
active_slot_names = {
    instance.coord_slot_by_trie_offset.get(trie_offset)
    for instance in node.instances
}
if teacher_slot_name is not None:
    if active_slot_names != {teacher_slot_name}:
        raise ValueError(
            f"coord_soft_ce requires same coordinate slot at trie offset {trie_offset}; "
            f"got {sorted(str(slot) for slot in active_slot_names)}"
        )
    for instance in node.instances:
        expected_token = instance.coord_token_by_slot[teacher_slot_name]
        active_child_token = instance.token_ids[trie_offset]
        if active_child_token != expected_token:
            raise ValueError(
                "coord_soft_ce trie metadata mismatch: child token does not match bbox slot"
            )
```

Then build object-uniform candidates:

```python
probability = 1.0 / float(len(node.instances))
coord_soft_targets = tuple(
    CoordSoftTargetCandidate(
        object_instance_id=instance.object_instance_id,
        slot_name=teacher_slot_name,
        bbox_xyxy=instance.bbox_xyxy,
        probability=probability,
    )
    for instance in node.instances
)
```

- [ ] **Step 6: Run metadata tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_prefix_rollin_dataset_alignment.py::test_recursive_coord_targets_carry_bbox_slot_soft_targets tests/test_recursive_detection_ce_loss_adapter.py::test_coord_soft_targets_reject_mixed_slots_at_same_trie_offset -q
```

Expected: PASS.

- [ ] **Step 7: Commit recursive target metadata**

Run:

```bash
git add src/detection/objective.py src/detection/__init__.py tests/test_prefix_rollin_dataset_alignment.py tests/test_recursive_detection_ce_loss_adapter.py
git commit -m "feat: attach bbox metadata to recursive coord targets"
```

Expected: commit succeeds with only target metadata files staged.

## Task 5: Recursive CE Loss Integration

**Files:**

- Modify: `src/detection/loss.py`
- Test: `tests/test_recursive_detection_ce_loss_adapter.py`
- Test: `tests/test_iou_gibbs_coord_softce.py`

- [ ] **Step 1: Write failing coordinate replacement tests**

Append tests to `tests/test_recursive_detection_ce_loss_adapter.py`:

```python
def test_coord_softce_preserves_recursive_support_balance_weights() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=0.0090909091,
        coord_token_start=10,
        coord_token_end=1009,
    )
    candidate = CoordSoftTargetCandidate("obj-0", "x1", (100, 100, 200, 200), 1.0)
    target = replace(
        _hard_target(position=1, teacher_token_id=110, semantic_role=SemanticRole.ENTRY_TRIE_DECISION),
        coord_slot_name="x1",
        coord_soft_targets=(candidate,),
    )
    logits = torch.zeros((1, 1020), dtype=torch.float32)

    result = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=(_targets(token_targets=(target,)),),
        weights=RecursiveDetectionLossWeights(support_weight=2.0, balance_weight=1.0),
        coord_soft_ce=cfg,
    )
    expected = full_vocab_coord_support_balance_ce(
        logits[0],
        (candidate,),
        cfg,
        support_weight=2.0,
        balance_weight=1.0,
    ).weighted_loss

    assert result.loss.item() == pytest.approx(expected.item())
```

```python
def test_coord_softce_requires_metadata_when_slot_present() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=0.0090909091,
        coord_token_start=10,
        coord_token_end=1009,
    )
    target = replace(_hard_target(position=1, teacher_token_id=110), coord_slot_name="x1")

    with pytest.raises(ValueError, match="missing coord_soft_targets"):
        compute_recursive_detection_ce_batch_loss(
            logits=torch.zeros((1, 1020), dtype=torch.float32),
            targets=(_targets(token_targets=(target,)),),
            coord_soft_ce=cfg,
        )
```

- [ ] **Step 2: Run loss tests and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py::test_coord_softce_preserves_recursive_support_balance_weights tests/test_recursive_detection_ce_loss_adapter.py::test_coord_softce_requires_metadata_when_slot_present -q
```

Expected: FAIL because `compute_recursive_detection_ce_batch_loss` has no `coord_soft_ce` argument and targets have no coord metadata fields.

- [ ] **Step 3: Extend batch loss API**

In `src/detection/loss.py`, import:

```python
from src.detection.coord_soft_targets import (
    CoordSoftCELoss,
    CoordSoftTargetRuntimeConfig,
    full_vocab_coord_support_balance_ce,
)
```

Extend:

```python
def compute_recursive_detection_ce_batch_loss(
    *,
    logits: torch.Tensor,
    targets: Sequence[RecursiveDetectionTargets],
    weights: RecursiveDetectionLossWeights | None = None,
    coord_soft_ce: CoordSoftTargetRuntimeConfig | None = None,
) -> RecursiveDetectionLossResult:
```

Pass `coord_soft_ce` into `_compute_sample_loss`.

- [ ] **Step 4: Replace coordinate main loss when enabled**

Before hard/trie branching in `_compute_sample_loss`:

```python
coord_loss = _maybe_compute_coord_softce_loss(
    logits=logits[target.position - 1],
    target=target,
    coord_soft_ce=coord_soft_ce,
    weights=weights,
)
if coord_loss is not None:
    position_loss = coord_loss.weighted_loss
    per_position_main_losses[target.position] = _loss_float(position_loss)
    per_position_losses[target.position] = _apply_type_gate_loss(
        position_loss,
        step_log_probs=step_log_probs,
        target=target,
        vocab_size=vocab_size,
    )
    coord_soft_stats.append(coord_loss)
    continue
```

Add helper:

```python
def _maybe_compute_coord_softce_loss(
    *,
    logits: torch.Tensor,
    target: TokenTarget,
    coord_soft_ce: CoordSoftTargetRuntimeConfig | None,
    weights: RecursiveDetectionLossWeights,
) -> CoordSoftCELoss | None:
    if coord_soft_ce is None:
        return None
    if target.coord_slot_name is None and not target.coord_soft_targets:
        return None
    if target.coord_slot_name is None:
        raise ValueError(f"coord_soft_ce target at position {target.position} is missing coord_slot_name")
    if not target.coord_soft_targets:
        raise ValueError(f"coord_soft_ce target at position {target.position} is missing coord_soft_targets")
    return full_vocab_coord_support_balance_ce(
        logits,
        target.coord_soft_targets,
        coord_soft_ce,
        support_weight=weights.support_weight,
        balance_weight=weights.balance_weight,
    )
```

- [ ] **Step 5: Add coordinate diagnostics**

Canonical event keys:

```text
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/support_loss
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/support_mass
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/outside_support_mass
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/balance_loss
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/weighted_loss
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/pure_soft_ce_equiv
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/target_entropy
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/kl_like
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/peak_prob
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/perplexity
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/effective_support_size
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/candidate_count
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/support_bin_count
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/target_std
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/enabled
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/tau
detection_sequence/objective/recursive_detection_ce/coord_soft_ce/support_mixture
```

Add assertions in tests that `RecursiveDetectionLossResult.metric_events`
contains the canonical namespace and finite values for these keys. Keep existing
hard-teacher coord CE summaries as diagnostics only.

- [ ] **Step 6: Run recursive CE tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py tests/test_iou_gibbs_coord_softce.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit loss integration**

Run:

```bash
git add src/detection/loss.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_iou_gibbs_coord_softce.py
git commit -m "feat: integrate iou gibbs coord softce into recursive loss"
```

Expected: commit succeeds with loss integration staged.

## Task 6: Runtime Resolution And Trainer Wiring

**Files:**

- Modify: `src/detection/runtime.py`
- Modify: `src/trainers/metrics/recursive_detection.py`
- Test: `tests/test_latest_training_config_contract.py`

- [ ] **Step 1: Write failing runtime resolution test**

Add:

```python
def test_recursive_runtime_resolves_iou_gibbs_coord_softce_from_token_rows() -> None:
    from src.detection.runtime import resolve_recursive_detection_ce_runtime_cfg

    payload = _latest_payload()
    payload["objective"] = {
        "id": "recursive_detection_ce",
        "variant": "random_permutation_et_rmp_ce",
        "trie_support_weight": 2.0,
        "trie_balance_weight": 1.0,
        "state_weighting": "uniform_permutation",
        "normalization": "semantic_image_bucket_balanced",
        "coord_soft_ce": {
            "enabled": True,
            "target_distribution": "iou_gibbs_v0",
            "tau": 0.0090909091,
            "tau_source": "train_one_token_iou_median_v0",
            "weighting": "preserve_recursive_support_balance",
        },
    }
    cfg = LatestDetectionTrainingConfig.from_mapping(payload)

    runtime = resolve_recursive_detection_ce_runtime_cfg(cfg)
    group = cfg.token_rows.groups["coord_geometry"]

    assert runtime is not None
    assert runtime.coord_soft_ce is not None
    assert runtime.coord_soft_ce.tau == pytest.approx(0.0090909091)
    assert runtime.coord_soft_ce.coord_token_start == group.expected_start
    assert runtime.coord_soft_ce.coord_token_end == group.expected_end
```

- [ ] **Step 2: Run runtime test and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_recursive_runtime_resolves_iou_gibbs_coord_softce_from_token_rows -q
```

Expected: FAIL because `RecursiveDetectionCERuntimeConfig.coord_soft_ce` does not exist.

- [ ] **Step 3: Add runtime field and resolver**

In `src/detection/runtime.py`, import `CoordSoftTargetRuntimeConfig` and add:

```python
coord_soft_ce: CoordSoftTargetRuntimeConfig | None = None
```

to `RecursiveDetectionCERuntimeConfig`.

Add resolver helper:

```python
def _resolve_coord_soft_ce_runtime_cfg(
    training_config: LatestDetectionTrainingConfig,
) -> CoordSoftTargetRuntimeConfig | None:
    cfg = getattr(training_config.objective, "coord_soft_ce", None)
    if cfg is None or not bool(cfg.enabled):
        return None
    group = training_config.token_rows.groups.get("coord_geometry")
    if group is None:
        raise ValueError("objective.coord_soft_ce requires token_rows.groups.coord_geometry")
    start = int(group.expected_start)
    end = int(group.expected_end)
    if end - start + 1 != 1000:
        raise ValueError("objective.coord_soft_ce requires exactly 1000 coord token rows")
    return CoordSoftTargetRuntimeConfig(
        target_distribution=cfg.target_distribution,
        tau=float(cfg.tau),
        coord_token_start=start,
        coord_token_end=end,
        weighting=cfg.weighting,
        apply_to_multi_positive=cfg.apply_to_multi_positive,
    )
```

Pass `coord_soft_ce=_resolve_coord_soft_ce_runtime_cfg(training_config)` into the
runtime dataclass.

- [ ] **Step 4: Wire trainer mixin call**

In `src/trainers/metrics/recursive_detection.py`, read:

```python
coord_soft_ce = getattr(cfg, "coord_soft_ce", None)
```

and call:

```python
loss_result = compute_recursive_detection_ce_batch_loss(
    logits=logits,
    targets=tuple(recursive_targets),
    weights=weights,
    coord_soft_ce=coord_soft_ce,
)
```

Add flat reporter aliases:

```python
"recursive_detection_ce/coord_soft_ce/enabled": float(coord_soft_ce is not None),
"recursive_detection_ce/coord_soft_ce/tau": float(getattr(coord_soft_ce, "tau", 0.0) or 0.0),
```

- [ ] **Step 5: Run runtime tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_recursive_runtime_resolves_iou_gibbs_coord_softce_from_token_rows tests/test_recursive_detection_ce_loss_adapter.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit runtime wiring**

Run:

```bash
git add src/detection/runtime.py src/trainers/metrics/recursive_detection.py tests/test_latest_training_config_contract.py
git commit -m "feat: route coord softce through recursive runtime"
```

Expected: commit succeeds with runtime and trainer wiring staged.

## Task 7: A5/A6 Configs And Fairness Contract

**Files:**

- Create: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_tiny.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_tiny.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_ddp4_preflight.yaml`
- Test: `tests/test_latest_training_config_contract.py`

- [ ] **Step 1: Add failing deep inheritance test**

Add:

```python
def _drop_allowed_continuous_softce_deltas(mapping: dict[str, object]) -> dict[str, object]:
    clone = copy.deepcopy(mapping)
    clone.get("training", {}).pop("artifact_subdir", None)
    clone.get("training", {}).pop("run_name", None)
    clone.get("objective", {}).pop("coord_soft_ce", None)
    clone.pop("experiment", None)
    return clone


def test_a5_iou_gibbs_config_extends_support2_without_training_drift() -> None:
    base_path = REPO_ROOT / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
    a5_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml"
    )
    base = ConfigLoader.load_yaml_with_extends(base_path)
    a5 = ConfigLoader.load_yaml_with_extends(a5_path)
    base_cfg = LatestDetectionTrainingConfig.from_mapping(base)
    a5_cfg = LatestDetectionTrainingConfig.from_mapping(a5)

    assert _drop_allowed_continuous_softce_deltas(a5) == _drop_allowed_continuous_softce_deltas(base)
    assert base_cfg.objective.trie_support_weight == a5_cfg.objective.trie_support_weight == 2.0
    assert base_cfg.objective.trie_balance_weight == a5_cfg.objective.trie_balance_weight == 1.0
    assert getattr(base_cfg.objective, "coord_soft_ce", None) is None
    assert a5_cfg.objective.coord_soft_ce is not None
    assert a5_cfg.objective.coord_soft_ce.target_distribution == "iou_gibbs_v0"
    assert a5_cfg.experiment.surface == "ablation"
    assert a5_cfg.experiment.claim_scope == "none"


def test_a6_ciou_gibbs_config_extends_support2_without_training_drift() -> None:
    base_path = REPO_ROOT / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml"
    a6_path = (
        REPO_ROOT
        / "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml"
    )
    base = ConfigLoader.load_yaml_with_extends(base_path)
    a6 = ConfigLoader.load_yaml_with_extends(a6_path)
    base_cfg = LatestDetectionTrainingConfig.from_mapping(base)
    a6_cfg = LatestDetectionTrainingConfig.from_mapping(a6)

    assert _drop_allowed_continuous_softce_deltas(a6) == _drop_allowed_continuous_softce_deltas(base)
    assert base_cfg.objective.trie_support_weight == a6_cfg.objective.trie_support_weight == 2.0
    assert base_cfg.objective.trie_balance_weight == a6_cfg.objective.trie_balance_weight == 1.0
    assert getattr(base_cfg.objective, "coord_soft_ce", None) is None
    assert a6_cfg.objective.coord_soft_ce is not None
    assert a6_cfg.objective.coord_soft_ce.target_distribution == "ciou_gibbs_v0"
    assert a6_cfg.experiment.surface == "ablation"
    assert a6_cfg.experiment.claim_scope == "none"
```

If `LatestDetectionTrainingConfig` exposes `experiment` differently, assert the
same values on the resolved mapping.

- [ ] **Step 2: Run config test and verify failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_a5_iou_gibbs_config_extends_support2_without_training_drift tests/test_latest_training_config_contract.py::test_a6_ciou_gibbs_config_extends_support2_without_training_drift -q
```

Expected: FAIL because A5/A6 configs do not exist.

- [ ] **Step 3: Create A5 YAML**

Create `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml`:

```yaml
# A5 production-scale ablation candidate: A2 support+balance ET-RMP with
# IoU-Gibbs coordinate soft targets. This extends the A2 support2 launch config
# and changes only run identity, objective.coord_soft_ce, and non-claiming
# experiment metadata.
extends:
  - ./compact_full_support2.yaml

training:
  artifact_subdir: compact_full_et_rmp_ce_support2_a5_iou_gibbs_v0_tau00909_softce_preserve_sb_bsz16_4epoch_tokenrows_v2
  run_name: compact-full-et-rmp-ce-support2-a5-iou-gibbs-v0-tau00909-softce-preserve-sb-bsz16-4epoch-tokenrows-v2

objective:
  id: recursive_detection_ce
  variant: random_permutation_et_rmp_ce
  trie_support_weight: 2.0
  trie_balance_weight: 1.0
  state_weighting: uniform_permutation
  normalization: semantic_image_bucket_balanced
  coord_soft_ce:
    enabled: true
    target_distribution: iou_gibbs_v0
    tau: 0.0090909091
    tau_source: train_one_token_iou_median_v0
    weighting: preserve_recursive_support_balance
    replace_coord_hard_ce: true
    apply_to_multi_positive: support_mixture

experiment:
  surface: ablation
  ablation_id: A5-iou-gibbs-softce-support2
  claim_scope: none
```

- [ ] **Step 4: Create A6 YAML**

Create `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml`:

```yaml
# A6 production-scale ablation candidate: A2 support+balance ET-RMP with
# CIoU-Gibbs coordinate soft targets. This extends the A2 support2 launch config
# and changes only run identity, objective.coord_soft_ce, and non-claiming
# experiment metadata.
extends:
  - ./compact_full_support2.yaml

training:
  artifact_subdir: compact_full_et_rmp_ce_support2_a6_ciou_gibbs_v0_tau00909_softce_preserve_sb_bsz16_4epoch_tokenrows_v2
  run_name: compact-full-et-rmp-ce-support2-a6-ciou-gibbs-v0-tau00909-softce-preserve-sb-bsz16-4epoch-tokenrows-v2

objective:
  id: recursive_detection_ce
  variant: random_permutation_et_rmp_ce
  trie_support_weight: 2.0
  trie_balance_weight: 1.0
  state_weighting: uniform_permutation
  normalization: semantic_image_bucket_balanced
  coord_soft_ce:
    enabled: true
    target_distribution: ciou_gibbs_v0
    tau: 0.0090909091
    tau_source: train_one_token_iou_median_v0
    weighting: preserve_recursive_support_balance
    replace_coord_hard_ce: true
    apply_to_multi_positive: support_mixture

experiment:
  surface: ablation
  ablation_id: A6-ciou-gibbs-softce-support2
  claim_scope: none
```

- [ ] **Step 5: Create tiny smoke configs**

Create `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_tiny.yaml`:

```yaml
extends:
  - ../prod/compact_full_support2_iou_gibbs_softce_a5.yaml

training:
  artifact_subdir: smoke_compact_full_support2_iou_gibbs_softce_a5_tiny
  run_name: smoke-compact-full-support2-a5-iou-gibbs-softce-tiny
  max_steps: 1
  per_device_train_batch_size: 1
  effective_batch_size: 1
  per_device_eval_batch_size: 1

debug:
  enabled: true
  train_sample_limit: 1
  val_sample_limit: 1

experiment:
  surface: smoke
  ablation_id: A5-iou-gibbs-softce-support2
  claim_scope: none
```

Create `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_tiny.yaml` with the same runtime overrides but extending
`../prod/compact_full_support2_ciou_gibbs_softce_a6.yaml`, using:

```yaml
training:
  artifact_subdir: smoke_compact_full_support2_ciou_gibbs_softce_a6_tiny
  run_name: smoke-compact-full-support2-a6-ciou-gibbs-softce-tiny

experiment:
  surface: smoke
  ablation_id: A6-ciou-gibbs-softce-support2
  claim_scope: none
```

- [ ] **Step 6: Create 4-GPU preflight configs**

Create `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight.yaml` by mirroring the existing
latest recursive detection DDP preflight shape, but extend A5 instead of A2 and keep:

```yaml
training:
  artifact_subdir: smoke_compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight
  run_name: smoke-compact-full-support2-a5-iou-gibbs-softce-ddp4-preflight
  max_steps: 4
  eval_steps: 1

debug:
  enabled: true
  train_sample_limit: 512
  val_sample_limit: 32

experiment:
  surface: smoke
  ablation_id: A5-iou-gibbs-softce-support2
  claim_scope: none
```

Create `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_ddp4_preflight.yaml` analogously for A6, using A6 run/artifact names and `ablation_id: A6-ciou-gibbs-softce-support2`.

- [ ] **Step 7: Run config tests and parse check**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_training_config_contract.py::test_a5_iou_gibbs_config_extends_support2_without_training_drift tests/test_latest_training_config_contract.py::test_a6_ciou_gibbs_config_extends_support2_without_training_drift -q
conda run -n ms python -c "from pathlib import Path; from src.config.loader import ConfigLoader; from src.config.schema import LatestDetectionTrainingConfig; path=Path('configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml'); cfg=LatestDetectionTrainingConfig.from_mapping(ConfigLoader.load_yaml_with_extends(path)); print(cfg.training.run_name); print(cfg.model['model']); print(cfg.objective.coord_soft_ce.target_distribution); print(cfg.objective.coord_soft_ce.tau)"
conda run -n ms python -c "from pathlib import Path; from src.config.loader import ConfigLoader; from src.config.schema import LatestDetectionTrainingConfig; path=Path('configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml'); cfg=LatestDetectionTrainingConfig.from_mapping(ConfigLoader.load_yaml_with_extends(path)); print(cfg.training.run_name); print(cfg.model['model']); print(cfg.objective.coord_soft_ce.target_distribution); print(cfg.objective.coord_soft_ce.tau)"
```

Expected output includes:

```text
compact-full-et-rmp-ce-support2-a5-iou-gibbs-v0-tau00909-softce-preserve-sb-bsz16-4epoch-tokenrows-v2
model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
iou_gibbs_v0
0.0090909091
compact-full-et-rmp-ce-support2-a6-ciou-gibbs-v0-tau00909-softce-preserve-sb-bsz16-4epoch-tokenrows-v2
model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
ciou_gibbs_v0
0.0090909091
```

- [ ] **Step 8: Commit A5/A6 configs**

Run:

```bash
git add configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_tiny.yaml configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_tiny.yaml configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight.yaml configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_ddp4_preflight.yaml tests/test_latest_training_config_contract.py
git commit -m "config: add a5 a6 continuous coord softce configs"
```

Expected: commit succeeds with config and config test files staged.

## Task 8: Documentation And Catalog

**Files:**

- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/README.md`
- Modify: `docs/catalog.yaml`

- [ ] **Step 1: Update Stage-1 objective docs**

Add an A5 candidate entry:

```markdown
### A5 Candidate: IoU-Gibbs Coordinate SoftCE

`A5` is an unlaunched production-scale ablation candidate. It keeps the A2
`compact_full` support+balance ET-RMP training setup and replaces
coordinate-token hard targets with `iou_gibbs_v0` soft targets. The optimized
coordinate loss preserves A2 recursive support/balance coefficients while
adopting dense geometry-valid coordinate support:

```text
support = -logsumexp_{k in valid coord support} log p_full(k)
balance = -sum_k q(k) * (log p_full(k) - log valid_mass)
L_coord = 2.0 * support + 1.0 * balance
```

The soft target is:

```text
q_r(k | b) = softmax_k(-(1 - IoU(valid_replace(b, r, k), b)) / 0.0090909091)
```

The temperature is the median one-coordinate-token IoU loss measured on
`public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`. Fixed Gaussian
`sigma`/`truncate` coordinate softCE is deprecated for latest recursive
detection. Non-coordinate recursive CE behavior is unchanged. This candidate
requires the tau calibration artifact, target-shape audit, tiny smoke, and DDP4
preflight before production launch; it carries no AP claim yet.
```

Add an A6 candidate entry:

```markdown
### A6 Candidate: CIoU-Gibbs Coordinate SoftCE

`A6` is an unlaunched production-scale ablation candidate paired with A5. It
keeps the same A2 `compact_full` support+balance ET-RMP setup and uses the same
dense-support coordinate softCE machinery, but changes the coordinate energy
from `1 - IoU` to `1 - CIoU` via `ciou_gibbs_v0`. It shares the same initial
data-derived `tau=0.0090909091` unless a later calibration artifact justifies a
separate CIoU-specific value. This candidate requires the same risk audit,
tiny smoke, and DDP4 preflight gates as A5; it carries no AP claim yet.
```

- [ ] **Step 2: Update training README route**

Add:

```markdown
- A5 IoU-Gibbs coordinate softCE production-scale ablation candidate:
  `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml`
  (requires tau calibration artifact, target-shape audit, tiny smoke, and DDP4
  preflight before production launch).
- A6 CIoU-Gibbs coordinate softCE production-scale ablation candidate:
  `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml`
  (paired with A5 for 4-GPU/4-GPU production comparison after smoke and risk
  audit gates).
```

- [ ] **Step 3: Update catalog with existing shape**

Register under `config_surfaces.training`:

```yaml
    - id: stage1_latest_compact_detection_iou_gibbs_coord_softce_a5
      status: implemented-unlaunched-ablation
      config: configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml
      authoring_snippets: configs/_shared/latest_detection/
      parser: src/config/schema.py::LatestDetectionTrainingConfig
      runtime: src/detection/runtime.py
      notes: "A5 production-scale ablation candidate: A2 support2 plus iou_gibbs_v0 coordinate soft targets preserving recursive support/balance coefficients; claim_scope remains none until measured."
    - id: stage1_latest_compact_detection_ciou_gibbs_coord_softce_a6
      status: implemented-unlaunched-ablation
      config: configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml
      authoring_snippets: configs/_shared/latest_detection/
      parser: src/config/schema.py::LatestDetectionTrainingConfig
      runtime: src/detection/runtime.py
      notes: "A6 production-scale ablation candidate: A2 support2 plus ciou_gibbs_v0 coordinate soft targets preserving recursive support/balance coefficients; intended to run beside A5 on a separate 4-GPU slice."
```

- [ ] **Step 4: Verify docs mention exact config**

Run:

```bash
conda run -n ms python -c "import yaml; yaml.safe_load(open('docs/catalog.yaml'))"
rg -n "stage1_latest_compact_detection_iou_gibbs_coord_softce_a5|stage1_latest_compact_detection_ciou_gibbs_coord_softce_a6|compact_full_support2_iou_gibbs_softce_a5|compact_full_support2_ciou_gibbs_softce_a6|A5.*IoU-Gibbs|A6.*CIoU-Gibbs|iou_gibbs_v0|ciou_gibbs_v0" docs configs/stage1/recursive_detection_ce_latest
```

Expected: hits in docs, catalog, prod config, and smoke configs.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add docs/training/STAGE1_OBJECTIVE.md docs/training/README.md docs/catalog.yaml
git commit -m "docs: register a5 a6 continuous coord softce"
```

Expected: commit succeeds with docs/catalog staged.

## Task 9: Verification And Launch Gates

**Files:**

- Read: all modified files
- Runtime-only symlinks, not staged: `model_cache`, `public_data`

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_iou_gibbs_coord_stats.py \
  tests/test_iou_gibbs_coord_softce.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_prefix_rollin_dataset_alignment.py \
  tests/test_latest_training_config_contract.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Create ignored worktree symlinks for heavy roots**

Use the `full-pipeline-smoke` skill path strategy: preserve authored config
paths and add local symlinks when ignored data/model roots are absent in the
worktree.

Run from the worktree root:

```bash
test -e model_cache || ln -s /data/home/xiaoyan/AIteam/data/CoordExp/model_cache model_cache
mkdir -p public_data/coco
test -e public_data/coco/rescale_32_1024_bbox_max60 || ln -s /data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60 public_data/coco/rescale_32_1024_bbox_max60
test -f public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
test -f model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp/config.json
```

Expected: all `test` commands pass. Do not stage the symlinks.

- [ ] **Step 3: Run model-innovation risk audit before smoke**

Use `model-innovation-risk-audit` over A5/A6 before smoke launch. Required
audit surfaces:

```text
config schema -> A5/A6 materialized configs -> target metadata -> helper math
-> recursive loss -> trainer metrics -> smoke configs -> artifact contracts
```

Launch-gate rule:

- all P0 findings fixed;
- all P1 findings fixed or explicitly accepted by the user;
- P2 findings either fixed or tracked in this plan/final report.

- [ ] **Step 4: Run parse smoke without heredocs**

Run:

```bash
conda run -n ms python -c "from pathlib import Path; from src.config.loader import ConfigLoader; from src.config.schema import LatestDetectionTrainingConfig; paths=['configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml','configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml','configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml']; [print(path, LatestDetectionTrainingConfig.from_mapping(ConfigLoader.load_yaml_with_extends(Path(path))).objective.variant, getattr(getattr(LatestDetectionTrainingConfig.from_mapping(ConfigLoader.load_yaml_with_extends(Path(path))).objective, 'coord_soft_ce', None), 'target_distribution', None)) for path in paths]"
```

Expected:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml random_permutation_et_rmp_ce None
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml random_permutation_et_rmp_ce iou_gibbs_v0
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml random_permutation_et_rmp_ce ciou_gibbs_v0
```

- [x] **Step 5: Run tiny trainer smokes for A5 and A6**

Run:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_tiny.yaml
PYTHONPATH=. conda run -n ms python -m src.sft --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_tiny.yaml
```

Expected for both:

- finite `loss/recursive_detection_ce`;
- nonzero `recursive_detection_ce/coord_soft_ce/enabled`;
- nonzero coordinate soft-target diagnostic coverage;
- finite support, balance, weighted, entropy, peak, and KL-like diagnostics;
- no missing-geometry fail-fast errors;
- normal run metadata and manifest files.

- [x] **Step 6: Run 4-GPU DDP preflights for A5 and A6**

Use direct `torchrun -m src.sft` for this latest recursive detection surface.
The production plan is 4 GPUs for A5 and 4 GPUs for A6, not one 8-GPU job.
For preflight, run each config on a 4-GPU slice:

```bash
PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n ms torchrun --master_port=29605 --nproc_per_node=4 -m src.sft --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_iou_gibbs_softce_a5_ddp4_preflight.yaml
PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n ms torchrun --master_port=29606 --nproc_per_node=4 -m src.sft --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_support2_ciou_gibbs_softce_a6_ddp4_preflight.yaml
```

Expected for both:

- all ranks enter and exit;
- finite recursive CE loss;
- coordinate softCE diagnostics present on rank-aggregated metrics;
- `support_mixture` is present;
- no missing geometry;
- eval step completes;
- rank heartbeats are present;
- manifests, resolved config, and output root are recorded in the preflight
  report.

- [x] **Step 7: Record smoke/risk-audit evidence**

In the final implementation report, include:

- symlink targets used for `model_cache` and `public_data`;
- model-innovation-risk-audit findings and disposition;
- model-diagnosis launch gate summary from tiny/DDP4 trends: objective-active
  composition metrics, finite loss trend, no malformed-output or artifact
  contradiction visible in the smoke scope, and named residual risks;
- A5/A6 tiny smoke output roots and key metrics;
- A5/A6 DDP4 preflight output roots and key metrics;
- confirmation that final production launch should use two 4-GPU jobs:
  A5 on one GPU slice, A6 on another.

Smoke/model-diagnosis evidence recorded on 2026-05-11:

- symlinks:
  - `model_cache -> /data/home/xiaoyan/AIteam/data/CoordExp/model_cache`
  - `public_data/coco/rescale_32_1024_bbox_max60 -> /data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60`
  - `public_data/coco/rescale_32_1024_bbox -> /data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox`
  - `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy -> /data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy`
- A5 tiny:
  `temp/recursive_detection_ce_latest/output/compact_full_iou_gibbs_softce_a5_tiny/smoke-compact-full-iou-gibbs-softce-a5-tiny/v0-20260511-171218`
  - `loss/recursive_detection_ce=19.93226814`
  - `recursive_detection_ce/coord_soft_ce/enabled=1.0`
  - `weighted_loss=34.11478995`, `support_loss=12.87118289`, `balance_loss=8.37242417`
  - `candidate_count=1.44444444`, `support_bin_count=609.05555556`, `support_mixture=0.16666667`
- A6 tiny:
  `temp/recursive_detection_ce_latest/output/compact_full_ciou_gibbs_softce_a6_tiny/smoke-compact-full-ciou-gibbs-softce-a6-tiny/v0-20260511-171258`
  - `loss/recursive_detection_ce=19.93226814`
  - `recursive_detection_ce/coord_soft_ce/enabled=1.0`
  - `weighted_loss=34.11479011`, `support_loss=12.87118289`, `balance_loss=8.37242433`
  - `candidate_count=1.44444444`, `support_bin_count=609.05555556`, `support_mixture=0.16666667`
- A5 DDP4 preflight:
  `temp/recursive_detection_ce_latest/output/compact_full_iou_gibbs_softce_a5_ddp4_preflight/smoke-compact-full-iou-gibbs-softce-a5-ddp4-preflight/v0-20260511-171459`
  - final train `loss/recursive_detection_ce=16.06588364`
  - final eval `eval_loss/recursive_detection_ce=16.40919495`
  - train/eval `coord_soft_ce/enabled=1.0`
  - final train `weighted_loss=31.80255890`, `candidate_count=1.53905976`, `support_bin_count=644.43377686`, `support_mixture=0.13986725`
  - final eval `weighted_loss=31.23566628`, `support_mixture=0.09474511`
- A6 DDP4 preflight:
  `temp/recursive_detection_ce_latest/output/compact_full_ciou_gibbs_softce_a6_ddp4_preflight/smoke-compact-full-ciou-gibbs-softce-a6-ddp4-preflight/v0-20260511-171440`
  - final train `loss/recursive_detection_ce=16.05656433`
  - final eval `eval_loss/recursive_detection_ce=16.38127518`
  - train/eval `coord_soft_ce/enabled=1.0`
  - final train `weighted_loss=31.78451538`, `candidate_count=1.53905976`, `support_bin_count=644.43377686`, `support_mixture=0.13986725`
  - final eval `weighted_loss=31.18028831`, `support_mixture=0.09474511`
- A2 pure-CE prodlike anchor:
  `temp/recursive_detection_ce_latest/output/compact_full_prodlike_single_gpu/smoke-compact-full-et-rmp-ce-prodlike-single-gpu/v0-20260511-172309`
  - `loss/recursive_detection_ce=12.28367639`
  - `recursive_detection_ce/coord_soft_ce/config_enabled=0.0`
  - `recursive_detection_ce/support_loss=8.24677403`
  - `coord_fraction=0.39652534`, `trie_multi_positive_fraction=0.16280275`
- artifact note: latest recursive CE wrote `resolved_config.json`,
  `runtime_env.json`, `effective_runtime.json`, data provenance,
  `experiment_manifest.json`, `run_metadata.json`, heartbeat, and
  `logging.jsonl`; it did not emit `pipeline_manifest.json` on these smokes.
  Treat that as a P2 artifact-contract gap, not a softCE loss blocker.
- launch-node filesystem note: A5/A6 inherit the existing production
  `output_root: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest`
  convention from A2/A3/A4. On this smoke node `/data/CoordExp` is absent,
  while the smoke overlay writes under `temp/`. Before production launch,
  confirm the launch node has this output root or create the standard
  `/data/CoordExp` mapping there; do not change A5/A6 YAML paths unless the
  production filesystem convention has changed for all latest recursive CE
  runs.
- broader pre-launch tests on 2026-05-11:
  - fixed an adjacent legacy bbox-loss import regression:
    `src/trainers/metrics/bbox_losses.py` now imports
    `get_coord_token_ids` for its tokenizer fallback path.
  - `conda run -n ms python -m pytest tests/test_iou_gibbs_coord_stats.py tests/test_iou_gibbs_coord_softce.py tests/test_latest_training_config_contract.py tests/test_training_config_strict_unknown_keys.py tests/test_legacy_config_contract.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_target_builder.py tests/test_coord_softce_w1_loss.py tests/test_coord_soft_ce_w1_collective_guard.py tests/coord_tokens/test_soft_ce_w1.py tests/test_bbox_size_aux_loss.py tests/test_bbox_format_switch.py -q`
    passed: `265 passed in 2.40s`.
  - `conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py tests/test_stage2_ab_profile_leaf_contract.py tests/test_training_runtime_profile.py tests/test_stage1_metric_key_parity.py -q`
    passed: `139 passed in 8.20s`.
  - `py_compile` passed for the touched analysis, detection, schema, recursive
    metrics, legacy coord-softCE, and bbox-loss Python modules.

Production launch shape after user approval:

```bash
PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n ms torchrun --master_port=29605 --nproc_per_node=4 -m src.sft --config configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_iou_gibbs_softce_a5.yaml
PYTHONPATH=. OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COORDEXP_TRAIN_HEARTBEAT=1 CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n ms torchrun --master_port=29606 --nproc_per_node=4 -m src.sft --config configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2_ciou_gibbs_softce_a6.yaml
```

Production launch state on 2026-05-11:

- Created launch-node mapping:
  `/data/CoordExp -> /data/home/xiaoyan/AIteam/data/CoordExp`.
- Started tmux sessions:
  - `coordexp_a5_iou_gibbs_prod`, GPUs `0,1,2,3`, port `29605`,
    log `temp/prod_launch/a5_iou_gibbs_prod_20260511_173916.log`.
  - `coordexp_a6_ciou_gibbs_prod`, GPUs `4,5,6,7`, port `29606`,
    log `temp/prod_launch/a6_ciou_gibbs_prod_20260511_173916.log`.
- Run roots:
  - A5:
    `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_iou_gibbs_softce_a5_support2_bsz16_4gpu_4epoch_tokenrows_v2/compact-full-et-rmp-iou-gibbs-softce-a5-support2-bsz16-4gpu-4epoch-tokenrows-v2/v0-20260511-173957`
  - A6:
    `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ciou_gibbs_softce_a6_support2_bsz16_4gpu_4epoch_tokenrows_v2/compact-full-et-rmp-ciou-gibbs-softce-a6-support2-bsz16-4gpu-4epoch-tokenrows-v2/v0-20260511-173957`
- First-step gate passed for both A5 and A6:
  `first_step_end` and `global_step/max_steps = 1/3664` were recorded,
  artifacts were written, and `coord_soft_ce/enabled=1.0` appeared in
  `logging.jsonl` for both runs.
- Continued verification after launch:
  - An expanded non-GPU regression sweep over coord-token modules, recursive
    CE wiring, manifests, encoded-cache runtime config, Stage-1 static-packing
    config, runtime integration, token metrics, tokenizer canaries, and dataset
    runtime contracts passed: `236 passed, 1 skipped in 3.83s`.
  - The non-failing remainder of `tests/test_training_config_hierarchy_contract.py`
    passed: `2 passed, 2 deselected in 0.45s`.
  - The two deselected hierarchy-contract failures are pre-existing A3/A4
    wrapper authoring checks, not A5/A6 softCE failures:
    `test_stage1_canonical_profiles_load_under_current_hierarchy` and
    `test_canonical_non_smoke_leaves_author_raw_run_identity_fields`.
    They point at Stage-1 non-smoke wrapper configs that inherit run identity
    fields instead of authoring them directly. Because A3/A4 bsz8 jobs were
    already running on another node, those configs were not changed during the
    A5/A6 launch.
  - Fixed compact detection suffix stripping so strict compact parsing trims
    whitespace left before terminal chat tokens; targeted parser contract test
    passed.
  - Fixed repo-local `scripts` package isolation for full-suite collection:
    added package markers for `scripts/` and `scripts/analysis/`, hardened
    `tests/conftest.py` against external editable `scripts` packages, and
    changed the public-data converter test to import its helper by explicit
    file location. Combined regression passed:
    `tests/coord_tokens/test_converter.py tests/test_manual_audit_reviewer.py`
    -> `7 passed`.
  - Public-data provenance manifests passed after adding the worktree symlinks:
    `tests/test_public_data_provenance_manifests.py` -> `6 passed in 1.65s`.
  - Full hidden-GPU suite command:
    `CUDA_VISIBLE_DEVICES= conda run -n ms python -m pytest tests -q -k 'not stage1_canonical_profiles_load_under_current_hierarchy and not canonical_non_smoke_leaves_author_raw_run_identity_fields'`.
    Initial post-launch result: `2090 passed, 8 skipped, 2 deselected, 16
    failed, 21 warnings in 229.80s`. After targeted residual fixes, the fresh
    whole-suite rerun produced `2095 passed, 8 skipped, 2 deselected, 11
    failed, 21 warnings in 229.24s`.
  - Continued residual triage fixed three narrow non-A5 regressions without
    changing A5/A6 semantics:
    restartable checkpoint preflight now references the public
    `save_model_only=true` setting in incomplete-checkpoint errors;
    coord-only loss-gradient monitoring no longer includes `text_gate`; and
    Stage-2 AB DDP phase config defaults cleanly for minimal executor mixins
    without `_ab_channel_b_get`.
  - Verified targeted fixes:
    `tests/test_checkpoint_weight_only_policy.py::test_restartable_checkpoint_preflight_rejects_artifact_only_sidecar`
    -> `1 passed`;
    `tests/test_loss_gradient_monitor.py::test_build_stage2_coord_monitor_terms_from_pipeline_excludes_text_terms`
    -> `1 passed`;
    `tests/test_stage2_ab_channel_a_pack_count_skew_barrier.py::test_stage2_ab_channel_a_calls_barrier_on_final_pack`
    -> `1 passed`.
  - Re-running the residual file group after those fixes reduced it to
    `11 failed, 54 passed, 1 warning in 8.12s`. Remaining failures are:
    raw coordjson parse inventory, duplication-collapse bootstrap requiring
    an old missing checkpoint path, max-pixels expectation mismatch, blanket
    exception policy inventory, Stage-2 prompt-alignment source-string check,
    and Stage-2 post-rollout bin-packing expectation drift.
  - `py_compile` passed for `src/trainers/final_checkpoint.py`,
    `src/trainers/monitoring/loss_gradient_monitor.py`, and
    `src/trainers/stage2_coordination.py`.
  - Live heartbeats after this sweep reached A5 `global_step=84` at
    `2026-05-11T18:19:43Z` and A6 `global_step=81` at
    `2026-05-11T18:19:18Z`; both tmux sessions remained present.

- [x] **Step 8: Inspect git diff**

Run:

```bash
git diff --stat
git status --short
```

Observed on 2026-05-11: `git diff --stat` showed only intended code, config,
test, doc, and diagnostic surfaces for the A5/A6 coordinate-softCE change.
Runtime symlinks for `model_cache` and the shared COCO data root remain ignored
and unstaged.

## Rollback Plan

If tests show loss instability or target metadata mismatch:

1. Disable only the A5 configs by not launching
   `compact_full_support2_iou_gibbs_softce_a5.yaml`.
2. Keep `coord_soft_ce` disabled-by-absence behavior for all existing configs.
3. Do not revert A2/support2 or prefix-rollin A3/A4 configs.
4. If metadata attachment breaks target construction, revert the target metadata
   commit and the loss/runtime commits together because the loss depends on
   `coord_soft_targets`.
5. Do not delete legacy Gaussian modules as part of rollback; they are
   compatibility surfaces.

## Self-Review Checklist

- Spec coverage: calibration, schema, deprecation, target metadata, IoU-Gibbs helper, recursive loss integration, runtime wiring, A5 configs, smoke gates, diagnostics, and docs are each mapped to tasks.
- No hand-tuned sigma/truncate/floor/cap policy appears in the latest implementation tasks.
- A5 keeps A2 training setup unchanged except run identity, `objective.coord_soft_ce`, and non-claiming experiment metadata.
- Coordinate optimized loss preserves A2 support2/balance1 coefficients while
  adopting dense geometry-valid coordinate support.
- Full-vocab soft target CE is specified so non-coordinate probability mass remains penalized.
- Multi-positive coordinate positions use support-mixture geometry metadata only after same-slot validation.
- The plan includes tau calibration and target-shape audit before production-scale training.
- Verification does not launch production training unless the user explicitly requests it after implementation and smoke gates.
