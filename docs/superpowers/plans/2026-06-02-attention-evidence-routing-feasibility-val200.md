# Attention Evidence Routing Feasibility, Val200, And Train Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first attention-focused mechanism lane for checkpoint-3664: a feasibility gate, a sharded `val200` observational attention atlas for missed-GT evidence routing, and a training-set teacher-forced anchor slice.

**Architecture:** Add a new analysis module that reuses Lane C prepared compact-full forward examples and Lane D compact role positions, then adds visual-token region mapping and region-level attention aggregation. The first phase is observational only: it records stop/bind decision context, candidate regions, and attention mass by layer/head/query role without launching training or causal intervention. `val200` remains the rollout-behavior surface; `train200_teacher_forced_anchor` is a separate stronger-control slice for seen-sample internal routing, not a rollout-recall claim.

**Tech Stack:** Python stdlib, PyYAML, PyTorch/Transformers Qwen3-VL forward passes, existing CoordExp Lane C/D helpers, pytest, tmux shell launcher.

---

## Scope

Primary checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Primary analysis root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200
```

New lane output root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing
```

Training anchor output root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_train200/attention_evidence_routing_teacher_forced_anchor
```

Input artifacts:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/rollout_anatomy/per_row.jsonl
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/x1_basin_attribution/per_case.jsonl
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/selected_cases.jsonl
/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/pred_token_trace.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
```

Evidence stance:

```text
primary objective: maximize labeled-GT recall and plausible visible-object coverage
hard negatives: same_desc_iou_gt_0p95_duplicate, format_or_geometry_invalid_extra
soft/neutral extras: other_extra_prediction
```

This plan must not start production training. It must not claim that attention visualizations alone prove a mechanism. It must not reintroduce complex FP taxonomy beyond the same-desc IoU>`0.95` duplicate guardrail.

## File Structure

Create:

```text
configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml
configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml
scripts/analysis/run_autoreg_attention_evidence_routing.py
scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
src/analysis/autoreg_attention_evidence_routing.py
tests/test_autoreg_attention_evidence_routing.py
```

Read/import:

```text
src/analysis/hard_ce_coord_logit_locality.py
src/analysis/autoreg_hidden_state_probe.py
src/analysis/autoreg_object_rollout.py
src/datasets/geometry.py
```

Do not modify:

```text
modeling_qwen3_vl.py
src/infer/*
src/eval/*
training code
```

## Artifact Contract

The new lane writes:

```text
attention_evidence_routing/shards_manifest.json
attention_evidence_routing/shards/shard_000-of-008/selected_cases.jsonl
attention_evidence_routing/shards/shard_000-of-008/candidate_region_rows.jsonl
attention_evidence_routing/shards/shard_000-of-008/feasibility_rows.jsonl
attention_evidence_routing/shards/shard_000-of-008/attention_region_rows.jsonl
attention_evidence_routing/shards/shard_000-of-008/decision_context_rows.jsonl
attention_evidence_routing/shards/shard_000-of-008/summary.json
attention_evidence_routing/selected_cases.jsonl
attention_evidence_routing/candidate_region_rows.jsonl
attention_evidence_routing/feasibility_rows.jsonl
attention_evidence_routing/attention_region_rows.jsonl
attention_evidence_routing/decision_context_rows.jsonl
attention_evidence_routing/summary.json
attention_evidence_routing/merge_summary.json
attention_evidence_routing/report.md
```

Shard selection must use `source_line_idx % num_shards`, so all cases from one image stay on one GPU.

## Task 1: CPU Schema, Sharding, And Simple Duplication Guardrail

**Files:**

- Create: `src/analysis/autoreg_attention_evidence_routing.py`
- Create: `tests/test_autoreg_attention_evidence_routing.py`

- [ ] **Step 1: Write failing tests for sharding and duplicate classification**

Add this to `tests/test_autoreg_attention_evidence_routing.py`:

```python
from __future__ import annotations

import pytest

from src.analysis.autoreg_attention_evidence_routing import (
    attention_record_selected,
    attention_shard_label,
    box_iou_xyxy,
    classify_extra_prediction,
    normalize_attention_shard,
)


def test_attention_shard_selection_keeps_images_together() -> None:
    assert attention_shard_label(0, 8) == "shard_000-of-008"
    assert attention_record_selected(16, shard_index=0, num_shards=8)
    assert not attention_record_selected(17, shard_index=0, num_shards=8)


def test_normalize_attention_shard_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="provided together"):
        normalize_attention_shard(shard_index=0, num_shards=None)
    with pytest.raises(ValueError, match="num_shards"):
        normalize_attention_shard(shard_index=0, num_shards=0)
    with pytest.raises(ValueError, match="shard_index"):
        normalize_attention_shard(shard_index=8, num_shards=8)


def test_box_iou_xyxy_uses_closed_open_area() -> None:
    assert box_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)
    assert box_iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0)


def test_classify_extra_prediction_only_marks_same_desc_iou_gt_0p95_duplicate() -> None:
    previous = [
        {"desc": "chair", "bbox_xyxy": [10, 10, 100, 100]},
        {"desc": "person", "bbox_xyxy": [300, 300, 500, 900]},
    ]
    assert (
        classify_extra_prediction(
            {"desc": "chair", "bbox_xyxy": [11, 10, 100, 100]},
            previous,
        )
        == "same_desc_iou_gt_0p95_duplicate"
    )
    assert (
        classify_extra_prediction(
            {"desc": "table", "bbox_xyxy": [10, 10, 100, 100]},
            previous,
        )
        == "other_extra_prediction"
    )
    assert (
        classify_extra_prediction(
            {"desc": "chair", "bbox_xyxy": [200, 200, 280, 280]},
            previous,
        )
        == "other_extra_prediction"
    )
    assert (
        classify_extra_prediction(
            {"desc": "chair", "bbox_xyxy": [100, 10, 10, 100]},
            previous,
        )
        == "format_or_geometry_invalid_extra"
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest \
  tests/test_autoreg_attention_evidence_routing.py::test_attention_shard_selection_keeps_images_together \
  tests/test_autoreg_attention_evidence_routing.py::test_classify_extra_prediction_only_marks_same_desc_iou_gt_0p95_duplicate \
  -q
```

Expected: import failure because `src.analysis.autoreg_attention_evidence_routing` does not exist.

- [ ] **Step 3: Implement CPU helpers**

Create `src/analysis/autoreg_attention_evidence_routing.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ATTENTION_STAGES = (
    "select_cases",
    "feasibility",
    "attention_atlas",
    "merge",
    "report",
)

ATTENTION_QUERY_ROLES = (
    "final_generated_prefix_state",
    "row_start",
    "desc_end",
    "box_start",
    "pre_x1",
)

ATTENTION_REGION_KINDS = (
    "target_gt",
    "same_desc_gt",
    "emitted_gt",
    "same_desc_iou_gt_0p95_duplicate_candidate",
    "context_ring",
    "far_background",
)


@dataclass(frozen=True)
class AttentionPaths:
    artifact_root: Path
    checkpoint: Path
    dataset_jsonl: Path
    lane_a_rollout_root: Path
    lane_c_per_case: Path
    lane_c_study_config: Path
    lane_d_selected_cases: Path
    self_rollout_root: Path


@dataclass(frozen=True)
class AttentionSelectionConfig:
    sample_limit: int
    max_cases: int
    prefer_prefix_mode: str
    min_remaining_gt: int


@dataclass(frozen=True)
class AttentionRegionConfig:
    context_expansion_norm1000: int
    duplicate_iou_threshold: float


@dataclass(frozen=True)
class AttentionExecutionConfig:
    batch_size: int
    attn_implementation: str
    torch_dtype: str
    max_feasibility_cases: int


@dataclass(frozen=True)
class AttentionConfig:
    paths: AttentionPaths
    selection: AttentionSelectionConfig
    regions: AttentionRegionConfig
    execution: AttentionExecutionConfig


def attention_shard_label(shard_index: int, num_shards: int) -> str:
    return f"shard_{int(shard_index):03d}-of-{int(num_shards):03d}"


def normalize_attention_shard(
    *, shard_index: int | None, num_shards: int | None
) -> tuple[int | None, int | None, str | None]:
    if shard_index is None and num_shards is None:
        return None, None, None
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together")
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive")
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")
    return int(shard_index), int(num_shards), attention_shard_label(shard_index, num_shards)


def attention_record_selected(
    source_line_idx: int, *, shard_index: int | None, num_shards: int | None
) -> bool:
    shard_index, num_shards, _ = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if shard_index is None or num_shards is None:
        return True
    return int(source_line_idx) % int(num_shards) == int(shard_index)


def _valid_box_xyxy(box: Sequence[object]) -> bool:
    if len(box) != 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in box]
    return x2 > x1 and y2 > y1


def box_iou_xyxy(left: Sequence[object], right: Sequence[object]) -> float:
    if not _valid_box_xyxy(left) or not _valid_box_xyxy(right):
        return 0.0
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _desc(value: Mapping[str, Any]) -> str:
    return str(value.get("desc", "")).strip().lower()


def _box(value: Mapping[str, Any]) -> Sequence[object]:
    raw = value.get("bbox_xyxy", value.get("bbox_2d", value.get("pred_points")))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return raw


def classify_extra_prediction(
    prediction: Mapping[str, Any],
    previous_predictions: Sequence[Mapping[str, Any]],
    *,
    duplicate_iou_threshold: float = 0.95,
) -> str:
    pred_box = _box(prediction)
    if not _valid_box_xyxy(pred_box):
        return "format_or_geometry_invalid_extra"
    pred_desc = _desc(prediction)
    for previous in previous_predictions:
        if pred_desc and pred_desc == _desc(previous):
            if box_iou_xyxy(pred_box, _box(previous)) > float(duplicate_iou_threshold):
                return "same_desc_iou_gt_0p95_duplicate"
    return "other_extra_prediction"
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
```

Expected: all tests in this file pass.

## Task 2: Config Loader, Dry Run, And CLI Skeleton

**Files:**

- Modify: `src/analysis/autoreg_attention_evidence_routing.py`
- Create: `configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml`
- Create: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Modify: `tests/test_autoreg_attention_evidence_routing.py`

- [ ] **Step 1: Add config loader tests**

Append:

```python
from pathlib import Path

from src.analysis.autoreg_attention_evidence_routing import (
    build_attention_dry_run_plan,
    load_attention_config,
)


def _write_attention_config(path: Path, *, artifact_root: Path | None = None) -> None:
    root = path.parent / "attention_root" if artifact_root is None else artifact_root
    path.write_text(
        f"""
paths:
  artifact_root: {root}
  checkpoint: {path.parent / "checkpoint"}
  dataset_jsonl: {path.parent / "val.coord.jsonl"}
  lane_a_rollout_root: {path.parent / "rollout_anatomy"}
  lane_c_per_case: {path.parent / "x1_basin_attribution" / "per_case.jsonl"}
  lane_c_study_config: {path.parent / "lane_c.yaml"}
  lane_d_selected_cases: {path.parent / "hidden_state_probe" / "selected_cases.jsonl"}
  self_rollout_root: {path.parent / "self_rollout"}
selection:
  sample_limit: 200
  max_cases: 512
  prefer_prefix_mode: self_prefix
  min_remaining_gt: 1
regions:
  context_expansion_norm1000: 64
  duplicate_iou_threshold: 0.95
execution:
  batch_size: 1
  attn_implementation: eager
  torch_dtype: bfloat16
  max_feasibility_cases: 4
""".lstrip(),
        encoding="utf-8",
    )


def test_load_attention_config_and_dry_run_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "attention.yaml"
    _write_attention_config(config_path)
    config = load_attention_config(config_path)
    assert config.paths.artifact_root == tmp_path / "attention_root"
    assert config.selection.sample_limit == 200
    assert config.regions.duplicate_iou_threshold == 0.95
    plan = build_attention_dry_run_plan(
        config,
        stages=("select_cases", "feasibility"),
        shard_index=0,
        num_shards=8,
    )
    assert plan["artifact_root"] == str(tmp_path / "attention_root")
    assert plan["shard_label"] == "shard_000-of-008"
    assert plan["stages"] == ["select_cases", "feasibility"]
```

- [ ] **Step 2: Implement config loader and dry run**

Add imports:

```python
import json

import yaml
```

Add helpers:

```python
def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _required_path(payload: Mapping[str, Any], key: str, *, base: Path) -> Path:
    raw = payload.get(key)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"missing required path: {key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def _required_int(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = int(payload.get(key))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def load_attention_config(path: Path | str) -> AttentionConfig:
    config_path = Path(path).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root if not config_path.is_absolute() else Path("/")
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"attention config must be a mapping: {config_path}")
    paths = _required_mapping(payload, "paths")
    selection = _required_mapping(payload, "selection")
    regions = _required_mapping(payload, "regions")
    execution = _required_mapping(payload, "execution")
    return AttentionConfig(
        paths=AttentionPaths(
            artifact_root=_required_path(paths, "artifact_root", base=base),
            checkpoint=_required_path(paths, "checkpoint", base=base),
            dataset_jsonl=_required_path(paths, "dataset_jsonl", base=base),
            lane_a_rollout_root=_required_path(paths, "lane_a_rollout_root", base=base),
            lane_c_per_case=_required_path(paths, "lane_c_per_case", base=base),
            lane_c_study_config=_required_path(paths, "lane_c_study_config", base=base),
            lane_d_selected_cases=_required_path(paths, "lane_d_selected_cases", base=base),
            self_rollout_root=_required_path(paths, "self_rollout_root", base=base),
        ),
        selection=AttentionSelectionConfig(
            sample_limit=_required_int(selection, "sample_limit", minimum=1),
            max_cases=_required_int(selection, "max_cases", minimum=1),
            prefer_prefix_mode=_required_str(selection, "prefer_prefix_mode"),
            min_remaining_gt=_required_int(selection, "min_remaining_gt", minimum=0),
        ),
        regions=AttentionRegionConfig(
            context_expansion_norm1000=_required_int(
                regions,
                "context_expansion_norm1000",
                minimum=0,
            ),
            duplicate_iou_threshold=float(regions.get("duplicate_iou_threshold", 0.95)),
        ),
        execution=AttentionExecutionConfig(
            batch_size=_required_int(execution, "batch_size", minimum=1),
            attn_implementation=_required_str(execution, "attn_implementation"),
            torch_dtype=_required_str(execution, "torch_dtype"),
            max_feasibility_cases=_required_int(
                execution,
                "max_feasibility_cases",
                minimum=1,
            ),
        ),
    )


def build_attention_dry_run_plan(
    config: AttentionConfig,
    *,
    stages: Sequence[str],
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    normalized_shard_index, normalized_num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    unknown = [stage for stage in stages if stage not in ATTENTION_STAGES]
    if unknown:
        raise ValueError(f"unknown attention stage(s): {unknown}")
    return {
        "artifact_root": str(config.paths.artifact_root),
        "stages": list(stages),
        "shard_index": normalized_shard_index,
        "num_shards": normalized_num_shards,
        "shard_label": shard_label,
        "sample_limit": config.selection.sample_limit,
        "max_cases": config.selection.max_cases,
        "attn_implementation": config.execution.attn_implementation,
    }
```

Add a helper for reusing the frozen Lane C config safely:

```python
def build_attention_lane_c_config(config: AttentionConfig) -> Any:
    from src.analysis.hard_ce_coord_logit_locality import load_study_config

    lane_c_config = load_study_config(config.paths.lane_c_study_config)
    return replace(
        lane_c_config,
        paths=replace(
            lane_c_config.paths,
            dataset_jsonl=config.paths.dataset_jsonl,
        ),
        model=replace(
            lane_c_config.model,
            attn_implementation=config.execution.attn_implementation,
            torch_dtype=config.execution.torch_dtype,
        ),
    )
```

This helper is required because the reused Lane C `StudyConfig`, `StudyPaths`,
and `StudyModelConfig` are frozen dataclasses. It also prevents the
`train200_teacher_forced_anchor` config from accidentally reading the `val200`
dataset declared in the base Lane C YAML.

- [ ] **Step 3: Create checkpoint-3664 YAML**

Create `configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml`:

```yaml
paths:
  artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing
  checkpoint: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
  dataset_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  lane_a_rollout_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/rollout_anatomy
  lane_c_per_case: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/x1_basin_attribution/per_case.jsonl
  lane_c_study_config: /data/CoordExp/configs/analysis/hard_ce_coord_logit_locality/ckpt3664_lane_c_val200.yaml
  lane_d_selected_cases: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/selected_cases.jsonl
  self_rollout_root: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
selection:
  sample_limit: 200
  max_cases: 512
  prefer_prefix_mode: self_prefix
  min_remaining_gt: 1
regions:
  context_expansion_norm1000: 64
  duplicate_iou_threshold: 0.95
execution:
  batch_size: 1
  attn_implementation: eager
  torch_dtype: bfloat16
  max_feasibility_cases: 4
```

- [ ] **Step 4: Create runner skeleton**

Create `scripts/analysis/run_autoreg_attention_evidence_routing.py`:

```python
#!/usr/bin/env python
"""Run autoregressive attention evidence-routing analysis stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoreg_attention_evidence_routing import (  # noqa: E402
    ATTENTION_STAGES,
    build_attention_dry_run_plan,
    load_attention_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    return parser


def _stages(raw: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not stages:
        parser.error("--stages must not be empty")
    unknown = [stage for stage in stages if stage not in ATTENTION_STAGES]
    if unknown:
        parser.error(f"unknown attention stage(s): {', '.join(unknown)}")
    return stages


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    stages = _stages(args.stages, parser)
    try:
        config = load_attention_config(args.config)
        if args.dry_run:
            payload = build_attention_dry_run_plan(
                config,
                stages=stages,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    parser.error("non-dry-run attention stages are added in later tasks of this plan")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run config and runner checks**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages select_cases,feasibility \
  --shard-index 0 \
  --num-shards 8 \
  --dry-run
```

Expected: pytest passes; dry-run JSON contains `"shard_label":"shard_000-of-008"` and `"attn_implementation":"eager"`.

## Task 3: Selected Cases And Candidate Region Ledger

**Files:**

- Modify: `src/analysis/autoreg_attention_evidence_routing.py`
- Modify: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Modify: `tests/test_autoreg_attention_evidence_routing.py`

- [ ] **Step 1: Add CPU tests for missed-GT selection and region rows**

Append:

```python
import json

from src.analysis.autoreg_attention_evidence_routing import (
    build_candidate_region_rows,
    select_attention_cases_from_rows,
)


def test_select_attention_cases_prefers_self_prefix_missed_gt() -> None:
    lane_d_cases = [
        {
            "case_id": "row0:self_prefix:depth2:gt3",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 2,
            "prefix_quality": "clean_prefix",
            "intended_target_gt_idx": 3,
            "target_desc": "vase",
            "x1_target_rank": 350,
            "x1_top_peak_attribution": "no_local_object_diffuse",
        },
        {
            "case_id": "row1:teacher_forced:depth0:gt0",
            "source_line_idx": 1,
            "prefix_mode": "teacher_forced",
            "prefix_depth": 0,
            "prefix_quality": "gt_prefix",
            "intended_target_gt_idx": 0,
            "target_desc": "person",
            "x1_target_rank": 1,
            "x1_top_peak_attribution": "target_gt_object",
        },
    ]
    rows = select_attention_cases_from_rows(
        lane_d_cases,
        shard_index=0,
        num_shards=8,
        max_cases=8,
        prefer_prefix_mode="self_prefix",
    )
    assert [row["case_id"] for row in rows] == ["row0:self_prefix:depth2:gt3"]
    assert rows[0]["attention_case_family"] == "missed_gt_evidence_routing"


def test_build_candidate_region_rows_emits_target_same_desc_and_context() -> None:
    selected_case = {
        "case_id": "row0:self_prefix:depth2:gt1",
        "source_line_idx": 0,
        "intended_target_gt_idx": 1,
        "target_desc": "vase",
        "prefix_depth": 2,
        "prefix_mode": "self_prefix",
        "prefix_quality": "fp_prefix",
    }
    dataset_row = {
        "objects": [
            {"desc": "chair", "bbox_2d": [0, 0, 100, 100]},
            {"desc": "vase", "bbox_2d": [200, 200, 260, 300]},
            {"desc": "vase", "bbox_2d": [700, 700, 760, 820]},
        ],
        "width": 1000,
        "height": 1000,
    }
    rows = build_candidate_region_rows(
        selected_case,
        dataset_row,
        context_expansion_norm1000=64,
        shard_index=0,
        num_shards=8,
    )
    kinds = {(row["region_kind"], row.get("gt_idx")) for row in rows}
    assert ("target_gt", 1) in kinds
    assert ("same_desc_gt", 2) in kinds
    assert ("context_ring", 1) in kinds
    assert all(row["case_id"] == selected_case["case_id"] for row in rows)
```

- [ ] **Step 2: Implement selection and region builders**

Add:

```python
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _clip_box_norm1000(box: Sequence[object]) -> list[int]:
    if not _valid_box_xyxy(box):
        raise ValueError(f"invalid xyxy box: {box!r}")
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    return [
        max(0, min(999, x1)),
        max(0, min(999, y1)),
        max(0, min(999, x2)),
        max(0, min(999, y2)),
    ]


def _expanded_context_box(box: Sequence[object], expansion: int) -> list[int]:
    x1, y1, x2, y2 = _clip_box_norm1000(box)
    return [
        max(0, x1 - int(expansion)),
        max(0, y1 - int(expansion)),
        min(999, x2 + int(expansion)),
        min(999, y2 + int(expansion)),
    ]


def select_attention_cases_from_rows(
    lane_d_cases: Sequence[Mapping[str, Any]],
    *,
    shard_index: int | None,
    num_shards: int | None,
    max_cases: int,
    prefer_prefix_mode: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in lane_d_cases:
        source_line_idx = int(row["source_line_idx"])
        if not attention_record_selected(
            source_line_idx,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue
        if str(row.get("prefix_mode")) != prefer_prefix_mode:
            continue
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen:
            continue
        target_idx = row.get("intended_target_gt_idx")
        if target_idx is None:
            continue
        seen.add(case_id)
        out = dict(row)
        out["attention_case_family"] = "missed_gt_evidence_routing"
        out["selection_policy"] = "lane_d_self_prefix_remaining_gt"
        selected.append(out)
        if len(selected) >= int(max_cases):
            break
    return selected


def build_candidate_region_rows(
    selected_case: Mapping[str, Any],
    dataset_row: Mapping[str, Any],
    *,
    context_expansion_norm1000: int,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    source_line_idx = int(selected_case["source_line_idx"])
    target_gt_idx = int(selected_case["intended_target_gt_idx"])
    objects = dataset_row.get("objects")
    if not isinstance(objects, Sequence):
        raise ValueError("dataset row missing objects")
    if target_gt_idx < 0 or target_gt_idx >= len(objects):
        raise ValueError(f"target_gt_idx out of range: {target_gt_idx}")
    target = objects[target_gt_idx]
    if not isinstance(target, Mapping):
        raise ValueError("target object must be a mapping")
    target_desc = _desc(target)
    target_box = _clip_box_norm1000(_box(target))
    shard_label = attention_shard_label(shard_index, num_shards)
    base = {
        "case_id": selected_case["case_id"],
        "source_line_idx": source_line_idx,
        "prefix_mode": selected_case["prefix_mode"],
        "prefix_depth": selected_case["prefix_depth"],
        "prefix_quality": selected_case["prefix_quality"],
        "target_gt_idx": target_gt_idx,
        "target_desc": selected_case.get("target_desc", target.get("desc")),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
    }
    rows: list[dict[str, Any]] = [
        {
            **base,
            "region_kind": "target_gt",
            "gt_idx": target_gt_idx,
            "desc": target.get("desc"),
            "bbox_xyxy": target_box,
        },
        {
            **base,
            "region_kind": "context_ring",
            "gt_idx": target_gt_idx,
            "desc": target.get("desc"),
            "bbox_xyxy": _expanded_context_box(target_box, context_expansion_norm1000),
            "exclude_bbox_xyxy": target_box,
        },
    ]
    for gt_idx, obj in enumerate(objects):
        if gt_idx == target_gt_idx or not isinstance(obj, Mapping):
            continue
        if target_desc and _desc(obj) == target_desc:
            rows.append(
                {
                    **base,
                    "region_kind": "same_desc_gt",
                    "gt_idx": gt_idx,
                    "desc": obj.get("desc"),
                    "bbox_xyxy": _clip_box_norm1000(_box(obj)),
                }
            )
    rows.append(
        {
            **base,
            "region_kind": "far_background",
            "gt_idx": None,
            "desc": None,
            "bbox_xyxy": [0, 0, 999, 999],
            "exclude_bbox_xyxy": target_box,
        }
    )
    return rows
```

- [ ] **Step 3: Add materialization function and CLI stage**

Add:

```python
def materialize_attention_select_cases_shard(
    config: AttentionConfig,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    assert shard_index is not None and num_shards is not None and shard_label is not None
    lane_d_cases = _read_jsonl(config.paths.lane_d_selected_cases)
    dataset_rows = _read_jsonl(config.paths.dataset_jsonl)
    selected = select_attention_cases_from_rows(
        lane_d_cases,
        shard_index=shard_index,
        num_shards=num_shards,
        max_cases=config.selection.max_cases,
        prefer_prefix_mode=config.selection.prefer_prefix_mode,
    )
    candidate_rows: list[dict[str, Any]] = []
    for case in selected:
        candidate_rows.extend(
            build_candidate_region_rows(
                case,
                dataset_rows[int(case["source_line_idx"])],
                context_expansion_norm1000=config.regions.context_expansion_norm1000,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        )
    shard_dir = config.paths.artifact_root / "shards" / shard_label
    _write_jsonl(shard_dir / "selected_cases.jsonl", selected)
    _write_jsonl(shard_dir / "candidate_region_rows.jsonl", candidate_rows)
    summary = {
        "stage": "attention_evidence_routing",
        "stages_completed": ["select_cases"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "row_counts": {
            "selected_cases": len(selected),
            "candidate_region_rows": len(candidate_rows),
        },
        "selection_policy": "lane_d_self_prefix_remaining_gt",
        "duplicate_policy": "same_desc_iou_gt_0p95",
    }
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
```

Update the runner:

```python
from src.analysis.autoreg_attention_evidence_routing import (  # noqa: E402
    ATTENTION_STAGES,
    build_attention_dry_run_plan,
    load_attention_config,
    materialize_attention_select_cases_shard,
)
```

In `main()` before the final `parser.error`:

```python
        if "select_cases" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error("select_cases requires --shard-index and --num-shards")
            payload = materialize_attention_select_cases_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
```

- [ ] **Step 4: Verify select-cases smoke**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages select_cases \
  --shard-index 0 \
  --num-shards 8
```

Expected: pytest passes; command writes `selected_cases.jsonl`, `candidate_region_rows.jsonl`, and `summary.json` under shard `000`.

## Task 4: Visual Token Mapping And Attention Feasibility Gate

**Files:**

- Modify: `src/analysis/autoreg_attention_evidence_routing.py`
- Modify: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Modify: `tests/test_autoreg_attention_evidence_routing.py`

- [ ] **Step 1: Add pure tests for visual-token span and patch region membership**

Append:

```python
from src.analysis.autoreg_attention_evidence_routing import (
    build_patch_region_membership,
    find_visual_token_spans,
)


def test_find_visual_token_spans_groups_contiguous_image_pad_tokens() -> None:
    spans = find_visual_token_spans([1, 9, 9, 2, 9, 9, 9, 3], image_token_id=9)
    assert spans == [(1, 3), (4, 7)]


def test_build_patch_region_membership_maps_grid_centers_to_regions() -> None:
    regions = [
        {
            "case_id": "case",
            "region_kind": "target_gt",
            "bbox_xyxy": [0, 0, 500, 500],
        },
        {
            "case_id": "case",
            "region_kind": "far_background",
            "bbox_xyxy": [0, 0, 999, 999],
            "exclude_bbox_xyxy": [0, 0, 500, 500],
        },
    ]
    membership = build_patch_region_membership(
        visual_token_start=10,
        grid_h=2,
        grid_w=2,
        region_rows=regions,
    )
    assert membership["target_gt"] == [10]
    assert sorted(membership["far_background"]) == [11, 12, 13]
```

- [ ] **Step 2: Implement visual-token mapping helpers**

Add:

```python
def find_visual_token_spans(input_ids: Sequence[int], *, image_token_id: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, token_id in enumerate(input_ids):
        if int(token_id) == int(image_token_id):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(input_ids)))
    return spans


def _point_in_box(x: float, y: float, box: Sequence[object]) -> bool:
    if not _valid_box_xyxy(box):
        return False
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1 <= x <= x2 and y1 <= y <= y2


def build_patch_region_membership(
    *,
    visual_token_start: int,
    grid_h: int,
    grid_w: int,
    region_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    membership: dict[str, list[int]] = {str(row["region_kind"]): [] for row in region_rows}
    for row in range(int(grid_h)):
        for col in range(int(grid_w)):
            token_index = int(visual_token_start) + row * int(grid_w) + col
            x = (col + 0.5) * 1000.0 / float(grid_w)
            y = (row + 0.5) * 1000.0 / float(grid_h)
            for region in region_rows:
                kind = str(region["region_kind"])
                if _point_in_box(x, y, region.get("bbox_xyxy", ())):
                    exclude = region.get("exclude_bbox_xyxy")
                    if isinstance(exclude, Sequence) and not isinstance(exclude, (str, bytes)):
                        if _point_in_box(x, y, exclude):
                            continue
                    membership.setdefault(kind, []).append(token_index)
    return membership
```

- [ ] **Step 3: Implement feasibility stage**

Add a GPU-capable function that loads one shard's selected cases, uses Lane C and Lane D to build matching prepared examples and position inventory, forces `output_attentions=True`, and writes one feasibility row per checked case:

```python
def materialize_attention_feasibility_shard(
    config: AttentionConfig,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    import torch

    from src.analysis.autoreg_hidden_state_probe import (
        build_lane_d_position_inventory_for_prepared_example,
        filter_lane_d_examples_for_selected_cases,
    )
    from src.analysis.hard_ce_coord_logit_locality import (
        _batch_pad_offset,
        _model_device,
        load_model_handle,
        prepare_lane_c_x1_basin_examples,
    )

    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    assert shard_index is not None and num_shards is not None and shard_label is not None
    shard_dir = config.paths.artifact_root / "shards" / shard_label
    selected = _read_jsonl(shard_dir / "selected_cases.jsonl")[
        : config.execution.max_feasibility_cases
    ]
    candidate_rows = _read_jsonl(shard_dir / "candidate_region_rows.jsonl")
    lane_c_config = build_attention_lane_c_config(config)
    model_handle = load_model_handle(lane_c_config)
    image_token_id = model_handle.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or int(image_token_id) < 0:
        raise RuntimeError("tokenizer cannot resolve <|image_pad|>")
    examples, _ = prepare_lane_c_x1_basin_examples(
        lane_c_config,
        model_handle=model_handle,
        limit=None,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    pairs = filter_lane_d_examples_for_selected_cases(examples, selected)
    rows: list[dict[str, Any]] = []
    for selected_case, example in pairs:
        image = __import__("PIL.Image", fromlist=["Image"]).open(example.image_path).convert("RGB")
        inputs = model_handle.processor(
            text=[example.full_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(_model_device(model_handle.model)) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model_handle.model(
                **inputs,
                use_cache=False,
                output_attentions=True,
            )
        attentions = getattr(outputs, "attentions", None)
        if not isinstance(attentions, tuple) or not attentions:
            raise RuntimeError("model forward did not return attentions")
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        visual_spans = find_visual_token_spans(input_ids, image_token_id=int(image_token_id))
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("processor output missing image_grid_thw")
        grid = image_grid_thw[0].detach().cpu().tolist()
        grid_t, grid_h, grid_w = [int(value) for value in grid]
        expected_visual_tokens = grid_t * grid_h * grid_w
        if not visual_spans or (visual_spans[0][1] - visual_spans[0][0]) != expected_visual_tokens:
            raise RuntimeError("visual token span does not match image_grid_thw")
        inventory = build_lane_d_position_inventory_for_prepared_example(
            example,
            selected_case,
            model_handle.tokenizer,
            shard_index=shard_index,
            num_shards=num_shards,
            shard_label=shard_label,
        )
        pad_offset = _batch_pad_offset(
            input_ids=inputs["input_ids"],
            batch_idx=0,
            expected_ids=example.full_input_ids,
        )
        rows.append(
            {
                "case_id": selected_case["case_id"],
                "source_line_idx": selected_case["source_line_idx"],
                "prefix_mode": selected_case["prefix_mode"],
                "prefix_depth": selected_case["prefix_depth"],
                "image_token_id": int(image_token_id),
                "visual_span_start": visual_spans[0][0],
                "visual_span_end": visual_spans[0][1],
                "grid_t": grid_t,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "attention_layer_count": len(attentions),
                "attention_shape_0": list(attentions[0].shape),
                "inventory_role_count": len(inventory),
                "pad_offset": int(pad_offset),
                "candidate_region_count": sum(
                    1 for row in candidate_rows if row["case_id"] == selected_case["case_id"]
                ),
                "shard_index": shard_index,
                "num_shards": num_shards,
                "shard_label": shard_label,
            }
        )
    _write_jsonl(shard_dir / "feasibility_rows.jsonl", rows)
    summary = {
        "stage": "attention_evidence_routing",
        "stages_completed": ["select_cases", "feasibility"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "row_counts": {"feasibility_rows": len(rows)},
        "runtime_kind": "attention_feasibility_forward",
    }
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
```

Update the runner import and stage branch:

```python
from src.analysis.autoreg_attention_evidence_routing import (  # noqa: E402
    ATTENTION_STAGES,
    build_attention_dry_run_plan,
    load_attention_config,
    materialize_attention_feasibility_shard,
    materialize_attention_select_cases_shard,
)
```

```python
        if "feasibility" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error("feasibility requires --shard-index and --num-shards")
            payload = materialize_attention_feasibility_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
```

- [ ] **Step 4: Run CPU tests and one-GPU feasibility smoke**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages select_cases \
  --shard-index 0 \
  --num-shards 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages feasibility \
  --shard-index 0 \
  --num-shards 8
```

Expected:

```text
tests pass
feasibility_rows.jsonl exists
every feasibility row has attention_layer_count > 0
visual_span_end - visual_span_start == grid_t * grid_h * grid_w
inventory_role_count >= 9
```

If this smoke fails because Qwen3-VL does not return attentions with `eager`, stop this plan before Task 5 and revise the mechanism lane toward occlusion/activation patching instead of attention weights.

## Task 5: Observational Attention Atlas Rows

**Files:**

- Modify: `src/analysis/autoreg_attention_evidence_routing.py`
- Modify: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Modify: `tests/test_autoreg_attention_evidence_routing.py`

- [ ] **Step 1: Add aggregation tests with fake attention**

Append:

```python
import torch

from src.analysis.autoreg_attention_evidence_routing import aggregate_attention_for_query


def test_aggregate_attention_for_query_sums_region_mass_per_head() -> None:
    attention = torch.zeros(1, 2, 6, 6)
    attention[0, 0, 5, 1] = 0.25
    attention[0, 0, 5, 2] = 0.25
    attention[0, 0, 5, 3] = 0.50
    attention[0, 1, 5, 1] = 0.10
    attention[0, 1, 5, 4] = 0.90
    rows = aggregate_attention_for_query(
        attention,
        batch_idx=0,
        query_index=5,
        layer_index=3,
        role="pre_x1",
        region_membership={"target_gt": [1, 2], "far_background": [4]},
        base_row={"case_id": "case", "source_line_idx": 0},
    )
    by_key = {(row["head"], row["region_kind"]): row for row in rows}
    assert by_key[(0, "target_gt")]["attention_mass"] == pytest.approx(0.5)
    assert by_key[(1, "far_background")]["attention_mass"] == pytest.approx(0.9)
```

- [ ] **Step 2: Implement attention aggregation helper**

Add:

```python
def aggregate_attention_for_query(
    attention: Any,
    *,
    batch_idx: int,
    query_index: int,
    layer_index: int,
    role: str,
    region_membership: Mapping[str, Sequence[int]],
    base_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    if not isinstance(attention, torch.Tensor):
        raise TypeError("attention must be a torch.Tensor")
    if attention.ndim != 4:
        raise ValueError(f"expected attention shape [batch, heads, query, key], got {tuple(attention.shape)}")
    rows: list[dict[str, Any]] = []
    head_count = int(attention.shape[1])
    key_count = int(attention.shape[3])
    for head in range(head_count):
        query_vector = attention[batch_idx, head, query_index].detach().float().cpu()
        denom = float(query_vector.sum().item())
        for region_kind, indices in region_membership.items():
            valid_indices = [int(index) for index in indices if 0 <= int(index) < key_count]
            mass = float(query_vector[valid_indices].sum().item()) if valid_indices else 0.0
            rows.append(
                {
                    **dict(base_row),
                    "layer": int(layer_index),
                    "head": int(head),
                    "role": role,
                    "query_index": int(query_index),
                    "region_kind": str(region_kind),
                    "region_token_count": len(valid_indices),
                    "attention_mass": mass,
                    "attention_mass_normalized": 0.0 if denom <= 0.0 else mass / denom,
                }
            )
    return rows
```

- [ ] **Step 3: Implement `attention_atlas` shard materialization**

Add `materialize_attention_atlas_shard()` by extending the feasibility code path
with this concrete implementation:

```python
def materialize_attention_atlas_shard(
    config: AttentionConfig,
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    import torch

    from src.analysis.autoreg_hidden_state_probe import (
        build_lane_d_position_inventory_for_prepared_example,
        filter_lane_d_examples_for_selected_cases,
    )
    from src.analysis.hard_ce_coord_logit_locality import (
        _batch_pad_offset,
        _model_device,
        load_model_handle,
        prepare_lane_c_x1_basin_examples,
    )

    shard_index, num_shards, shard_label = normalize_attention_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    assert shard_index is not None and num_shards is not None and shard_label is not None
    shard_dir = config.paths.artifact_root / "shards" / shard_label
    selected = _read_jsonl(shard_dir / "selected_cases.jsonl")
    candidate_rows_all = _read_jsonl(shard_dir / "candidate_region_rows.jsonl")
    candidate_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows_all:
        candidate_by_case.setdefault(str(row["case_id"]), []).append(row)
    lane_c_config = build_attention_lane_c_config(config)
    model_handle = load_model_handle(lane_c_config)
    image_token_id = int(model_handle.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    examples, _ = prepare_lane_c_x1_basin_examples(
        lane_c_config,
        model_handle=model_handle,
        limit=None,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    pairs = filter_lane_d_examples_for_selected_cases(examples, selected)
    attention_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for selected_case, example in pairs:
        image = __import__("PIL.Image", fromlist=["Image"]).open(example.image_path).convert("RGB")
        inputs = model_handle.processor(
            text=[example.full_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(_model_device(model_handle.model)) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model_handle.model(
                **inputs,
                use_cache=False,
                output_attentions=True,
            )
        attentions = getattr(outputs, "attentions", None)
        if not isinstance(attentions, tuple) or not attentions:
            raise RuntimeError("model forward did not return attentions")
        input_ids_tensor = inputs["input_ids"]
        input_ids = input_ids_tensor[0].detach().cpu().tolist()
        visual_span = find_visual_token_spans(input_ids, image_token_id=image_token_id)[0]
        grid_t, grid_h, grid_w = [
            int(value) for value in inputs["image_grid_thw"][0].detach().cpu().tolist()
        ]
        membership = build_patch_region_membership(
            visual_token_start=visual_span[0],
            grid_h=grid_h,
            grid_w=grid_w,
            region_rows=candidate_by_case[str(selected_case["case_id"])],
        )
        inventory = build_lane_d_position_inventory_for_prepared_example(
            example,
            selected_case,
            model_handle.tokenizer,
            shard_index=shard_index,
            num_shards=num_shards,
            shard_label=shard_label,
        )
        pad_offset = _batch_pad_offset(
            input_ids=input_ids_tensor,
            batch_idx=0,
            expected_ids=example.full_input_ids,
        )
        inv_by_role = {str(row["role"]): row for row in inventory}
        base = {
            "case_id": selected_case["case_id"],
            "source_line_idx": selected_case["source_line_idx"],
            "prefix_mode": selected_case["prefix_mode"],
            "prefix_depth": selected_case["prefix_depth"],
            "prefix_quality": selected_case["prefix_quality"],
            "target_gt_idx": selected_case.get("intended_target_gt_idx"),
            "target_desc": selected_case.get("target_desc"),
            "x1_target_rank": selected_case.get("x1_target_rank"),
            "x1_top_peak_attribution": selected_case.get("x1_top_peak_attribution"),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "shard_label": shard_label,
        }
        decision_rows.append(
            {
                **base,
                "visual_span_start": visual_span[0],
                "visual_span_end": visual_span[1],
                "grid_t": grid_t,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "attention_layer_count": len(attentions),
                "query_roles": list(ATTENTION_QUERY_ROLES),
            }
        )
        for role in ATTENTION_QUERY_ROLES:
            inv = inv_by_role.get(role)
            if inv is None:
                continue
            hidden_index = inv.get("prediction_token_index")
            if hidden_index is None:
                hidden_index = inv.get("absolute_token_index")
            query_index = int(pad_offset) + int(hidden_index)
            for layer_index, attention in enumerate(attentions):
                attention_rows.extend(
                    aggregate_attention_for_query(
                        attention,
                        batch_idx=0,
                        query_index=query_index,
                        layer_index=layer_index,
                        role=role,
                        region_membership=membership,
                        base_row=base,
                    )
                )
    _write_jsonl(shard_dir / "attention_region_rows.jsonl", attention_rows)
    _write_jsonl(shard_dir / "decision_context_rows.jsonl", decision_rows)
    summary = {
        "stage": "attention_evidence_routing",
        "stages_completed": ["select_cases", "attention_atlas"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_label": shard_label,
        "row_counts": {
            "selected_cases": len(selected),
            "candidate_region_rows": len(candidate_rows_all),
            "attention_region_rows": len(attention_rows),
            "decision_context_rows": len(decision_rows),
        },
        "runtime_kind": "attention_atlas_forward",
    }
    (shard_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
```

Update runner import and branch:

```python
from src.analysis.autoreg_attention_evidence_routing import (  # noqa: E402
    ATTENTION_STAGES,
    build_attention_dry_run_plan,
    load_attention_config,
    materialize_attention_atlas_shard,
    materialize_attention_feasibility_shard,
    materialize_attention_select_cases_shard,
)
```

```python
        if "attention_atlas" in stages:
            if args.shard_index is None or args.num_shards is None:
                parser.error("attention_atlas requires --shard-index and --num-shards")
            payload = materialize_attention_atlas_shard(
                config,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return 0
```

- [ ] **Step 4: Run single-shard atlas smoke**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 8
```

Expected: `attention_region_rows.jsonl` and `decision_context_rows.jsonl` exist for shard 0. Row count must be nonzero.

## Task 6: Merge, Summary, And Report

**Files:**

- Modify: `src/analysis/autoreg_attention_evidence_routing.py`
- Modify: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Modify: `tests/test_autoreg_attention_evidence_routing.py`

- [ ] **Step 1: Add merge tests with two fake shards**

Append:

```python
from src.analysis.autoreg_attention_evidence_routing import merge_attention_shards


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_merge_attention_shards_requires_expected_shards(tmp_path: Path) -> None:
    root = tmp_path / "attention"
    for shard_idx in range(2):
        label = f"shard_{shard_idx:03d}-of-002"
        shard = root / "shards" / label
        _write_jsonl(shard / "selected_cases.jsonl", [{"case_id": f"case-{shard_idx}", "source_line_idx": shard_idx}])
        _write_jsonl(shard / "candidate_region_rows.jsonl", [])
        _write_jsonl(shard / "feasibility_rows.jsonl", [])
        _write_jsonl(shard / "attention_region_rows.jsonl", [])
        _write_jsonl(shard / "decision_context_rows.jsonl", [])
        (shard / "summary.json").write_text(
            json.dumps({"shard_label": label, "row_counts": {"selected_cases": 1}}),
            encoding="utf-8",
        )
    summary = merge_attention_shards(root, expected_shards=2)
    assert summary["expected_shards"] == 2
    assert summary["row_counts"]["selected_cases"] == 2
    assert (root / "merge_summary.json").exists()
```

- [ ] **Step 2: Implement merge and report helpers**

Add:

```python
MERGE_JSONL_FILES = (
    "selected_cases.jsonl",
    "candidate_region_rows.jsonl",
    "feasibility_rows.jsonl",
    "attention_region_rows.jsonl",
    "decision_context_rows.jsonl",
)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def merge_attention_shards(root: Path, *, expected_shards: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    shard_labels = [attention_shard_label(index, expected_shards) for index in range(expected_shards)]
    row_counts: dict[str, int] = {}
    for filename in MERGE_JSONL_FILES:
        out_path = root / filename
        with out_path.open("w", encoding="utf-8") as out:
            for label in shard_labels:
                in_path = root / "shards" / label / filename
                if not in_path.exists():
                    raise FileNotFoundError(f"missing shard file: {in_path}")
                text = in_path.read_text(encoding="utf-8")
                if text and not text.endswith("\n"):
                    text += "\n"
                out.write(text)
        row_counts[filename.removesuffix(".jsonl")] = _jsonl_count(out_path)
    summary = {
        "stage": "attention_evidence_routing",
        "expected_shards": expected_shards,
        "merged_shards": shard_labels,
        "row_counts": row_counts,
        "duplicate_policy": "same_desc_iou_gt_0p95",
        "causal_status": "not_in_scope_for_this_plan",
    }
    (root / "merge_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_attention_report(root: Path) -> Path:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    report = root / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Attention Evidence Routing Report",
                "",
                "## Scope",
                "",
                "- checkpoint: checkpoint-3664",
                "- dataset_slice: val200",
                "- evidence_scope: observational_attention_only",
                "- duplicate_policy: same-desc IoU > 0.95",
                "- causal_status: not_in_scope_for_this_plan",
                "",
                "## Row Counts",
                "",
                "```json",
                json.dumps(summary.get("row_counts", {}), sort_keys=True, indent=2),
                "```",
                "",
                "## Interpretation Bounds",
                "",
                "- Attention rows are mechanism candidates, not causal proof.",
                "- Forced continuation is not used as evidence of solved recall.",
                "- Other extra predictions are soft/neutral under partial COCO labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report
```

Update runner:

```python
from src.analysis.autoreg_attention_evidence_routing import (  # noqa: E402
    ATTENTION_STAGES,
    build_attention_dry_run_plan,
    load_attention_config,
    materialize_attention_atlas_shard,
    materialize_attention_feasibility_shard,
    materialize_attention_select_cases_shard,
    merge_attention_shards,
    write_attention_report,
)
```

```python
        if args.merge_shards:
            if args.num_shards is None:
                parser.error("--merge-shards requires --num-shards")
            summary = merge_attention_shards(
                config.paths.artifact_root,
                expected_shards=args.num_shards,
            )
            if "report" in stages:
                write_attention_report(config.paths.artifact_root)
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
            return 0
```

- [ ] **Step 3: Verify merge tests**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py::test_merge_attention_shards_requires_expected_shards -q
```

Expected: test passes.

## Task 7: 8-GPU tmux Launcher

**Files:**

- Create: `scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh`

- [ ] **Step 1: Create dry-run-safe launcher**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-autoreg_attention_ckpt3664}"
NUM_SHARDS="${NUM_SHARDS:-8}"
CONFIG="${CONFIG:-configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml}"
REPO_ROOT="${REPO_ROOT:-/data/CoordExp}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_FEASIBILITY_ONLY="${RUN_FEASIBILITY_ONLY:-0}"
ROOT="${ROOT:-}"

CONFIG_ROOT="$(
  CONFIG_PATH="$CONFIG" REPO_ROOT="$REPO_ROOT" python - <<'PY'
import os
from pathlib import Path
import yaml

repo = Path(os.environ["REPO_ROOT"]).expanduser().resolve()
config_path = Path(os.environ["CONFIG_PATH"]).expanduser()
if not config_path.is_absolute():
    config_path = repo / config_path
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
paths = payload.get("paths") if isinstance(payload, dict) else None
if not isinstance(paths, dict) or not paths.get("artifact_root"):
    raise SystemExit(f"config missing paths.artifact_root: {config_path}")
root = Path(str(paths["artifact_root"])).expanduser()
if not root.is_absolute():
    root = repo / root
print(root.resolve(strict=False))
PY
)"

if [[ -z "$ROOT" ]]; then
  ROOT="$CONFIG_ROOT"
fi
if [[ "$ROOT" != "$CONFIG_ROOT" ]]; then
  echo "Attention ROOT mismatch: ROOT=$ROOT CONFIG_ROOT=$CONFIG_ROOT" >&2
  exit 1
fi

LOG_DIR="$ROOT/logs"
SHARDS_DIR="$ROOT/shards"
COMMAND_FILE="${COMMAND_FILE:-$LOG_DIR/${SESSION}_commands.sh}"
RUNNER="$REPO_ROOT/scripts/analysis/run_autoreg_attention_evidence_routing.py"

existing_outputs=()
for output_path in \
  "$ROOT/selected_cases.jsonl" \
  "$ROOT/candidate_region_rows.jsonl" \
  "$ROOT/feasibility_rows.jsonl" \
  "$ROOT/attention_region_rows.jsonl" \
  "$ROOT/decision_context_rows.jsonl" \
  "$ROOT/summary.json" \
  "$ROOT/merge_summary.json" \
  "$ROOT/report.md" \
  "$COMMAND_FILE"
do
  if [[ -e "$output_path" ]]; then
    existing_outputs+=("$output_path")
  fi
done
if [[ -d "$SHARDS_DIR" ]]; then
  for shard_dir in "$SHARDS_DIR"/shard_[0-9][0-9][0-9]-of-[0-9][0-9][0-9]; do
    if [[ -d "$shard_dir" ]]; then
      existing_outputs+=("$shard_dir")
    fi
  done
fi

if (( ${#existing_outputs[@]} > 0 )) && [[ "$DRY_RUN" != "1" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "existing attention outputs found under $ROOT; set ALLOW_OVERWRITE=1 to remove stale outputs." >&2
  printf '  %s\n' "${existing_outputs[@]}" >&2
  exit 1
fi

if (( ${#existing_outputs[@]} > 0 )) && [[ "$DRY_RUN" != "1" && "$ALLOW_OVERWRITE" == "1" ]]; then
  rm -rf "$ROOT"
fi

mkdir -p "$LOG_DIR" "$SHARDS_DIR"

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n\n'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'mkdir -p %q %q\n\n' "$LOG_DIR" "$SHARDS_DIR"
  printf '# CPU selected-case materialization is serial for deterministic manifests.\n'
  for ((i = 0; i < NUM_SHARDS; i += 1)); do
    label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
    select_log="$LOG_DIR/${label}_select_cases.log"
    printf 'CUDA_VISIBLE_DEVICES= PYTHONPATH=%q python %q --config %q --stages select_cases --shard-index %q --num-shards %q > %q 2>&1\n' \
      "$REPO_ROOT" "$RUNNER" "$CONFIG" "$i" "$NUM_SHARDS" "$select_log"
  done
  printf '\n'
  printf 'pids=()\n'
  for ((i = 0; i < NUM_SHARDS; i += 1)); do
    label="$(printf "shard_%03d-of-%03d" "$i" "$NUM_SHARDS")"
    if [[ "$RUN_FEASIBILITY_ONLY" == "1" ]]; then
      stage="feasibility"
      log_file="$LOG_DIR/${label}_feasibility.log"
    else
      stage="attention_atlas"
      log_file="$LOG_DIR/${label}_attention_atlas.log"
    fi
    printf '(\n'
    printf '  CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q python %q --config %q --stages %q --shard-index %q --num-shards %q\n' \
      "$i" "$REPO_ROOT" "$RUNNER" "$CONFIG" "$stage" "$i" "$NUM_SHARDS"
    printf ') > %q 2>&1 &\n' "$log_file"
    printf 'pids+=("$!")\n\n'
  done
  printf 'status=0\n'
  printf 'for pid in "${pids[@]}"; do\n'
  printf '  wait "$pid" || status=1\n'
  printf 'done\n'
  printf 'if [[ "$status" -ne 0 ]]; then\n'
  printf '  echo "one or more attention shards failed; inspect %s" >&2\n' "$LOG_DIR"
  printf '  exit "$status"\n'
  printf 'fi\n\n'
  printf 'PYTHONPATH=%q python %q --config %q --stages merge,report --merge-shards --num-shards %q > %q 2>&1\n' \
    "$REPO_ROOT" "$RUNNER" "$CONFIG" "$NUM_SHARDS" "$LOG_DIR/merge.log"
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Attention tmux command file: $COMMAND_FILE"
if [[ "$DRY_RUN" == "1" ]]; then
  cat "$COMMAND_FILE"
  exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE'"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Inspect with: tmux capture-pane -pt $SESSION:0 -S -200"
echo "Tail logs with: tail -f $LOG_DIR/shard_000-of-$(printf "%03d" "$NUM_SHARDS")_attention_atlas.log"
```

- [ ] **Step 2: Verify launcher syntax and dry run**

Run:

```bash
bash -n scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
DRY_RUN=1 bash scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
```

Expected: syntax check passes; dry run prints serial `select_cases`, parallel `attention_atlas`, and merge/report commands without starting tmux.

## Task 8: Training-Set Teacher-Forced Anchor Slice

**Files:**

- Create: `configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml`
- Modify: `src/analysis/autoreg_attention_evidence_routing.py`
- Modify: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Modify: `tests/test_autoreg_attention_evidence_routing.py`

This task adds a seen-sample control surface. It does not claim train-set rollout recall, because no train self-rollout artifact is required by this plan. It asks a sharper internal question: on samples from the checkpoint's own training distribution, do target-object attention and `pre_x1` routing look healthier than `val200`, or do they show the same foreground/context/history diversion?

- [ ] **Step 1: Add config fields for attention case source**

Extend `AttentionSelectionConfig`:

```python
@dataclass(frozen=True)
class AttentionSelectionConfig:
    sample_limit: int
    max_cases: int
    prefer_prefix_mode: str
    min_remaining_gt: int
    case_source: str
    scope_label: str
```

Update `load_attention_config()`:

```python
        selection=AttentionSelectionConfig(
            sample_limit=_required_int(selection, "sample_limit", minimum=1),
            max_cases=_required_int(selection, "max_cases", minimum=1),
            prefer_prefix_mode=_required_str(selection, "prefer_prefix_mode"),
            min_remaining_gt=_required_int(selection, "min_remaining_gt", minimum=0),
            case_source=str(selection.get("case_source", "lane_d_self_prefix_remaining_gt")).strip(),
            scope_label=str(selection.get("scope_label", "val200_self_rollout")).strip(),
        ),
```

Update `build_attention_dry_run_plan()`:

```python
        "case_source": config.selection.case_source,
        "scope_label": config.selection.scope_label,
```

- [ ] **Step 2: Add tests for teacher-forced anchor case construction**

Append:

```python
from src.analysis.autoreg_attention_evidence_routing import (
    select_teacher_forced_anchor_cases_from_dataset_rows,
)


def test_select_teacher_forced_anchor_cases_uses_all_gt_objects() -> None:
    dataset_rows = [
        {
            "objects": [
                {"desc": "person", "bbox_2d": [10, 20, 100, 200]},
                {"desc": "vase", "bbox_2d": [300, 400, 350, 500]},
            ],
            "width": 1000,
            "height": 1000,
        },
        {
            "objects": [
                {"desc": "chair", "bbox_2d": [100, 100, 200, 300]},
            ],
            "width": 1000,
            "height": 1000,
        },
    ]
    cases = select_teacher_forced_anchor_cases_from_dataset_rows(
        dataset_rows,
        lane_c_config=None,
        shard_index=0,
        num_shards=8,
        max_cases=8,
        scope_label="train200_teacher_forced_anchor",
    )
    assert [case["case_id"] for case in cases] == [
        "row0:teacher_forced:depth0:gt0",
        "row0:teacher_forced:depth1:gt1",
    ]
    assert cases[0]["prefix_mode"] == "teacher_forced"
    assert cases[0]["attention_case_family"] == "train_teacher_forced_anchor"
```

- [ ] **Step 3: Implement teacher-forced anchor case selection**

Add:

```python
def select_teacher_forced_anchor_cases_from_dataset_rows(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    lane_c_config: Any,
    shard_index: int | None,
    num_shards: int | None,
    max_cases: int,
    scope_label: str,
) -> list[dict[str, Any]]:
    from src.analysis.hard_ce_coord_logit_locality import (
        _lane_c_teacher_prefix_state,
        _ordering_plan,
        normalize_detection_row,
        parse_raw_detection_row,
        select_lane_c_intended_target_gt_idx,
    )

    selected: list[dict[str, Any]] = []
    for source_line_idx, row in enumerate(dataset_rows):
        if not attention_record_selected(
            source_line_idx,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            continue
        if lane_c_config is None:
            objects = row.get("objects")
            if not isinstance(objects, Sequence):
                continue
            teacher_order_gt_indices = tuple(range(len(objects)))
            objects_by_source = {
                gt_idx: obj
                for gt_idx, obj in enumerate(objects)
                if isinstance(obj, Mapping)
            }
        else:
            raw = parse_raw_detection_row(row)
            normalized = normalize_detection_row(
                raw,
                object_ordering=_ordering_plan(lane_c_config, row_index=source_line_idx),
            )
            teacher_order_gt_indices = tuple(
                int(obj.source_object_index) for obj in normalized.objects
            )
            objects_by_source = {
                int(obj.source_object_index): {
                    "desc": str(obj.desc),
                    "bbox_2d": list(obj.bbox_2d),
                }
                for obj in normalized.objects
            }
        gt_count = len(teacher_order_gt_indices)
        for depth in range(gt_count):
            prefix_state = _lane_c_teacher_prefix_state(
                teacher_order_gt_indices,
                depth=depth,
                gt_count=gt_count,
            )
            selection = select_lane_c_intended_target_gt_idx(
                teacher_order_gt_indices,
                prefix_state,
            )
            gt_idx = selection.intended_target_gt_idx
            if gt_idx is None or int(gt_idx) not in objects_by_source:
                continue
            obj = objects_by_source[int(gt_idx)]
            if not _valid_box_xyxy(_box(obj)):
                continue
            selected.append(
                {
                    "case_id": (
                        f"row{source_line_idx}:teacher_forced:"
                        f"depth{depth}:gt{int(gt_idx)}"
                    ),
                    "source_line_idx": source_line_idx,
                    "prefix_mode": "teacher_forced",
                    "prefix_depth": depth,
                    "prefix_quality": "gt_prefix",
                    "intended_target_gt_idx": int(gt_idx),
                    "target_desc": obj.get("desc"),
                    "attention_case_family": "train_teacher_forced_anchor",
                    "selection_policy": "teacher_forced_all_gt_objects",
                    "scope_label": scope_label,
                }
            )
            if len(selected) >= int(max_cases):
                return selected
    return selected
```

Update `materialize_attention_select_cases_shard()`:

```python
    if config.selection.case_source == "teacher_forced_anchor":
        lane_c_config = build_attention_lane_c_config(config)
        selected = select_teacher_forced_anchor_cases_from_dataset_rows(
            dataset_rows[: config.selection.sample_limit],
            lane_c_config=lane_c_config,
            shard_index=shard_index,
            num_shards=num_shards,
            max_cases=config.selection.max_cases,
            scope_label=config.selection.scope_label,
        )
    else:
        lane_d_cases = _read_jsonl(config.paths.lane_d_selected_cases)
        selected = select_attention_cases_from_rows(
            lane_d_cases,
            shard_index=shard_index,
            num_shards=num_shards,
            max_cases=config.selection.max_cases,
            prefer_prefix_mode=config.selection.prefer_prefix_mode,
        )
```

The rest of `materialize_attention_select_cases_shard()` continues to call `build_candidate_region_rows()` for every selected case.

- [ ] **Step 4: Create train200 teacher-forced anchor config**

Create `configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml`:

```yaml
paths:
  artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_train200/attention_evidence_routing_teacher_forced_anchor
  checkpoint: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
  dataset_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
  lane_a_rollout_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/rollout_anatomy
  lane_c_per_case: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/x1_basin_attribution/per_case.jsonl
  lane_c_study_config: /data/CoordExp/configs/analysis/hard_ce_coord_logit_locality/ckpt3664_lane_c_val200.yaml
  lane_d_selected_cases: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/selected_cases.jsonl
  self_rollout_root: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
selection:
  sample_limit: 200
  max_cases: 512
  prefer_prefix_mode: teacher_forced
  min_remaining_gt: 0
  case_source: teacher_forced_anchor
  scope_label: train200_teacher_forced_anchor
regions:
  context_expansion_norm1000: 64
  duplicate_iou_threshold: 0.95
execution:
  batch_size: 1
  attn_implementation: eager
  torch_dtype: bfloat16
  max_feasibility_cases: 4
```

`lane_a_rollout_root`, `lane_c_per_case`, `lane_d_selected_cases`, and `self_rollout_root` stay present because the first loader version requires a full path block. The teacher-forced anchor selection path must not read them.

- [ ] **Step 5: Add train-anchor forward path**

The `attention_atlas` and `feasibility` stages already filter Lane C x1-basin
examples. Keep that path for `case_source: teacher_forced_anchor`, but make sure
the examples come from `build_attention_lane_c_config(config)`, which points the
frozen Lane C config at `train.coord.jsonl`. Do not use
`prepare_teacher_forced_examples()` here: it renders the full assistant answer
and only gives an unambiguous row-local target for the final object, while this
anchor needs Lane C's per-prefix teacher-forced case IDs.

```python
from src.analysis.hard_ce_coord_logit_locality import (
    _batch_pad_offset,
    _model_device,
    load_model_handle,
    prepare_lane_c_x1_basin_examples,
)
```

In `materialize_attention_atlas_shard()` and
`materialize_attention_feasibility_shard()`, keep example construction as:

```python
    examples, _ = prepare_lane_c_x1_basin_examples(
        lane_c_config,
        model_handle=model_handle,
        limit=config.selection.sample_limit,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    pairs = filter_lane_d_examples_for_selected_cases(examples, selected)
```

For `teacher_forced_anchor`, selected case IDs must match Lane C's
`row{source}:teacher_forced:depth{depth}:gt{target}` convention, so Lane D's
existing `build_lane_d_position_inventory_for_prepared_example()` case-id guard
remains valid. The query roles should be restricted to row-local roles:

```python
query_roles = ("desc_end", "box_start", "pre_x1", "post_x1", "post_y1")
```

For `lane_d_self_prefix_remaining_gt`, keep `ATTENTION_QUERY_ROLES`.

- [ ] **Step 6: Verify train-anchor CPU and feasibility smoke**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
  --stages select_cases \
  --shard-index 0 \
  --num-shards 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
  --stages feasibility \
  --shard-index 0 \
  --num-shards 8
```

Expected: selected train-anchor cases are written under `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_train200/attention_evidence_routing_teacher_forced_anchor`; feasibility rows are nonempty and carry `scope_label=train200_teacher_forced_anchor`.

## Task 9: Full Verification Sequence

**Files:**

- No new files.

- [ ] **Step 1: Run unit tests**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_attention_evidence_routing.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run py_compile**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m py_compile \
  src/analysis/autoreg_attention_evidence_routing.py \
  scripts/analysis/run_autoreg_attention_evidence_routing.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run single-GPU feasibility gate**

Run:

```bash
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages select_cases \
  --shard-index 0 \
  --num-shards 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages feasibility \
  --shard-index 0 \
  --num-shards 8
```

Expected: `feasibility_rows.jsonl` exists and each row has valid attention layer count and visual-token span.

- [ ] **Step 4: Run one-shard observational smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_feasibility_val200.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 8
```

Expected: shard 0 writes nonempty `attention_region_rows.jsonl` and `decision_context_rows.jsonl`.

- [ ] **Step 5: Run train200 teacher-forced anchor smoke**

Run:

```bash
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
  --stages select_cases \
  --shard-index 0 \
  --num-shards 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
  --stages feasibility \
  --shard-index 0 \
  --num-shards 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_attention_evidence_routing.py \
  --config configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
  --stages attention_atlas \
  --shard-index 0 \
  --num-shards 8
```

Expected: train-anchor shard 0 writes nonempty `attention_region_rows.jsonl` and `decision_context_rows.jsonl`. Report interpretation must label this as `train200_teacher_forced_anchor`, not rollout recall.

- [ ] **Step 6: Run 8-GPU dry-run for val200 and train anchor**

Run:

```bash
DRY_RUN=1 bash scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
DRY_RUN=1 \
CONFIG=configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
SESSION=autoreg_attention_train_anchor_ckpt3664 \
bash scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
```

Expected: command file includes 8 serial select-case commands, 8 parallel GPU commands, and one merge/report command.

- [ ] **Step 7: Only after Steps 1-6 pass, launch val200 atlas**

Run:

```bash
ALLOW_OVERWRITE=1 bash scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
```

Expected: tmux session starts. This is not production training; it is 8-way analysis forward-pass sharding.

Monitor:

```bash
tmux attach -t autoreg_attention_ckpt3664
tmux capture-pane -pt autoreg_attention_ckpt3664:0 -S -200
tail -f /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing/logs/shard_000-of-008_attention_atlas.log
```

Expected final artifacts:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing/merge_summary.json
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing/report.md
```

- [ ] **Step 8: Optionally launch train200 teacher-forced anchor atlas**

Run this after the val200 atlas if the train-anchor one-shard smoke was informative:

```bash
ALLOW_OVERWRITE=1 \
CONFIG=configs/analysis/autoreg_attention_evidence_routing/ckpt3664_train200_teacher_forced_anchor.yaml \
SESSION=autoreg_attention_train_anchor_ckpt3664 \
bash scripts/analysis/launch_autoreg_attention_evidence_routing_tmux.sh
```

Expected final artifacts:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_train200/attention_evidence_routing_teacher_forced_anchor/merge_summary.json
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_train200/attention_evidence_routing_teacher_forced_anchor/report.md
```

This result is a training-distribution attention anchor. It is stronger evidence for seen-sample internal routing, but it must not be reported as train-set rollout recall unless a separate train self-rollout artifact is generated and analyzed with the same schema.

## Resolved Follow-Up Decision: FN-Rescue Continuation

Decision:

`FN-rescue continuation` is a diagnostic GT-hint counterfactual experiment, not a deployable inference improvement. For a GT object that is false-negative under self-rollout matching, the experiment appends a constrained continuation after the same rollout prefix and measures whether minimal hints recover the missed object.

Planned hint tiers:

- `desc_only`: self-rollout prefix plus the target GT description; generate the bbox.
- `desc_x1`: self-rollout prefix plus the target GT description and target GT `x1`; generate `y1/x2/y2`.
- `desc_x1_wrong_control`: self-rollout prefix plus the target GT description and an intentionally wrong or competing `x1`; verify that recovery is not just coordinate prior leakage.

Rationale:

The purpose is to localize recall failure into mechanism stages: next-object desc proposal, x1 instance binding, or later bbox decoding. A rescue after `desc_only` argues against pure visual invisibility and points to proposal/enumeration failure. A rescue only after `desc_x1` points to x1 binding failure. Failure under both tiers remains ambiguous and must be interpreted with prefix quality, same-desc competition, and format validity controls.

Consequence:

First-pass reporting must label this as a counterfactual diagnostic surface with GT leakage. It must not be mixed with production rollout metrics or presented as an inference-time recall gain. Any later deployable variant must replace GT hints with model-derived top-k desc proposals, attention-derived region proposals, or multi-sample inventories and should be evaluated separately.

Success criteria:

Primary rescue success is `valid_parse == true`, target description is preserved by the forced hint condition, and `IoU(generated_box, target_gt_box) >= 0.5`. The first-pass report must also emit auxiliary thresholds at `IoU >= 0.3` and `IoU >= 0.75`, plus a duplicate guard that flags whether the generated box is a same-desc copy of an existing rollout prediction under `same-desc IoU > 0.95`.

Interpretation:

- `IoU >= 0.5`: main evidence that the missed target can be recovered under the hint condition.
- `IoU >= 0.3`: weak coarse-localization evidence, useful for distinguishing partial visual grounding from total failure.
- `IoU >= 0.75`: high-quality localization evidence.
- same-desc duplicate flag: prevents counting a copied existing prediction as target rescue.

Evidence:

- Scope: `none-yet`
- Handles: checkpoint `checkpoint-3664`, prior attention atlas roots under `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing` and `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_train200/attention_evidence_routing_teacher_forced_anchor`

## Self-Review Notes

Spec coverage:

- Missed-GT evidence routing: covered by Task 3 selected cases and candidate ledger.
- Same-desc IoU>`0.95` duplication guardrail: covered by Task 1 helper and Task 3 ledger policy.
- Attention feasibility before long run: covered by Task 4.
- Val200 observational lane: covered by Tasks 5-9.
- Training-distribution anchor: covered by Task 8 and final verification Step 5/8.
- 8-card efficiency: covered by Task 7 launcher.
- No production training: stated in scope and verification.
- No causal claim from attention-only evidence: recorded in report and summary as `causal_status=not_in_scope_for_this_plan`.

Execution stop condition:

Do not launch the 8-GPU atlas if the feasibility gate cannot prove that Qwen3-VL returns usable `attentions`, that `<|image_pad|>` spans match `image_grid_thw`, and that Lane D role positions align with the padded forward inputs.
