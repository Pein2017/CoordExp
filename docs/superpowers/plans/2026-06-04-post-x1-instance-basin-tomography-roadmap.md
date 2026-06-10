# Post-X1 Instance-Basin Tomography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build A3.3 post-`x1` instance-basin tomography for three checkpoints, comparing mechanism traits through slot-level posterior trajectories rather than detector accuracy.

**Architecture:** Create a new idea-wise analysis surface under `post_x1_instance_basin_tomography` instead of extending A3.2.  The pipeline first materializes a GT-controlled same-desc case universe and prefix-mode states, then runs sharded GPU slot-posterior readout, then aggregates trajectories, attraction matrices, prefix sensitivity, secondary greedy continuation, reports, galleries, and status gates.  Per-checkpoint template contracts are first-class and every row records checkpoint role plus template contract provenance.

**Tech Stack:** Python, PyTorch/Qwen3-VL HF runtime, YAML/JSONL artifacts, existing CoordExp compact-full rendering and image processing helpers, pytest, tmux, 8 single-GPU analysis shards.

## Preflight Audit Incorporation

This roadmap has absorbed the read-only preflight audit from five lanes:

- Research semantics: Lane 1 is a per-instance slot-posterior trajectory test; Lane 2 is a cluster-level aggregation of Lane 1, not an independent proof. Reports must state null/alternative hypotheses and avoid detector-accuracy ranking language.
- Template contracts: primary rows are rendered per checkpoint at runtime from semantic `prefix_objects`; do not persist checkpoint-agnostic `prefix_text` as runtime truth. Every runtime row records prompt/template hashes and exact row-separator contract.
- Posterior math: runtime must expose both full-vocab next-token logits and conditional coordinate-token logits. `coord_vocab_mass` is measured from the full vocabulary, not from a 1000-bin conditional softmax.
- Data/schema: real len12000 JSONL stores object boxes as `bbox_2d` coord-token strings. Normalization must preserve source surface/provenance and must not silently assume numeric pixel boxes.
- Artifact/status: sharded GPU output must materialize per-shard rows plus shard summaries and a merge manifest; status gates must compare shard totals to merged totals and validate per-role downstream non-empty counts.

---

## Scope Lock

Implement the design in:

- `docs/superpowers/specs/2026-06-04-post-x1-instance-basin-tomography-design.md`

Do not implement visual ledger / painting interventions or new training objectives in this plan.

Do not present this as production training, production eval, or a three-checkpoint accuracy ranking.

Primary clean pair:

- `fullobj_random_pure_ce_ckpt3668`
- `fullobj_sorted_pure_ce_ckpt3668`

Reference anchor:

- `et_rmp_ce_ckpt3664`

Primary artifact roots:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke
/data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3
```

Required interpretation labels:

- Pure-CE pair: `clean_pair`
- Pure-CE pair controlled group: `pure_ce_sorted_vs_random_no_newline`
- ET-RMP-CE: `reference_anchor`
- ET-RMP-CE caveat: `template_objective_confounded_reference`

## File Structure

Create a new analysis surface:

- Create: `src/analysis/post_x1_instance_basin_tomography/__init__.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/config.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/data_root_audit.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/case_universe.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/prefix_modes.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/posterior.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/runtime.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/trajectory.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/greedy.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/merge_report.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/gallery.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/status.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/runner.py`

Create scripts and configs:

- Create: `scripts/analysis/post_x1_instance_basin_tomography/run.py`
- Create: `scripts/analysis/post_x1_instance_basin_tomography/status.py`
- Create: `scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh`
- Create: `configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke.yaml`
- Create: `configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3.yaml`

Create tests:

- Create: `tests/analysis/post_x1_instance_basin_tomography/test_config.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_case_universe.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_prefix_modes.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_posterior.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_trajectory.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_greedy.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_merge_report.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_status.py`
- Create: `tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py`

Allowed reuse:

- `src/analysis/sorted_random_no_newline_phenotype/data_root_audit.py` behavior, copied or imported only if it stays generic.
- `src/analysis/sorted_random_no_newline_phenotype/fn_hint_runtime.py` coord-token scaling and bbox IoU semantics, but do not inherit its FN-specific buckets.
- `src/analysis/sorted_random_no_newline_phenotype/paired_probe.py` model/logit helper patterns, but do not import private A3.2 helpers into A3.3 public code.
- `src/analysis/prefix_state_transition_tomography/prefix_rendering.py` compact row rendering if it supports explicit row separator.
- `src/datasets/geometry.py` for bbox geometry where available.

Avoid:

- changing upstream HF model files;
- adding production dependencies;
- changing official inference/eval metrics;
- reusing A3.2 global `template_contract.row_separator == none` validation.

## Subagent Lanes For Implementation

Use at most six active subagents:

- Lane 1: config, provenance, status gates, runner CLI.
- Lane 2: case universe, same-desc GT selection, prefix modes.
- Lane 3: posterior math, R95, slot and trajectory taxonomy.
- Lane 4: real GPU posterior runtime and greedy continuation runtime.
- Lane 5: merge/report/gallery and plots.
- Lane 6: independent audit for template contracts, GT-only primary labels, and interpretation boundaries.

GPU work starts only after config tests, CPU case/prefix dry-run, and status index gates pass.

---

### Task 1: Config, Identity, And Per-Checkpoint Template Contracts

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/__init__.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/config.py`
- Create: `configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke.yaml`
- Create: `configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3.yaml`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create tests asserting the smoke config resolves exactly:

```python
from pathlib import Path

import pytest

from src.analysis.post_x1_instance_basin_tomography.config import load_config


CONFIG = Path(
    "configs/analysis/post_x1_instance_basin_tomography/"
    "three_ckpt_phase_a3_3_smoke.yaml"
)


def test_a3_3_config_resolves_three_roles_and_contracts():
    config = load_config(CONFIG)

    assert config.project_id == "post_x1_instance_basin_tomography"
    assert config.phase_id == "phase_a3_3"
    assert config.schema_version == "a3.3.v1"
    assert config.run_id == "three_ckpt_phase_a3_3_smoke"
    assert tuple(config.checkpoints) == (
        "fullobj_random_pure_ce_ckpt3668",
        "fullobj_sorted_pure_ce_ckpt3668",
        "et_rmp_ce_ckpt3664",
    )

    random = config.checkpoints["fullobj_random_pure_ce_ckpt3668"]
    sorted_ = config.checkpoints["fullobj_sorted_pure_ce_ckpt3668"]
    et = config.checkpoints["et_rmp_ce_ckpt3664"]

    assert random.comparison_role == "clean_pair"
    assert sorted_.comparison_role == "clean_pair"
    assert et.comparison_role == "reference_anchor"
    assert random.controlled_comparison_group == "pure_ce_sorted_vs_random_no_newline"
    assert sorted_.controlled_comparison_group == "pure_ce_sorted_vs_random_no_newline"
    assert et.controlled_comparison_group == "reference_anchor_not_controlled"

    assert random.template_contract.row_separator == "none"
    assert sorted_.template_contract.row_separator == "none"
    assert et.template_contract.row_separator == "newline"
    assert et.template_contract.contract_provenance == "legacy_compact_full_default_inferred"

    assert config.case_sampling.max_images == 64
    assert config.case_sampling.max_target_instances == 256
    assert config.case_sampling.min_same_desc_count == 3
    assert config.case_sampling.split_quotas["train"] > 0
    assert config.case_sampling.desc_cap_per_split > 0
    assert config.prefix.rollout_prefix_missing_policy_smoke == "skip_with_manifest"
    assert config.prefix.rollout_prefix_missing_policy_full == "fail"
    assert config.runtime.num_shards == 8
    assert config.greedy.sample_fraction == pytest.approx(0.10)
```

Add rejection tests:

```python
def test_config_rejects_missing_et_template_contract(tmp_path):
    path = tmp_path / "bad.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace("row_separator: newline", "row_separator: none", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="et_rmp_ce_ckpt3664.*row_separator"):
        load_config(path)


def test_config_rejects_global_template_contract(tmp_path):
    path = tmp_path / "bad.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        "checkpoints:",
        "template_contract:\\n  row_separator: none\\ncheckpoints:",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="per-checkpoint template_contract"):
        load_config(path)
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_config.py -q
```

Expected: fail because `post_x1_instance_basin_tomography.config` does not exist.

- [ ] **Step 3: Implement config dataclasses and validation**

Implement `src/analysis/post_x1_instance_basin_tomography/__init__.py`:

```python
PROJECT_ID = "post_x1_instance_basin_tomography"
PHASE_ID = "phase_a3_3"
SCHEMA_VERSION = "a3.3.v1"
SMOKE_RUN_ID = "three_ckpt_phase_a3_3_smoke"
FULL_RUN_ID = "three_ckpt_phase_a3_3"

CHECKPOINT_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
    "et_rmp_ce_ckpt3664",
)

PURE_CE_ROLES = (
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668",
)
ET_RMP_ROLE = "et_rmp_ce_ckpt3664"
```

Implement `config.py` dataclasses:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import CHECKPOINT_ROLES, ET_RMP_ROLE, FULL_RUN_ID, PHASE_ID, PROJECT_ID, SCHEMA_VERSION, SMOKE_RUN_ID


@dataclass(frozen=True)
class TemplateContractConfig:
    template_contract_id: str
    detection_sequence_format: str
    coordinate_surface: str
    bbox_format: str
    row_separator: str
    contract_provenance: str


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_path: Path
    training_ordering: str
    comparison_role: str
    controlled_comparison_group: str
    template_contract: TemplateContractConfig


@dataclass(frozen=True)
class CaseSamplingConfig:
    max_images: int
    max_target_instances: int
    min_same_desc_count: int
    min_object_count: int
    easy_sanity_max_fraction: float
    seed: int
    splits: tuple[str, ...]
    split_quotas: dict[str, int]
    desc_cap_per_split: int
    object_count_buckets: tuple[str, ...]


@dataclass(frozen=True)
class PrefixConfig:
    modes: tuple[str, ...]
    rollout_prefix_source_jsonl: Path | None
    rollout_prefix_missing_policy_smoke: str
    rollout_prefix_missing_policy_full: str


@dataclass(frozen=True)
class PosteriorConfig:
    strict_r95_axis_fraction: float
    strict_r95_cap_bins: int
    peak_mass_floor: float
    low_margin_threshold: float
    coord_mass_low_threshold: float


@dataclass(frozen=True)
class GreedyConfig:
    enabled: bool
    sample_fraction: float
    decode_policy: str
    constraint_policy: str


@dataclass(frozen=True)
class RuntimeConfig:
    num_shards: int
    max_new_tokens: int
    torch_dtype: str
    device_map: str


@dataclass(frozen=True)
class A33Config:
    project_id: str
    phase_id: str
    schema_version: str
    run_id: str
    artifact_root: Path
    train_jsonl: Path
    val_jsonl: Path
    image_root: Path
    checkpoints: dict[str, CheckpointConfig]
    case_sampling: CaseSamplingConfig
    prefix: PrefixConfig
    posterior: PosteriorConfig
    greedy: GreedyConfig
    runtime: RuntimeConfig
```

Validation rules:

```python
def _validate_config(config: A33Config, raw: Mapping[str, Any]) -> None:
    if "template_contract" in raw:
        raise ValueError("A3.3 requires per-checkpoint template_contract, not a global template_contract")
    if tuple(config.checkpoints) != CHECKPOINT_ROLES:
        raise ValueError("checkpoint roles must be ordered exactly as CHECKPOINT_ROLES")
    for role, checkpoint in config.checkpoints.items():
        contract = checkpoint.template_contract
        if contract.detection_sequence_format != "compact_full":
            raise ValueError(f"{role} detection_sequence_format must be compact_full")
        if contract.coordinate_surface != "coord_token":
            raise ValueError(f"{role} coordinate_surface must be coord_token")
        if contract.bbox_format != "xyxy":
            raise ValueError(f"{role} bbox_format must be xyxy")
        if role == ET_RMP_ROLE and contract.row_separator != "newline":
            raise ValueError("et_rmp_ce_ckpt3664 template_contract.row_separator must be newline")
        if role != ET_RMP_ROLE and contract.row_separator != "none":
            raise ValueError(f"{role} template_contract.row_separator must be none")
        if role != ET_RMP_ROLE and checkpoint.controlled_comparison_group != "pure_ce_sorted_vs_random_no_newline":
            raise ValueError(f"{role} must stay in the pure-CE controlled comparison group")
        if role == ET_RMP_ROLE and checkpoint.controlled_comparison_group != "reference_anchor_not_controlled":
            raise ValueError("ET-RMP-CE must be labeled as a non-controlled reference anchor")
        if not checkpoint.checkpoint_path.is_dir():
            raise ValueError(f"{role} checkpoint_path must exist: {checkpoint.checkpoint_path}")
    if config.runtime.num_shards != 8:
        raise ValueError("runtime.num_shards must be 8")
    if "len12000" in config.image_root.name:
        raise ValueError("image_root must not point to the len12000 JSONL directory")
```

- [ ] **Step 4: Add smoke and full YAMLs**

Smoke YAML:

```yaml
project_id: post_x1_instance_basin_tomography
phase_id: phase_a3_3
schema_version: a3.3.v1
run_id: three_ckpt_phase_a3_3_smoke
artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke
train_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
val_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
image_root: /data/CoordExp/public_data/coco/rescale_32_1024_bbox
case_sampling:
  max_images: 64
  max_target_instances: 256
  min_same_desc_count: 3
  min_object_count: 6
  easy_sanity_max_fraction: 0.10
  seed: 333
  splits: [train, val]
  split_quotas:
    train: 48
    val: 16
  desc_cap_per_split: 12
  object_count_buckets: ["6-9", "10-19", "20+"]
prefix:
  modes:
    - minimal_or_empty_prefix
    - canonical_sorted_gt_prefix_before_target
    - clean_non_target_same_desc_prefix
    - duplicate_same_desc_prefix
  rollout_prefix_source_jsonl:
  rollout_prefix_missing_policy_smoke: skip_with_manifest
  rollout_prefix_missing_policy_full: fail
posterior:
  strict_r95_axis_fraction: 0.04
  strict_r95_cap_bins: 8
  peak_mass_floor: 0.002
  low_margin_threshold: 0.05
  coord_mass_low_threshold: 0.01
greedy:
  enabled: true
  sample_fraction: 0.10
  decode_policy: free_text_unconstrained_greedy_temp0
  constraint_policy: none
runtime:
  num_shards: 8
  max_new_tokens: 64
  torch_dtype: bfloat16
  device_map: single_gpu
checkpoints:
  fullobj_random_pure_ce_ckpt3668:
    checkpoint_path: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
    training_ordering: random_permutation
    comparison_role: clean_pair
    controlled_comparison_group: pure_ce_sorted_vs_random_no_newline
    template_contract:
      template_contract_id: compact_full_no_newline_native_v1
      detection_sequence_format: compact_full
      coordinate_surface: coord_token
      bbox_format: xyxy
      row_separator: none
      contract_provenance: user_reported_training_contract
  fullobj_sorted_pure_ce_ckpt3668:
    checkpoint_path: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
    training_ordering: sorted
    comparison_role: clean_pair
    controlled_comparison_group: pure_ce_sorted_vs_random_no_newline
    template_contract:
      template_contract_id: compact_full_no_newline_native_v1
      detection_sequence_format: compact_full
      coordinate_surface: coord_token
      bbox_format: xyxy
      row_separator: none
      contract_provenance: user_reported_training_contract
  et_rmp_ce_ckpt3664:
    checkpoint_path: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
    training_ordering: random_permutation
    comparison_role: reference_anchor
    controlled_comparison_group: reference_anchor_not_controlled
    template_contract:
      template_contract_id: compact_full_newline_native_v1
      detection_sequence_format: compact_full
      coordinate_surface: coord_token
      bbox_format: xyxy
      row_separator: newline
      contract_provenance: legacy_compact_full_default_inferred
```

Full YAML uses the same roles and contracts, with:

```yaml
run_id: three_ckpt_phase_a3_3
artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3
case_sampling:
  max_images: 1024
  max_target_instances: 4096
  min_same_desc_count: 3
  min_object_count: 6
  easy_sanity_max_fraction: 0.10
  seed: 333
  splits: [train, val]
prefix:
  modes:
    - minimal_or_empty_prefix
    - canonical_sorted_gt_prefix_before_target
    - clean_non_target_same_desc_prefix
    - duplicate_same_desc_prefix
    - wrong_instance_same_desc_prefix
    - rollout_native_prefix_with_quality_label
greedy:
  enabled: true
  sample_fraction: 0.20
```

- [ ] **Step 5: Run config tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_config.py -q
```

Expected: all config tests pass.

---

### Task 2: Data Root Audit And Same-Desc Case Universe

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/data_root_audit.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/case_universe.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_case_universe.py`

- [ ] **Step 1: Write failing case-universe tests**

Test same-desc hard case selection:

```python
from src.analysis.post_x1_instance_basin_tomography.case_universe import (
    build_case_universe_rows,
    normalize_sample_objects,
)


def test_case_universe_selects_same_desc_targets_and_competitors():
    sample = {
        "image": "train/demo.jpg",
        "width": 1000,
        "height": 800,
        "objects": [
            {"desc": "person", "bbox": [100, 100, 200, 300]},
            {"desc": "person", "bbox": [300, 120, 420, 320]},
            {"desc": "person", "bbox": [600, 130, 760, 340]},
            {"desc": "chair", "bbox": [50, 500, 160, 700]},
            {"desc": "table", "bbox": [400, 500, 700, 760]},
            {"desc": "cup", "bbox": [710, 510, 750, 570]},
        ],
    }
    rows = build_case_universe_rows([sample], split="train", max_images=1, max_target_instances=12)
    target_rows = [row for row in rows if row["desc"] == "person"]
    assert len(target_rows) == 3
    for row in target_rows:
        assert row["same_desc_count"] == 3
        assert row["target_gt_idx"] in {0, 1, 2}
        assert set(row["competitor_gt_indices"]) == ({0, 1, 2} - {row["target_gt_idx"]})
        assert row["primary_basin_label_source"] == "same_desc_gt_instances"
```

Test real len12000 `bbox_2d` coord-token surface support and preserve provenance:

```python
def test_normalize_sample_objects_accepts_bbox_2d_coord_tokens():
    sample = {
        "image": "train/demo.jpg",
        "objects": [
            {
                "desc": "person",
                "bbox_2d": [
                    "<|coord_010|>",
                    "<|coord_020|>",
                    "<|coord_110|>",
                    "<|coord_220|>",
                ],
            },
        ]
    }
    objects = normalize_sample_objects(sample, split="train", source_line_id=7)
    assert objects[0]["bbox_coord_token_xyxy"] == [10, 20, 110, 220]
    assert objects[0]["bbox_surface"] == "bbox_2d_coord_token_xyxy"
    assert objects[0]["bbox_source_field"] == "bbox_2d"
    assert objects[0]["source_line_id"] == 7
```

Keep legacy numeric support only for unit fixtures:

```python
def test_normalize_sample_objects_accepts_numeric_fixture_surfaces():
    sample = {"objects": [{"desc": "person", "bbox": [10, 20, 110, 220]}]}
    objects = normalize_sample_objects(sample, split="unit", source_line_id=0)
    assert objects[0]["bbox_coord_token_xyxy"] == [10, 20, 110, 220]
    assert objects[0]["bbox_surface"] == "numeric_xyxy_fixture"
```

- [ ] **Step 2: Run case-universe tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_case_universe.py -q
```

Expected: fail because `case_universe.py` does not exist.

- [ ] **Step 3: Implement object normalization and case rows**

Implement functions:

```python
def normalize_sample_objects(sample: Mapping[str, Any], *, split: str, source_line_id: int) -> list[dict[str, Any]]:
    objects = []
    for gt_idx, raw in enumerate(sample.get("objects") or []):
        desc = str(raw.get("desc") or raw.get("label") or "").strip()
        bbox, surface, source_field = _bbox_coord_token_xyxy_from_raw(raw)
        if not desc or bbox is None or not _valid_xyxy(bbox):
            continue
        objects.append(
            {
                "gt_idx": gt_idx,
                "desc": desc,
                "bbox_coord_token_xyxy": bbox,
                "bbox_surface": surface,
                "bbox_source_field": source_field,
                "split": split,
                "source_line_id": source_line_id,
            }
        )
    return objects
```

`_bbox_coord_token_xyxy_from_raw` must accept, in this priority order:

- `bbox_2d`: coord-token strings like `"<|coord_699|>"`; this is the primary len12000 surface.
- `bbox` or `bbox_xyxy`: numeric fixture surfaces only.
- `points: [[x1, y1], [x2, y2]]`: fixture/legacy surface only.

Do not import A3.2 `_coerce_bbox` for this parser because it does not handle coord-token strings. Prefer a local parser patterned after `prefix_index._parse_bbox_2d`.

Implement `build_case_universe_rows` so each target row includes:

```python
{
    "case_id": "a33-{split}-{image_id}-{desc}-{target_gt_idx}",
    "split": split,
    "image_id": image_id,
    "image_path": image_path,
    "width": width,
    "height": height,
    "desc": desc,
    "target_gt_idx": target_gt_idx,
    "target_bbox_coord_token_xyxy": target_bbox,
    "target_bbox_surface": target_object["bbox_surface"],
    "target_bbox_source_field": target_object["bbox_source_field"],
    "same_desc_gt_indices": same_desc_indices,
    "competitor_gt_indices": competitor_indices,
    "same_desc_competitor_bboxes": [...],
    "same_desc_count": len(same_desc_indices),
    "object_count": len(objects),
    "primary_basin_label_source": "same_desc_gt_instances",
    "x1_anchor_unique_under_r95": bool,
    "anchor_ambiguity_bucket": "unique" | "near_collision" | "exact_x1_collision",
    "primary_denominator_eligible": bool,
    "source_jsonl_path": str(source_jsonl_path),
    "source_jsonl_sha256": sha256,
    "source_line_id": source_line_id,
}
```

Rows with exact/near x1 collisions remain materialized for audit, but `primary_denominator_eligible` is `False` so they do not decide primary basin-rate summaries.

- [ ] **Step 4: Implement data-root audit**

`data_root_audit.py` writes:

```python
{
    "status": "ok",
    "train_jsonl": str(config.train_jsonl),
    "val_jsonl": str(config.val_jsonl),
    "image_root": str(config.image_root),
    "row_counts": {"train": train_count, "val": val_count},
    "same_desc_hard_candidate_count": count,
    "missing_image_examples": missing[:10],
}
```

Reject `image_root` if it points to `*_len12000`.

- [ ] **Step 5: Run case-universe tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_case_universe.py -q
```

Expected: all tests pass.

---

### Task 3: Prefix Modes And Per-Checkpoint Row Separator Rendering

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/prefix_modes.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_prefix_modes.py`

- [ ] **Step 1: Write failing prefix-mode tests**

Test separator handling:

```python
from src.analysis.post_x1_instance_basin_tomography.prefix_modes import (
    render_compact_prefix_rows,
    render_forced_state_prompt,
)


OBJECTS = [
    {"desc": "person", "bbox_xyxy": [100, 100, 200, 300]},
    {"desc": "chair", "bbox_xyxy": [400, 500, 600, 700]},
]


def test_render_prefix_uses_no_newline_separator_for_pure_ce():
    text = render_compact_prefix_rows(OBJECTS, row_separator="none")
    assert "\n" not in text
    assert text.count("<|object_ref_start|>") == 2


def test_render_prefix_uses_newline_separator_for_et_rmp():
    text = render_compact_prefix_rows(OBJECTS, row_separator="newline")
    assert "\n" in text
    assert text.count("\n") == 1


def test_forced_partial_uses_separator_between_completed_rows_and_forced_row_only():
    text = render_forced_state_prompt(
        prefix_objects=OBJECTS,
        desc="person",
        forced_x1=123,
        forced_state="post_x1",
        row_separator="newline",
    )
    assert text.count("\n") == 2
    assert "<|box_start|><|coord_123|>" in text
```

Test prefix buckets:

```python
from src.analysis.post_x1_instance_basin_tomography.prefix_modes import build_prefix_mode_rows


def test_prefix_modes_do_not_include_target_for_recall_modes():
    case = {
        "case_id": "case-1",
        "desc": "person",
        "target_gt_idx": 1,
        "same_desc_gt_indices": [0, 1, 2],
        "competitor_gt_indices": [0, 2],
        "objects": [
            {"gt_idx": 0, "desc": "person", "bbox_xyxy": [10, 10, 50, 80]},
            {"gt_idx": 1, "desc": "person", "bbox_xyxy": [100, 10, 150, 90]},
            {"gt_idx": 2, "desc": "person", "bbox_xyxy": [200, 10, 250, 90]},
        ],
    }
    rows = build_prefix_mode_rows(case, modes=("canonical_sorted_gt_prefix_before_target", "clean_non_target_same_desc_prefix"))
    for row in rows:
        assert row["target_leak"] is False
        assert 1 not in row["prefix_gt_indices"]
```

- [ ] **Step 2: Run prefix tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_prefix_modes.py -q
```

Expected: fail because `prefix_modes.py` does not exist.

- [ ] **Step 3: Implement rendering**

Implement:

```python
def render_compact_prefix_rows(objects: Sequence[Mapping[str, Any]], *, row_separator: str) -> str:
    rendered = [_render_one_compact_row(obj) for obj in objects]
    if row_separator == "none":
        return "".join(rendered)
    if row_separator == "newline":
        return "\n".join(rendered)
    raise ValueError(f"unsupported row_separator: {row_separator}")
```

`_render_one_compact_row` must output:

```text
<|object_ref_start|>{desc}<|object_ref_end|><|box_start|><|box_{x1}|><|box_{y1}|><|box_{x2}|><|box_{y2}|><|box_end|>
```

Use coord-token integer boxes in `0..999`.

Implement `render_forced_state_prompt(...)` from semantic objects, not from a persisted cross-checkpoint `prefix_text`:

```python
def render_forced_state_prompt(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    desc: str,
    forced_x1: int,
    forced_state: str,
    row_separator: str,
) -> str:
    prefix = render_compact_prefix_rows(prefix_objects, row_separator=row_separator)
    forced = f"<|object_ref_start|>{desc}<|object_ref_end|><|box_start|><|coord_{forced_x1:03d}|>"
    if not prefix:
        return forced
    separator = "" if row_separator == "none" else "\n"
    return prefix + separator + forced
```

Separator rule: separators appear between completed compact rows and between the last completed prefix row and the forced partial row; never insert a separator inside one object row.

- [ ] **Step 4: Implement prefix modes**

`build_prefix_mode_rows` creates rows with:

```python
{
    "prefix_mode": mode,
    "prefix_source_kind": "gt_synthetic" | "rollout_native" | "none",
    "prefix_source_artifact": str | None,
    "prefix_source_line_id": int | None,
    "prefix_quality": "good" | "bad" | "target_leak",
    "prefix_gt_indices": [...],
    "prefix_objects": [...],
    "target_leak": bool,
    "duplicate_source_gt_idx": int | None,
    "wrong_instance_replacement_gt_idx": int | None,
    "match_iou_policy": "gt_exact" | "rollout_iou50" | "not_applicable",
    "denominator_eligible": bool,
    "bad_prefix_bucket": None | "duplicate_prefix" | "wrong_instance_prefix" | "target_leak_prefix" | "rollout_unmatched_prefix" | "rollout_duplicate_prefix",
}
```

Minimum modes for smoke:

- `minimal_or_empty_prefix`
- `canonical_sorted_gt_prefix_before_target`
- `clean_non_target_same_desc_prefix`
- `duplicate_same_desc_prefix`

Full adds:

- `wrong_instance_same_desc_prefix`
- `rollout_native_prefix_with_quality_label`

If `rollout_native_prefix_with_quality_label` has no configured source JSONL:

- smoke: emit `prefix_mode_skipped_rows.jsonl` with `skip_reason="rollout_prefix_source_missing"` and continue;
- full: fail before GPU unless the config explicitly changes `rollout_prefix_missing_policy_full`.

The prefix stage must also write `prefix_mode_summary.json` with `prefix_modes_requested`, `prefix_modes_materialized`, and `prefix_modes_skipped`.

- [ ] **Step 5: Run prefix tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_prefix_modes.py -q
```

Expected: all tests pass.

---

### Task 4: Slot Posterior Math, R95, And Basin Classification

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/posterior.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_posterior.py`

- [ ] **Step 1: Write failing posterior tests**

```python
import torch

from src.analysis.post_x1_instance_basin_tomography.posterior import (
    axis_len_for_slot,
    classify_slot_posterior,
    strict_r95_radius,
)


def test_strict_r95_uses_axis_fraction_and_cap():
    assert strict_r95_radius(axis_len=20, fraction=0.04, cap=8) == 0
    assert strict_r95_radius(axis_len=50, fraction=0.04, cap=8) == 2
    assert strict_r95_radius(axis_len=200, fraction=0.04, cap=8) == 8
    assert strict_r95_radius(axis_len=400, fraction=0.04, cap=8) == 8


def test_axis_len_for_slot_uses_matching_bbox_axis():
    bbox = [10, 20, 210, 120]
    assert axis_len_for_slot("x1", bbox) == 200
    assert axis_len_for_slot("x2", bbox) == 200
    assert axis_len_for_slot("y1", bbox) == 100
    assert axis_len_for_slot("y2", bbox) == 100


def test_classify_slot_posterior_prefers_target_over_competitor():
    full_vocab_logits = torch.full((1300,), -10.0)
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits[200] = 5.0
    full_vocab_logits[400] = 3.0
    full_vocab_logits[42] = 1.0
    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="y1",
        target_value=100,
        target_axis_len=80,
        competitors=[{"gt_idx": 2, "value": 300, "axis_len": 90}],
        low_margin_threshold=0.05,
    )
    assert row["winner_bucket"] == "target_instance"
    assert row["winner_instance_id"] == "target"
    assert row["target_r95_hit"] is True
    assert row["best_competitor_r95_hit"] is False
    assert row["boundary_extreme_flag"] is False
    assert 0.0 < row["coord_vocab_mass"] < 1.0
    assert row["noncoord_top_token_id"] == 42
```

Boundary extreme test:

```python
def test_classify_slot_posterior_marks_boundary_extreme():
    full_vocab_logits = torch.full((1300,), -10.0)
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits[1099] = 5.0
    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x2",
        target_value=650,
        target_axis_len=100,
        competitors=[],
        low_margin_threshold=0.05,
    )
    assert row["winner_bucket"] == "boundary_extreme"
    assert row["boundary_extreme_flag"] is True
```

Low-margin and other-desc tests:

```python
def test_classify_slot_posterior_marks_low_margin_tied_state():
    full_vocab_logits = torch.full((1300,), -10.0)
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits[200] = 5.0
    full_vocab_logits[400] = 4.99
    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x1",
        target_value=100,
        target_axis_len=100,
        competitors=[{"gt_idx": 2, "desc": "person", "value": 300, "axis_len": 100}],
        low_margin_threshold=0.05,
    )
    assert row["low_margin_flag"] is True
    assert row["slot_taxonomy"] == "ambiguous_tied"


def test_classify_slot_posterior_can_label_other_desc_object():
    full_vocab_logits = torch.full((1300,), -10.0)
    coord_token_ids = list(range(100, 1100))
    full_vocab_logits[700] = 5.0
    row = classify_slot_posterior(
        full_vocab_logits=full_vocab_logits,
        coord_token_ids=coord_token_ids,
        slot="x1",
        target_value=100,
        target_axis_len=80,
        competitors=[],
        other_desc_objects=[{"gt_idx": 9, "desc": "chair", "value": 600, "axis_len": 80}],
        low_margin_threshold=0.05,
    )
    assert row["winner_bucket"] == "other_desc_object"
    assert row["winner_instance_id"] == 9
```

- [ ] **Step 2: Run posterior tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_posterior.py -q
```

Expected: fail because `posterior.py` does not exist.

- [ ] **Step 3: Implement posterior classifier**

Implement:

```python
def strict_r95_radius(*, axis_len: int, fraction: float, cap: int) -> int:
    return int(min(cap, max(0, int(axis_len * fraction))))
```

Implement two posterior views:

- full-vocab view: `full_vocab_probs = softmax(full_vocab_logits)`;
- conditional coord view: gather `coord_token_ids` into `coord_full_vocab_probs`, then renormalize to `coord_conditional_probs`.

`coord_vocab_mass` is `coord_full_vocab_probs.sum()` from the full-vocab view. It must not be computed after the 1000-bin renormalization.

Implement mass extraction over conditional coord bins `0..999`:

```python
def neighborhood_mass(probs: torch.Tensor, center: int, radius: int) -> float:
    lo = max(0, int(center) - int(radius))
    hi = min(999, int(center) + int(radius))
    return float(probs[lo : hi + 1].sum().item())
```

`classify_slot_posterior` returns fields:

```python
{
    "slot": slot,
    "artifact_schema_version": "a3.3.v1",
    "row_schema_version": "slot_posterior.v1",
    "target_slot_mass": target_mass,
    "best_same_desc_competitor_slot_mass": competitor_mass,
    "target_vs_competitor_margin": target_mass - competitor_mass,
    "winner_instance_id": "target" | competitor_gt_idx | None,
    "winner_bucket": "target_instance" | "same_desc_competitor" | "other_desc_object" | "background_or_outlier" | "boundary_extreme",
    "slot_taxonomy": "target_dependent" | "competitor_dependent" | "other_desc_dependent" | "background_dependent" | "boundary_dependent" | "ambiguous_tied" | "invalid_low_coord_mass",
    "coord_vocab_mass": float(coord_full_vocab_probs.sum().item()),
    "coord_mass_low_flag": coord_vocab_mass < coord_mass_low_threshold,
    "noncoord_top_token_id": int,
    "noncoord_top_prob": float,
    "top_peak_value": top_idx,
    "top_peak_mass": top_prob,
    "top_peak_logit": top_coord_logit,
    "target_center_logit": target_logit,
    "best_competitor_center_logit": competitor_logit,
    "target_rank": rank,
    "low_margin_flag": bool,
    "target_r95_hit": bool(abs(top_idx - target_value) <= target_radius),
    "best_competitor_r95_hit": bool,
    "boundary_extreme_flag": top_idx in {0, 999},
    "background_or_outlier_flag": winner_bucket == "background_or_outlier",
}
```

Boundary extremes are orthogonal flags. If a true target or competitor x coordinate is legitimately at 0 or 999, preserve `winner_bucket=target_instance` or `same_desc_competitor` and set `boundary_extreme_flag=True`; do not erase identity by forcing `winner_bucket=boundary_extreme`.

- [ ] **Step 4: Run posterior tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_posterior.py -q
```

Expected: all tests pass.

---

### Task 5: Trajectory Rows And Basin Attraction Matrix

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/trajectory.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_trajectory.py`

- [ ] **Step 1: Write failing trajectory tests**

```python
from src.analysis.post_x1_instance_basin_tomography.trajectory import (
    build_attraction_matrix_rows,
    classify_trajectory,
)


def test_classify_trajectory_stays_target_all_slots():
    slots = [
        {"slot": "y1", "winner_bucket": "target_instance", "winner_instance_id": "target"},
        {"slot": "x2", "winner_bucket": "target_instance", "winner_instance_id": "target"},
        {"slot": "y2", "winner_bucket": "target_instance", "winner_instance_id": "target"},
    ]
    row = classify_trajectory(slots)
    assert row["trajectory_bucket"] == "stay_target_all_slots"
    assert row["basin_stay_rate"] == 1.0


def test_classify_trajectory_detects_early_switch():
    slots = [
        {"slot": "y1", "winner_bucket": "same_desc_competitor", "winner_instance_id": 2},
        {"slot": "x2", "winner_bucket": "same_desc_competitor", "winner_instance_id": 2},
        {"slot": "y2", "winner_bucket": "same_desc_competitor", "winner_instance_id": 2},
    ]
    row = classify_trajectory(slots)
    assert row["trajectory_bucket"] == "early_switch"
    assert row["switch_slot"] == "y1"
    assert row["switched_to_instance_id"] == 2


def test_classify_trajectory_precedence_marks_invalid_low_coord_mass_first():
    slots = [
        {"slot": "y1", "winner_bucket": "same_desc_competitor", "winner_instance_id": 2, "coord_mass_low_flag": True},
        {"slot": "x2", "winner_bucket": "target_instance", "winner_instance_id": "target", "coord_mass_low_flag": False},
        {"slot": "y2", "winner_bucket": "target_instance", "winner_instance_id": "target", "coord_mass_low_flag": False},
    ]
    row = classify_trajectory(slots)
    assert row["trajectory_bucket"] == "invalid_low_coord_mass"
```

Attraction matrix test:

```python
def test_attraction_matrix_records_forced_anchor_and_slot_winners():
    trajectory_rows = [
        {
            "case_id": "c1",
            "checkpoint_role": "fullobj_sorted_pure_ce_ckpt3668",
            "desc": "person",
            "forced_anchor_gt_idx": 1,
            "slot_winners": {"y1": 1, "x2": 1, "y2": 2},
            "trajectory_bucket": "partial_target_then_switch",
        }
    ]
    rows = build_attraction_matrix_rows(trajectory_rows)
    assert rows[0]["forced_anchor_gt_idx"] == 1
    assert rows[0]["winner_y2_gt_idx"] == 2
    assert rows[0]["diagonal_y1"] is True
    assert rows[0]["diagonal_y2"] is False
```

- [ ] **Step 2: Run trajectory tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_trajectory.py -q
```

Expected: fail because `trajectory.py` does not exist.

- [ ] **Step 3: Implement trajectory taxonomy**

Implement buckets:

```python
SLOT_ORDER = ("y1", "x2", "y2")

def classify_trajectory(slot_rows):
    buckets = [row["winner_bucket"] for row in slot_rows]
    if any(row.get("coord_mass_low_flag") for row in slot_rows):
        return {"trajectory_bucket": "invalid_low_coord_mass", "switch_slot": None, "basin_stay_rate": _target_rate(slot_rows)}
    if any(row.get("low_margin_flag") for row in slot_rows):
        return {"trajectory_bucket": "ambiguous_tied", "switch_slot": None, "basin_stay_rate": _target_rate(slot_rows)}
    if all(bucket == "target_instance" for bucket in buckets):
        return {"trajectory_bucket": "stay_target_all_slots", "switch_slot": None, "basin_stay_rate": 1.0}
    if buckets[0] == "same_desc_competitor":
        return {"trajectory_bucket": "early_switch", "switch_slot": slot_rows[0]["slot"], "basin_stay_rate": 0.0}
    if "same_desc_competitor" in buckets:
        first = next(row for row in slot_rows if row["winner_bucket"] == "same_desc_competitor")
        return {"trajectory_bucket": "partial_target_then_switch", "switch_slot": first["slot"], "basin_stay_rate": _target_rate(slot_rows)}
    if all(bucket == "boundary_extreme" for bucket in buckets):
        return {"trajectory_bucket": "boundary_extreme_dominated", "switch_slot": None, "basin_stay_rate": 0.0}
    if any(bucket == "background_or_outlier" for bucket in buckets):
        return {"trajectory_bucket": "background_drift", "switch_slot": None, "basin_stay_rate": _target_rate(slot_rows)}
    if any(bucket == "other_desc_object" for bucket in buckets):
        return {"trajectory_bucket": "other_desc_drift", "switch_slot": None, "basin_stay_rate": _target_rate(slot_rows)}
    return {"trajectory_bucket": "ambiguous_tied", "switch_slot": None, "basin_stay_rate": _target_rate(slot_rows)}
```

Lane 2 attraction matrix is cluster-level aggregation over Lane 1 rows:

- diagonal cells: forced anchor `i` remains winner `i`;
- off-diagonal cells: forced anchor `i` is pulled to same-desc competitor `j`;
- report `diagonal_rate_by_checkpoint`, `off_diagonal_mass_by_checkpoint`, and `top_off_diagonal_edges`.

Null hypothesis: after forced `desc+x1_i`, the model remains in target basin across `y1/x2/y2` independent of prefix quality. Alternative: the model frequently switches to competitor/background/other-desc basins, meaning x1 alone is not a stable instance binder.

- [ ] **Step 4: Run trajectory tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_trajectory.py -q
```

Expected: all tests pass.

---

### Task 6: Real Slot-Posterior Runtime

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/runtime.py`
- Modify only if needed: `scripts/analysis/post_x1_instance_basin_tomography/run.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py`

- [ ] **Step 1: Write runtime interface tests with a fake model scorer**

```python
import torch

from src.analysis.post_x1_instance_basin_tomography.runtime import score_forced_slot_state


class FakeSlotScorer:
    def score(self, prompt_text, image_path, slot):
        full_vocab_logits = torch.full((1300,), -10.0)
        coord_token_ids = list(range(100, 1100))
        full_vocab_logits[223] = 5.0
        return {"full_vocab_logits": full_vocab_logits, "coord_token_ids": coord_token_ids}


def test_score_forced_slot_state_records_contract_and_checkpoint_role(tmp_path):
    row = score_forced_slot_state(
        scorer=FakeSlotScorer(),
        checkpoint_role="fullobj_random_pure_ce_ckpt3668",
        template_contract={"row_separator": "none", "template_contract_id": "compact_full_no_newline_native_v1"},
        case_row={
            "case_id": "c1",
            "image_path": str(tmp_path / "x.jpg"),
            "desc": "person",
            "target_gt_idx": 1,
            "target_bbox_coord_token_xyxy": [100, 120, 200, 260],
            "competitor_gt_indices": [],
        },
        prefix_row={"prefix_mode": "minimal_or_empty_prefix", "prefix_objects": []},
        forced_state="post_x1",
        slot="y1",
        target_value=123,
        target_axis_len=140,
        competitors=[],
        low_margin_threshold=0.05,
        strict_r95_axis_fraction=0.04,
        strict_r95_cap_bins=8,
    )
    assert row["checkpoint_role"] == "fullobj_random_pure_ce_ckpt3668"
    assert row["template_contract"]["row_separator"] == "none"
    assert row["slot"] == "y1"
    assert row["winner_bucket"] == "target_instance"
    assert row["template_prompt_hash"]
    assert row["forced_prompt_sha256"]
    assert row["runtime_kind"] == "mock_runtime"
```

- [ ] **Step 2: Run runtime interface tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py::test_score_forced_slot_state_records_contract_and_checkpoint_role -q
```

Expected: fail because runtime interface is not implemented.

- [ ] **Step 3: Implement runtime interface**

Implement pure Python wrapper first:

```python
def score_forced_slot_state(
    *,
    scorer,
    checkpoint_role: str,
    template_contract: Mapping[str, Any],
    case_row: Mapping[str, Any],
    prefix_row: Mapping[str, Any],
    forced_state: str,
    slot: str,
    target_value: int,
    target_axis_len: int,
    competitors: Sequence[Mapping[str, Any]],
    low_margin_threshold: float,
    strict_r95_axis_fraction: float,
    strict_r95_cap_bins: int,
) -> dict[str, Any]:
    prompt_text = render_forced_state_prompt(
        prefix_objects=prefix_row.get("prefix_objects") or (),
        desc=str(case_row["desc"]),
        forced_x1=int(case_row["target_bbox_coord_token_xyxy"][0]),
        forced_state=forced_state,
        row_separator=str(template_contract["row_separator"]),
    )
    scored = scorer.score(prompt_text, case_row["image_path"], slot)
    posterior = classify_slot_posterior(
        full_vocab_logits=scored["full_vocab_logits"],
        coord_token_ids=scored["coord_token_ids"],
        slot=slot,
        target_value=target_value,
        target_axis_len=target_axis_len,
        competitors=competitors,
        low_margin_threshold=low_margin_threshold,
        strict_r95_axis_fraction=strict_r95_axis_fraction,
        strict_r95_cap_bins=strict_r95_cap_bins,
    )
    return {
        **posterior,
        "runtime_kind": getattr(scorer, "runtime_kind", "mock_runtime"),
        "runtime_id": getattr(scorer, "runtime_id", "fake-slot-scorer"),
        "checkpoint_role": checkpoint_role,
        "template_contract": dict(template_contract),
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "template_prompt_hash": sha256_json(template_contract),
        "assistant_prefix_sha256": sha256_json(prefix_row.get("prefix_objects") or []),
        "forced_prompt_sha256": sha256_text(prompt_text),
        "case_id": case_row["case_id"],
        "prefix_mode": prefix_row["prefix_mode"],
        "forced_state": forced_state,
        "target_gt_idx": case_row["target_gt_idx"],
    }
```

Implement `QwenSlotScorer` after fake tests pass.  It must:

- load one checkpoint per process/GPU;
- build image+text inputs through the existing Qwen processor;
- compute one forward pass for each forced state;
- extract the logits for the next coordinate-token position;
- map coord token IDs to bins `0..999`;
- return full-vocab logits plus `coord_token_ids`, not only a length-1000 tensor;
- never constrain generation because this is posterior readout, not decode.
- record checkpoint fingerprint, processor/chat-template fingerprint, `mock_runtime=False`, `dry_run=False`, and device/shard metadata in row provenance.

- [ ] **Step 4: Run runtime interface tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py::test_score_forced_slot_state_records_contract_and_checkpoint_role -q
```

Expected: pass.

---

### Task 7: Secondary Greedy Continuation

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/greedy.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_greedy.py`

- [ ] **Step 1: Write failing greedy parse/classification tests**

```python
from src.analysis.post_x1_instance_basin_tomography.greedy import classify_greedy_continuation


def test_greedy_continuation_matches_target_box():
    row = classify_greedy_continuation(
        generated_box=[100, 120, 200, 260],
        target_box=[100, 120, 200, 260],
        competitor_boxes=[{"gt_idx": 2, "bbox_xyxy": [300, 120, 420, 260]}],
    )
    assert row["greedy_bucket"] == "target_iou50"
    assert row["matched_instance_id"] == "target"


def test_greedy_continuation_matches_competitor_box():
    row = classify_greedy_continuation(
        generated_box=[300, 120, 420, 260],
        target_box=[100, 120, 200, 260],
        competitor_boxes=[{"gt_idx": 2, "bbox_xyxy": [300, 120, 420, 260]}],
    )
    assert row["greedy_bucket"] == "same_desc_competitor_iou50"
    assert row["matched_instance_id"] == 2
```

- [ ] **Step 2: Run greedy tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_greedy.py -q
```

Expected: fail because `greedy.py` does not exist.

- [ ] **Step 3: Implement greedy classifier and runtime hook**

Implement:

```python
def classify_greedy_continuation(*, generated_box, target_box, competitor_boxes):
    target_iou = bbox_iou_xyxy(generated_box, target_box)
    best_competitor = _best_competitor_iou(generated_box, competitor_boxes)
    if target_iou >= 0.50:
        bucket = "target_iou50"
        matched = "target"
    elif best_competitor["iou"] >= 0.50:
        bucket = "same_desc_competitor_iou50"
        matched = best_competitor["gt_idx"]
    else:
        bucket = "background_or_invalid"
        matched = None
    return {
        "greedy_bucket": bucket,
        "matched_instance_id": matched,
        "target_iou": target_iou,
        "best_same_desc_competitor_iou": best_competitor["iou"],
    }
```

Implement real greedy runtime as secondary stage only:

- deterministic free-text generation;
- max_new_tokens from config;
- parse compact coord tokens;
- record parse failures without removing posterior rows.

- [ ] **Step 4: Run greedy tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_greedy.py -q
```

Expected: all tests pass.

---

### Task 8: Runner, Stages, Status, And Artifact Gates

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/runner.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/status.py`
- Create: `scripts/analysis/post_x1_instance_basin_tomography/run.py`
- Create: `scripts/analysis/post_x1_instance_basin_tomography/status.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_status.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py`

- [ ] **Step 1: Write failing runner dry-run tests**

```python
from pathlib import Path

from src.analysis.post_x1_instance_basin_tomography.runner import run_stages


CONFIG = Path("configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke.yaml")


def test_runner_dry_run_lists_a3_3_stages():
    result = run_stages(CONFIG, stages="data_root_audit,case_universe,prefix_modes", dry_run=True)
    assert result["project_id"] == "post_x1_instance_basin_tomography"
    assert result["phase_id"] == "phase_a3_3"
    assert result["checkpoint_roles"] == [
        "fullobj_random_pure_ce_ckpt3668",
        "fullobj_sorted_pure_ce_ckpt3668",
        "et_rmp_ce_ckpt3664",
    ]
    assert result["stage_results"]["case_universe"]["dry_run"] is True
```

Status test:

```python
from src.analysis.post_x1_instance_basin_tomography.status import evaluate_status


def test_status_rejects_missing_template_contracts(tmp_path):
    status = evaluate_status(tmp_path)
    assert status["status"] == "incomplete"
    assert "template_contracts_present" in status["failed_gates"]
```

- [ ] **Step 2: Run runner/status tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py \
  tests/analysis/post_x1_instance_basin_tomography/test_status.py -q
```

Expected: fail because runner/status do not exist.

- [ ] **Step 3: Implement stage graph**

Stages:

```python
STAGE_NAMES = (
    "data_root_audit",
    "case_universe",
    "prefix_modes",
    "validate_index",
    "slot_posterior",
    "slot_merge",
    "trajectory",
    "attraction_matrix",
    "prefix_sensitivity",
    "greedy_continuation",
    "report",
    "gallery",
    "finalize",
)
```

GPU stages:

```python
GPU_STAGES = {"slot_posterior", "greedy_continuation"}
SHARDED_GPU_STAGES = {"slot_posterior"}
```

`run_stages` rules:

- CPU stages may run without `--launch-context`.
- `slot_posterior` requires `--shard-id` or launch context.
- real GPU stages require `--real-runtime`.
- `--mock-runtime` can materialize deterministic fake rows for tests only.

- [ ] **Step 4: Implement status gates**

Required index artifacts:

```python
INDEX_READY_ARTIFACTS = (
    "config_resolved.json",
    "template_contracts.json",
    "data_root_audit.json",
    "case_universe.jsonl",
    "prefix_modes.jsonl",
    "case_universe_summary.json",
)
```

Required final artifacts:

```python
FINAL_ARTIFACTS = INDEX_READY_ARTIFACTS + (
    "slot_posterior_rows.jsonl",
    "slot_posterior_shard_summaries.jsonl",
    "merge_manifest.json",
    "trajectory_rows.jsonl",
    "basin_attraction_matrix.jsonl",
    "prefix_sensitivity_rows.jsonl",
    "greedy_continuation_rows.jsonl",
    "summary.json",
    "report.md",
    "gallery/index.md",
)
```

Semantic gates:

- `config_resolved.json`, `template_contracts.json`, `summary.json`, `report.md`, and `gallery/index.md` include `artifact_schema_version`, `config_path`, `config_sha256`, `code_revision`, checkpoint paths/fingerprints, template contracts, `runtime_kind`, `runtime_id`, `mock_runtime`, and `dry_run` where applicable;
- exact template contract validation is per row: checkpoint role, `template_contract_id`, row separator, coordinate surface, bbox format, prompt hashes, and rendered forced prompt hash must agree;
- every slot row has one of the three checkpoint roles;
- every slot row has `template_contract.template_contract_id`;
- ET-RMP rows carry `comparison_role=reference_anchor`;
- primary labels use `primary_basin_label_source=same_desc_gt_instances`;
- shard artifacts exist as `slot_posterior_shards/shard_0.jsonl` through `slot_posterior_shards/shard_7.jsonl`;
- `slot_posterior_shard_summaries.jsonl` row totals equal merged `slot_posterior_rows.jsonl` totals;
- `merge_manifest.json` records every input shard path, sha256, row count, and merge timestamp;
- every downstream stage has non-empty rows for each materialized checkpoint role where denominator rows exist: slot posterior, trajectory, attraction matrix, prefix sensitivity, and greedy continuation;
- `prefix_mode_summary.json` records `prefix_modes_requested`, `prefix_modes_materialized`, and `prefix_modes_skipped`;
- boundary extremes are counted in `summary.json`;
- report contains `not by final detector accuracy`;
- report does not contain uncaveated ranking/detector metric language such as `ET-RMP is better`, `sorted is best`, `AP`, `AR100`, `mAP`, `F1-score`, or `wins`.

- [ ] **Step 5: Run runner/status tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py \
  tests/analysis/post_x1_instance_basin_tomography/test_status.py -q
```

Expected: all tests pass.

---

### Task 9: Merge, Report, Gallery, And Plots

**Files:**
- Create: `src/analysis/post_x1_instance_basin_tomography/merge_report.py`
- Create: `src/analysis/post_x1_instance_basin_tomography/gallery.py`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_merge_report.py`

- [ ] **Step 1: Write failing report tests**

```python
from src.analysis.post_x1_instance_basin_tomography.merge_report import build_summary, build_report_markdown


def test_report_preserves_reference_anchor_caveat():
    summary = build_summary(
        slot_rows=[
            {"checkpoint_role": "et_rmp_ce_ckpt3664", "comparison_role": "reference_anchor", "winner_bucket": "target_instance", "slot": "y1"},
            {"checkpoint_role": "fullobj_sorted_pure_ce_ckpt3668", "comparison_role": "clean_pair", "winner_bucket": "same_desc_competitor", "slot": "y1"},
        ],
        trajectory_rows=[],
        prefix_sensitivity_rows=[],
        greedy_rows=[],
    )
    report = build_report_markdown(summary)
    assert "not by final detector accuracy" in report
    assert "template_objective_confounded_reference" in report
    assert "reference_anchor" in report
```

- [ ] **Step 2: Run report tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_merge_report.py -q
```

Expected: fail because merge/report does not exist.

- [ ] **Step 3: Implement summary and report**

`summary.json` must include:

```python
{
    "artifact_schema_version": "a3.3.v1",
    "scope": "a3_3_post_x1_instance_basin_tomography",
    "comparison_semantics": "mechanism_traits_not_detector_accuracy",
    "config_path": str(config_path),
    "config_sha256": config_sha256,
    "code_revision": git_revision,
    "checkpoint_fingerprints": {...},
    "row_counts": {...},
    "checkpoint_roles": {...},
    "controlled_comparison_groups": {...},
    "slot_bucket_counts_by_checkpoint": {...},
    "coord_vocab_mass_summary_by_checkpoint": {...},
    "low_margin_counts_by_checkpoint": {...},
    "trajectory_bucket_counts_by_checkpoint": {...},
    "diagonal_rate_by_checkpoint": {...},
    "off_diagonal_mass_by_checkpoint": {...},
    "prefix_sensitivity_counts": {...},
    "boundary_extreme_counts_by_checkpoint": {...},
    "template_contracts": {...},
    "prompt_hash_coverage": {...},
    "shard_merge": {...},
    "status_gate_result": {...},
    "interpretation_boundaries": [
        "ET-RMP-CE is reference_anchor, not a clean controlled baseline.",
        "IoU50 is secondary to slot-level and trajectory-level taxonomy.",
    ],
}
```

`report.md` sections:

- Scope and caveats.
- Template contracts.
- Lane 1 basin trajectory findings with null/alternative.
- Lane 2 attraction matrix findings as cluster-level aggregation.
- Lane 3 prefix-quality perturbation findings, using `prefix_recovery_delta`/`bad_to_good_basin_recovery_rate` naming rather than FN-rescue wording.
- Secondary greedy continuation.
- Residual risks and next probes.

The report linter must reject uncaveated detector ranking language. Allowed detector references are only scope caveats such as "not by final detector accuracy".

- [ ] **Step 4: Implement gallery**

Gallery requirements:

- write `gallery/index.md`;
- create representative per-image cards;
- include target and same-desc competitor GT boxes;
- include slot winners by checkpoint;
- include prefix mode and template contract in visible text;
- do not hide ET-RMP caveat.

- [ ] **Step 5: Run report tests and verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_merge_report.py -q
```

Expected: all tests pass.

---

### Task 10: Tmux Launcher And GPU Execution Plan

**Files:**
- Create: `scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh`
- Test: `tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py`

- [ ] **Step 1: Write launcher contract test**

Test the script text contains the expected session and shard policy:

```python
from pathlib import Path


def test_launcher_uses_eight_single_gpu_shards():
    path = Path("scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh")
    text = path.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=${GPU_ID}" in text
    assert "--shard-id ${SHARD_ID}" in text
    assert "slot_posterior" in text
    assert "greedy_continuation" in text
```

- [ ] **Step 2: Implement launcher**

Launcher behavior:

```bash
CONFIG=${CONFIG:?CONFIG is required}
SESSION=${SESSION:-a3_3_post_x1_basin}
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
REQUIRE_SMOKE_PASS=${REQUIRE_SMOKE_PASS:-1}
```

Stages:

1. CPU index stages in tmux window `index`:
   - `data_root_audit`
   - `case_universe`
   - `prefix_modes`
   - `validate_index`
2. Eight shard windows for `slot_posterior`.
   - each shard writes `slot_posterior_shards/shard_${SHARD_ID}.jsonl`;
   - each shard writes an append-safe row to `slot_posterior_shard_summaries.jsonl` or writes a per-shard summary that merge consolidates.
3. Merge windows:
   - `slot_merge`
   - `trajectory`
   - `attraction_matrix`
   - `prefix_sensitivity`
4. Greedy continuation windows:
   - sharded or sampled GPU work using free-text deterministic decode.
5. Final windows:
   - `report`
   - `gallery`
   - `finalize`

Each GPU shard command must set exactly one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} \
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/run.py \
  --config "${CONFIG}" \
  --stages slot_posterior \
  --shard-id "${SHARD_ID}" \
  --real-runtime \
  --launch-context
```

- [ ] **Step 3: Run launcher contract test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography/test_runner_cli.py::test_launcher_uses_eight_single_gpu_shards -q
```

Expected: pass.

Launcher safety rules:

- do not set `ALLOW_OVERWRITE=1` in the launcher by default;
- full config launch requires a smoke pass marker or final smoke status file unless `REQUIRE_SMOKE_PASS=0` is explicitly set;
- `DRY_RUN=1` prints the full tmux plan and exits without creating GPU shard windows;
- real full launches use 8 single-GPU shards for analysis, not 8-GPU production training.

---

### Task 11: Verification, Smoke, Full Launch, And Documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-06-04-post-x1-instance-basin-tomography-design.md` only if implementation exposes a necessary design correction.
- Create after results: `progress/diagnostics/2026-06-04_post_x1_instance_basin_tomography_smoke_findings.md`

- [ ] **Step 1: Run full unit suite for A3.3**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/post_x1_instance_basin_tomography -q
```

Expected: all A3.3 tests pass.

- [ ] **Step 2: Run compile check**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  src/analysis/post_x1_instance_basin_tomography/*.py \
  scripts/analysis/post_x1_instance_basin_tomography/*.py
```

Expected: exit code 0.

- [ ] **Step 3: Run dry-run**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/run.py \
  --config /data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke.yaml \
  --stages data_root_audit,case_universe,prefix_modes,validate_index \
  --dry-run
```

Expected: JSON output includes all three checkpoint roles and per-checkpoint template contracts.

- [ ] **Step 4: Run CPU index stages**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/run.py \
  --config /data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke.yaml \
  --stages data_root_audit,case_universe,prefix_modes,validate_index
```

Expected artifacts:

```text
.../three_ckpt_phase_a3_3_smoke/config_resolved.json
.../three_ckpt_phase_a3_3_smoke/template_contracts.json
.../three_ckpt_phase_a3_3_smoke/data_root_audit.json
.../three_ckpt_phase_a3_3_smoke/case_universe.jsonl
.../three_ckpt_phase_a3_3_smoke/prefix_modes.jsonl
.../three_ckpt_phase_a3_3_smoke/case_universe_summary.json
```

- [ ] **Step 5: Check status after CPU index**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke
```

Expected:

```json
{
  "status": "index_ready_pending_gpu",
  "failed_gates": []
}
```

- [ ] **Step 6: Launch smoke tmux**

Run:

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke.yaml \
SESSION=a3_3_post_x1_basin_smoke \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh
```

If the smoke root already exists and rerun is intended, prepend `ALLOW_OVERWRITE=1` explicitly after reviewing the existing status.

Monitor:

```bash
tmux capture-pane -pt a3_3_post_x1_basin_smoke:0 -S -200
```

- [ ] **Step 7: Verify smoke final artifacts**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke
```

Expected:

```json
{
  "status": "final_artifacts_present",
  "failed_gates": []
}
```

Inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/data/CoordExp/outputs/analysis/autoreg_object_rollout/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3_smoke')
print(json.dumps(json.loads((root / 'summary.json').read_text()), indent=2)[:5000])
print((root / 'report.md').read_text()[:4000])
PY
```

- [ ] **Step 8: Record smoke findings**

Create:

```text
progress/diagnostics/2026-06-04_post_x1_instance_basin_tomography_smoke_findings.md
```

Include:

- scope label `a3_3_smoke`;
- config path;
- artifact root;
- checkpoint roles and template contracts;
- row counts;
- shard merge manifest summary;
- prompt/template hash coverage;
- slot bucket counts;
- trajectory bucket counts;
- prefix sensitivity counts;
- boundary extreme counts;
- explicit caveat that ET-RMP is `reference_anchor`.

- [ ] **Step 9: Dry-run full config**

Run:

```bash
DRY_RUN=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3.yaml \
SESSION=a3_3_post_x1_basin_full \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh
```

Expected: launch plan prints 8 slot-posterior shards and secondary greedy continuation plan.

- [ ] **Step 10: Launch full run only after smoke review**

Run:

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3.yaml \
SESSION=a3_3_post_x1_basin_full \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh
```

If the full root already exists and rerun is intended:

```bash
ALLOW_OVERWRITE=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/post_x1_instance_basin_tomography/three_ckpt_phase_a3_3.yaml \
SESSION=a3_3_post_x1_basin_full \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/post_x1_instance_basin_tomography/launch_a3_3_tmux.sh
```

---

## Acceptance Criteria

- A3.3 unit tests pass:

```text
tests/analysis/post_x1_instance_basin_tomography
```

- Compile check passes for A3.3 source and scripts.
- Smoke CPU index reaches `index_ready_pending_gpu` with no failed gates.
- Smoke tmux reaches `final_artifacts_present` with no failed gates.
- Smoke artifacts contain non-empty rows for all three checkpoint roles.
- Every slot posterior row records:
  - `artifact_schema_version`
  - `row_schema_version`
  - `checkpoint_role`
  - `comparison_role`
  - `controlled_comparison_group`
  - `template_contract`
  - `template_prompt_hash`
  - `assistant_prefix_sha256`
  - `forced_prompt_sha256`
  - `runtime_kind`
  - `runtime_id`
  - `mock_runtime`
  - `dry_run`
  - `primary_basin_label_source=same_desc_gt_instances`
  - `slot`
  - `forced_state`
  - `winner_bucket`
  - `slot_taxonomy`
  - `coord_vocab_mass`
  - `coord_mass_low_flag`
  - `noncoord_top_token_id`
  - `low_margin_flag`
- ET-RMP rows are labeled `reference_anchor`.
- ET-RMP rows are labeled `reference_anchor_not_controlled` under `controlled_comparison_group`.
- Pure-CE rows use `row_separator=none`.
- ET-RMP rows use `row_separator=newline`.
- `case_universe.jsonl` preserves `bbox_2d` coord-token surface/provenance and marks x1 anchor ambiguity.
- `prefix_modes.jsonl` stores semantic `prefix_objects`, not checkpoint-agnostic rendered `prefix_text` as runtime truth.
- `slot_posterior_shards/shard_0.jsonl` through `slot_posterior_shards/shard_7.jsonl` exist in real shard runs.
- `merge_manifest.json` proves shard row counts equal merged slot rows.
- `summary.json` and `report.md` include interpretation boundaries.
- `report.md` does not contain uncaveated detector ranking language.
- Full run is launched only after smoke passes and the user agrees to proceed.

## Residual Risks To Report

- ET-RMP remains a reference anchor with objective/template/data confounds.
- `len12000` compatibility for ET-RMP may need a legacy-control lane if prompt/data mismatch appears.
- Posterior readout can reveal basin geometry but does not prove attention-head causality.
- Greedy continuation parser failures are secondary and must not invalidate posterior evidence.
- Same-desc GT primary labels avoid unmatched ambiguity but cannot label unlabeled COCO objects.

## Suggested Commit Groups

When execution starts, keep commits scoped:

1. Config and data/case universe.
2. Prefix modes and posterior math.
3. Runtime and trajectory aggregation.
4. Runner/status/launcher.
5. Report/gallery/docs and smoke findings.

Use scoped staging only.  Do not stage unrelated files.
