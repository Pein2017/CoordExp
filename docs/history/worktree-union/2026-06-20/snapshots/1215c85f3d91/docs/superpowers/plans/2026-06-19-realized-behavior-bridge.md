# Realized Behavior Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 4 of the checkpoint-928 binding-template study: a deterministic realized before/after behavior bridge that turns candidate-only rank/identity evidence into real short-continuation behavior rows before any hidden-state patching or training.

**Architecture:** Reuse the existing `codex/autoregressive-binding-template-study` worktree and Phase 3 analysis helpers. Add a focused Phase 4 module under `src/analysis/autoregressive_binding_template_ablation/`, a thin CLI under `scripts/analysis/`, tests under `tests/analysis/`, and branch-provenance findings under `progress/diagnostics/`. The new surface must produce `realized_before_after_behavior_rows.jsonl` and control/promotion summaries under `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/`.

**Tech Stack:** Python 3.12, PyTorch/Qwen3-VL only for actual short-continuation execution, existing CoordExp detection template/runtime utilities, pytest, JSONL/JSON/Markdown artifacts, deterministic `temperature=0.0`, `repetition_penalty=1.10`, `max_new_tokens=3084` decode contract.

---

## Research Contract

This plan continues from:

```text
/data/CoordExp/.worktrees/autoregressive-binding-template-study/progress/diagnostics/2026-06-18_binding_mechanism_recursive_convergence.md
/data/CoordExp/.worktrees/autoregressive-binding-template-study/progress/diagnostics/2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md
```

The accepted grill decisions are:

```text
worktree: reuse /data/CoordExp/.worktrees/autoregressive-binding-template-study
phase name: realized_behavior_bridge
stage boundary: Stage A only; no latent patching, no attention patching, no training
primary output root: /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge
canonical bridge artifact: realized_before_after_behavior_rows.jsonl
next stage gate: promote to latent_behavior_patch only after real next-object behavior movement beats controls
```

### Stage Split

```text
Stage A: realized_behavior_bridge
  input: candidate-only identity/rank/coverage rows plus original rollout/probe context
  output: realized_before_after_behavior_rows.jsonl
  allowed interventions: prefix/language/coordinate-side guidance only
  allowed readouts: guidance separability matrix and read-only object pointer trajectory tomography
  forbidden: hidden-state patching, attention patching, model weight updates, micro-training

Stage B: latent_behavior_patch
  input: Stage A rows with behavior movement or near movement
  output: causal_patch_rows.jsonl
  allowed only after Stage A promotion gate passes

Stage C: bounded_micro_training_probe
  input: stable Stage A/B causal target
  output: bounded dry-run or tiny training artifacts
  allowed only after a later findings note explicitly promotes it
```

### First-Class Deep Mechanism Readouts

Phase 4 must include two readouts, not only a before/after behavior table.

#### Guidance Separability Matrix

The bridge must classify each case by which prefix-side guidance channel moves
real next-object behavior:

```text
none moves:
  evidence against simple prefix recoverability; inspect visual absence or termination basin

desc-only moves:
  language-side identity guidance unlocks behavior

coord-only moves:
  coordinate/spatial basin guidance unlocks behavior

desc+coord moves only jointly:
  semantic and spatial states are present but weakly synchronized

wrong controls move:
  bridge is contaminated or guidance is too strong
```

The matrix is a required output:

```text
guidance_separability_matrix.json
guidance_separability_matrix.md
```

#### Object Pointer Trajectory Tomography

The bridge must also prepare a read-only trajectory view of the model's
candidate object state across family-specific emission events. This is not a
patching experiment. It may use logits, candidate mass, role rows, coord rows,
or hidden-state captures if already available or captured read-only during
short continuation. It must not alter activations.

Trajectory events:

```text
pre_object
pre_desc
desc_end
pre_box_start
pre_x1
post_x1
post_y1
post_x2
row_end
next_pre_object
```

Pointer trajectory rows should expose the split state hypothesis:

```text
semantic pointer: desc-side target/same-class/competitor mass
spatial pointer: coord-side target/previous-anchor/remaining-GT basin mass
coverage pointer: emitted-vs-remaining polarity
termination pointer: stop/wrapper/next-object transition pressure
```

Required outputs:

```text
object_pointer_trajectory_rows.jsonl
object_pointer_trajectory_summary.json
object_pointer_trajectory_summary.md
```

### Source Inputs

Use the Phase 3 adjusted root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
```

Required inputs:

```text
identity_posterior_rows.jsonl
candidate_step_rows_matched_target_only.jsonl
coverage_polarity_rows_matched_target_only.jsonl
upstream_onset_label_rows.jsonl
repetition_penalty_gate_rows.jsonl
split_manifest.json
behavioral_patch/behavioral_patch_plan.json
```

Config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

### Case Selection Contract

Build a new stratified case manifest. Do not reuse only the 4 candidate-only cases from `behavioral_patch_plan.json`.

Minimum manifest strata:

```text
families: desc_first, geometry_first
splits: discovery, reserve, val200_remainder
prediction_kind: new_gt, repeated_gt, duplicate_iou70, unmatched when available
identity_margin_bin: high, medium, low_or_ambiguous
onset_label: next_step_duplicate_onset, next_step_unmatched_onset, neutral
selection_event:
  desc_first: pre_desc, desc_end
  geometry_first: pre_box_start, pre_x1
```

Recommended size:

```text
smoke: 16 cases
scaled: 96 to 128 cases after smoke sanity passes
```

### Before And After Semantics

`before` is the model's deterministic short free continuation from the original rendered prefix at the selected family-specific event.

`after` is a deterministic short free continuation from the same prefix plus one allowed prefix-side guidance intervention.

Allowed intervention arms:

```text
self_noop
target_desc_seed
target_coord_seed
target_desc_plus_coord_seed
target_object_start_seed
wrong_image_same_family
same_image_wrong_object_idx
same_class_competitor
shuffled_desc_label
coord_jitter_nearby_wrong_gt
post_commit_too_late
repetition_penalty_on
repetition_penalty_off
```

Guidance may reveal or hint one role, but must not inject the full object span. This phase measures recoverability and transition behavior, not answer copying.

### Required Output Artifacts

```text
bridge_case_manifest.json
bridge_case_manifest.md
rendered_prefix_interventions.jsonl
realized_before_after_behavior_rows.jsonl
realized_before_after_behavior_summary.json
control_outcome_rows.jsonl
guidance_separability_matrix.json
guidance_separability_matrix.md
object_pointer_trajectory_rows.jsonl
object_pointer_trajectory_summary.json
object_pointer_trajectory_summary.md
promotion_gate_summary.json
run_manifest.json
```

Every behavior row must include:

```text
schema_version
case_id
family
split
image_id
source_line_idx
object_idx
selection_event
prediction_kind
target_gt_idx
target_desc
target_bbox
identity_margin_bin
onset_label
intervention_arm
control_family
before.parse_ok
before.next_object_completed
before.next_prediction_kind
before.next_gt_idx
before.next_gt_iou
before.emitted_desc
before.emitted_bbox
before.token_count
after.parse_ok
after.next_object_completed
after.next_prediction_kind
after.next_gt_idx
after.next_gt_iou
after.emitted_desc
after.emitted_bbox
after.token_count
changed_next_object_behavior
changed_to_target_gt
duplicate_to_new_gt
unmatched_to_target_gt
stop_or_invalid_to_valid
valid_to_invalid_regression
decode.temperature
decode.repetition_penalty
decode.max_new_tokens
model_perturbation_ran
training_ran
```

`model_perturbation_ran` and `training_ran` must be `false` for every Stage A row.

### Promotion Gate

Stage A promotes to latent Stage B only if all are true:

```text
parse_preserved_rate >= 0.80 on target-guided arms
target guidance beats noop and wrong-object controls
guidance separability is interpretable rather than control-contaminated
at least one family has >= 3 nontrivial changed_to_target_gt cases
effect appears outside discovery, preferably reserve or val200_remainder
wrong-image, shuffled-label, coord-jitter, and post-commit controls remain mostly null
no promotion based only on target_desc_seed if it copies descriptor without box grounding
object pointer trajectory does not contradict the claimed guidance channel
```

### Negative Outcome Map

```text
no movement under desc or coord seed:
  branch toward visual/perceptual absence or parser/termination basin

movement under desc seed only:
  language-side guidance can recover identity; inspect identity-binding and FN guidance

movement under coord seed only:
  coordinate basin/spatial anchor can unlock behavior; inspect coordinate manifold and slot basin

movement under combined seed only:
  identity and geometry are jointly weak; inspect synchronization/binding timing

movement also under wrong controls:
  bridge is contaminated or prompt guidance is too strong; demote bridge until controls are fixed
```

---

## File Structure

Create:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
progress/diagnostics/2026-06-19_realized_behavior_bridge_findings.md
```

Modify:

```text
progress/diagnostics/README.md
progress/index.yaml
```

Do not modify:

```text
upstream HF model files
model_cache/
outputs/stage1_2b/
output/stage1_2b/
```

Do not edit `src/analysis/autoregressive_binding_template_ablation/behavioral_patch.py` unless feeding the new `realized_before_after_behavior_rows.jsonl` into the existing short-rollout scorer exposes a contract bug. The bridge generator should live in its own file.

---

### Task 1: Schema And Case Manifest Builder

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- Create: `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- Test: `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

- [ ] **Step 1: Write the failing manifest tests**

Add tests for deterministic case IDs, family-specific selection validation, margin bins, and stratified smoke limits:

```python
from __future__ import annotations

import json
from pathlib import Path

from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    build_bridge_case_manifest,
    classify_identity_margin,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def test_classify_identity_margin_bins() -> None:
    assert classify_identity_margin(0.75) == "high"
    assert classify_identity_margin(0.30) == "medium"
    assert classify_identity_margin(0.05) == "low_or_ambiguous"


def test_build_bridge_case_manifest_keeps_family_selection_events(tmp_path: Path) -> None:
    identity_rows = tmp_path / "identity.jsonl"
    _write_jsonl(
        identity_rows,
        [
            {
                "family": "desc_first",
                "image_id": 139,
                "source_line_idx": 0,
                "object_idx": 0,
                "split": "reserve",
                "selection_event": "desc_end",
                "prediction_kind": "new_gt",
                "probability_mode": "penalty_adjusted",
                "target_candidate_gt_idx": 0,
                "target_desc": "clock",
                "target_identity_bucket_mass": 0.90,
                "bucket_target_margin": 0.75,
                "same_class_count": 1,
                "same_desc_candidate_count": 1,
            },
            {
                "family": "geometry_first",
                "image_id": 2685,
                "source_line_idx": 7,
                "object_idx": 4,
                "split": "val200_remainder",
                "selection_event": "pre_x1",
                "prediction_kind": "duplicate_iou70",
                "probability_mode": "penalty_adjusted",
                "target_candidate_gt_idx": 3,
                "target_desc": "person",
                "target_identity_bucket_mass": 0.42,
                "bucket_target_margin": 0.30,
                "same_class_count": 4,
                "same_desc_candidate_count": 2,
            },
        ],
    )

    manifest = build_bridge_case_manifest(
        identity_rows_path=identity_rows,
        upstream_label_rows_path=None,
        limit_cases=16,
    )

    assert manifest["schema_version"] == 1
    assert manifest["case_count"] == 2
    assert manifest["cases"][0]["case_id"] == "desc_first-139-0-0-desc_end"
    assert manifest["cases"][0]["identity_margin_bin"] == "high"
    assert manifest["cases"][1]["case_id"] == "geometry_first-2685-7-4-pre_x1"
    assert manifest["cases"][1]["identity_margin_bin"] == "medium"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py::test_build_bridge_case_manifest_keeps_family_selection_events -q
```

Expected: fail with missing module/function.

- [ ] **Step 3: Implement manifest builder**

Implement:

```python
FAMILY_SELECTION_EVENTS = {
    "desc_first": {"pre_desc", "desc_end"},
    "geometry_first": {"pre_box_start", "pre_x1"},
}


def classify_identity_margin(value: object) -> str:
    margin = float(value)
    if margin >= 0.50:
        return "high"
    if margin >= 0.15:
        return "medium"
    return "low_or_ambiguous"
```

`build_bridge_case_manifest()` must:

```text
read identity_posterior_rows.jsonl
drop rows whose selection_event is invalid for family
prefer probability_mode == penalty_adjusted
join upstream labels when upstream_label_rows_path is provided
create deterministic case_id: {family}-{image_id}-{source_line_idx}-{object_idx}-{selection_event}
deduplicate by case_id
record input paths and skipped counters
apply limit_cases only after deterministic sorting by split, family, image_id, object_idx, selection_event
```

- [ ] **Step 4: Add CLI dry-run**

CLI stages:

```text
--stage manifest
--stage render-dry-run
--stage score-offline
```

For Task 1 implement only `--stage manifest`.

Command:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage manifest \
  --identity-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/identity_posterior_rows.jsonl \
  --upstream-label-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/upstream_onset_label_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge \
  --limit-cases 16
```

Expected files:

```text
bridge_case_manifest.json
bridge_case_manifest.md
```

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py --stage manifest --identity-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/identity_posterior_rows.jsonl --upstream-label-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/upstream_onset_label_rows.jsonl --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge --limit-cases 16
git add src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
git commit -m "feat: add realized behavior bridge manifest"
```

---

### Task 2: Prefix Intervention Spec And Render Dry-Run

**Files:**
- Modify: `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- Modify: `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- Test: `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

- [ ] **Step 1: Write failing intervention tests**

Add tests that each case expands into the required intervention arms and controls:

```python
from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    build_prefix_intervention_specs,
)


def test_build_prefix_intervention_specs_include_required_arms() -> None:
    case = {
        "case_id": "desc_first-139-0-0-desc_end",
        "family": "desc_first",
        "target_desc": "clock",
        "target_bbox": [699, 150, 770, 225],
        "target_gt_idx": 0,
        "selection_event": "desc_end",
    }

    specs = build_prefix_intervention_specs([case])
    arms = {row["intervention_arm"] for row in specs}

    assert {
        "self_noop",
        "target_desc_seed",
        "target_coord_seed",
        "target_desc_plus_coord_seed",
        "target_object_start_seed",
        "wrong_image_same_family",
        "same_image_wrong_object_idx",
        "same_class_competitor",
        "shuffled_desc_label",
        "coord_jitter_nearby_wrong_gt",
        "post_commit_too_late",
        "repetition_penalty_on",
        "repetition_penalty_off",
    }.issubset(arms)
    assert all(row["model_perturbation_ran"] is False for row in specs)
    assert all(row["training_ran"] is False for row in specs)
```

- [ ] **Step 2: Implement intervention specs**

`build_prefix_intervention_specs(cases)` must emit one row per case and arm. Each row must include:

```text
case_id
family
selection_event
intervention_arm
control_family
guidance_kind
guidance_desc
guidance_bbox
decode_temperature
decode_repetition_penalty
decode_max_new_tokens
model_perturbation_ran: false
training_ran: false
```

Use constants:

```python
DEFAULT_DECODE = {
    "temperature": 0.0,
    "repetition_penalty": 1.10,
    "max_new_tokens": 3084,
}
```

- [ ] **Step 3: Implement render dry-run**

`--stage render-dry-run` reads `bridge_case_manifest.json`, writes:

```text
rendered_prefix_interventions.jsonl
rendered_prefix_interventions_summary.json
```

This stage may render placeholder prompt fragments if full model prompt rendering is not yet wired, but it must preserve exact case IDs, family, object order, guidance arm, target desc, and target bbox. Mark rows:

```text
render_status: dry_run_only
requires_model_execution: true
```

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py --stage render-dry-run --case-manifest /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/bridge_case_manifest.json --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge
git add src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
git commit -m "feat: add realized behavior intervention dry run"
```

---

### Task 3: Offline Scoring For Provided Before/After Rows

**Files:**
- Modify: `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- Modify: `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- Test: `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

- [ ] **Step 1: Write failing scoring tests**

Test behavior scoring without requiring a model:

```python
from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    score_realized_before_after_rows,
)


def test_score_realized_before_after_rows_promotes_target_hit_with_parse_preserved() -> None:
    rows = [
        {
            "case_id": "desc_first-139-0-0-desc_end",
            "family": "desc_first",
            "split": "reserve",
            "intervention_arm": "target_desc_plus_coord_seed",
            "control_family": "target_guided",
            "target_gt_idx": 0,
            "before": {
                "parse_ok": True,
                "next_object_completed": True,
                "next_prediction_kind": "duplicate_iou70",
                "next_gt_idx": 4,
                "next_gt_iou": 0.72,
                "emitted_desc": "clock",
                "emitted_bbox": [10, 10, 30, 30],
                "token_count": 28,
            },
            "after": {
                "parse_ok": True,
                "next_object_completed": True,
                "next_prediction_kind": "new_gt",
                "next_gt_idx": 0,
                "next_gt_iou": 0.63,
                "emitted_desc": "clock",
                "emitted_bbox": [699, 150, 770, 225],
                "token_count": 30,
            },
        }
    ]

    scored = score_realized_before_after_rows(rows)

    assert scored[0]["changed_next_object_behavior"] is True
    assert scored[0]["changed_to_target_gt"] is True
    assert scored[0]["duplicate_to_new_gt"] is True
    assert scored[0]["valid_to_invalid_regression"] is False
```

- [ ] **Step 2: Implement offline scorer**

`score_realized_before_after_rows(rows)` must:

```text
preserve every input field
flatten before/after parse and next-object fields into canonical top-level columns
set changed_next_object_behavior when kind or gt_idx changes
set changed_to_target_gt only when after.parse_ok is true and after.next_gt_idx == target_gt_idx
set duplicate_to_new_gt only for duplicate_iou70 to new_gt with parse preserved
set unmatched_to_target_gt only for unmatched to target GT with parse preserved
set stop_or_invalid_to_valid when before invalid/incomplete and after parse_ok plus completed
set valid_to_invalid_regression when before parse_ok and after parse fails or no object completed
force model_perturbation_ran=false and training_ran=false
serialize with allow_nan=False
```

- [ ] **Step 3: Implement `--stage score-offline`**

The CLI reads a JSONL file of provided before/after rows and writes:

```text
realized_before_after_behavior_rows.jsonl
realized_before_after_behavior_summary.json
control_outcome_rows.jsonl
promotion_gate_summary.json
```

This is the first executable bridge artifact that can feed:

```bash
python scripts/analysis/run_autoregressive_binding_behavioral_patch.py \
  --stage short-rollout \
  --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --gate-summary /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/promotion_gate_summary.json \
  --patch-cases /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/realized_before_after_behavior_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/behavioral_patch_short_rollout
```

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
git add src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
git commit -m "feat: score realized behavior bridge rows"
```

---

### Task 4: Short-Continuation Execution Harness

**Files:**
- Modify: `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- Modify: `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- Test: `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

- [ ] **Step 1: Write execution manifest tests**

Add tests for a dry execution manifest that is deterministic and shardable:

```python
from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    build_execution_manifest,
)


def test_build_execution_manifest_shards_by_case_and_arm() -> None:
    intervention_rows = [
        {"case_id": "a", "intervention_arm": "self_noop", "family": "desc_first"},
        {"case_id": "a", "intervention_arm": "target_desc_seed", "family": "desc_first"},
        {"case_id": "b", "intervention_arm": "self_noop", "family": "geometry_first"},
    ]

    manifest = build_execution_manifest(
        intervention_rows=intervention_rows,
        available_gpus=[0, 1],
        output_root="/tmp/bridge",
    )

    assert manifest["schema_version"] == 1
    assert manifest["execution_stage"] == "short_continuation"
    assert manifest["gpu_shards"]["0"]
    assert manifest["gpu_shards"]["1"]
    assert manifest["model_perturbation_ran"] is False
    assert manifest["training_ran"] is False
```

- [ ] **Step 2: Implement manifest-only execution planning**

Add `--stage execution-manifest` that writes:

```text
run_manifest.json
shards/shard_000.jsonl
shards/shard_001.jsonl
...
```

It must not load a model. It only partitions already-rendered intervention rows.

- [ ] **Step 3: Implement smoke execution hook**

Add `--stage execute-smoke` with these safety rules:

```text
requires --allow-model-load
requires --max-cases <= 16
requires rendered_prefix_interventions.jsonl to exist
writes raw_continuation_rows.jsonl
writes realized_before_after_behavior_rows.jsonl only after parse/scoring succeeds
records model_perturbation_ran=false and training_ran=false
```

If model prompt rendering is not yet reliable, this task may stop at `execution-manifest` and record `blocked_by_prompt_renderer` in `run_manifest.json`. Do not fake model rows.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py --stage execution-manifest --rendered-interventions /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/rendered_prefix_interventions.jsonl --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge --available-gpus 0,1,2,3,4,5,6,7
git add src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
git commit -m "feat: plan realized behavior bridge execution"
```

---

### Task 5: Promotion Summary And Findings Note

**Files:**
- Modify: `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- Modify: `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- Create: `progress/diagnostics/2026-06-19_realized_behavior_bridge_findings.md`
- Modify: `progress/diagnostics/README.md`
- Modify: `progress/index.yaml`
- Test: `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

- [ ] **Step 1: Write promotion summary tests**

```python
from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    summarize_bridge_promotion_gates,
)


def test_summarize_bridge_promotion_gates_requires_controls_and_holdout() -> None:
    rows = [
        {
            "family": "desc_first",
            "split": "reserve",
            "intervention_arm": "target_desc_plus_coord_seed",
            "control_family": "target_guided",
            "after_parse_ok": True,
            "changed_to_target_gt": True,
        },
        {
            "family": "desc_first",
            "split": "reserve",
            "intervention_arm": "self_noop",
            "control_family": "noop",
            "after_parse_ok": True,
            "changed_to_target_gt": False,
        },
    ]

    summary = summarize_bridge_promotion_gates(rows)

    assert summary["schema_version"] == 1
    assert summary["parse_preserved_rate_target_guided"] == 1.0
    assert summary["effect_visible_outside_discovery"] is True
    assert summary["promote_to_latent_behavior_patch"] is False
```

- [ ] **Step 2: Implement promotion gates**

Summary fields:

```text
row_count
target_guided_row_count
control_row_count
parse_preserved_rate_target_guided
changed_to_target_gt_by_family
changed_to_target_gt_by_split
changed_to_target_gt_target_guided_count
changed_to_target_gt_control_count
effect_visible_outside_discovery
controls_mostly_null
descriptor_only_copy_risk
promote_to_latent_behavior_patch
promotion_block_reasons
```

- [ ] **Step 3: Write findings note**

Write `progress/diagnostics/2026-06-19_realized_behavior_bridge_findings.md` after smoke/scaled execution. Required sections:

```text
Scope
Resolved Design Decisions
Artifact Root
Input Handles
Case Manifest
Intervention Arms
Smoke Or Scaled Execution
Behavior Summary
Controls
Promotion Gate
Interpretation
Next Branch
Verification
```

If no model execution happened, set:

```text
evidence_scope: none-yet
status: branch-plan
```

If smoke executed, use:

```text
evidence_scope: smoke
status: branch-provenance
```

- [ ] **Step 4: Update routers**

Add the findings/plan note to:

```text
progress/diagnostics/README.md
progress/index.yaml
```

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
git diff --check
git add src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py progress/diagnostics/2026-06-19_realized_behavior_bridge_findings.md progress/diagnostics/README.md progress/index.yaml
git commit -m "docs: record realized behavior bridge findings"
```

---

### Task 6: Guidance Separability And Object Pointer Trajectory Readouts

**Files:**
- Modify: `src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py`
- Modify: `scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py`
- Test: `tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py`

- [ ] **Step 1: Write guidance matrix tests**

```python
from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    build_guidance_separability_matrix,
)


def test_build_guidance_separability_matrix_classifies_channel_unlocks() -> None:
    rows = [
        {
            "case_id": "case-a",
            "family": "desc_first",
            "split": "reserve",
            "intervention_arm": "self_noop",
            "control_family": "noop",
            "changed_to_target_gt": False,
            "after_parse_ok": True,
        },
        {
            "case_id": "case-a",
            "family": "desc_first",
            "split": "reserve",
            "intervention_arm": "target_desc_seed",
            "control_family": "target_guided",
            "changed_to_target_gt": True,
            "after_parse_ok": True,
        },
        {
            "case_id": "case-a",
            "family": "desc_first",
            "split": "reserve",
            "intervention_arm": "target_coord_seed",
            "control_family": "target_guided",
            "changed_to_target_gt": False,
            "after_parse_ok": True,
        },
    ]

    matrix = build_guidance_separability_matrix(rows)

    assert matrix["schema_version"] == 1
    assert matrix["case_count"] == 1
    assert matrix["cases"][0]["guidance_class"] == "desc_only"
    assert matrix["summary"]["desc_only_count"] == 1
    assert matrix["summary"]["control_contaminated_count"] == 0
```

- [ ] **Step 2: Implement guidance matrix**

`build_guidance_separability_matrix(rows)` must group scored behavior rows by
`case_id` and classify each case into exactly one guidance class:

```text
none
desc_only
coord_only
desc_and_coord_independent
desc_coord_joint_only
object_start_only
control_contaminated
invalid_or_unparseable
```

Classification rules:

```text
control_contaminated if any wrong/control arm changes_to_target_gt
invalid_or_unparseable if target-guided parse_ok rate for the case is zero
desc_only if target_desc_seed changes and target_coord_seed does not
coord_only if target_coord_seed changes and target_desc_seed does not
desc_and_coord_independent if both desc-only and coord-only move
desc_coord_joint_only if only target_desc_plus_coord_seed moves
object_start_only if only target_object_start_seed moves
none otherwise
```

Write:

```text
guidance_separability_matrix.json
guidance_separability_matrix.md
```

- [ ] **Step 3: Write pointer trajectory tests**

```python
from src.analysis.autoregressive_binding_template_ablation.realized_behavior_bridge import (
    build_object_pointer_trajectory_rows,
)


def test_build_object_pointer_trajectory_rows_tracks_split_pointer_state() -> None:
    cases = [
        {
            "case_id": "case-a",
            "family": "desc_first",
            "image_id": 139,
            "source_line_idx": 0,
            "object_idx": 0,
            "target_gt_idx": 0,
            "target_desc": "clock",
            "selection_event": "desc_end",
        }
    ]
    observations = [
        {
            "case_id": "case-a",
            "event": "pre_desc",
            "target_identity_mass": 0.20,
            "same_class_competitor_mass": 0.10,
            "target_coord_mass": 0.05,
            "previous_anchor_mass": 0.30,
            "remaining_gt_mass": 0.15,
            "stop_or_end_mass": 0.01,
            "next_object_start_mass": 0.40,
        },
        {
            "case_id": "case-a",
            "event": "desc_end",
            "target_identity_mass": 0.55,
            "same_class_competitor_mass": 0.08,
            "target_coord_mass": 0.12,
            "previous_anchor_mass": 0.22,
            "remaining_gt_mass": 0.20,
            "stop_or_end_mass": 0.02,
            "next_object_start_mass": 0.45,
        },
    ]

    rows = build_object_pointer_trajectory_rows(cases=cases, observations=observations)

    assert rows[0]["schema_version"] == 1
    assert rows[0]["semantic_pointer_margin"] == 0.10
    assert rows[1]["semantic_pointer_margin"] == 0.47
    assert rows[1]["spatial_pointer_margin"] == -0.10
    assert rows[1]["model_perturbation_ran"] is False
```

- [ ] **Step 4: Implement read-only pointer trajectory builder**

`build_object_pointer_trajectory_rows(cases, observations)` must merge case
metadata with event observations and compute:

```text
semantic_pointer_margin = target_identity_mass - same_class_competitor_mass
spatial_pointer_margin = target_coord_mass - previous_anchor_mass
coverage_pointer_margin = remaining_gt_mass - previous_anchor_mass
termination_pointer_margin = next_object_start_mass - stop_or_end_mass
```

Rows must include:

```text
schema_version
case_id
family
split
image_id
source_line_idx
object_idx
event
target_gt_idx
target_desc
target_identity_mass
same_class_competitor_mass
target_coord_mass
previous_anchor_mass
remaining_gt_mass
stop_or_end_mass
next_object_start_mass
semantic_pointer_margin
spatial_pointer_margin
coverage_pointer_margin
termination_pointer_margin
model_perturbation_ran: false
training_ran: false
readout_only: true
```

If read-only hidden/logit observations are unavailable after execution, write a
manifest with:

```text
status: requires_tomography_capture
blocked_reason: missing_read_only_event_observations
```

Do not synthesize fake pointer rows.

- [ ] **Step 5: Add CLI stage**

Add:

```text
--stage guidance-matrix
--stage pointer-trajectory
```

`guidance-matrix` reads `realized_before_after_behavior_rows.jsonl`.
`pointer-trajectory` reads a case manifest plus read-only event observation
JSONL. It may refuse with a structured manifest if observations are missing.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
git diff --check
git add src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
git commit -m "feat: add bridge guidance and pointer readouts"
```

---

## Execution Policy

Run order:

```text
1. CPU manifest and render dry-run
2. tiny smoke on <= 16 cases and <= 1 GPU, only with --allow-model-load
3. artifact sanity: parse counters, no hidden patch/training flags, controls present
4. guidance separability matrix over smoke behavior rows
5. read-only object pointer trajectory rows or explicit capture-required manifest
6. scaled 8-GPU continuation only if smoke sanity passes
7. findings note and promotion gate summary
```

Do not ask for another manual decision between smoke and scale unless:

```text
prompt rendering is ambiguous
parse drops exceed 20 percent
controls move at a rate comparable to target-guided arms
model execution would require changing production inference or model files
```

## Self-Review Checklist

- The plan reuses the current worktree and does not create a new branch by default.
- The plan creates a bridge artifact before any latent patching or training.
- The plan makes guidance separability and object pointer trajectory first-class Phase 4 outputs.
- The plan separates candidate-only evidence, realized behavior, interpretation, and promotion.
- The plan records exact artifact paths and tests.
- The plan does not promote OpenSpec because this is branch-provenance research, not a stable compatibility contract.
