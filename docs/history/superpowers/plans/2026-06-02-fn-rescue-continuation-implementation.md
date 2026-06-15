# FN-Rescue Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-pass checkpoint-3664 FN-rescue continuation diagnostic: deterministic GT-hint counterfactual generation, bounded attention capture, rescue scoring, and curated visual audit artifacts over the existing val200 attention-atlas linked FN sample.

**Architecture:** Keep FN-rescue as a separate analysis CLI and module. Reuse existing CoordExp compact row constants, model loading, attention aggregation, visual-token mapping, and geometry helpers; add only the missing partial-row hint construction, deterministic generation, bounded attention query capture, stratified source selection, scoring, merge/report, and curated gallery logic.

**Tech Stack:** Python, PyTorch/HF Qwen3-VL, existing CoordExp compact-full detection rows, JSONL artifacts, tmux sharding, pytest-style unit tests.

---

## Pre-Implementation Gate

No production code changes should start until the pre-implementation audit findings have been integrated into this plan.

Resolved experiment decisions live in:

- `/data/CoordExp/docs/superpowers/plans/2026-06-02-attention-evidence-routing-feasibility-val200.md`

The first-pass FN-rescue source ledger is:

- `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing/selected_cases.jsonl`

Evidence scope label:

```text
val200_attention_atlas_linked_fn_stratified
```

This scope is not the complete val200 FN universe. Reports must keep aggregate rescue rates secondary to strata-level evidence.

## Audit Integration Notes

Four read-only subagents audited the plan before implementation. All returned `GO_WITH_GUARDS`; implementation may proceed only if these guardrails are preserved.

- Template/hint guardrails: hints are partial compact-full assistant continuations, not complete rows. Use `src.common.detection_compact_rows.render_compact_row`, `OBJECT_REF_START_TOKEN`, `BOX_START_TOKEN`, and `src.tokens.coord.codec.int_to_token`; never call strict complete-row renderers on `desc_only` or `desc_x1` prefixes. Marker-delimited compact-full rows concatenate with no newline separator. Strip any `<|im_end|>` or padding suffix before appending the rescue row. Strict `parse_compact_full(..., mode="marker_delimited_strict")` is only valid after the generated tail completes the row.
- Ledger/scoring guardrails: `selected_cases.jsonl` is an index and stratification ledger, not enough to reconstruct prefixes or score rescue. Join attention candidate regions, rollout anatomy rows, scored infer rows, prediction token trace, and dataset/image provenance. Fail fast if selected/candidate rows disagree on `case_id`, `source_line_idx`, `target_gt_idx`, or `target_desc`.
- Runtime/attention guardrails: do not use `generate(output_attentions=True)` as evidence. Pass A performs deterministic greedy decode and persists prompt ids, generated ids, generated text, stop information, and parse outcome. Pass B replays the exact prefix with `use_cache=false`, `output_attentions=true`, eager attention, and token-id equality checks before writing attention rows. `pre_y1` must be an explicit role or a recorded alias to `post_x1`; prefer explicit `pre_y1`.
- Geometry/provenance guardrails: bounded attention only bounds persisted rows, not GPU memory. Keep batch size one, record attention shapes and visual grid/span provenance, require selected attention backend to be exactly eager, and record `processor_do_resize=false` or the exact replay policy used.
- Wrong-control guardrails: materialize a concrete wrong-control source region for every emitted wrong-control row. The broad `far_background` exclusion region `[0,0,999,999]` is not a valid inserted-`x1` source until converted into a concrete low-overlap sampled region outside the target context ring. Random `x1` remains out of scope.
- Gallery/report guardrails: gallery is qualitative only. Reuse canonical visualization normalization/rendering in `src.vis.gt_vs_pred`, attach explicit `debug.visual_roles` / provenance, preserve source prediction order, and put denominator tables before gallery examples. Gallery rows must carry `visual_note: qualitative_only_not_metric_source`.

Integrated audit fixes before implementation:

- Prefix reconstruction must use raw compact coord tokens from scored inference or token trace artifacts. Pixel `pred_points` must never be directly converted into coord tokens.
- `object_count_bucket` is part of the stratum key and report schema.
- Selected-case rows must normalize `intended_target_gt_idx` to canonical `target_gt_idx` before joins.
- `rescue_rows.jsonl` is the denominator ledger for emitted and skipped planned tiers.
- Merge validation covers generation, replay, decision, wrong-control, candidate, and attention rows with file-specific uniqueness keys, row-count invariants, source/output hashes, and atomic writes.
- Region attention uses de-duplicated per-kind union rows for metric claims and per-instance rows only for diagnostics.
- Duplicate-copy rejection uses strict `max_same_desc_existing_iou > duplicate_iou_threshold` and persists duplicate source provenance.
- Launcher overwrite behavior is allowlisted and guarded against source-root deletion.

## File Structure

Create:

- `/data/CoordExp/src/analysis/autoreg_fn_rescue_continuation.py`
  - Owns FN-rescue config dataclasses, source-case selection, artifact joins, partial compact-row hint construction, prefix reconstruction from rollout/infer artifacts, wrong-control source selection, generation-tail parsing, rescue scoring, two-pass attention replay, merge, summary, report, and gallery selection.
- `/data/CoordExp/scripts/analysis/run_autoreg_fn_rescue_continuation.py`
  - Thin CLI wrapper for stages: `select_cases`, `feasibility`, `rescue_decode`, `attention_replay`, `merge`, `report`, `gallery`.
- `/data/CoordExp/scripts/analysis/launch_autoreg_fn_rescue_continuation_tmux.sh`
  - Sharded tmux launcher for smoke and full linked-sample runs. It must separate `NUM_SHARDS` from `GPU_LIST`.
- `/data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml`
  - Tiny/smoke config using the existing linked source ledger and a small per-stratum cap.
- `/data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked.yaml`
  - First-pass linked val200 config.
- `/data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py`
  - Unit tests for config parsing, stratified selection, partial-row construction, wrong-control source priority, generated-coordinate parsing, scoring, region ledger construction, merge summaries, and report/gallery bucket logic.

Modify only if reuse requires it:

- `/data/CoordExp/src/analysis/autoreg_attention_evidence_routing.py`
  - Allow import/reuse of already-public helper functions. Do not change existing atlas artifact semantics.
- `/data/CoordExp/src/datasets/geometry.py`
  - Only if a missing pure geometry helper is necessary. Existing `valid_xyxy_box` and `box_iou_xyxy` should be sufficient.

## Artifact Contract

Artifact root for smoke:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation_smoke
```

Artifact root for first-pass linked run:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation
```

Merged artifact files:

```text
selected_rescue_cases.jsonl
rescue_rows.jsonl
rescue_generation_rows.jsonl
rescue_replay_prefix_rows.jsonl
rescue_attention_region_rows.jsonl
rescue_decision_context_rows.jsonl
rescue_candidate_region_rows.jsonl
wrong_control_rows.jsonl
merge_summary.json
summary.json
report.md
shards_manifest.json
gallery/gallery_index.jsonl
gallery/gallery_summary.json
gallery/vis_resources/*.jsonl
gallery/images/*/*.png
```

Required row fields for `rescue_generation_rows.jsonl`:

```python
{
    "case_id": str,
    "source_line_idx": int,
    "target_gt_idx": int,
    "target_desc": str,
    "prefix_depth": int,
    "prefix_quality": str,
    "object_count_bucket": str,
    "x1_top_peak_attribution": str | None,
    "rescue_tier": "desc_only" | "desc_x1" | "desc_x1_wrong_control",
    "hint_x1": int | None,
    "wrong_control_source_kind": str | None,
    "wrong_control_source_bbox_xyxy": list[int] | None,
    "generated_tail_text": str,
    "generated_coord_tokens": list[str],
    "generated_box_xyxy": list[int] | None,
    "valid_parse": bool,
    "target_iou": float,
    "success_iou30": bool,
    "success_iou50": bool,
    "success_iou75": bool,
    "same_desc_duplicate_iou95": bool,
    "max_same_desc_existing_iou": float,
    "duplicate_source_kind": str | None,
    "duplicate_source_raw_pred_idx": int | None,
    "duplicate_source_guarded_pred_idx": int | None,
    "duplicate_source_bbox_xyxy": list[int] | None,
    "primary_rescue_success": bool,
    "outcome_bucket": str,
    "source_artifacts": dict,
    "prefix_objects": list[dict],
    "parse_errors": list[str],
}
```

Required row fields for `rescue_rows.jsonl`:

```python
{
    "case_id": str,
    "source_line_idx": int,
    "target_gt_idx": int,
    "target_desc": str,
    "prefix_depth": int,
    "prefix_quality": str,
    "binding_bucket": str,
    "depth_bucket": str,
    "object_count_bucket": str,
    "planned_rescue_tier": "desc_only" | "desc_x1" | "desc_x1_wrong_control",
    "attempt_status": "emitted" | "skipped",
    "emitted": bool,
    "skip_reason": str | None,
    "prefix_reconstruction_status": str,
    "wrong_control_status": str | None,
    "source_artifacts": dict,
    "config_sha256": str,
    "shard_label": str,
}
```

`rescue_rows.jsonl` is the denominator ledger. All report denominators must come from this file, not from generation rows alone.

Required attention query roles:

```text
desc_only/pre_x1
desc_x1/pre_y1
desc_x1_wrong_control/pre_y1
```

Required region kinds:

```text
target_gt
context_ring
far_background
same_desc_competitor_gt_object
previous_generated_object
same_desc_rollout_prediction
wrong_control_source_region
```

Each region row must include `region_instance_id`, `source_index`, and `aggregation_scope`. Attention output must include both per-instance rows and per-kind union rows. Density claims in `report.md` must use union rows with de-duplicated visual-token indices; per-instance rows are diagnostics.

## Task 1: Unit-Test The Pure FN-Rescue Contract

**Files:**

- Create: `/data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py`

- [ ] **Step 1: Write tests for partial-row hint construction**

Add tests that force exact compact-full marker usage:

```python
from src.analysis.autoreg_fn_rescue_continuation import (
    coord_token,
    render_rescue_hint_row_prefix,
)


def test_render_desc_only_hint_row_prefix_uses_compact_markers():
    assert (
        render_rescue_hint_row_prefix("vase", tier="desc_only", x1=None)
        == "<|object_ref_start|>vase<|box_start|>"
    )


def test_render_desc_x1_hint_row_prefix_appends_x1_coord_token():
    assert (
        render_rescue_hint_row_prefix("person", tier="desc_x1", x1=7)
        == "<|object_ref_start|>person<|box_start|><|coord_7|>"
    )


def test_coord_token_rejects_out_of_range_values():
    for bad in [-1, 1000, 1200]:
        try:
            coord_token(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_render_hint_rejects_compact_forbidden_desc_tokens():
    for bad in ["bad\nrow", "bad\rrow", "bad\trow", "<|im_end|>", "<|coord_7|>", "<|im_start|>"]:
        try:
            render_rescue_hint_row_prefix(bad, tier="desc_only", x1=None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
```

- [ ] **Step 2: Write tests for generated bbox parsing**

```python
from src.analysis.autoreg_fn_rescue_continuation import parse_generated_bbox


def test_parse_desc_only_requires_four_leading_coord_tokens():
    parsed = parse_generated_bbox(
        tier="desc_only",
        generated_tail_text="<|coord_10|><|coord_20|><|coord_30|><|coord_40|>extra",
        hint_x1=None,
    )
    assert parsed.valid_parse is True
    assert parsed.generated_box_xyxy == [10, 20, 30, 40]


def test_parse_desc_x1_prepends_hint_x1_and_requires_three_generated_coords():
    parsed = parse_generated_bbox(
        tier="desc_x1",
        generated_tail_text="<|coord_20|><|coord_30|><|coord_40|>",
        hint_x1=10,
    )
    assert parsed.valid_parse is True
    assert parsed.generated_box_xyxy == [10, 20, 30, 40]


def test_parse_rejects_non_positive_area_box():
    parsed = parse_generated_bbox(
        tier="desc_only",
        generated_tail_text="<|coord_10|><|coord_20|><|coord_10|><|coord_40|>",
        hint_x1=None,
    )
    assert parsed.valid_parse is False
    assert parsed.generated_box_xyxy is None
```

- [ ] **Step 3: Write tests for rescue scoring and duplicate rejection**

```python
from src.analysis.autoreg_fn_rescue_continuation import score_rescue_box


def test_score_rescue_success_and_duplicate_flag_are_separate():
    score = score_rescue_box(
        generated_box=[100, 100, 200, 200],
        target_box=[100, 100, 200, 200],
        same_desc_rollout_boxes=[[100, 100, 200, 200]],
        duplicate_iou_threshold=0.95,
    )
    assert score.target_iou == 1.0
    assert score.success_iou50 is True
    assert score.same_desc_duplicate_iou95 is True
    assert score.primary_rescue_success is False


def test_duplicate_threshold_is_strictly_greater_than_policy():
    exact = score_rescue_box(
        generated_box=[0, 0, 100, 100],
        target_box=[0, 0, 100, 100],
        same_desc_rollout_boxes=[[0, 0, 95, 100]],
        duplicate_iou_threshold=0.95,
    )
    over = score_rescue_box(
        generated_box=[0, 0, 100, 100],
        target_box=[0, 0, 100, 100],
        same_desc_rollout_boxes=[[0, 0, 96, 100]],
        duplicate_iou_threshold=0.95,
    )
    assert exact.same_desc_duplicate_iou95 is False
    assert over.same_desc_duplicate_iou95 is True
```

- [ ] **Step 4: Write tests for wrong-control source priority**

```python
from src.analysis.autoreg_fn_rescue_continuation import choose_wrong_control_source


def test_wrong_control_prefers_same_desc_gt_over_rollout_prediction():
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[[300, 100, 400, 200]],
        same_desc_rollout_boxes=[[500, 100, 600, 200]],
        all_gt_boxes=[[100, 100, 200, 200], [300, 100, 400, 200]],
        context_expansion_norm1000=64,
    )
    assert source is not None
    assert source.kind == "same_desc_competitor_gt_object"
    assert source.x1 == 300


def test_wrong_control_skips_when_only_target_overlap_is_available():
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[[105, 105, 205, 205]],
        same_desc_rollout_boxes=[],
        all_gt_boxes=[[100, 100, 200, 200]],
        context_expansion_norm1000=64,
    )
    assert source is None


def test_wrong_control_far_background_fallback_is_concrete_not_broad_placeholder():
    source = choose_wrong_control_source(
        target_box=[100, 100, 200, 200],
        same_desc_gt_boxes=[],
        same_desc_rollout_boxes=[],
        all_gt_boxes=[[100, 100, 200, 200]],
        context_expansion_norm1000=64,
    )
    assert source is not None
    assert source.kind == "far_background_concrete"
    assert source.bbox_xyxy != [0, 0, 999, 999]
```

- [ ] **Step 5: Write tests for source-ledger normalization and stratification**

```python
from src.analysis.autoreg_fn_rescue_continuation import (
    canonical_target_gt_idx,
    fn_rescue_stratum_key,
)


def test_canonical_target_gt_idx_accepts_intended_key_from_atlas_ledger():
    row = {"intended_target_gt_idx": 19}
    assert canonical_target_gt_idx(row) == 19


def test_object_count_bucket_is_part_of_stratum_key():
    base = {
        "prefix_quality": "clean_prefix",
        "x1_top_peak_attribution": "no_local_object_diffuse",
        "x1_target_rank": 100,
        "prefix_depth": 2,
    }
    small = fn_rescue_stratum_key({**base, "dataset_gt_count": 3})
    crowded = fn_rescue_stratum_key({**base, "dataset_gt_count": 20})
    assert small != crowded
```

- [ ] **Step 6: Write tests for region-union attention membership**

```python
from src.analysis.autoreg_fn_rescue_continuation import build_rescue_region_membership


def test_same_kind_union_deduplicates_overlapping_region_tokens():
    rows = [
        {"region_kind": "same_desc_competitor_gt_object", "region_instance_id": "a", "token_indices": [1, 2, 3]},
        {"region_kind": "same_desc_competitor_gt_object", "region_instance_id": "b", "token_indices": [3, 4]},
    ]
    membership = build_rescue_region_membership(rows)
    assert membership["instance:same_desc_competitor_gt_object:a"] == [1, 2, 3]
    assert membership["instance:same_desc_competitor_gt_object:b"] == [3, 4]
    assert membership["union:same_desc_competitor_gt_object"] == [1, 2, 3, 4]
```

- [ ] **Step 7: Run tests and confirm they fail before implementation**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest /data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py -q
```

Expected before implementation: import errors for `src.analysis.autoreg_fn_rescue_continuation`.

## Task 2: Implement Pure Config, Selection, Hint, Parse, Score, And Region Helpers

**Files:**

- Create: `/data/CoordExp/src/analysis/autoreg_fn_rescue_continuation.py`
- Test: `/data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py`

- [ ] **Step 1: Add config dataclasses and loader**

Implement dataclasses with these required fields:

```python
@dataclass(frozen=True)
class FnRescuePaths:
    artifact_root: Path
    checkpoint: Path
    dataset_jsonl: Path
    attention_atlas_root: Path
    source_selected_cases: Path
    source_candidate_regions: Path
    rollout_anatomy_per_row: Path
    gt_vs_pred_scored: Path
    pred_token_trace: Path
    infer_resolved_config: Path | None
    lane_c_study_config: Path | None


@dataclass(frozen=True)
class FnRescueSelectionConfig:
    evidence_scope: str
    sample_limit: int
    per_stratum_cap: int
    duplicate_iou_threshold: float
    wrong_control_overlap_threshold: float
    context_expansion_norm1000: int


@dataclass(frozen=True)
class FnRescueExecutionConfig:
    attn_implementation: str
    torch_dtype: str
    decoding: str
    do_sample: bool
    num_beams: int
    max_new_tokens_desc_only: int
    max_new_tokens_desc_x1: int
    gallery_per_bucket: int


@dataclass(frozen=True)
class FnRescueConfig:
    paths: FnRescuePaths
    selection: FnRescueSelectionConfig
    execution: FnRescueExecutionConfig
```

`load_fn_rescue_config(path: Path) -> FnRescueConfig` must fail fast when required files do not exist, `evidence_scope` is not `val200_attention_atlas_linked_fn_stratified`, `attn_implementation` is not `eager`, `decoding` is not `greedy`, `do_sample` is not `false`, `num_beams` is not `1`, `wrong_control_overlap_threshold` is not in `[0.0, 1.0]`, or token counts are not `4` and `3`. The dry-run payload must echo every source artifact path, row count when applicable, byte size, and sha256.

- [ ] **Step 2: Implement compact coord and partial-row helpers**

Use existing constants:

```python
from src.common.detection_compact_rows import (
    BOX_START_TOKEN,
    COMPACT_DESC_FORBIDDEN_SUBSTRINGS,
    OBJECT_REF_START_TOKEN,
    render_compact_row,
)
from src.tokens.coord.codec import int_to_token
```

Implement:

```python
def coord_token(value: int) -> str:
    value = int(value)
    if value < 0 or value > 999:
        raise ValueError(f"coord token out of norm1000 range: {value}")
    return int_to_token(value)


def render_rescue_hint_row_prefix(desc: str, *, tier: str, x1: int | None) -> str:
    clean_desc = str(desc).strip()
    if not clean_desc:
        raise ValueError("target desc must be nonempty")
    if any(item in clean_desc for item in COMPACT_DESC_FORBIDDEN_SUBSTRINGS):
        raise ValueError("target desc is not compact-row safe")
    if tier == "desc_only":
        if x1 is not None:
            raise ValueError("desc_only must not receive x1")
        return render_compact_row(
            clean_desc,
            (),
            include_object_ref_marker=True,
            include_bbox_start_marker=True,
        )
    if tier in {"desc_x1", "desc_x1_wrong_control"}:
        if x1 is None:
            raise ValueError(f"{tier} requires x1")
        return render_compact_row(
            clean_desc,
            (coord_token(int(x1)),),
            include_object_ref_marker=True,
            include_bbox_start_marker=True,
        )
    raise ValueError(f"unknown rescue tier: {tier}")
```

- [ ] **Step 3: Implement generated-coordinate parser**

Parse only leading strict coord tokens. For `desc_only`, require four generated tokens. For `desc_x1` and `desc_x1_wrong_control`, require three generated tokens and prepend `hint_x1`.

The output dataclass must include:

```python
@dataclass(frozen=True)
class ParsedRescueGeneration:
    valid_parse: bool
    generated_coord_tokens: tuple[str, ...]
    generated_box_xyxy: list[int] | None
```

- [ ] **Step 4: Implement score helper**

Use `/data/CoordExp/src/datasets/geometry.py::box_iou_xyxy`. Same-desc duplicate policy is strict: `same_desc_duplicate_iou95 == True` only when `max_same_desc_existing_iou > duplicate_iou_threshold`. Primary rescue success is:

```python
valid_parse and target_iou >= 0.5 and not same_desc_duplicate_iou95
```

Also compute `success_iou30`, `success_iou50`, and `success_iou75`.

Persist duplicate provenance in generation rows:

```text
max_same_desc_existing_iou
duplicate_source_kind
duplicate_source_raw_pred_idx
duplicate_source_guarded_pred_idx
duplicate_source_bbox_xyxy
```

- [ ] **Step 5: Implement deterministic stratified source selection**

Load `selected_cases.jsonl`, canonicalize `target_gt_idx = row.get("target_gt_idx", row["intended_target_gt_idx"])`, join object counts from `rollout_anatomy/per_row.jsonl` (`gt_count` or `dataset_gt_count`) or the scored dataset row, deduplicate on `(case_id, rescue_tier)` later, and build a deterministic stratum key:

```python
prefix_quality_bucket = row["prefix_quality"]
binding_bucket = (
    "same_desc_competitor"
    if row.get("x1_top_peak_attribution") == "same_desc_competitor_gt_object"
    else "no_local_object_diffuse"
    if row.get("x1_top_peak_attribution") == "no_local_object_diffuse"
    else "target_low_rank"
    if int(row.get("x1_target_rank") or 10**9) > 50
    else "other"
)
depth_bucket = "d0" if depth == 0 else "d1_3" if depth <= 3 else "d4_7" if depth <= 7 else "d8_plus"
object_count_bucket = "gt_1_5" if gt_count <= 5 else "gt_6_15" if gt_count <= 15 else "gt_16_plus"
```

Sort rows by `(source_line_idx, prefix_depth, target_gt_idx, case_id)`, then take up to `per_stratum_cap` from each `(prefix_quality_bucket, binding_bucket, depth_bucket, object_count_bucket)` stratum until `sample_limit`. `selected_rescue_cases.jsonl`, `rescue_rows.jsonl`, `summary.json`, and `report.md` must expose all four stratum components.

- [ ] **Step 6: Implement wrong-control source selection**

Priority:

1. Same-image same-desc competitor GT box with IoU below duplicate threshold against target.
2. Same-image same-desc parsed prefix/rollout prediction box with IoU below duplicate threshold against target.
3. Deterministic far-background corner candidate not overlapping target context ring, any GT, or any prefix/rollout prediction above `wrong_control_overlap_threshold`.
4. Skip wrong-control tier if no source exists.

Do not use random coordinates.

Far-background fallback algorithm:

```text
1. Let w = max(16, target_x2 - target_x1) and h = max(16, target_y2 - target_y1).
2. Enumerate clipped target-sized boxes in this fixed order: top_left, top_right, bottom_left, bottom_right, center_left, center_right.
3. Reject a candidate if it overlaps target context ring, any GT box, or any existing prefix/rollout prediction with IoU > wrong_control_overlap_threshold.
4. Select the first remaining candidate and record all rejected candidate labels plus rejection reasons.
5. If none remain, emit skip reason `wrong_control_unavailable`.
```

- [ ] **Step 7: Run pure helper tests**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest /data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py -q
```

Expected after Task 2: pure helper tests pass; model-dependent tests may be skipped or not yet present.

## Task 3: Implement Prefix Reconstruction, Feasibility Gate, Deterministic Generation, And Bounded Attention

**Files:**

- Modify: `/data/CoordExp/src/analysis/autoreg_fn_rescue_continuation.py`
- Test: `/data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py`

- [ ] **Step 1: Reconstruct self-prefix text from raw compact tokens**

Use existing code paths:

```python
from src.analysis.hard_ce_coord_logit_locality import load_model_handle
from src.common.detection_compact_rows import render_compact_row
```

For each selected case, reconstruct prior self-rollout rows from raw compact coord tokens, not from pixel `pred_points`. Preferred sources, in order:

1. `/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/gt_vs_pred_scored.jsonl` `raw_output_json.objects[raw_pred_idx].bbox_2d`
2. `/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/pred_token_trace.jsonl` reconstructed from `generated_token_text`
3. Pixel `pred_points` only as a fallback after converting pixel xyxy to norm1000 with the scored row width/height and proving equality to the raw compact tokens for the same `(source_line_idx, raw_pred_idx)`.

The implementation must never call `int_to_token()` on pixel `pred_points`.

```python
def prefix_text_from_raw_compact_predictions(
    *,
    rollout_rows: list[dict],
    scored_row: dict,
    token_trace_row: dict,
    prefix_depth: int,
) -> str:
    prefix_rows = [
        row
        for row in rows
        if row.get("raw_pred_idx") is not None
        and int(row["raw_pred_idx"]) < int(prefix_depth)
        and not row.get("suppressed_by_guard", False)
    ]
    prefix_rows.sort(key=lambda row: int(row["raw_pred_idx"]))
    rendered: list[str] = []
    for row in prefix_rows:
        raw_pred_idx = int(row["raw_pred_idx"])
        raw_obj = scored_row["raw_output_json"]["objects"][raw_pred_idx]
        bbox_tokens = tuple(str(token) for token in raw_obj["bbox_2d"])
        if len(bbox_tokens) != 4 or not all(token.startswith("<|coord_") for token in bbox_tokens):
            raise ValueError("raw compact prediction is missing four coord tokens")
        if str(raw_obj["desc"]) != str(row["pred_desc"]):
            raise ValueError("raw compact desc does not match rollout row desc")
        rendered.append(
            render_compact_row(
                str(raw_obj["desc"]),
                bbox_tokens,
                include_object_ref_marker=True,
                include_bbox_start_marker=True,
            )
        )
    return "".join(rendered)
```

The returned prefix must be marker-delimited strict compact-full: direct row concatenation, no newline, no trailing space, no `<|im_end|>`. Cross-check the reconstructed prefix against `pred_token_trace.generated_token_text` for the same source row. For the first current artifact row, a fixture must prove that row 0 pred 0 reconstructs `<|coord_457|><|coord_509|><|coord_546|><|coord_747|>`, not pixel points `[570,423,682,621]`. If the prefix cannot be proven, emit a `rescue_rows.jsonl` skip row with reason `prefix_reconstruction_failed` instead of guessing.

Add tests for empty prefix, one row, multi-row direct concatenation, token-trace cross-checking, terminal stripping, suppressed-row exclusion, and pixel fallback rejection when raw-token equality cannot be proven.

- [ ] **Step 2: Build continuation assistant text**

For each case/tier:

```python
prefix_text = strip_generation_terminal(prefix_text_from_rollout_rows(rows, prefix_depth=prefix_depth))
hint_row = render_rescue_hint_row_prefix(target_desc, tier=tier, x1=hint_x1)
assistant_prefix = f"{prefix_text}{hint_row}" if prefix_text else hint_row
```

This is the only allowed hint insertion policy.

- [ ] **Step 3: Add chat-template continuation feasibility gate**

Implement a probe that calls:

```python
model_handle.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    continue_final_message=True,
)
```

The gate must verify:

- The rendered prompt ends with the exact `assistant_prefix`.
- The rendered prompt does not end with `<|im_end|>`.
- Tokenized input's last non-pad token decodes to the last hint token (`<|box_start|>` for `desc_only`, `<|coord_x|>` for x1 tiers).

If `continue_final_message=True` is unsupported or fails these checks, fail the feasibility stage rather than silently stripping stop tokens.

- [ ] **Step 4: Implement Pass A greedy rescue decode**

Use model generation with:

```python
generation_kwargs = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 4 if tier == "desc_only" else 3,
    "pad_token_id": model_handle.tokenizer.pad_token_id,
    "eos_token_id": model_handle.tokenizer.convert_tokens_to_ids("<|im_end|>"),
}
```

Decode only the generated tail. Do not count stop/extra text as rescue evidence; parse only the leading strict coord tokens. Persist, per case/tier:

```text
assistant_prefix_text
assistant_prefix_token_ids
generated_token_ids
generated_tail_text
stop_reason
parse_status
parsed_box
```

Do not request or trust attention from `generate()`.

- [ ] **Step 5: Implement Pass B bounded attention replay**

Run a separate forward pass on the exact stored continuation prompt with `output_attentions=True`, `use_cache=False`, and eager attention. Before writing attention rows, re-tokenize the replay prompt and assert the replay prefix token ids equal the persisted Pass A `assistant_prefix_token_ids`.

Query index:

```python
query_index = last_non_pad_input_index
role = "pre_x1" if tier == "desc_only" else "pre_y1"
```

`pre_x1` means the state that predicts the first generated coord token. `pre_y1` means the state that predicts the generated `y1` token after a forced `x1`; if implemented by aliasing an existing `post_x1` inventory state, record `pre_y1_alias_role: "post_x1"` in every manifest and attention row. Prefer an explicit `pre_y1` role.

Reuse:

- `find_visual_token_spans`
- visual-grid logic equivalent to `_visual_grid_from_thw`; either promote that pure helper to public reuse or copy a tiny local pure helper with tests
- attention aggregation logic equivalent to `aggregate_attention_for_query`, but with FN-rescue-specific region membership keys that include per-instance and union scopes

Do not call `_prepare_attention_pairs` from `autoreg_attention_evidence_routing.py`; it rebuilds atlas selection internals and is not the FN-rescue source of truth. If prepared examples are needed, implement an explicit FN-rescue-local preparation path or promote a small public helper with an explicit contract.

Each attention row must include `rescue_tier`, `role`, `aggregation_scope`, `region_kind`, `region_instance_id`, `layer`, `head`, `attention_mass`, and `attention_mass_normalized`. For union rows, visual-token indices must be de-duplicated before summing attention mass.

Fail fast if:

- selected attention backend is not `eager`
- `outputs.attentions` is absent
- visual span count is not exactly one
- visual token count does not match `image_grid_thw` adjusted by merge size
- replay prefix token ids differ from Pass A persisted ids

- [ ] **Step 6: Run one-case feasibility**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python /data/CoordExp/scripts/analysis/run_autoreg_fn_rescue_continuation.py \
  --config /data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml \
  --stages select_cases,feasibility \
  --shard-index 0 \
  --num-shards 8
```

Expected: feasibility writes a small JSON object proving continuation prompt tail, no terminal marker in the prefix, direct marker-delimited row concatenation, replay prefix token equality, visual span, attention layer count, selected eager backend, and query role alignment.

## Task 4: Implement Rescue Case Materialization, Region Ledger, Scoring Rows, Merge, Summary, Report, And Gallery Selection

**Files:**

- Modify: `/data/CoordExp/src/analysis/autoreg_fn_rescue_continuation.py`
- Test: `/data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py`

- [ ] **Step 1: Materialize selected rescue cases**

Write `selected_rescue_cases.jsonl` with normalized selected cases before tier expansion. Canonicalize `target_gt_idx = row.get("target_gt_idx", row["intended_target_gt_idx"])`, join object counts, and fail fast if any joined candidate row disagrees on `case_id`, `source_line_idx`, `target_gt_idx`, or `target_desc`.

Write `rescue_rows.jsonl` with one denominator row per `(case_id, planned_rescue_tier)`. Emit `desc_x1_wrong_control` as `attempt_status="skipped"` with `skip_reason="wrong_control_unavailable"` when a valid wrong-control source does not exist. Reports must use this ledger for denominators.

- [ ] **Step 2: Materialize rescue candidate regions**

Join `selected_cases.jsonl`, `candidate_region_rows.jsonl`, `rollout_anatomy/per_row.jsonl`, and scored infer rows. Reuse target/context/far-background logic from the attention atlas, but normalize names to the required FN-rescue region kinds:

```text
target_gt
context_ring
far_background
same_desc_competitor_gt_object
previous_generated_object
same_desc_rollout_prediction
wrong_control_source_region
```

If a region is unavailable for a case, omit that region row rather than emitting invalid geometry. Every row must include stable `region_instance_id`, `source_index`, `aggregation_scope="instance"`, and enough provenance to trace it to GT, raw prediction, guarded prediction, prefix object, or wrong-control source.

Build a second in-memory membership surface for attention aggregation with `aggregation_scope="union"` per `region_kind`. Union membership must de-duplicate visual-token indices before mass summation. Persist union attention rows; persist per-instance attention rows as diagnostics.

For emitted wrong-control rows, require exactly one concrete `wrong_control_source_region`. If fallback source is far-background, first materialize a concrete low-overlap box outside the target context ring and record its provenance. Do not use the broad attention-atlas `far_background` placeholder as an inserted `x1` source.

- [ ] **Step 3: Write generation, replay, attention, and decision rows**

For each emitted tier:

- Write or update exactly one `rescue_rows.jsonl` denominator row with `attempt_status="emitted"`.
- Write exactly one `rescue_generation_rows.jsonl` row.
- Write exactly one `rescue_replay_prefix_rows.jsonl` row with Pass A prompt ids, Pass B prompt ids, token-id equality result, selected attention backend, processor/image provenance, and query role.
- Write one `rescue_decision_context_rows.jsonl` row with visual span, grid, query role, and generated parse status.
- Write attention rows for all heads/layers over the bounded query role.
- Write exactly one `wrong_control_rows.jsonl` row for every emitted `desc_x1_wrong_control` tier.

- [ ] **Step 4: Implement merge with manifest validation**

`merge_fn_rescue_shards(root, expected_shards)` must:

- Require all expected shard labels.
- Validate every shard summary before opening or replacing root-level merged files.
- Reject duplicate `(case_id, planned_rescue_tier)` in `rescue_rows.jsonl`.
- Reject duplicate `(case_id, rescue_tier)` in `rescue_generation_rows.jsonl`, `rescue_replay_prefix_rows.jsonl`, `rescue_decision_context_rows.jsonl`, and `wrong_control_rows.jsonl`.
- Reject duplicate `(case_id, rescue_tier, region_kind, region_instance_id)` candidate rows.
- Reject duplicate `(case_id, rescue_tier, role, layer, head, aggregation_scope, region_kind, region_instance_id)` attention rows.
- Validate per-shard row counts against actual JSONL line counts and merged counts against shard-count sums.
- Validate invariants: generation/replay/decision rows equal emitted decode attempts, wrong-control rows equal emitted `desc_x1_wrong_control` attempts, and denominator rows include both emitted and skipped planned tiers.
- Record source artifacts and merged outputs with path, row count when applicable, byte size, and sha256.
- Write merged outputs through temp files or a temp merge root and atomically replace final files only after all checks pass.
- Write `merge_summary.json` with row counts, source hashes, output hashes, config hash, expected shard labels, and validation status.

- [ ] **Step 5: Implement summary and report**

`summary.json` and `report.md` must include:

- evidence scope
- checkpoint path
- source selected cases path
- source artifact hashes and row counts
- selected case counts by prefix quality, binding bucket, depth bucket, and object-count bucket
- emitted tier counts
- attempted, skipped, invalid-parse, target-desc-not-preserved, geometry-invalid, wrong-control-unavailable denominators
- parse-valid rates by tier
- `IoU>=0.3/0.5/0.75` rates by tier and stratum
- raw rescue success and duplicate-guarded primary rescue success, reported separately
- duplicate-rejected counts
- wrong-control source distribution
- bounded attention target-vs-competitor density summaries by tier/role/layer group using union rows for metric claims
- per-instance attention diagnostics marked separately from union metric summaries
- interpretation bounds: GT leakage, linked-stratified scope, not deployable inference

- [ ] **Step 6: Implement curated gallery selection**

Select up to `gallery_per_bucket` examples per bucket:

```text
desc_only_success
desc_x1_only_success
both_fail
wrong_control_binds_competitor
duplicate_copy_rejected
```

Render each image with target GT, original parsed prefix/rollout predictions, rescue box, and wrong-control source box when present. Use `src.vis.gt_vs_pred` canonical normalization/rendering rather than bespoke drawing. Gallery-only records must carry `source_kind: "fn_rescue_gallery"`, `debug.visual_roles`, source `target_gt_idx`, and `visual_note: "qualitative_only_not_metric_source"`. Use tables/JSONL as source of truth; gallery images are qualitative audit aids.

## Task 5: Add CLI, Configs, And tmux Launcher

**Files:**

- Create: `/data/CoordExp/scripts/analysis/run_autoreg_fn_rescue_continuation.py`
- Create: `/data/CoordExp/scripts/analysis/launch_autoreg_fn_rescue_continuation_tmux.sh`
- Create: `/data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml`
- Create: `/data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked.yaml`

- [ ] **Step 1: CLI stages**

The CLI must accept:

```text
--config PATH
--stages select_cases,feasibility,rescue_decode,attention_replay,merge,report,gallery
--shard-index INT
--num-shards INT
--merge-shards
--dry-run
```

`select_cases`, `feasibility`, `rescue_decode`, and `attention_replay` require shard args. `merge`, `report`, and `gallery` run on merged artifacts.

- [ ] **Step 2: Smoke config**

Use:

```yaml
paths:
  artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation_smoke
  checkpoint: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
  dataset_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  attention_atlas_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing
  source_selected_cases: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing/selected_cases.jsonl
  source_candidate_regions: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/attention_evidence_routing/candidate_region_rows.jsonl
  rollout_anatomy_per_row: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/rollout_anatomy/per_row.jsonl
  gt_vs_pred_scored: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/gt_vs_pred_scored.jsonl
  pred_token_trace: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/pred_token_trace.jsonl
  infer_resolved_config: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/resolved_config.json
  lane_c_study_config: null
selection:
  evidence_scope: val200_attention_atlas_linked_fn_stratified
  sample_limit: 24
  per_stratum_cap: 4
  duplicate_iou_threshold: 0.95
  wrong_control_overlap_threshold: 0.05
  context_expansion_norm1000: 64
execution:
  attn_implementation: eager
  torch_dtype: bfloat16
  decoding: greedy
  do_sample: false
  num_beams: 1
  max_new_tokens_desc_only: 4
  max_new_tokens_desc_x1: 3
  gallery_per_bucket: 2
```

- [ ] **Step 3: Full linked config**

Same paths as smoke, but:

```yaml
paths:
  artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation
selection:
  sample_limit: 512
  per_stratum_cap: 64
execution:
  gallery_per_bucket: 8
```

- [ ] **Step 4: tmux launcher**

Default to using the available 8 GPUs as parallel analysis shards, not 8-card production training:

```bash
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
CONFIG="${CONFIG:-configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked.yaml}"
SESSION="${SESSION:-autoreg_fn_rescue_ckpt3664}"
```

The launcher must write its exact commands under `${artifact_root}/logs/`.

Safety requirements:

- Default `ALLOW_OVERWRITE=0`.
- Support `DRY_RUN=1`, and require operators to run dry-run before the first full linked launch.
- Refuse to run when resolved `artifact_root` is equal to `attention_atlas_root`, equal to any source artifact root, a parent of any source artifact path, or inside a source artifact root.
- Refuse `ROOT`/environment overrides that disagree with `config.paths.artifact_root`.
- When `ALLOW_OVERWRITE=1`, delete only enumerated FN-rescue outputs and `shards/shard_NNN-of-MMM` directories under the resolved FN-rescue root.
- Write `${artifact_root}/logs/${SESSION}_commands.sh`.
- Redirect each shard/stage stdout and stderr to per-shard/per-stage logs.

It must schedule one GPU process per shard wave and use all available analysis GPUs by default:

```bash
CUDA_VISIBLE_DEVICES=<gpu> ... --stages rescue_decode --shard-index <i> --num-shards <n>
CUDA_VISIBLE_DEVICES=<gpu> ... --stages attention_replay --shard-index <i> --num-shards <n>
```

`rescue_decode` and `attention_replay` may run in separate waves so replay can validate persisted decode prefixes. The launcher must not assume `NUM_SHARDS == len(GPU_LIST)`.

`shards_manifest.json` must record at least:

```text
analysis_name
artifact_schema_version
config_path
config_sha256
git_sha
git_dirty
checkpoint
source_selected_cases_sha256
source_candidate_regions_sha256
rollout_anatomy_per_row_sha256
gt_vs_pred_scored_sha256
pred_token_trace_sha256
attn_implementation_requested
attn_implementation_selected
decoding
do_sample
num_beams
max_new_tokens_desc_only
max_new_tokens_desc_x1
processor_do_resize_or_policy
pre_y1_definition
expected_shard_labels
```

## Task 6: Verification And First-Pass Run

**Files:**

- Uses files created in Tasks 1-5.

- [ ] **Step 1: Unit tests**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest /data/CoordExp/tests/test_autoreg_fn_rescue_continuation.py -q
```

Expected: all tests pass.

- [ ] **Step 2: CLI dry run**

Run:

```bash
PYTHONPATH=/data/CoordExp python /data/CoordExp/scripts/analysis/run_autoreg_fn_rescue_continuation.py \
  --config /data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml \
  --stages select_cases \
  --shard-index 0 \
  --num-shards 8 \
  --dry-run
```

Expected: JSON payload includes checkpoint, artifact root, shard label, stages, evidence scope, every source artifact path, every source artifact sha256, row counts where applicable, selected `attn_implementation`, `decoding`, `do_sample`, `num_beams`, and the exact shard labels expected by merge.

- [ ] **Step 3: One-shard smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/CoordExp python /data/CoordExp/scripts/analysis/run_autoreg_fn_rescue_continuation.py \
  --config /data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml \
  --stages select_cases,feasibility,rescue_decode,attention_replay \
  --shard-index 0 \
  --num-shards 8
```

Expected: shard output contains nonempty selected cases, generation rows, decision rows, candidate region rows, and bounded attention rows.

- [ ] **Step 4: Merge smoke**

Run all smoke shards, then:

```bash
PYTHONPATH=/data/CoordExp python /data/CoordExp/scripts/analysis/run_autoreg_fn_rescue_continuation.py \
  --config /data/CoordExp/configs/analysis/autoreg_fn_rescue_continuation/ckpt3664_val200_linked_smoke.yaml \
  --stages merge,report,gallery \
  --merge-shards \
  --num-shards 8
```

Expected:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation_smoke/summary.json
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation_smoke/report.md
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation_smoke/gallery/gallery_index.jsonl
```

- [ ] **Step 5: Full linked run in tmux**

First inspect the exact command plan:

```bash
DRY_RUN=1 bash /data/CoordExp/scripts/analysis/launch_autoreg_fn_rescue_continuation_tmux.sh
```

Then launch without overwrite for the first run:

```bash
bash /data/CoordExp/scripts/analysis/launch_autoreg_fn_rescue_continuation_tmux.sh
```

Use `ALLOW_OVERWRITE=1` only for an intentional rerun after inspecting the dry-run output list and confirming it only targets FN-rescue outputs.

Monitor:

```bash
tmux capture-pane -pt autoreg_fn_rescue_ckpt3664:0 -S -120
tail -f /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/logs/shard_000-of-008_attention_replay.log
```

Expected final artifacts:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/merge_summary.json
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/summary.json
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/report.md
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation/gallery/gallery_index.jsonl
```

## Self-Review

Spec coverage:

- Diagnostic GT-hint counterfactual, not deployable inference: covered by scope labels, report bounds, and config evidence scope.
- E4 excluded: no sampling stage, no multi-sample proposal inventory.
- Deterministic greedy decoding: covered by generation kwargs and max-new-token contract.
- Hint tiers: covered by `desc_only`, `desc_x1`, and `desc_x1_wrong_control`.
- Success criteria: covered by IoU 0.3/0.5/0.75, valid parse, and same-desc duplicate rejection.
- Existing source ledger: covered by `source_selected_cases`.
- Source ledger is not self-contained: covered by required joins to candidate regions, rollout anatomy, scored infer rows, prediction token trace, and dataset/image provenance.
- Canonical selected-case target index: covered by `target_gt_idx = target_gt_idx or intended_target_gt_idx` normalization and join fail-fast checks.
- Object-count stratification: covered by `object_count_bucket` in selection, denominator rows, summaries, and reports.
- Compact-full object-row insertion: covered by partial-row helper and continuation feasibility gate.
- Raw compact-token prefix reconstruction: covered by `raw_output_json.objects[raw_pred_idx].bbox_2d` and `pred_token_trace` cross-checks; pixel `pred_points` are never directly converted to coord tokens.
- Marker-delimited no-newline contract: covered by prefix reconstruction and hint construction.
- Wrong-control priority: covered by deterministic source selection.
- Concrete far-background fallback: covered by fixed candidate enumeration, overlap rejection, and no-random policy.
- Bounded attention: covered by two-pass `rescue_decode` plus `attention_replay`, `pre_x1` and `pre_y1` query roles, token-id equality checks, and eager fail-fast.
- Rescue region ledger: covered by required region kinds, stable region ids, per-instance diagnostics, and union rows for de-duplicated density claims.
- Reproducible denominator/merge: covered by `rescue_rows.jsonl`, file-specific uniqueness keys, row-count invariants, source/output hashes, and atomic merge writes.
- Strict duplicate policy: covered by strict `max_same_desc_existing_iou > duplicate_iou_threshold` plus duplicate provenance fields.
- Concrete far-background wrong-control: covered by region materialization guardrails.
- Val200-only first pass: covered by config paths and dataset scope.
- Curated gallery: covered by bucket selection, canonical visualization reuse, visual-role provenance, and qualitative-only output contract.
- 8-GPU resource use without production training: covered by tmux launcher and shard semantics.

Placeholder scan:

- No unresolved implementation placeholders are intentionally present.
- Any future implementer must preserve exact artifact names and evidence labels above unless a new documented decision changes them.

Type consistency:

- Config class names use `FnRescue*`.
- Tier names use `desc_only`, `desc_x1`, and `desc_x1_wrong_control`.
- Primary success field is `primary_rescue_success`; report-visible thresholds are `success_iou30`, `success_iou50`, and `success_iou75`.
