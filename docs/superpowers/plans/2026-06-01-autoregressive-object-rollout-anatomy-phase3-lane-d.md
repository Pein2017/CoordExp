# Autoregressive Object Rollout Anatomy Phase 3 Lane D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shard-safe compact-full Lane D hidden-state probe and causal-patching experiment for checkpoint-3664 without reusing the older JSON-oriented instance-binding rendering contract.

**Architecture:** Add a compact-full Lane D adapter that consumes the completed Lane A/C artifacts, materializes selected cases, records CPU-testable compact role positions, then runs GPU hidden-state extraction and patching in immutable shard directories. Reuse existing model-loading and Lane C forced-continuation builders where they preserve compact-full semantics, but keep Lane D outputs under `hidden_state_probe/` with their own shard/merge gates.

**Tech Stack:** Python stdlib, PyYAML, PyTorch/Transformers through existing CoordExp helpers, existing compact-full detection template/tokenization helpers, pytest, tmux shell launcher.

---

## Scope And Evidence Inputs

Primary analysis root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200
```

Required upstream artifacts:

```text
rollout_anatomy/per_row.jsonl
x1_basin_attribution/per_case.jsonl
x1_basin_attribution/summary.json
```

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Dataset:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
```

Lane D output root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe
```

Do not modify upstream HF model files, including `modeling_qwen3_vl.py`.

## Files

- Create: `configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml`
- Create: `scripts/analysis/run_autoreg_hidden_state_probe.py`
- Create: `scripts/analysis/launch_autoreg_object_rollout_lane_d_tmux.sh`
- Create: `src/analysis/autoreg_hidden_state_probe.py`
- Create: `tests/test_autoreg_hidden_state_probe.py`
- Read/import where useful: `src/analysis/hard_ce_coord_logit_locality.py`
- Read only: `src/analysis/qwen3_vl_instance_binding.py`

## Contract

Lane D must not use these JSON-oriented position roles as its compact-full role set:

```text
desc_closing_quote
field_delimiter
bbox_key
bbox_open_bracket
```

The compact-full Lane D roles are:

```text
prompt_end
row_start
desc_end
box_start
pre_x1
post_x1
post_y1
row_end_or_separator
final_generated_prefix_state
```

Each `position_inventory` row must record:

```text
case_id
source_line_idx
prefix_mode
prefix_depth
prefix_quality
render_source
role
absolute_token_index
prediction_token_index
assistant_relative_token_index
assistant_start_token_index
prefix_state_kind
token_text
separator_kind
row_end_source_kind
shard_index
num_shards
shard_label
```

For generated-token roles, `absolute_token_index` names the label/boundary token
associated with the role, while `prediction_token_index` names the hidden/logits
row that predicts it. `prediction_token_index` must be `absolute_token_index - 1`
for `pre_x1`, `post_x1`, `post_y1`, and `desc_end`. `box_start` is defined as the
state after the complete `<|box_start|>` marker, so for multi-token box markers
it shares the x1 prediction boundary with `pre_x1`. Boundary roles such as
`prompt_end`, `row_start`, `row_end_or_separator`, and
`final_generated_prefix_state` may use `prediction_token_index=null`.
`assistant_relative_token_index` may be `null` for `prompt_end`, because it is
outside the assistant span, and for `final_generated_prefix_state` only when
`prefix_state_kind` is `empty_prefix_prompt_end`. All non-null token indices must
be nonnegative integers.

`render_source` must be explicit on every row and one of
`strict_compact_full`, `lane_c_forced_continuation`, or
`generated_prefix_replay`. `separator_kind` must also be explicit on every row
and one of `none_marker_delimited` or `newline`. Source-aware validation applies:
`strict_compact_full` plus `prefix_mode=teacher_forced` requires
`none_marker_delimited`, because objects are adjacent marker-delimited rows with
no newline separator. `lane_c_forced_continuation` and
`generated_prefix_replay` may use either supported separator kind, because the
field must reflect the actual rendered text. Do not normalize strict
teacher-forced text into newline rows.
`row_end_source_kind` must distinguish whether `row_end_or_separator` points to
the next token after the target row or falls back to the last target-row token.

`prefix_state_kind` must be explicit on every row. Ordinary roles use
`not_applicable`. `final_generated_prefix_state` uses
`teacher_forced_prefix_boundary` after a non-empty teacher-forced prefix,
`empty_prefix_prompt_end` for empty prefixes, and later generated-prefix stages
may use `generated_prefix_boundary` or `partial_row`.

Layer groups must come from config and survive into every shard summary:

```yaml
layer_groups:
  early: [0, 1]
  middle: [12, 13]
  late: [24, 25, 26, 27]
  last: [-4, -3, -2, -1]
```

Sharded stage layout:

```text
hidden_state_probe/shards_manifest.json
hidden_state_probe/shards/shard_000-of-008/selected_cases.jsonl
hidden_state_probe/shards/shard_000-of-008/position_inventory.jsonl
hidden_state_probe/shards/shard_000-of-008/probe_rows.jsonl
hidden_state_probe/shards/shard_000-of-008/patch_rows.jsonl
hidden_state_probe/shards/shard_000-of-008/summary.json
hidden_state_probe/selected_cases.jsonl
hidden_state_probe/position_inventory.jsonl
hidden_state_probe/probe_rows.jsonl
hidden_state_probe/patch_rows.jsonl
hidden_state_probe/summary.json
hidden_state_probe/merge_summary.json
hidden_state_probe/report.md
```

Shard selection must be by stable `source_line_idx % num_shards`, not by individual case ordinal, so all Lane D cases from the same image stay on the same GPU.

## Task 1: CPU Case Selection And Compact Role Inventory

**Files:**

- Create: `src/analysis/autoreg_hidden_state_probe.py`
- Create: `tests/test_autoreg_hidden_state_probe.py`

- [ ] **Step 1: Write failing tests for shard selection**

Add tests:

```python
def test_lane_d_record_selected_keeps_images_together() -> None:
    assert lane_d_record_selected(0, shard_index=0, num_shards=2)
    assert lane_d_record_selected(2, shard_index=0, num_shards=2)
    assert not lane_d_record_selected(1, shard_index=0, num_shards=2)


def test_normalize_lane_d_shard_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="shard_index"):
        normalize_lane_d_shard(shard_index=2, num_shards=2)
    with pytest.raises(ValueError, match="num_shards"):
        normalize_lane_d_shard(shard_index=0, num_shards=0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_hidden_state_probe.py::test_lane_d_record_selected_keeps_images_together tests/test_autoreg_hidden_state_probe.py::test_normalize_lane_d_shard_rejects_invalid_args -q
```

Expected before implementation: import/function-not-found failure.

- [ ] **Step 3: Implement shard helpers**

Implement:

```python
def lane_d_shard_label(shard_index: int, num_shards: int) -> str:
    return f"shard_{shard_index:03d}-of-{num_shards:03d}"


def normalize_lane_d_shard(
    *, shard_index: int | None, num_shards: int | None
) -> tuple[int | None, int | None, str | None]:
    if shard_index is None and num_shards is None:
        return None, None, None
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return shard_index, num_shards, lane_d_shard_label(shard_index, num_shards)


def lane_d_record_selected(
    source_line_idx: int, *, shard_index: int | None, num_shards: int | None
) -> bool:
    shard_index, num_shards, _ = normalize_lane_d_shard(
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if shard_index is None or num_shards is None:
        return True
    return int(source_line_idx) % int(num_shards) == int(shard_index)
```

- [ ] **Step 4: Write failing compact inventory validation tests**

Add tests. Keep the embedded examples concise; the canonical implementation and
complete test helper now live in
`src/analysis/autoreg_hidden_state_probe.py` and
`tests/test_autoreg_hidden_state_probe.py`.

```python
def _inventory_rows_for_roles(
    *roles: str,
    render_source: str = "generated_prefix_replay",
    separator_kind: str = "none_marker_delimited",
) -> list[dict[str, object]]:
    # Required row keys include render_source, separator_kind, and
    # prefix_state_kind. Inventory contexts are source-aware:
    context_key = (
        source_line_idx,
        case_id,
        prefix_mode,
        prefix_depth,
        render_source,
    )

    # prompt_end is outside the assistant span. final_generated_prefix_state is
    # nullable only when it represents an empty-prefix prompt-end state.
    assistant_relative_token_index = (
        None
        if role == "prompt_end"
        or prefix_state_kind == "empty_prefix_prompt_end"
        else assistant_relative_index
    )
```

Assert the source-aware inventory contract:

- every row validates explicit `render_source`, `separator_kind`, and
  `prefix_state_kind`;
- `strict_compact_full` plus `prefix_mode=teacher_forced` rejects `newline` and
  accepts `none_marker_delimited`;
- `lane_c_forced_continuation` and `generated_prefix_replay` accept `newline`
  when that is the actual rendered separator;
- `prompt_end` requires nullable assistant-relative state;
- `final_generated_prefix_state` requires nullable assistant-relative state only
  with `prefix_state_kind="empty_prefix_prompt_end"`;
- the same case and role can appear under different `render_source` contexts
  without being treated as a duplicate, provided each context has the complete
  compact role set.

- [ ] **Step 5: Implement inventory validation**

Implement source-aware validation. This is pseudo-code only; keep
`src/analysis/autoreg_hidden_state_probe.py` canonical.

```python
LANE_D_COMPACT_ROLES = (
    "prompt_end",
    "row_start",
    "desc_end",
    "box_start",
    "pre_x1",
    "post_x1",
    "post_y1",
    "row_end_or_separator",
    "final_generated_prefix_state",
)


def validate_lane_d_position_inventory(rows: Sequence[Mapping[str, Any]]) -> None:
    roles_by_context: dict[
        tuple[int, str, str, int, str], dict[str, Mapping[str, Any]]
    ] = {}
    for row in rows:
        case_id = require_nonempty_string(row, "case_id")
        source_line_idx = require_nonnegative_int(row, "source_line_idx")
        prefix_mode = require_nonempty_string(row, "prefix_mode")
        prefix_depth = require_nonnegative_int(row, "prefix_depth")
        render_source = require_one_of(row, "render_source", LANE_D_RENDER_SOURCES)
        separator_kind = require_one_of(row, "separator_kind", LANE_D_SEPARATOR_KINDS)
        prefix_state_kind = require_one_of(
            row,
            "prefix_state_kind",
            LANE_D_PREFIX_STATE_KINDS,
        )
        role = require_one_of(row, "role", LANE_D_COMPACT_ROLES)

        context_key = (
            source_line_idx,
            case_id,
            prefix_mode,
            prefix_depth,
            render_source,
        )
        context_roles = roles_by_context.setdefault(context_key, {})
        if role not in LANE_D_COMPACT_ROLES:
            raise ValueError(f"unknown compact role for {case_id}: {role}")
        if role in context_roles:
            raise ValueError(f"duplicate compact role for {case_id}: {role}")

        if render_source == "strict_compact_full" and prefix_mode == "teacher_forced":
            if separator_kind != "none_marker_delimited":
                raise ValueError("strict teacher-forced rows are marker-delimited")

        relative = require_present(row, "assistant_relative_token_index")
        if role == "prompt_end":
            require_none(relative)
        elif (
            role == "final_generated_prefix_state"
            and prefix_state_kind == "empty_prefix_prompt_end"
        ):
            require_none(relative)
        else:
            require_nonnegative_int_value(relative)

        context_roles[role] = row

    for (
        source_line_idx,
        case_id,
        prefix_mode,
        prefix_depth,
        render_source,
    ), context_roles in roles_by_context.items():
        missing = [role for role in LANE_D_COMPACT_ROLES if role not in context_roles]
        if missing:
            raise ValueError(
                f"case {case_id} missing compact roles for "
                f"source_line_idx={source_line_idx}, prefix_mode={prefix_mode}, "
                f"prefix_depth={prefix_depth}, render_source={render_source}: "
                f"{', '.join(missing)}"
            )
```

- [ ] **Step 6: Write failing selected-case tests from Lane C per-case rows**

Create fixture Lane C rows and assert:

```python
def test_select_lane_d_cases_prefers_x1_failure_cohorts_and_preserves_case_fields(tmp_path: Path) -> None:
    path = tmp_path / "per_case.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "row0:self_prefix:depth2:gt19",
                "source_line_idx": 0,
                "prefix_mode": "self_prefix",
                "prefix_depth": 2,
                "prefix_quality": "fp_prefix",
                "intended_target_gt_idx": 19,
                "target_desc": "vase",
                "x1": {"top_peak_attribution": "same_desc_competitor_gt_object", "target_rank": 350},
            },
            {
                "case_id": "row1:teacher_forced:depth0:gt0",
                "source_line_idx": 1,
                "prefix_mode": "teacher_forced",
                "prefix_depth": 0,
                "prefix_quality": "gt_prefix",
                "intended_target_gt_idx": 0,
                "target_desc": "person",
                "x1": {"top_peak_attribution": "target_gt_object", "target_rank": 1},
            },
        ],
    )
    rows = select_lane_d_cases(path, max_cases=1, shard_index=None, num_shards=None)
    assert rows == [
        {
            "case_id": "row0:self_prefix:depth2:gt19",
            "source_line_idx": 0,
            "prefix_mode": "self_prefix",
            "prefix_depth": 2,
            "prefix_quality": "fp_prefix",
            "intended_target_gt_idx": 19,
            "target_desc": "vase",
            "x1_top_peak_attribution": "same_desc_competitor_gt_object",
            "x1_target_rank": 350,
            "selection_reason": "x1_non_target_or_low_rank",
        }
    ]
```

- [ ] **Step 7: Implement selected-case reader**

Implement `select_lane_d_cases(...)` to:

- read Lane C `per_case.jsonl`;
- preserve case identity and prefix metadata;
- rank cases with non-target `x1.top_peak_attribution` or `x1.target_rank > 32` first;
- apply `lane_d_record_selected(source_line_idx, ...)`;
- respect `max_cases`;
- return deterministic rows sorted by `(priority, source_line_idx, prefix_mode, prefix_depth, case_id)`.

- [ ] **Step 8: Verify Task 1**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_hidden_state_probe.py -q
```

Expected: Task 1 tests pass.

## Task 2: Config And CLI Dry-Run Contract

**Files:**

- Create: `configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml`
- Create: `scripts/analysis/run_autoreg_hidden_state_probe.py`
- Modify: `src/analysis/autoreg_hidden_state_probe.py`
- Modify: `tests/test_autoreg_hidden_state_probe.py`

- [ ] **Step 1: Write failing config-load test**

Add a test that writes a tiny YAML and asserts parsed fields:

```python
def test_load_lane_d_config_parses_roots_roles_and_layer_groups(tmp_path: Path) -> None:
    cfg = tmp_path / "lane_d.yaml"
    cfg.write_text(
        f"""
paths:
  artifact_root: {tmp_path / "hidden_state_probe"}
  checkpoint: /ckpt
  dataset_jsonl: /data.jsonl
  self_rollout_root: /rollout
  lane_a_rollout_root: /analysis/rollout_anatomy
  lane_c_per_case: /analysis/x1_basin_attribution/per_case.jsonl
selection:
  max_cases: 12
positions:
  roles: [prompt_end, row_start, desc_end, box_start, pre_x1, post_x1, post_y1, row_end_or_separator, final_generated_prefix_state]
  layer_groups:
    late: [24, 25, 26, 27]
execution:
  batch_size: 2
""".lstrip(),
        encoding="utf-8",
    )
    loaded = load_lane_d_config(cfg)
    assert loaded.paths.artifact_root == tmp_path / "hidden_state_probe"
    assert loaded.selection.max_cases == 12
    assert loaded.positions.roles == LANE_D_COMPACT_ROLES
    assert loaded.positions.layer_groups == {"late": (24, 25, 26, 27)}
```

- [ ] **Step 2: Implement dataclasses and config loader**

Implement dataclasses:

```python
@dataclass(frozen=True)
class LaneDPaths:
    artifact_root: Path
    checkpoint: Path
    dataset_jsonl: Path
    self_rollout_root: Path
    lane_a_rollout_root: Path
    lane_c_per_case: Path


@dataclass(frozen=True)
class LaneDSelectionConfig:
    max_cases: int = 512


@dataclass(frozen=True)
class LaneDPositionsConfig:
    roles: tuple[str, ...] = LANE_D_COMPACT_ROLES
    layer_groups: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: {...})


@dataclass(frozen=True)
class LaneDExecutionConfig:
    batch_size: int = 1


@dataclass(frozen=True)
class LaneDConfig:
    config_path: Path
    paths: LaneDPaths
    selection: LaneDSelectionConfig
    positions: LaneDPositionsConfig
    execution: LaneDExecutionConfig
```

`load_lane_d_config` must reject unknown or duplicate roles by calling `validate_configured_lane_d_roles`.

- [ ] **Step 3: Add the default checkpoint-3664 YAML**

Create:

```yaml
paths:
  artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe
  checkpoint: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
  dataset_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  self_rollout_root: /data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
  lane_a_rollout_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/rollout_anatomy
  lane_c_per_case: /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/x1_basin_attribution/per_case.jsonl
selection:
  max_cases: 512
positions:
  roles:
    - prompt_end
    - row_start
    - desc_end
    - box_start
    - pre_x1
    - post_x1
    - post_y1
    - row_end_or_separator
    - final_generated_prefix_state
  layer_groups:
    early: [0, 1]
    middle: [12, 13]
    late: [24, 25, 26, 27]
    last: [-4, -3, -2, -1]
execution:
  batch_size: 1
```

- [ ] **Step 4: Add CLI with non-model dry-run**

Create `scripts/analysis/run_autoreg_hidden_state_probe.py` accepting:

```text
--config
--stages select_cases,position_inventory,hidden_states,patching,merge,report
--shard-index
--num-shards
--dry-run
--merge-shards
```

`--dry-run` must load config, print selected shard labels and output paths, and must not import/load model code.

- [ ] **Step 5: Verify config and CLI**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_hidden_state_probe.py -q
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_hidden_state_probe.py \
  --config configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml \
  --stages select_cases,position_inventory \
  --dry-run \
  --num-shards 8
```

Expected: tests pass and dry-run prints 8 unique `shard_XXX-of-008` paths.

## Task 3: Hidden-State Shard Extraction And Merge Gate

**Files:**

- Modify: `src/analysis/autoreg_hidden_state_probe.py`
- Modify: `tests/test_autoreg_hidden_state_probe.py`

- [ ] **Step 1: Write failing merge tests**

Add fixture shard dirs:

```text
shards/shard_000-of-002/{selected_cases,position_inventory,probe_rows,patch_rows,summary}.*
shards/shard_001-of-002/{selected_cases,position_inventory,probe_rows,patch_rows,summary}.*
```

Assert merge:

```python
summary = merge_lane_d_shards(root=tmp_path / "hidden_state_probe", expected_shards=2)
assert summary["row_counts"]["probe_rows"] == 2
assert (tmp_path / "hidden_state_probe/probe_rows.jsonl").exists()
```

Duplicate keys must fail:

```python
with pytest.raises(ValueError, match="duplicate Lane D row key"):
    merge_lane_d_shards(root=root, expected_shards=2)
```

The duplicate key domain is:

```text
selected_cases: (source_line_idx, case_id)
position_inventory: (source_line_idx, case_id, prefix_mode, prefix_depth, render_source, role)
probe_rows: (source_line_idx, case_id, prefix_mode, prefix_depth, task, role, layer_group, model_layer, slot, target_label_source)
patch_rows: (source_line_idx, case_id, prefix_mode, prefix_depth, patch_policy, donor_policy, donor_case_id, role, layer_group, model_layer, slot)
```

- [ ] **Step 2: Implement merge gate**

`merge_lane_d_shards` must:

- require exactly `expected_shards` immediate shard directories;
- reject missing, unexpected, or malformed shard labels;
- require and validate `shards_manifest.json`;
- require every shard to contain `summary.json`;
- concatenate `selected_cases.jsonl`, `position_inventory.jsonl`, `probe_rows.jsonl`, and `patch_rows.jsonl`;
- validate merged position inventory;
- reject duplicate row keys;
- write `merge_summary.json` with shard labels, row counts, source summaries,
  config/checkpoint, selected-cases hash, compact role set, layer groups,
  duplicate-key domains, and output paths.

- [ ] **Step 3: Implement GPU hidden extraction stage**

Use `src.analysis.hard_ce_coord_logit_locality` helpers to preserve compact-full rendering:

```python
from src.analysis.hard_ce_coord_logit_locality import (
    load_model_handle,
    prepare_lane_c_x1_basin_examples,
)
```

For each selected case:

- build the same Lane C forced-continuation example;
- compute compact roles with `build_lane_d_position_inventory_for_example`;
- run model forward with `output_hidden_states=True`;
- write one `probe_rows.jsonl` row per `(case_id, role, layer_group, model_layer)`;
- include a minimal probe target ledger: `object_count`, `remaining_count`, `intended_target_gt_idx`, `prefix_quality`, `x1_top_peak_attribution`, and `x1_target_rank`.

Do not interpret probe separability in this stage; write vectors or scalar summaries only.

- [ ] **Step 4: Verify GPU stage with dry-run and CPU merge fixtures**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_hidden_state_probe.py -q
PYTHONPATH=/data/CoordExp python scripts/analysis/run_autoreg_hidden_state_probe.py \
  --config configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml \
  --stages hidden_states \
  --dry-run \
  --shard-index 0 \
  --num-shards 8
```

Expected: tests pass and dry-run does not load model.

## Task 4: Causal Patching Rows

**Files:**

- Modify: `src/analysis/autoreg_hidden_state_probe.py`
- Modify: `tests/test_autoreg_hidden_state_probe.py`

- [ ] **Step 1: Write failing patch-policy validation tests**

Patch policies must include:

```text
self_noop
attenuate_pre_x1
copy_same_image_same_desc
copy_wrong_image_same_desc
```

Unknown policies must fail config load.

- [ ] **Step 2: Implement first-pass patch rows**

Implement a conservative first-pass causal patching stage:

- `self_noop`: rerun logits with no hook and record numerical drift control;
- `attenuate_pre_x1`: zero or scale hidden states at configured layers for `pre_x1`;
- `copy_same_image_same_desc`: copy donor hidden vector from another same-image same-desc case at the same role/layer;
- `copy_wrong_image_same_desc`: copy donor hidden vector from a different image same-desc case.

Each patch row must record:

```text
case_id
donor_case_id
patch_policy
role
layer_group
model_layers
baseline_x1_target_margin
patched_x1_target_margin
target_margin_delta
baseline_top_peak_attribution
patched_top_peak_attribution
```

- [ ] **Step 3: Verify patch rows on fixture summaries**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_hidden_state_probe.py -q
```

Expected: tests pass. Do not launch patching on GPU until both reviews approve.

## Task 5: tmux Launcher And Review Gate

**Files:**

- Create: `scripts/analysis/launch_autoreg_object_rollout_lane_d_tmux.sh`
- Modify: `tests/test_autoreg_hidden_state_probe.py`

- [ ] **Step 1: Create launch script**

The script must:

- use `set -euo pipefail`;
- default `SESSION=autoreg_lane_d_ckpt3664`;
- default `NUM_SHARDS=8`;
- default `CONFIG=configs/analysis/autoreg_hidden_state_probe/ckpt3664_lane_d_val200.yaml`;
- parse YAML `paths.artifact_root` and require it equals `$ROOT`;
- scan all immediate directories under `$ROOT/shards`;
- support `DRY_RUN=1`;
- support `ALLOW_OVERWRITE=1`, but never delete anything during dry-run;
- launch one shard per GPU with `CUDA_VISIBLE_DEVICES=$i`;
- run merge after all shards complete;
- write logs under `$ROOT/logs`;
- print attach/capture/tail instructions.

- [ ] **Step 2: Run launcher dry-run checks**

Run:

```bash
bash -n scripts/analysis/launch_autoreg_object_rollout_lane_d_tmux.sh
DRY_RUN=1 bash scripts/analysis/launch_autoreg_object_rollout_lane_d_tmux.sh
```

Expected: exits 0, prints 8 `CUDA_VISIBLE_DEVICES=` shard commands and one merge command.

- [ ] **Step 3: Dispatch two fresh review subagents**

Spec reviewer:

```text
Review Lane D compact-full inventory/shard/merge/launcher against the design spec.
Read-only. Do not run GPU jobs.
Return APPROVED or CHANGES_REQUESTED.
```

Code-quality reviewer:

```text
Review Lane D implementation for stale shard hazards, duplicate merge keys,
dry-run non-destructiveness, test coverage, and JSON-oriented role leakage.
Read-only. Do not run GPU jobs.
Return APPROVED or CHANGES_REQUESTED.
```

- [ ] **Step 4: Fix review findings**

If either reviewer returns `CHANGES_REQUESTED`, fix with targeted tests and rerun the review.

## Task 6: Launch Lane D Only After Gates Pass

- [ ] **Step 1: Final preflight**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tmux has-session -t autoreg_lane_d_ckpt3664 2>/dev/null; echo $?
DRY_RUN=1 bash scripts/analysis/launch_autoreg_object_rollout_lane_d_tmux.sh
```

Expected:

- 8 GPUs visible and idle enough for one worker per card;
- `tmux has-session` returns nonzero;
- dry-run prints 8 unique shard paths.

- [ ] **Step 2: Launch tmux**

Run:

```bash
bash scripts/analysis/launch_autoreg_object_rollout_lane_d_tmux.sh
```

Expected: starts `autoreg_lane_d_ckpt3664`.

- [ ] **Step 3: Verify live run**

Run:

```bash
tmux capture-pane -pt autoreg_lane_d_ckpt3664:0 -S -200
pgrep -af 'run_autoreg_hidden_state_probe|autoreg_lane_d'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

Expected: 8 shard processes running, GPU memory nonzero, no immediate OOM/path failure.

- [ ] **Step 4: Handoff commands**

Tell the user:

```bash
tmux attach -t autoreg_lane_d_ckpt3664
tmux capture-pane -pt autoreg_lane_d_ckpt3664:0 -S -200
tail -f /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/logs/lane_d_shard_000-of-008.log
tail -f /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/logs/merge.log
```

## Verification Summary

Before interpreting Lane D:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_autoreg_hidden_state_probe.py -q
python -m json.tool /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/merge_summary.json
python -m json.tool /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/hidden_state_probe/summary.json
```

Do not claim H1/H2/H3 convergence unless the merged Lane D probe and patch rows both exist and the report explicitly compares probe evidence with causal patch controls.
