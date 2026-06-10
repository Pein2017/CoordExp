# Prefix-State Transition Tomography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase A3.1 prefix-state transition tomography for paired
ET-RMP-CE vs pure-CE checkpoint-3664 diagnostics.

**Architecture:** Create a new idea-wise analysis project named
`prefix_state_transition_tomography`. The pipeline first materializes
teacher-controlled prefix-state rows on CPU, applies deterministic launch
gates, then runs paired checkpoint GPU readouts for boundary full-desc span
scores and forced-desc `pre_x1` posteriors. Reports center residual-state
alignment, same-desc and different-desc transition splits, paired deltas, and a
four-quadrant mechanism table.

**Tech Stack:** Python, JSON/JSONL, PyYAML, pytest, matplotlib/Pillow for
sampled gallery, PyTorch/Qwen3-VL only in GPU probe runtime, tmux shell
launcher for 8-card analysis shards.

---

## Scope And Stop Conditions

This plan implements the experiment design recorded in:

```text
/data/CoordExp/.worktrees/fn-rescue-attention-probes/docs/superpowers/specs/2026-06-03-candidate-field-cardinality-tomography-design.md
```

Implementation scope:

- Build the Phase A3.1 index-only CPU stage.
- Build deterministic sampling and launch gates.
- Build paired ET-RMP-CE and pure-CE GPU readouts for 4096 prefix-state rows.
- Build merge, report, summary, and sampled gallery artifacts.
- Do not dump full attention in the first 4096-row probe.
- Do not start training, painting, category-sorted SFT, object-marginal
  training, or production jobs.

Stop after implementation when:

- Unit tests and py_compile pass.
- Index-only stage produces `launch_eligible=true` or a clear failed-gate
  summary.
- If eligible, the linked tmux run starts the paired GPU probe and writes
  status/log handles.
- Final report states whether the GPU run is complete or still running.

## Source Checkpoints And Artifact Root

ET-RMP-CE checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Pure-CE checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664
```

Primary artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096
```

## File Structure

Create this project namespace:

```text
/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/
/data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/
/data/CoordExp/.worktrees/fn-rescue-attention-probes/src/analysis/prefix_state_transition_tomography/
/data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/
```

Planned files and responsibilities:

```text
configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml
  Main executable config for Phase A3.1 paired 4096-row probe.

scripts/analysis/prefix_state_transition_tomography/run.py
  Thin CLI wrapper around runner.run_from_config.

scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh
  Linked tmux launcher: index stage, gate check, GPU shard wave, merge, report, gallery.

scripts/analysis/prefix_state_transition_tomography/status.py
  Compact status reader for logs, summary files, launch gates, and shard completion.

src/analysis/prefix_state_transition_tomography/__init__.py
  Project constants and public project id.

src/analysis/prefix_state_transition_tomography/config.py
  YAML parsing and dataclass config objects.

src/analysis/prefix_state_transition_tomography/jsonl.py
  Small JSONL read/write helpers used by index, merge, and tests.

src/analysis/prefix_state_transition_tomography/prefix_state_index.py
  CPU-only teacher prefix-state generation, deterministic sampler, launch-gate summary.

src/analysis/prefix_state_transition_tomography/prefix_rendering.py
  Compact-full teacher prefix, boundary prompt, and forced-desc prompt text rendering.

src/analysis/prefix_state_transition_tomography/boundary_scoring.py
  Full-desc span scoring interface and GPU runtime implementation.

src/analysis/prefix_state_transition_tomography/x1_readout.py
  Forced-desc pre-x1 posterior extraction and x1 peak attribution.

src/analysis/prefix_state_transition_tomography/paired_probe.py
  Sharded paired checkpoint GPU probe orchestration.

src/analysis/prefix_state_transition_tomography/merge_report.py
  Shard merge, paired-row validation, metrics, quadrant table, and report generation.

src/analysis/prefix_state_transition_tomography/gallery.py
  Sampled high-value manual-review gallery.

src/analysis/prefix_state_transition_tomography/runner.py
  Stage dispatcher used by CLI and tests.

tests/analysis/prefix_state_transition_tomography/test_config.py
tests/analysis/prefix_state_transition_tomography/test_prefix_state_index.py
tests/analysis/prefix_state_transition_tomography/test_prefix_rendering.py
tests/analysis/prefix_state_transition_tomography/test_boundary_scoring.py
tests/analysis/prefix_state_transition_tomography/test_x1_readout.py
tests/analysis/prefix_state_transition_tomography/test_merge_report.py
tests/analysis/prefix_state_transition_tomography/test_runner_cli.py
```

## Artifact Contract

Index-only outputs:

```text
resolved_config.yaml
prefix_state_index.jsonl
prefix_state_index_summary.json
prefix_state_sampled_rows.jsonl
```

GPU shard outputs:

```text
shards/shard_00/boundary_score_rows.jsonl
shards/shard_00/forced_x1_rows.jsonl
shards/shard_00/shard_summary.json
...
shards/shard_07/boundary_score_rows.jsonl
shards/shard_07/forced_x1_rows.jsonl
shards/shard_07/shard_summary.json
```

Merged outputs:

```text
boundary_score_rows.jsonl
forced_x1_rows.jsonl
paired_state_rows.jsonl
quadrant_rows.jsonl
summary.json
report.md
merge_summary.json
gallery/gallery_rows.jsonl
gallery/index.md
gallery/images/*.jpg
manifest.json
```

Every row must include:

```text
schema_version
project_id = prefix_state_transition_tomography
phase_id = phase_a3_1
run_id
checkpoint_id
checkpoint_role = et_rmp_ce | pure_ce
split
source_dataset_jsonl
source_line_idx
image_id
image_path
prefix_state_id
transition_type = same_desc_transition | different_desc_transition
prefix_condition
prefix_depth
prefix_order_policy_id
emitted_gt_indices
residual_gt_indices
emitted_descs
residual_descs
probe_desc
probe_desc_role = target_residual_desc | hard_competitor_desc
readout_type = boundary_full_desc_span | forced_desc_pre_x1
shard_id
```

Index-stage exception:

- `prefix_state_index.jsonl` and `prefix_state_sampled_rows.jsonl` are CPU
  planning rows.  They use `checkpoint_role = paired_index` and
  `readout_type = prefix_state_index`.
- GPU readout rows must use `checkpoint_role = et_rmp_ce | pure_ce` and
  `readout_type = boundary_full_desc_span | forced_desc_pre_x1`.

Paired-key fields:

```text
image_id
source_line_idx
prefix_state_id
prefix_condition
prefix_depth
prefix_order_policy_id
probe_desc
readout_type
```

## Task 1: Project Skeleton, Config, And CLI Dry Run

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/__init__.py`
- Create: `src/analysis/prefix_state_transition_tomography/config.py`
- Create: `src/analysis/prefix_state_transition_tomography/jsonl.py`
- Create: `src/analysis/prefix_state_transition_tomography/runner.py`
- Create: `scripts/analysis/prefix_state_transition_tomography/run.py`
- Create: `configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml`
- Create: `tests/analysis/prefix_state_transition_tomography/test_config.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_runner_cli.py`

- [ ] **Step 1: Write config parse tests**

Create `tests/analysis/prefix_state_transition_tomography/test_config.py` with
tests that parse a minimal config and reject wrong `project_id`.

Expected config dataclass fields:

```python
from pathlib import Path

from src.analysis.prefix_state_transition_tomography.config import load_config


def test_load_config_parses_phase_a3_pair(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_id: prefix_state_transition_tomography
artifact_root: /tmp/a3
train_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
val_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
stages: [prefix_state_index]
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et/checkpoint-3664
  pure_ce:
    checkpoint_path: /ckpts/pure/checkpoint-3664
sampling:
  max_prefix_states: 4096
  num_shards: 8
  seed: 3664
peak:
  absolute_mass_floor: 0.002
  relative_floor: 0.10
  primary_merge_radius: 24
  gt_x1_neighborhood_radius: 24
  raw_topk_k: 32
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.project_id == "prefix_state_transition_tomography"
    assert cfg.artifact_root == Path("/tmp/a3")
    assert cfg.checkpoints["et_rmp_ce"].checkpoint_path == Path("/ckpts/et/checkpoint-3664")
    assert cfg.sampling.max_prefix_states == 4096
    assert cfg.sampling.num_shards == 8
    assert cfg.peak.primary_merge_radius == 24


def test_load_config_rejects_wrong_project_id(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
project_id: candidate_field_cardinality_tomography
artifact_root: /tmp/a3
train_jsonl: /tmp/train.jsonl
val_jsonl: /tmp/val.jsonl
checkpoints:
  et_rmp_ce:
    checkpoint_path: /ckpts/et
  pure_ce:
    checkpoint_path: /ckpts/pure
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as exc:
        assert "project_id must be prefix_state_transition_tomography" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_config.py \
  -q
```

Expected: fail with missing module or missing `load_config`.

- [ ] **Step 3: Implement minimal project constants and config parser**

Implement:

```python
# src/analysis/prefix_state_transition_tomography/__init__.py
PROJECT_ID = "prefix_state_transition_tomography"
PHASE_ID = "phase_a3_1"
SCHEMA_VERSION = "a3.1.v1"
```

`config.py` should define:

```python
@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_path: Path

@dataclass(frozen=True)
class SamplingConfig:
    max_prefix_states: int = 4096
    num_shards: int = 8
    seed: int = 3664
    easy_sanity_max_fraction: float = 0.20

@dataclass(frozen=True)
class PeakConfig:
    absolute_mass_floor: float = 0.002
    relative_floor: float = 0.10
    primary_merge_radius: int = 24
    gt_x1_neighborhood_radius: int = 24
    raw_topk_k: int = 32

@dataclass(frozen=True)
class PrefixStateTransitionConfig:
    project_id: str
    artifact_root: Path
    train_jsonl: Path
    val_jsonl: Path
    stages: tuple[str, ...]
    checkpoints: dict[str, CheckpointConfig]
    sampling: SamplingConfig
    peak: PeakConfig
```

Known stages:

```python
KNOWN_STAGES = {
    "prefix_state_index",
    "paired_checkpoint_probe",
    "merge",
    "report",
    "gallery",
    "validate",
}
```

- [ ] **Step 4: Add JSONL helpers**

`jsonl.py` should contain:

```python
def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 5: Add CLI wrapper and dry-run runner**

`runner.py` should load config and return a JSON-safe dry-run summary when
`dry_run=True`. `scripts/.../run.py` should mirror the existing candidate-field
thin CLI and support `--config`, `--stages`, `--dry-run`, `--allow-overwrite`,
and `--shard-id`.

- [ ] **Step 6: Add real A3 config**

Create `ckpt3664_et_vs_purece_phase_a3_4096.yaml` with the exact checkpoint
paths and artifact root from this plan. Initial stages:

```yaml
stages:
  - prefix_state_index
  - validate
```

The linked launcher can override stages later.

- [ ] **Step 7: Verify Task 1**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_config.py \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_runner_cli.py \
  -q
```

Expected: all tests pass.

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/run.py \
  --config /data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
  --dry-run
```

Expected: JSON output with `project_id`, `artifact_root`, `stages`, and both
checkpoint roles.

## Task 2: Prefix-State Index And Launch Gates

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/prefix_state_index.py`
- Modify: `src/analysis/prefix_state_transition_tomography/runner.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_prefix_state_index.py`

- [ ] **Step 1: Write fixture-driven index tests**

Use small JSONL records with `objects` containing `desc` and `bbox_2d`.
Include:

- one same-desc hard image with at least 4 `person` objects and another desc;
- one mixed-desc hard image with at least 3 desc groups and 6 objects;
- one easy count-1 image.

Test expectations:

```python
rows, summary = build_prefix_state_index(
    train_jsonl=train_path,
    val_jsonl=val_path,
    run_id="test-run",
    max_prefix_states=32,
    seed=3664,
)

assert any(row["transition_type"] == "same_desc_transition" for row in rows)
assert any(row["transition_type"] == "different_desc_transition" for row in rows)
assert {"shallow_1", "mid_half", "late_one_left"} & {
    row["prefix_depth"] for row in rows
}
assert "launch_eligible" in summary
assert "failed_launch_gates" in summary
```

- [ ] **Step 2: Implement prefix-state row generation**

`build_prefix_state_index(...)` should:

- read train and val JSONLs;
- group objects by canonical desc using lower/strip/collapse whitespace;
- construct rows for:
  - `same_desc_prefix_k`;
  - `different_desc_prefix_k`;
  - `class_block_prefix`;
  - `spatial_prefix`;
  - `size_salience_prefix`;
  - `original_order_prefix`;
- assign prefix depths:
  - `empty`;
  - `shallow_1`;
  - `mid_half`;
  - `late_one_left`;
  - `class_block_done`;
- classify rows as `headline_hard` or `easy_sanity`;
- select deterministic sampled rows up to `max_prefix_states`.

Each row must include the minimum fields listed in the Artifact Contract.

- [ ] **Step 3: Implement launch gate summary**

Summary must include:

```python
{
    "launch_eligible": bool,
    "failed_launch_gates": list[str],
    "row_counts": {
        "indexed_prefix_states": int,
        "sampled_prefix_states": int,
        "easy_sanity_rows": int,
        "headline_hard_rows": int,
    },
    "by_split_transition_type": {...},
    "by_prefix_depth": {...},
    "sampling_adjustments": list[dict[str, Any]],
}
```

Gate names:

```text
missing_train_same_desc_transition
missing_train_different_desc_transition
missing_val_same_desc_transition
missing_val_different_desc_transition
insufficient_train_prefix_depth_coverage
insufficient_val_prefix_depth_coverage
same_desc_headline_residual_count_lt2
easy_sanity_fraction_gt20pct
```

Mixed-desc hard ratios are reported but do not fail the gate unless there are
zero mixed-desc hard rows.

- [ ] **Step 4: Wire runner stage**

`runner.py` stage `prefix_state_index` writes:

```text
resolved_config.yaml
prefix_state_index.jsonl
prefix_state_index_summary.json
prefix_state_sampled_rows.jsonl
```

Respect `allow_overwrite=False` by failing if the artifact root already exists.

- [ ] **Step 5: Verify Task 2**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_prefix_state_index.py \
  -q
```

Expected: all tests pass.

Run real index-only stage:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/run.py \
  --config /data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
  --stages prefix_state_index,validate \
  --allow-overwrite
```

Expected:

- `prefix_state_index_summary.json` exists;
- `prefix_state_sampled_rows.jsonl` has at most 4096 rows;
- `launch_eligible` is present and deterministic.

## Task 3: Prefix Rendering And Boundary Candidate Universe

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/prefix_rendering.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_prefix_rendering.py`

- [ ] **Step 1: Write rendering tests**

Test that teacher prefix rows preserve object order and that boundary and
forced-desc prompts end at the expected places.

Expected helper API:

```python
render_compact_object_row(desc: str, bbox_xyxy: Sequence[int]) -> str
render_teacher_prefix(rows: Sequence[Mapping[str, Any]]) -> str
render_boundary_assistant_text(prefix_rows: Sequence[Mapping[str, Any]]) -> str
render_forced_desc_pre_x1_assistant_text(prefix_rows: Sequence[Mapping[str, Any]], desc: str) -> str
```

Assertions:

```python
assert render_compact_object_row("person", [10, 20, 30, 40]).startswith("<|object_ref_start|>person")
assert render_forced_desc_pre_x1_assistant_text(prefix_rows, "chair").endswith("<|box_start|>")
assert "<|box_end|>" in render_boundary_assistant_text(prefix_rows)
```

- [ ] **Step 2: Implement rendering helpers**

Use compact-full desc-first xyxy coord-token text compatible with existing
candidate-field prompting:

```text
<|object_ref_start|>{desc}<|object_ref_end|><|box_start|><|x1|><|y1|><|x2|><|y2|><|box_end|>
```

Use the exact coord-token string convention already used by CoordExp templates.
If existing helper functions are available, import them rather than duplicating
serialization. If not, implement one local helper and test it.

- [ ] **Step 3: Implement image-local desc candidate universe**

Function:

```python
image_local_desc_groups(objects: Sequence[Mapping[str, Any]]) -> list[str]
```

Return canonical descs sorted by stable canonical text. Do not include COCO80
categories absent from the image.

- [ ] **Step 4: Verify Task 3**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_prefix_rendering.py \
  -q
```

Expected: all tests pass.

## Task 4: Boundary Full-Desc Span Scoring

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/boundary_scoring.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_boundary_scoring.py`

- [ ] **Step 1: Write pure scoring reduction tests**

Create tests for:

```python
length_normalized_logprob([-1.0, -3.0]) == -2.0
rank_desc_scores({"person": -1.0, "chair": -2.0})[0]["desc"] == "person"
```

Test boundary alignment classification:

```python
summary = summarize_boundary_alignment(
    desc_scores=[
        {"desc": "person", "role": "residual", "score": -1.0},
        {"desc": "chair", "role": "emitted", "score": -3.0},
    ],
    eos_score=-2.5,
)
assert summary["boundary_alignment"] == "residual_favored"
```

- [ ] **Step 2: Implement pure scoring utilities**

Implement:

```python
def length_normalized_logprob(token_logprobs: Sequence[float]) -> float
def summarize_boundary_alignment(desc_scores: Sequence[Mapping[str, Any]], eos_score: float) -> dict[str, Any]
```

Boundary alignment labels:

```text
residual_favored
emitted_favored
eos_favored
mixed_or_tied
no_residual_candidate
```

- [ ] **Step 3: Implement GPU runtime adapter**

Implement:

```python
def score_boundary_desc_spans(
    *,
    model_handle: Mapping[str, Any],
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    boundary_assistant_text: str,
    candidate_descs: Sequence[str],
    eos_token_ids: Sequence[int],
) -> dict[str, Any]
```

Runtime behavior:

- Build one chat prompt per candidate desc path:
  `<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>`.
- Teacher-force only the candidate suffix tokens.
- Return length-normalized suffix logprob per desc.
- Compute EOS/stop logprob from the exact same boundary prompt.
- Do not dump attention.

Use mocked model tests for shape/JSON safety if full Qwen model is unavailable
inside unit tests.

- [ ] **Step 4: Verify Task 4**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_boundary_scoring.py \
  -q
```

Expected: all tests pass.

## Task 5: Forced-Desc Pre-X1 Readout And Peak Attribution

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/x1_readout.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_x1_readout.py`

- [ ] **Step 1: Write x1 partition tests**

Test partitions:

```python
peaks = [{"x1_bin": 100, "mass": 0.1}, {"x1_bin": 250, "mass": 0.05}, {"x1_bin": 999, "mass": 0.03}]
emitted = [{"gt_idx": 0, "x1": 100}]
residual = [{"gt_idx": 1, "x1": 252}]
assigned = partition_x1_peaks(peaks, emitted, residual, radius=24)
assert assigned[0]["partition"] == "emitted_same_desc_x1_peak"
assert assigned[1]["partition"] == "residual_same_desc_x1_peak"
assert assigned[2]["partition"] == "boundary_artifact_x1_peak"
```

- [ ] **Step 2: Reuse Phase A2 peak extraction**

Import `extract_x1_peaks` from:

```python
src.analysis.candidate_field_cardinality_tomography.x1_candidate_field
```

Do not fork peak policy unless tests reveal incompatible row shapes. Keep:

```text
absolute_mass_floor = 0.002
relative_floor = 0.10
primary_merge_radius = 24
gt_x1_neighborhood_radius = 24
raw_topk_k = 32
```

- [ ] **Step 3: Implement forced-desc x1 runtime**

Implement:

```python
def probe_forced_desc_pre_x1(
    *,
    model_handle: Mapping[str, Any],
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    forced_desc_assistant_text: str,
    emitted_same_desc: Sequence[Mapping[str, Any]],
    residual_same_desc: Sequence[Mapping[str, Any]],
    peak_config: PeakConfig,
) -> dict[str, Any]
```

Return:

```text
coord_vocab_mass
top_bins
merged_peaks
partitioned_peaks
forced_x1_residual_coverage
emitted_attraction_rate
boundary_artifact_peak_count
unmatched_x1_peak_count
```

- [ ] **Step 4: Verify Task 5**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_x1_readout.py \
  -q
```

Expected: all tests pass.

## Task 6: Paired GPU Probe Orchestration

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/paired_probe.py`
- Modify: `src/analysis/prefix_state_transition_tomography/runner.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_runner_cli.py`

- [ ] **Step 1: Write mocked paired-probe tests**

Test that for one sampled prefix state and two checkpoint roles, the probe
emits:

```text
boundary_score_rows.jsonl: 2 checkpoint roles x >=2 descs
forced_x1_rows.jsonl: 2 checkpoint roles x >=2 descs
shard_summary.json
```

Use a fake runtime object that returns deterministic boundary and x1 rows.

- [ ] **Step 2: Implement shard selection**

Function:

```python
def rows_for_shard(rows: Sequence[Mapping[str, Any]], shard_id: int, num_shards: int) -> list[Mapping[str, Any]]:
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_id]
```

Each shard processes the same prefix-state rows for both checkpoints to
preserve paired comparability.

- [ ] **Step 3: Implement model-handle cache**

Load one checkpoint at a time per shard to control memory. Recommended loop:

```python
for checkpoint_role in ("et_rmp_ce", "pure_ce"):
    model_handle = load_checkpoint(checkpoint_config)
    process all shard rows for checkpoint_role
    release model_handle
    torch.cuda.empty_cache()
```

If memory allows both checkpoints at once, do not add that optimization in this
first implementation.

- [ ] **Step 4: Wire runner stage `paired_checkpoint_probe`**

Preconditions:

- `prefix_state_index_summary.json` exists;
- `launch_eligible` is `true`;
- `prefix_state_sampled_rows.jsonl` exists.

If `launch_eligible=false`, return JSON:

```json
{
  "stage": "paired_checkpoint_probe",
  "status": "blocked",
  "reason": "prefix_state_index_not_launch_eligible"
}
```

Do not perform model forward in blocked state.

- [ ] **Step 5: Verify Task 6 with mocked tests**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_runner_cli.py \
  -q
```

Expected: mocked paired probe test passes.

## Task 7: Merge, Metrics, Quadrants, And Report

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/merge_report.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_merge_report.py`

- [ ] **Step 1: Write paired merge tests**

Create fake ET and pure rows sharing the paired key. Test:

- unpaired rows are excluded from headline paired rows;
- quadrant assignment works;
- same-desc and different-desc metrics are separate;
- train and val metrics are separate.

Quadrant expected logic:

```python
assert assign_quadrant(boundary_good=True, x1_good=True) == "boundary_good_x1_good"
assert assign_quadrant(boundary_good=True, x1_good=False) == "boundary_good_x1_bad"
assert assign_quadrant(boundary_good=False, x1_good=True) == "boundary_bad_x1_good"
assert assign_quadrant(boundary_good=False, x1_good=False) == "boundary_bad_x1_bad"
```

- [ ] **Step 2: Implement merge**

Merge all shard files into:

```text
boundary_score_rows.jsonl
forced_x1_rows.jsonl
paired_state_rows.jsonl
quadrant_rows.jsonl
```

Validate that each headline paired key has:

```text
2 checkpoint roles
boundary_full_desc_span row
forced_desc_pre_x1 row
```

- [ ] **Step 3: Implement summary metrics**

`summary.json` must include:

```text
row_counts
paired_row_counts
by_split
by_transition_type
by_prefix_depth
paired_delta_metrics
quadrant_counts
category_sorted_hypothesis_evidence
unpaired_sidecar_counts
```

Primary metrics:

```text
boundary_alignment
forced_x1_residual_coverage
emitted_attraction_rate
state_transition_delta
eos_stop_tendency
unmatched_x1_peak_rate
boundary_artifact_x1_peak_rate
```

- [ ] **Step 4: Implement Markdown report**

`report.md` sections:

```text
# Prefix-State Transition Tomography Phase A3.1
## Scope
## Checkpoints
## Index And Launch Gates
## Paired Coverage
## Same-Desc Transition
## Different-Desc Transition
## Four-Quadrant Mechanism Table
## Category-Sorted Hypothesis Evidence
## Train/Val Split
## ET-RMP-CE vs Pure-CE Paired Deltas
## Evidence Boundaries
## Residual Risks
```

Do not claim training recommendations. Phrase category sorting as
`Phase B consideration` only when evidence supports it.

- [ ] **Step 5: Verify Task 7**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_merge_report.py \
  -q
```

Expected: all tests pass.

## Task 8: Sampled Manual Review Gallery

**Files:**

- Create: `src/analysis/prefix_state_transition_tomography/gallery.py`
- Create: `tests/analysis/prefix_state_transition_tomography/test_gallery.py`

- [ ] **Step 1: Write gallery sampling tests**

Given fake quadrant rows, gallery sampler must prioritize:

```text
boundary_bad_x1_good
boundary_good_x1_bad
paired ET-vs-pure quadrant disagreements
high emitted_attraction_rate same-desc cases
surprising class_block_done cases
```

Expected output fields:

```text
gallery_case_id
prefix_state_id
transition_type
quadrant_et_rmp_ce
quadrant_pure_ce
image_path
render_path
manual_label
manual_notes
```

- [ ] **Step 2: Implement gallery rendering**

Render each sampled case with:

- teacher-prefix emitted GT boxes;
- residual GT boxes;
- forced-desc x1 vertical lines colored by partition;
- boundary desc score table for image-local descs;
- EOS/stop score;
- checkpoint labels.

Avoid legend overlap by placing metadata in a right-side panel or below the
image, not over the top of the image.

- [ ] **Step 3: Verify Task 8**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography/test_gallery.py \
  -q
```

Expected: all tests pass.

## Task 9: Launcher, Status, And End-To-End Smoke

**Files:**

- Create: `scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh`
- Create: `scripts/analysis/prefix_state_transition_tomography/status.py`
- Modify: `src/analysis/prefix_state_transition_tomography/runner.py`

- [ ] **Step 1: Implement status script**

Status output must show:

```text
artifact_root
launch_eligible
failed_launch_gates
sampled_prefix_state_count
shard summaries present/missing
merged outputs present/missing
report path
gallery path
```

- [ ] **Step 2: Implement linked tmux launcher**

Launcher environment variables:

```text
CONFIG
SESSION
ALLOW_OVERWRITE=0|1
DRY_RUN=0|1
NUM_SHARDS=8
```

Launcher sequence:

```text
prefix_state_index
validate
if launch_eligible != true: stop
paired_checkpoint_probe shards 0..7
merge
report
gallery
validate
```

Each GPU shard command must set one GPU:

```bash
CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/run.py \
  --config "$CONFIG" \
  --stages paired_checkpoint_probe \
  --shard-id "$shard_id"
```

- [ ] **Step 3: Run compile checks**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m py_compile \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/src/analysis/prefix_state_transition_tomography/*.py \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/*.py
```

Expected: no output and exit code 0.

- [ ] **Step 4: Run full unit test slice**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Dry-run linked launcher**

Run:

```bash
DRY_RUN=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
SESSION=prefix_state_transition_a3_4096_dryrun \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh
```

Expected: printed plan with index, gate, 8 shard commands, merge, report,
gallery, validate. No model forward.

- [ ] **Step 6: Run index-only real stage**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/run.py \
  --config /data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
  --stages prefix_state_index,validate \
  --allow-overwrite
```

Then inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/data/CoordExp/outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096')
summary = json.loads((root / 'prefix_state_index_summary.json').read_text())
print(json.dumps({
    'launch_eligible': summary.get('launch_eligible'),
    'failed_launch_gates': summary.get('failed_launch_gates'),
    'row_counts': summary.get('row_counts'),
}, indent=2, ensure_ascii=False))
PY
```

Expected: `launch_eligible` is present. If false, GPU stage does not run.

- [ ] **Step 7: Start real tmux run only if launch eligible**

Run:

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
SESSION=prefix_state_transition_a3_4096_ckpt3664 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh
```

Monitor:

```bash
tmux capture-pane -pt prefix_state_transition_a3_4096_ckpt3664:0 -S -200

PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096
```

Expected: 8 analysis shards use GPUs 0-7 as shard capacity, not production
training.

## Task 10: Progress Note And Final Verification

**Files:**

- Create or update:
  `progress/diagnostics/2026-06-04_prefix_state_transition_tomography_phase_a3.md`
- Modify:
  `progress/diagnostics/README.md`
  if this repo's diagnostics README indexes new notes explicitly.

- [ ] **Step 1: Create progress note after index-only stage**

Record:

```text
checkpoint paths
artifact root
config path
evidence scope = phase_a3_1_index_only or phase_a3_1_4096_paired
prefix_state_index row counts
launch_eligible
failed_launch_gates
sampling_adjustments
```

- [ ] **Step 2: Update progress note after GPU probe completes**

Record:

```text
boundary_score_rows count
forced_x1_rows count
paired_state_rows count
quadrant rows count
same-desc and different-desc headline metrics
train/val split
ET-vs-pure paired deltas
report path
gallery path
residual risks
```

- [ ] **Step 3: Final verification commands**

Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography \
  -q

PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m py_compile \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/src/analysis/prefix_state_transition_tomography/*.py \
  /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/*.py
```

Expected: tests pass and py_compile exits 0.

- [ ] **Step 4: Report residual risks**

Final response must include:

```text
changed files
verification commands run
artifact root
tmux session name if launched
whether full 4096 run is complete or still running
whether launch gates passed
known residual risks
```

## Suggested Subagent Split For Execution

Use up to six active subagents:

1. `index-contract-agent`: Task 2 prefix-state generation and launch gates.
2. `prompt-boundary-agent`: Task 3 and Task 4 prompt rendering and boundary scoring.
3. `x1-readout-agent`: Task 5 forced-desc x1 posterior and peak partitioning.
4. `probe-runtime-agent`: Task 6 paired checkpoint GPU runtime and sharding.
5. `report-gallery-agent`: Task 7 and Task 8 merge/report/gallery.
6. `launcher-audit-agent`: Task 9 and Task 10 launcher/status/progress and final verification.

Before GPU launch, run an audit pass over:

```text
prefix_state_index_summary.json
prefix_state_sampled_rows.jsonl
ckpt3664_et_vs_purece_phase_a3_4096.yaml
launch_prefix_state_transition_tomography_tmux.sh
```

The audit should check paired-key stability, no attention dump in first probe,
no production training command, no checkpoint mix-up, and no silent geometry
resizing.

## Self-Review Checklist

- Spec coverage:
  - teacher GT prefix first source of truth: Task 2 and Task 3.
  - self-rollout deferred: no task implements self-rollout.
  - dual readout: Task 4 and Task 5.
  - same-desc and different-desc peer experiments: Task 2 and Task 7.
  - fixed-depth grid: Task 2.
  - paired checkpoint rows: Task 6 and Task 7.
  - full-desc span scoring: Task 4.
  - image-local desc universe: Task 3 and Task 4.
  - x1-only residual/emitted partition: Task 5.
  - EOS as boundary competitor only: Task 4 and Task 7.
  - category-sorted sidecar evidence: Task 7.
  - train+val split reports: Task 2 and Task 7.
  - prefix-state sampling unit: Task 2.
  - automatic launch gates: Task 2 and Task 9.
  - 4096 paired scale: Task 1 config and Task 9 launcher.
  - sampled gallery: Task 8.
  - independent project namespace: Task 1.
  - no full attention in first probe: Task 6 and Task 9 audit.

- Placeholder scan:
  - No placeholder markers or unspecified later behavior should remain in this
    plan.

- Type consistency:
  - Use `prefix_state_id`, `transition_type`, `prefix_depth`,
    `prefix_order_policy_id`, `probe_desc`, `readout_type`, and
    `checkpoint_role` consistently across index, probe, merge, and gallery.
