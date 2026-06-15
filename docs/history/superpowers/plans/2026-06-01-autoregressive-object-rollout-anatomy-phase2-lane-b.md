# Autoregressive Object Rollout Anatomy Phase 2 Lane B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Lane B prefix-boundary causality safe to launch on 8 GPUs by adding shard/range execution, deterministic merge, and a tmux launch handoff.

**Architecture:** Extend the existing prefix-rollin diagnostic without changing its core scoring logic. Each GPU writes an immutable shard directory, then a CPU merge stage writes the canonical `prefix_boundary/per_case.jsonl`, `summary.json`, and `merge_summary.json`.

**Tech Stack:** Python stdlib, existing `src.analysis.prefix_rollin_teacher_forced_diagnostic`, pytest, tmux shell launch.

---

## Files

- Modify: `src/analysis/prefix_rollin_teacher_forced_diagnostic.py`
- Modify: `tests/test_prefix_rollin_teacher_forced_diagnostic.py`
- Create: `scripts/analysis/launch_autoreg_object_rollout_lane_b_tmux.sh`
- Create or update output-only at run time: `/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/prefix_boundary`

## Task 1: Lane B Shard And Merge Contract

- [ ] **Step 1: Add selection helpers and tests**

Add tests in `tests/test_prefix_rollin_teacher_forced_diagnostic.py` for:

```python
assert _record_selected_for_prefix_probe(0, limit=10, shard_index=0, num_shards=2)
assert not _record_selected_for_prefix_probe(1, limit=10, shard_index=0, num_shards=2)
assert _record_selected_for_prefix_probe(9, limit=10, shard_index=1, num_shards=2)
assert not _record_selected_for_prefix_probe(10, limit=10, shard_index=0, num_shards=2)
```

Add tests for invalid shard arguments:

```python
with pytest.raises(ValueError):
    _normalize_prefix_probe_shard(shard_index=2, num_shards=2)
with pytest.raises(ValueError):
    _normalize_prefix_probe_shard(shard_index=0, num_shards=0)
```

- [ ] **Step 2: Implement helpers**

Implement:

```python
def _normalize_prefix_probe_shard(*, shard_index: int | None, num_shards: int | None) -> tuple[int | None, int | None, str | None]: ...
def _record_selected_for_prefix_probe(record_idx: int, *, limit: int, shard_index: int | None, num_shards: int | None) -> bool: ...
def _prefix_probe_shard_label(shard_index: int, num_shards: int) -> str: ...
```

Selection must be `record_idx % num_shards == shard_index`, because every selected image needs its full `K=0..N` curve.

- [ ] **Step 3: Add shard metadata to `run_prefix_rollin_teacher_forced_probe`**

Extend the function signature with optional:

```python
shard_index: int | None = None
num_shards: int | None = None
```

Skip unselected records before opening images. Add these fields to every output row and summary:

```text
shard_index
num_shards
shard_label
source_line_idx
selected_record_count
```

When not sharded, these fields should be `null` except `source_line_idx`, which equals `record_idx`.

- [ ] **Step 4: Add CLI args**

Add:

```text
--shard-index
--num-shards
--merge-shards
--shards-dir
--expected-shards
```

Normal scoring mode must not require merge args. Merge mode must not load a model.

- [ ] **Step 5: Add merge implementation and tests**

Implement:

```python
def merge_prefix_rollin_shards(*, shards_dir: Path, output_dir: Path, expected_shards: int) -> tuple[Path, Path]: ...
```

It must:

- require exactly `expected_shards` `summary.json` files under `shard_XXX-of-YYY`;
- fail if any expected shard is missing;
- fail if unexpected shard labels exist;
- concatenate rows into `output_dir/per_case.jsonl`;
- fail on duplicate `(record_idx, prefix_mode, k, boundary_kind, branch_kind, position, object_instance_id)`;
- write `output_dir/summary.json` with merged `summarize_prefix_rollin_probe_rows(rows)`;
- write `output_dir/merge_summary.json` with shard labels, row counts, selected record count, and source summaries.

Add fixture tests with two shard dirs and duplicate-key failure.

- [ ] **Step 6: Run verification**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_prefix_rollin_teacher_forced_diagnostic.py -q
```

Expected: all tests pass. If the pre-existing boundary test still fails, report `DONE_WITH_CONCERNS` and include the exact failure; do not hide it.

## Task 2: tmux Launcher Handoff Script

- [ ] **Step 1: Create launch script**

Create `scripts/analysis/launch_autoreg_object_rollout_lane_b_tmux.sh`.

It must:

- use `set -euo pipefail`;
- default `SESSION=autoreg_lane_b_ckpt3664`;
- default `NUM_SHARDS=8`;
- default `ROOT=/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200`;
- write logs under `$ROOT/logs/lane_b`;
- launch one process per GPU with `CUDA_VISIBLE_DEVICES=$i`;
- write each shard under `$ROOT/prefix_boundary/shards/shard_$(printf "%03d" "$i")-of-$(printf "%03d" "$NUM_SHARDS")`;
- run merge after all shards complete;
- print attach instructions.

Use the existing checkpoint and artifacts:

```text
CHECKPOINT=/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
CONFIG=configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml
DECODE_ARTIFACT=/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/gt_vs_pred.jsonl
TRACE_ARTIFACT=/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu/pred_token_trace.jsonl
```

The command should use `--prefix-modes gt_prefix,generated_prefix`, `--k-values every`, `--limit 200`, and `--device cuda:0`.

- [ ] **Step 2: Script dry-run safety**

Support `DRY_RUN=1` so the script prints the tmux command file without starting the session.

- [ ] **Step 3: Verify dry-run**

Run:

```bash
DRY_RUN=1 bash scripts/analysis/launch_autoreg_object_rollout_lane_b_tmux.sh
```

Expected: exits 0, prints the generated command file path, and includes 8 `CUDA_VISIBLE_DEVICES=` commands plus one merge command.

## Task 3: Launch Decision

- [ ] **Step 1: Re-run tests**

Run:

```bash
PYTHONPATH=/data/CoordExp python -m pytest tests/test_prefix_rollin_teacher_forced_diagnostic.py tests/test_autoreg_object_rollout.py -q
```

Expected: tests pass, except a clearly reported pre-existing boundary test may remain as `DONE_WITH_CONCERNS`.

- [ ] **Step 2: If tests pass, start tmux**

Run:

```bash
bash scripts/analysis/launch_autoreg_object_rollout_lane_b_tmux.sh
```

Expected: creates tmux session `autoreg_lane_b_ckpt3664` and starts the 8-shard Lane B run.

- [ ] **Step 3: Handoff commands**

Tell the user:

```bash
tmux attach -t autoreg_lane_b_ckpt3664
tmux capture-pane -pt autoreg_lane_b_ckpt3664:0 -S -200
tail -f /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/logs/lane_b/shard_000-of-008.log
```

Do not launch Lane C/D until Lane B merge succeeds or the user explicitly redirects.
