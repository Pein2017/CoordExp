---
title: Prefix-State Transition Tomography Phase A3.1 Launch
date: 2026-06-04
status: completed-diagnostic
owner: codex
depends_on:
  - docs/superpowers/specs/2026-06-03-candidate-field-cardinality-tomography-design.md
  - docs/superpowers/plans/2026-06-04-prefix-state-transition-tomography-implementation.md
  - configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml
  - outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096/prefix_state_index_summary.json
  - outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096/prefix_state_sampled_rows.jsonl
---

# Prefix-State Transition Tomography Phase A3.1 Launch

This note records implementation and launch evidence for Phase A3.1.  It is an
experiment execution record, not a mechanism conclusion.

## Scope

Experiment namespace:

`prefix_state_transition_tomography`

Artifact root:

`outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096`

Compared checkpoints:

- `et_rmp_ce`: `outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`
- `pure_ce`: `outputs/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664`

Dataset roots:

- `train_jsonl`: `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `val_jsonl`: `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- `image_root`: `public_data/coco/rescale_32_1024_bbox_max60`

## Implementation Guards

The GPU readout stage records paired checkpoint rows for:

- `boundary_full_desc_span`: boundary-level image-local desc candidate scoring plus EOS score.
- `forced_desc_pre_x1`: forced desc prefix ending at `<|box_start|>` with x1 posterior partitioning.

Runtime guards added before launch:

- strict JSONL persistence with atomic `.inprogress` shard writes;
- `prefix_state_index` CPU rows use sentinel `checkpoint_role=paired_index`;
- `validate` checks all sampled image paths before loading checkpoints;
- rendered chat template must preserve assistant continuation tail;
- forced pre-x1 context must end at `<|box_start|>`;
- boundary suffix scoring asserts prefix/full token alignment before scoring suffix tokens;
- launcher blocks GPU shards unless `prefix_state_index_summary.json` has `launch_eligible=true`;
- launcher defaults to `REUSE_READY_INDEX=1`, so a validated existing index is reused on restart.

## Gate Evidence

CPU gate command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/run.py \
  --config /data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
  --stages prefix_state_index,validate \
  --allow-overwrite
```

Observed gate result:

- `prefix_state_index_rows`: `1332051`
- `prefix_state_sampled_rows`: `4096`
- `launch_eligible`: `true`
- `failed_launch_gates`: `[]`
- image validation `checked_rows`: `4096`
- image validation `missing_rows`: `0`

## Verification

Unit and compile checks run before launch:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python - <<'PY'
import pytest, sys
sys.exit(pytest.main([
    '/data/CoordExp/.worktrees/fn-rescue-attention-probes/tests/analysis/prefix_state_transition_tomography',
    '-q',
]))
PY
```

Result:

`53 passed`

Compile check:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m py_compile $(find src/analysis/prefix_state_transition_tomography scripts/analysis/prefix_state_transition_tomography -name '*.py' -print)
```

Result:

`passed`

Launcher dry-run:

```bash
DRY_RUN=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
SESSION=prefix_state_transition_ckpt3664_phase_a3_4096 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh
```

Result:

`passed`

## Tmux Launch

Session:

`prefix_state_transition_ckpt3664_phase_a3_4096`

Launch command:

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/prefix_state_transition_tomography/ckpt3664_et_vs_purece_phase_a3_4096.yaml \
SESSION=prefix_state_transition_ckpt3664_phase_a3_4096 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/launch_prefix_state_transition_tomography_tmux.sh
```

Status command:

```bash
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/prefix_state_transition_tomography/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/prefix_state_transition_tomography/et_rmp_ce_vs_purece_ckpt3664_phase_a3_4096 \
  --log-root /data/CoordExp/.worktrees/fn-rescue-attention-probes/logs/prefix_state_transition_ckpt3664_phase_a3_4096
```

Launch status after restart:

- `stage_status`: `paired_probe_shards_running`
- `alive_shard_processes`: `8`
- `expected_shards`: `8`
- shard logs: `logs/prefix_state_transition_ckpt3664_phase_a3_4096/prefix_state_paired_probe_shard_*.log`

## Known Execution Notes

The first GPU launch exposed a guard bug: prefix ids were on CPU and full ids
were on CUDA during `torch.equal`.  The guard now moves both slices to CPU
before exact comparison.  The failed first launch was killed and restarted.

The active run is a 4096 paired analysis run, not production training.  It uses
the eight GPUs as independent shard workers.

## Completion Update

The tmux run completed with all eight paired-probe shards exiting `0`.

Final status:

- `stage_status`: `final_artifacts_present`
- `final_ready`: `true`
- `shards_complete`: `true`
- `alive_shard_processes`: `0`
- `failed_launch_gates`: `[]`

The analysis summary is recorded in:

`progress/diagnostics/2026-06-04_prefix_state_transition_phase_a3_1_analysis.md`
