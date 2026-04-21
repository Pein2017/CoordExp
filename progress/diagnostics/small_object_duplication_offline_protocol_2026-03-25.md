# Small Object Duplication Offline Protocol (2026-03-25)

## Goal

Provide a fixed-checkpoint offline harness for diagnosing `small object duplication` with:

- monitor-dump cohort mining,
- image-only decode sweeps,
- prefix-conditioned continuation probes,
- and teacher-forced candidate scoring.

The current target checkpoint is:

- `output/stage2_ab/prod/pseudo_positive-ckpt_300_merged-v1`

The initial monitor-dump source is:

- `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.85-epoch_1-global_struct_ce_fixed-stronger_FN-from_300_v1/v0-20260325-025922/monitor_dumps`

## Entry Points

- Study module: [small_object_duplication_study.py](/data/CoordExp/src/analysis/small_object_duplication_study.py)
- Default config: [default.yaml](/data/CoordExp/configs/analysis/small_object_duplication/default.yaml)
- Cohort-only smoke: [smoke.yaml](/data/CoordExp/configs/analysis/small_object_duplication/smoke.yaml)
- Model-backed smoke: [smoke_model.yaml](/data/CoordExp/configs/analysis/small_object_duplication/smoke_model.yaml)

## What The Study Produces

The study writes to `output/analysis/<run_name>/` and currently materializes:

- `cohort/all_samples.jsonl`
- `cohort/duplication_cases.jsonl`
- `cohort/crowded_controls.jsonl`
- `decode/results.jsonl`
- `prefix/results.jsonl`
- `score/results.jsonl`
- per-stage `summary.json`
- top-level `summary.json`

## Core Measurements

### Cohort mining

Per monitor-dump sample, the harness records:

- existing duplication metrics from the dump,
- local duplicate-like graph metrics over predicted objects,
- small-object-specific duplicate-like pair counts,
- focus-match selection for `earliest_matched_small_or_first_matched`.

### Decode

Image-only generation under temperature sweeps on the fixed checkpoint.

### Prefix

Continuation from:

- the first matched small predicted object, if one exists,
- otherwise the first matched predicted object,
- plus controlled bbox jitters.

### Score

Teacher-forced scoring under the same image and prefix context for candidate families:

- `remaining_gt`
- `duplicate_jitter`
- `close`

The main summary margins are:

- `best_remaining_gt_full - best_duplicate_full`
- `close_full - best_duplicate_full`

## Commands

Run the cohort-only smoke:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms python -m src.analysis.small_object_duplication_study \
  --config configs/analysis/small_object_duplication/smoke.yaml
```

Run the model-backed smoke:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms python -m src.analysis.small_object_duplication_study \
  --config configs/analysis/small_object_duplication/smoke_model.yaml
```

Run the full default study:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms python -m src.analysis.small_object_duplication_study \
  --config configs/analysis/small_object_duplication/default.yaml
```

## Validation Status

The following checks have been completed:

- unit tests for the new helper logic,
- regression test on the existing rollout FN-factor study test file,
- cohort-only smoke config,
- model-backed smoke config on `cuda:0`.

The current model-backed smoke output lives at:

- `/data/CoordExp/output/analysis/small-object-duplication-smoke-model`
