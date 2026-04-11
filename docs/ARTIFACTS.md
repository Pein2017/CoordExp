---
doc_id: docs.artifacts
layer: docs
doc_type: artifacts-reference
status: canonical
domain: repo
summary: Runtime artifacts, logging controls, and provenance surfaces.
updated: 2026-03-29
---

# Artifacts & Provenance

This page documents the observable runtime artifacts CoordExp writes during
training, inference, post-processing, and evaluation.

Artifact names remain stable even though ownership moved into narrower helper
modules such as `src/bootstrap/`, `src/infer/artifacts.py`, and
`src/eval/artifacts.py`.

If you are looking for metric-key meaning, start here:
- `docs/training/METRICS.md`

If you are looking for the end-to-end system flow rather than artifact behavior,
start with:
- `docs/SYSTEM_OVERVIEW.md`

---

## Inference, Confidence, And Evaluation Artifacts

During inference and offline evaluation, CoordExp writes reproducibility and
analysis artifacts into the resolved run directory and its eval subdirectory.

- `gt_vs_pred.jsonl`
  - Base inference artifact with inline GT and parsed predictions per sample.
- `pred_token_trace.jsonl`
  - Optional per-sample generation trace artifact for later rollout inspection.
- `pred_confidence.jsonl`
  - Confidence post-op intermediate keyed to the base inference artifact.
- `gt_vs_pred_scored.jsonl`
  - Score-provenanced artifact consumed by COCO evaluation and official
    submission export.
- `confidence_postop_summary.json`
  - Post-op summary describing score materialization and drop counts.
- `vis_resources/gt_vs_pred.jsonl`
  - Derived canonical visualization sidecar used by the shared GT-vs-Pred
    reviewer and evaluator overlay path.
- `summary.json`
  - Inference-stage summary emitted by the YAML infer pipeline.
  - Check `infer.prompt_variant`, `infer.object_field_order`, and
    `infer.object_ordering` when reproducing prompt-sensitive evaluations.
- `resolved_config.json`
  - Canonical snapshot of the resolved infer pipeline config.
  - Check `infer.prompt_variant`, `infer.object_field_order`, and
    `infer.object_ordering` before launching long evaluation jobs.
- `resolved_config.path`
  - Pointer sidecar written next to `gt_vs_pred.jsonl` so downstream eval or
    visualization jobs can recover the authoritative `resolved_config.json`
    even when they start from an artifact path outside the original `run_dir`.
- `metrics.json`
  - Offline evaluator metrics and diagnostic counters.
- `per_image.json`
  - Per-image evaluator diagnostics for the standard single-artifact evaluation
    flow.
- `per_class.csv`
  - Per-class COCO export summary when classed evaluation is enabled.
- `coco_gt.json`
  - Deterministic COCO-format GT projection used by offline evaluation.
- `coco_preds.json`
  - Deterministic COCO-format prediction export, including score-aware ranking
    when the scored artifact is used.
- `matches.jsonl`
  - F1-ish primary-threshold match diagnostics.
- `matches@<thr>.jsonl`
  - Additional F1-ish match diagnostics when multiple IoU thresholds are
    requested.

These standard artifacts remain unchanged when Oracle-K is enabled elsewhere;
Oracle-K is an additive workflow rather than a replacement for the current
evaluator.

Current helper ownership for these artifacts:

- infer summary / resolved metadata:
  - `src/infer/artifacts.py`
- backend generation:
  - `src/infer/backends.py`
- confidence post-op scoring:
  - `src/eval/confidence_postop.py`
  - `src/eval/bbox_confidence.py`
- evaluation save/report materialization:
  - `src/eval/orchestration.py`
  - `src/eval/artifacts.py`

---

## Official Submission Export Artifacts

When the COCO test-dev submission workflow is used, the export step additionally
writes:

- `coco_submission.json`
  - official server-upload payload projected back to original COCO test-dev
    resolution
- `submission_summary.json`
  - export summary and provenance for the submission payload

---

## Oracle-K Analysis Artifacts

Oracle-K writes a dedicated analysis directory under its configured `out_dir`.
The v1 workflow focuses on object-level recoverability under repeated stochastic
sampling.

- `summary.json`
  - Aggregate Oracle-K report with baseline vs Oracle-K recall-style counts at
    each configured IoU threshold.
  - Includes `oracle_run_count` plus recoverable and systematic false-negative
    counts at the primary threshold.
- `per_image.json`
  - Per-image baseline false-negative totals with recoverable and systematic
    breakdowns for:
    - location-only
    - semantic+location
- `fn_objects.jsonl`
  - One row per baseline false-negative GT object, keyed by `record_idx` and
    `gt_idx`.
  - Includes per-run object-level pairing, `ever recovered`, `recover_count`,
    and `recover_fraction`.
- `materialized/<label>/` when Oracle-K is asked to generate runs
  - Persisted labeled inference artifacts for the baseline or Oracle runs
    before aggregation begins.

Oracle-K v1 may preserve run-level provenance such as `pred_token_trace.jsonl`
and `resolved_config.json` paths when available.
It does not require exact token-span-to-object alignment; object-level pairing
is the v1 contract boundary.

---

## Training Artifacts (Rank 0)

During training (`python -m src.sft ...`), rank 0 writes reproducibility
artifacts into `training.output_dir` before training starts:

- `resolved_config.json`
  - Canonical, serialized snapshot of the resolved training config.
  - Includes `schema_version` (current `1`), `config_path`,
    `base_config_path`, and `dataset_seed`.
- `runtime_env.json`
  - Runtime environment metadata snapshot (selected env vars + platform info).
- `effective_runtime.json`
  - Executed runtime payload after bootstrap / launcher mutation.
  - Use this instead of only `resolved_config.json` when debugging the true
    launched topology or runtime knobs.
- `pipeline_manifest.json`
  - First-class pipeline identity / manifest artifact assembled from
    `src/bootstrap/pipeline_manifest.py`.
- `train_data_provenance.json`
  - Stable train-split source identity and optional digests for supported
    local inputs.
- `eval_data_provenance.json`
  - Stable eval-split source identity and optional digests when eval data is
    configured.
- `run_metadata.json`
  - Best-effort provenance and run identity:
    - `git_sha`, `git_branch`, `git_dirty`, `git_status_porcelain` (truncated)
    - `created_at` (UTC ISO8601)
    - `upstream` dependency provenance (ms-swift / transformers, etc.)
    - launcher metadata (Stage-2 server/learner topology), when present
  - emitted via `src/bootstrap/run_metadata.py`
- `config_source.yaml` / `base_config_source.yaml`
  - Best-effort copies of the YAML sources used to build the run.
- `monitor_dumps/` when either
  `rollout_matching.train_monitor_dump.enabled: true` or
  `rollout_matching.eval_monitor_dump.enabled: true`
  - Qualitative rollout diagnostics written as `.json` and optional `.md`.
  - `eval_step` uses the configured eval-window cadence (`every_evals`).
  - `stage2_two_channel` Channel-B `train_step` writes only suspicious
    duplicate-heavy rollouts for the current optimizer step.
    `train_monitor_dump.every_channel_b_steps` counts realized Channel-B
    rollout steps when set; otherwise the trainer falls back to `every_steps`.
  - Channel-B `prepare_failures/` dumps preserve both token IDs and decoded
    rollout/prefix text so malformed JSON failure modes can be inspected without
    manual retokenization.
  - These remain raw telemetry artifacts; shared GT-vs-Pred review rendering
    uses an explicit normalized `vis_resources/gt_vs_pred.jsonl` sidecar
    instead of taking ownership of the monitor-dump path layout.
- `eval_detection/step_<global_step>/` when Stage-1 `custom.eval_detection.enabled: true`
  - Generation-backed eval-step artifacts for standard Stage-1 SFT runs.
  - Writes `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `infer_summary.json`,
    `metrics.json`, `per_image.json`, and standard evaluator sidecars for that
    eval window.
- `eval_detection/step_<global_step>/` during Stage-2 rollout-aware eval
  - Stage-2 writes this directory when
    `rollout_matching.eval_detection.materialize_artifacts: true`
    (default).
  - The intent is parity with the offline infer/eval pipeline so each eval
    window can be inspected with the same artifact readers used for standalone
    inference.
  - Writes:
    - `gt_vs_pred.jsonl`
    - `gt_vs_pred_scored.jsonl`
    - `infer_summary.json`
    - `metrics.json`
    - `per_image.json`
    - standard evaluator sidecars such as `coco_gt.json`, `coco_preds.json`,
      `per_class.csv`, `vis_resources/gt_vs_pred.jsonl`, and `matches*.jsonl`
      when requested by the evaluator mode
    - `raw_rollouts.jsonl` with per-sample rollout text, token IDs, scoring
      metadata, parsing diagnostics, and match details
    - `pred_token_trace.jsonl` when traced rollout outputs are available for the
      eval window (for example confidence-postop-backed scoring)

Notes:

- If `training.add_version: true` (default in `configs/base.yaml`), ms-swift
  scopes outputs under a versioned run directory.
- If `training.logging_dir` is explicitly set together with `training.run_name`,
  CoordExp writes TensorBoard event files under `<logging_dir>/<run_name>/` so
  `tensorboard --logdir <logging_dir>` shows the authored run name instead of a
  bare `.` root entry.
- If `training.output_dir` is not set, training fails fast because these
  artifacts are required for reproducibility.
- `training.checkpoint_mode: restartable` is the explicit opt-in mode for
  restart fidelity. It requires optimizer, scheduler, RNG, trainer state, and
  repo-owned runtime-sidecar artifacts in each checkpoint; `artifact_only`
  remains the compatibility-preserving default.

---

## Logging Controls

### All-Rank Logging

`src.sft` defaults to rank-0-only logging under distributed launchers.

- Use `--verbose` to enable logging from all ranks when diagnosing deadlocks or
  per-rank divergence.

### Mirror Logs Into `output_dir` (Optional)

You can optionally mirror logs into `training.output_dir` via
`custom.extra.log_file` (rank 0 only):

```yaml
custom:
  extra:
    log_file:
      enabled: true
      filename: train.log
      capture_stdout: false
      capture_stderr: false
```

This is intended for quick one-folder remote debugging workflows.

---

## Callbacks (Repo-Specific)

CoordExp uses a small set of in-repo callbacks under `src/callbacks/` for
reproducibility and monitoring.
They are wired by the training entrypoint (`src/sft.py`) and specific trainer
variants.

Common ones you may see in logs or artifacts:

- `SaveDelayCallback`
  - checkpoint throttling and save-delay behavior
- `TrainHeartbeatCallback`
  - lightweight heartbeat for long runs
- `DetectionEvalCallback`
  - offline detection evaluation helper; logs `eval_det_*` keys

Stage-2 trainers also emit rollout-specific metrics directly
(see `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS.md`).

- `stage2_two_channel` includes clean-prefix Channel-B duplicate-collapse
  diagnostics under:
  - `dup/*`
  - `stage2_ab/channel_b/dup/N_*`
  - `train/optimization/loss_duplicate_burst_unlikelihood`
  - `stage2_ab/channel_b/closure_supervision/N_drop` for the
    legacy-named closure-resolution fallback activation counter

---

## Related Docs

- Data contract: `docs/data/CONTRACT.md`
- Packing guide: `docs/data/PACKING.md`
- Stage-2 runbook: `docs/training/STAGE2_RUNBOOK.md`
- Metric keys: `docs/training/METRICS.md`
- Offline evaluator: `docs/eval/README.md`
