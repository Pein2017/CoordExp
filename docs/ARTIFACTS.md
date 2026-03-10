---
doc_id: docs.artifacts
layer: docs
doc_type: artifacts-reference
status: canonical
domain: repo
summary: Runtime artifacts, logging controls, and provenance surfaces.
updated: 2026-03-09
---

# Artifacts & Provenance

This page documents the *observable* runtime artifacts CoordExp writes during training/inference,
and the knobs that control logging behavior. It is intentionally contract-focused and paper-ready.

If you are looking for metric-key meaning, start here:
- `docs/training/METRICS.md`

If you are looking for the end-to-end system flow rather than artifact behavior, start with:
- `docs/SYSTEM_OVERVIEW.md`

---

## Training Artifacts (Rank 0)

During training (`python -m src.sft ...`), rank 0 writes reproducibility artifacts into
`training.output_dir` before training starts:

- `resolved_config.json`
  - Canonical, serialized snapshot of the resolved training config.
  - Includes `schema_version` (current `1`), `config_path`, `base_config_path`, and `dataset_seed`.
- `runtime_env.json`
  - Runtime environment metadata snapshot (selected env vars + platform info).
- `run_metadata.json`
  - Best-effort provenance and run identity:
    - `git_sha`, `git_branch`, `git_dirty`, `git_status_porcelain` (truncated)
    - `created_at` (UTC ISO8601)
    - `upstream` dependency provenance (ms-swift / transformers, etc.)
    - launcher metadata (Stage-2 server/learner topology), when present
- `config_source.yaml` / `base_config_source.yaml`
  - Best-effort copies of the YAML sources used to build the run (optional; failures do not abort training).
- `monitor_dumps/` when either `rollout_matching.train_monitor_dump.enabled: true`
  or `rollout_matching.eval_monitor_dump.enabled: true`
  - Qualitative rollout diagnostics written as `.json` and optional `.md`.
  - `eval_step` uses the configured eval-window cadence (`every_evals`).
  - `stage2_two_channel` Channel-B `train_step` writes only suspicious duplicate-heavy rollouts for the current optimizer step.

Notes:
- If `training.add_version: true` (default in `configs/base.yaml`), ms-swift scopes outputs under a versioned run directory.
- If `training.logging_dir` is explicitly set together with `training.run_name`, CoordExp writes TensorBoard
  event files under `<logging_dir>/<run_name>/` so `tensorboard --logdir <logging_dir>` shows the authored run
  name instead of a bare `.` root entry.
- If `training.output_dir` is not set, training fails fast (we treat these artifacts as required for reproducibility).

---

## Logging Controls

### All-rank logging

`src.sft` defaults to rank-0-only logging under distributed launchers.

- Use `--verbose` to enable logging from all ranks (helpful for diagnosing deadlocks / per-rank divergence).

### Mirror logs into output_dir (optional)

You can optionally mirror logs into `training.output_dir` via `custom.extra.log_file` (rank 0 only):

```yaml
custom:
  extra:
    log_file:
      enabled: true
      filename: train.log
      capture_stdout: false
      capture_stderr: false
```

This is intended for quick “open one folder and read everything” workflows on remote machines.

---

## Callbacks (Repo-Specific)

CoordExp uses a small set of in-repo callbacks under `src/callbacks/` for reproducibility and monitoring.
They are wired by the training entrypoint (`src/sft.py`) and/or specific trainer variants.

Common ones you may see in logs/artifacts:
- `SaveDelayCallback`: checkpoint throttling / save-delay behavior (see `training.save_delay_steps`).
- `TrainHeartbeatCallback`: emits a lightweight heartbeat for long runs.
- `DetectionEvalCallback`: offline detection evaluation helper; logs `eval_det_*` keys (see `docs/eval/README.md`).

Stage-2 trainers also emit rollout-specific metrics directly (see `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS.md`).
- `stage2_two_channel` now includes clean-prefix Channel-B duplicate-collapse diagnostics under:
  - `dup/*`
  - `stage2_ab/channel_b/dup/N_*`
  - `loss/B_rollout_text/duplicate_ul`
  - `stage2_ab/channel_b/closure_supervision/N_drop` for the legacy-named closure-resolution fallback activation counter

---

## Related Docs

- Data contract: `docs/data/CONTRACT.md`
- Packing guide: `docs/data/PACKING.md`
- Stage-2 runbook: `docs/training/STAGE2_RUNBOOK.md`
- Metric keys: `docs/training/METRICS.md`
- Offline evaluator: `docs/eval/README.md`
