# Architecture (Logging & Reproducibility)

This page documents the *observable* runtime artifacts CoordExp writes during training/inference,
and the knobs that control logging behavior. It is intentionally contract-focused and paper-ready.

If you are looking for metric-key meaning, start here:
- `docs/training/METRICS_LOSSES.md`

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

Notes:
- If `training.add_version: true` (default in `configs/base.yaml`), ms-swift scopes outputs under a versioned run directory.
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

Stage-2 trainers also emit rollout-specific metrics directly (see `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS_LOSSES.md`).

---

## Related Docs

- Data contract: `docs/data/JSONL_CONTRACT.md`
- Packing guide: `docs/data/PACKING.md`
- Stage-2 runbook: `docs/training/STAGE2_RUNBOOK.md`
- Metric keys: `docs/training/METRICS_LOSSES.md`
- Offline evaluator: `docs/eval/README.md`
