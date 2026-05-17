---
doc_id: docs.artifacts
layer: docs
doc_type: artifacts-reference
status: canonical
domain: repo
summary: Runtime artifacts, logging controls, and provenance surfaces.
updated: 2026-05-16
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

If you are looking for non-code asset ownership and Baidu Netdisk backup rules,
start with:
- `docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md`

---

## Non-Code Asset Ownership

CoordExp separates large non-code assets by whether they are recoverable,
reproducible, or experiment-specific:

- `model_cache/` is local pretrained cache state and is prepared per machine.
- raw `public_data/` is fetched from dataset sources or local mirrors per
  machine.
- processed `public_data/` directories are documented through git-tracked
  provenance manifests under `manifests/public_data_provenance/`.
- `output/` is the Baidu Netdisk sync surface and maps to `/CoordExp/output/`.

This keeps full raw-data hashing and generic large-asset sync out of the
routine backup path.

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
- `gt_vs_pred_guarded.jsonl`
  - Optional offline duplicate-control guarded companion for raw evaluation
    inputs.
- `gt_vs_pred_scored_guarded.jsonl`
  - Optional offline duplicate-control guarded companion for score-aware COCO
    evaluation inputs.
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
- `metrics_guarded.json`
  - Guarded companion metrics emitted when offline duplicate control is
    enabled.
- `per_image.json`
  - Per-image evaluator diagnostics for the standard single-artifact evaluation
    flow.
- `per_image_guarded.json`
  - Guarded companion per-image diagnostics emitted when offline duplicate
    control is enabled.
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
- `matches_guarded.jsonl` / `matches@<thr>_guarded.jsonl`
  - Guarded companions emitted when duplicate control is enabled together with
    match export.
- `duplicate_guard_report.json`
  - Deterministic report describing how many predictions and records changed
    under the offline duplicate-control guard.

Raw plus guarded rule:

- the raw artifact remains authoritative for model-output inspection and
  research debugging
- guarded artifacts are additive post-op views for safety and deployment-style
  evaluation
- score-aware jobs keep using the scored artifact family and therefore emit
  guarded scored companions rather than switching back to raw inputs

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

Training artifact policy is clean-write / tolerant-read:

- New runs write current artifact names and typed observability surfaces.
- Readers may tolerate historical flat metric keys, older diagnostic payloads,
  or migration-only Stage-2 policy metadata when explicitly documented.
- Tolerant reads do not authorize new writers to emit removed training
  mechanisms or legacy policy names.

Resolved config artifacts are the primary bridge between configs, metrics, and
runtime behavior:

- `resolved_config.json` records the exact resolved training config.
- `effective_runtime.json` records the executed runtime after bootstrap,
  launcher mutation, and derived runtime decisions.
- `experiment_manifest.json` is the first human orientation artifact and points
  back to authoritative sibling artifacts.
- `config_source.yaml` and `base_config_source.yaml` keep best-effort authored
  YAML copies when the source files are readable.
- Shadow unified-training configs record the `surface.id` direction through the
  resolved domains `run`, `surface`, `data`, `template`, `supervision`,
  `objectives`, `observability`, `artifacts`, and `runtime`; optional
  `experimental` remains an explicit opt-in escape hatch, not a hidden store.

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
  - Latest compact detection runs also record:
    - `latest_detection_objective`: objective id/variant, template id,
      coordinate surface, bbox format, state weighting, normalization,
      support/balance weights, roll-in source, type-gate mode, and EOS token.
    - `effective_batch_size` and `effective_batch_size_source`, because
      `gradient_accumulation_steps` is derived when effective batch is authored.
    - `actual_global_effective_batch_size`, `world_size`, and
      `effective_batch_rounding`, because non-divisible launch shapes can require
      ceil-derived accumulation; the actual global value is the run-time truth.
    - `model_source`: best-effort path identity for the base model/cache path.
    - `token_rows.expected_trainable_row_count`; compact-full token-row runs
      should report `1002` rows (1000 coord rows plus
      `<|object_ref_start|>` and `<|box_start|>`).
- `pipeline_manifest.json`
  - First-class pipeline identity / manifest artifact assembled from
    `src/bootstrap/pipeline_manifest.py`.
  - Stage-1 latest compact detection runs may not have a pipeline manifest; do
    not fabricate one. Treat absence as not applicable unless the runtime
    explicitly writes a real pipeline manifest.
- `experiment_manifest.json`
  - Primary run-level overview artifact for retrospective analysis.
  - Combines:
    - authored `experiment.*` narrative when present,
    - run identity (`config_path`, `run_name`, `output_dir`, `dataset_seed`),
    - executed-runtime summary,
    - provenance summary,
    - pointers to the authoritative artifact files.
  - emitted via `src/bootstrap/experiment_manifest.py`
  - The runtime summary mirrors the latest compact detection identity fields
    from `effective_runtime.json` so an experiment manifest can identify the
    exact objective surface without reopening the full resolved config.
- Inference summaries and resolved metadata include
  `generation.qwen_chat_generation`:
  - `eos_token: "<|im_end|>"` and `pad_token: "<|endoftext|>"`;
  - resolved token ids when a tokenizer is available;
  - `stop_tokens: ["<|im_end|>"]`;
  - `processor_do_resize: false`.
  This applies to HF and vLLM metadata so eval artifacts can prove the decode
  contract did not drift from training.
- `train_data_provenance.json`
  - Stable train-split source identity and optional digests for supported
    local inputs.
- `eval_data_provenance.json`
  - Stable eval-split source identity and optional digests when eval data is
    configured.
- `run_metadata.json`
  - Low-level provenance artifact:
    - `git_sha`, `git_branch`, `git_dirty`, `git_status_porcelain` (truncated)
    - `created_at` (UTC ISO8601)
    - `upstream` dependency provenance (ms-swift / transformers, etc.)
    - launcher metadata (Stage-2 server/learner topology), when present
    - encoded-sample-cache runtime metadata, when present
  - treat this as the detailed provenance sidecar; use
    `experiment_manifest.json` as the first run-level orientation artifact.
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

### Artifact/Provenance Freeze

The unified training infrastructure refactor freezes the current rank-0
artifact names and owners so later surface / runtime hierarchy cleanup cannot
silently weaken reproducibility. Future refactors may move the owner only with
an explicit replacement artifact and tests that preserve or deliberately
migrate the name.

| Artifact | Current owner | Compatibility decision |
| --- | --- | --- |
| `resolved_config.json` | `src/utils/run_manifest.py::write_run_manifest_files` | Preserve the exact filename as the canonical resolved training config snapshot. |
| `runtime_env.json` | `src/utils/run_manifest.py::write_run_manifest_files` | Preserve the exact filename and whitelisted-env behavior. |
| `effective_runtime.json` | `src/utils/run_manifest.py::write_run_manifest_files`; payload built in `src/sft.py::_build_effective_runtime_payload` | Preserve the exact filename as the executed-runtime truth after bootstrap / launcher mutation. |
| `pipeline_manifest.json` | `src/bootstrap/pipeline_manifest.py::build_pipeline_manifest` plus `src/utils/run_manifest.py::write_run_manifest_files` | Preserve the exact filename when the runtime surface has a real pipeline manifest; do not fabricate it for not-applicable surfaces. |
| `experiment_manifest.json` | `src/bootstrap/experiment_manifest.py::write_experiment_manifest_file` | Preserve the exact filename as the first run-level orientation artifact. |
| `run_metadata.json` | `src/bootstrap/run_metadata.py::write_run_metadata_file_from_payload` | Preserve the exact filename as the detailed git / dependency / launcher / cache provenance sidecar. |
| `train_data_provenance.json` | `src/utils/run_manifest.py::write_run_manifest_files`; source identity assembled in `src/sft.py` | Preserve the exact filename and split wrapper for train data identity. |
| `eval_data_provenance.json` | `src/utils/run_manifest.py::write_run_manifest_files`; source identity assembled in `src/sft.py` | Preserve the exact filename and split wrapper when eval data is configured. |
| `config_source.yaml` | `src/utils/run_manifest.py::write_run_manifest_files` | Preserve the exact filename for the authored config copy when the source path is locally readable. |
| `base_config_source.yaml` | `src/utils/run_manifest.py::write_run_manifest_files` | Preserve the exact filename for the authored base-config copy when the source path is locally readable. |

The run-manifest file map returned by `write_run_manifest_files` is also part of
the compatibility surface because `run_metadata.json` and
`experiment_manifest.json` consume it to point at authoritative sibling
artifacts.

Stage-2 eval artifact materialization is likewise frozen at the current
default-on location:

- `rollout_matching.eval_detection.materialize_artifacts: true`
- `training.output_dir/eval_detection/step_<global_step>/`
- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`
- `infer_summary.json`
- `metrics.json`
- `per_image.json`
- `raw_rollouts.jsonl`
- `pred_token_trace.jsonl` when trace metadata is available

No Stage-2 executable-path migration, resolver default flip, or observability
rewrite should pass review unless it preserves these artifacts or deliberately
migrates them with explicit docs and tests.

### Stage-2 Policy Provenance Migration Target

Stage-2 assignment, duplicate filtering, and object ordering need first-class
policy provenance as the architecture moves from legacy trainer internals to
the reusable `src/training/stage2/` planning stack. Current runs may expose
these policies through `resolved_config.json`, eval-step rollout diagnostics,
or planner metadata; the `stage2_policy_provenance.*` fields are migration
targets and are not yet written by all rank-0 manifests.

| Policy surface | Current owner / location | Current artifact visibility | Compatibility decision |
| --- | --- | --- | --- |
| `stage2_policy_provenance.assignment_strategy` | Target direction: `src/training/stage2/assignment.py::GreedyIoUAssignment` through `src/training/stage2/planners.py::Stage2GreedyIoUShadowPlanner`; migration-only legacy reader: `src/trainers/rollout_matching/matching.py::hungarian_match_maskiou` | Blocking migration gap: rank-0 manifests do not yet always write a first-class assignment-strategy field. Current evidence is resolved config, code path, and eval-step rollout diagnostics. | Preserve artifact visibility while migrating to `greedy_iou` over the post-duplicate survivor set. Hungarian is compatibility/migration-only until the remaining adapters and historical comparisons are removed; it is not the target architecture for new Stage-2 planning. |
| `stage2_policy_provenance.duplicate_filter_strategy` | `src/training/stage2/duplicate_filter.py::DuplicateFilter`; live compatibility owner `src/config/schema.py::Stage2ABChannelBDuplicateControlConfig`; `src/trainers/stage2_two_channel/target_builder.py::_apply_channel_b_duplicate_control` | Partial visibility through `resolved_config.json` at `stage2_ab.channel_b.duplicate_control.{iou_threshold,center_radius_scale}`; Blocking migration gap: no explicit strategy id is written to every rank-0 manifest. | Duplicate filtering must run before assignment and target realization. Preserve thresholds and diagnostic counters, or add an explicit replacement field plus tests before changing filtering order or semantics. |
| `stage2_policy_provenance.object_ordering_policy` | `src/training/ordering.py`; `src/sft.py` injection into rollout configs; `src/trainers/stage2_two_channel/target_builder.py` for `stage2_ab.channel_b.insertion_order` | Partial visibility through `resolved_config.json` at `custom.object_ordering` and `stage2_ab.channel_b.insertion_order`; Stage-2 eval summaries also record rollout object ordering when materialized. | Preserve Channel-B final-target insertion ordering: default `tail_append` keeps retained accepted rollout objects first and appends false-negative GT objects; `sorted` applies final top-left ordering over retained accepted objects plus inserted false negatives. |

Target manifest field names are reserved as
`stage2_policy_provenance.assignment_strategy`,
`stage2_policy_provenance.duplicate_filter_strategy`, and
`stage2_policy_provenance.object_ordering_policy` unless a later OpenSpec
migration deliberately replaces them. The current absence of all three
first-class fields from rank-0 manifests is a blocking migration gap for any
Stage-2 assignment, duplicate-filtering, or object-ordering rewrite.

### Diagnostic Compatibility Freeze

The infrastructure cleanup must keep high-value structured diagnostics, not only
scalar metrics. Current diagnostic surfaces map to future bounded writers as
follows:

`MetricEvent` remains the scalar metric contract. `DiagnosticEvent` is the
bounded structured-diagnostic contract with `off`, `standard`, and `debug`
profiles. Future diagnostic writers may be richer than scalar metrics, but they
must remain bounded by profile and keep enough provenance to locate the source
artifact, run config, and rollout/eval window.

| Current diagnostic surface | Current owner / producer | Future bounded-writer compatibility decision |
| --- | --- | --- |
| `monitor_dumps/` | Stage-2 rollout monitor dumping under the trainer `rollout_matching.*monitor_dump` configs | Keep a bounded qualitative rollout dump writer with the same directory-level discoverability, cadence controls, and raw text/token visibility. |
| `prepare_failures/` | Channel-B rollout preparation failure dumps under `monitor_dumps/prepare_failures/` | Keep structured malformed-rollout evidence with token IDs, decoded rollout text, prefix text, and parse/error classes. |
| `raw_rollouts.jsonl` | Stage-2 eval artifact materialization under `eval_detection/step_<global_step>/` | Keep per-sample rollout text, token IDs, parse diagnostics, match diagnostics, score metadata, and pre/post score prediction views. |
| `pred_token_trace.jsonl` | Inference and traced Stage-2 eval generation paths | Keep line-aligned token text/logprob traces whenever trace metadata is available; downstream confidence and rollout inspection depend on this name. |
| guarded eval/post-op artifacts (`gt_vs_pred_guarded.jsonl`, `gt_vs_pred_scored_guarded.jsonl`, `metrics_guarded.json`, `per_image_guarded.json`) | `src/eval/detection_duplicate_guard.py`, `src/eval/artifacts.py`, and confidence/eval orchestration | Keep guarded artifacts additive to the raw/scored families; do not replace authoritative raw artifacts with guarded-only outputs. |
| duplicate/EOS diagnostic probes | `src/analysis/duplication_collapse_analysis.py`, `src/analysis/small_object_duplication_diagnostics.py`, prefix-rollin / raw-text coordinate analysis probes, and related progress notes | Keep probe outputs as structured analysis artifacts with explicit source artifact roots, checkpoint/config handles, token-trace links, duplicate counters, and EOS/continue evidence. |

Notes:

- If `training.add_version: true` (default in `configs/base.yaml`), ms-swift
  scopes outputs under a versioned run directory.
- If `training.logging_dir` is explicitly set together with `training.run_name`,
  CoordExp writes TensorBoard event files under `<logging_dir>/<run_name>/` so
  `tensorboard --logdir <logging_dir>` shows the authored run name instead of a
  bare `.` root entry.
- If `training.output_dir` is not set, training fails fast because these
  artifacts are required for reproducibility.
- `training.save_model_only: true` is the public opt-in for restartable
  checkpoints. Each saved checkpoint must include model/adapter artifacts,
  tokenizer files, optimizer, scheduler, RNG, trainer state, and repo-owned
  runtime sidecars. `training.save_model_only: false` keeps inference-only
  artifacts and accepts that an interrupted run may need to restart from the
  base checkpoint.

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

- `stage2_two_channel` includes clean-prefix Channel-B duplicate-control
  diagnostics under:
  - `dup/raw/*`
  - `stage2_ab/channel_b/dup/N_*`
  - `stage2_ab/channel_b/closure_supervision/N_drop` for the
    legacy-named closure-resolution fallback activation counter

---

## Related Docs

- Data contract: `docs/data/CONTRACT.md`
- Packing guide: `docs/data/PACKING.md`
- Stage-2 runbook: `docs/training/STAGE2_RUNBOOK.md`
- Metric keys: `docs/training/METRICS.md`
- Offline evaluator: `docs/eval/README.md`
