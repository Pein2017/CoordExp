## ADDED Requirements

### Requirement: SupervisedTrainer Owns Loop Only

`SupervisedTrainer` SHALL own the supervised execution loop: step iteration,
calls into pack streams, Qwen forward, loss runner, runtime backward, optimizer
boundary, scheduler boundary, eval triggers, checkpoint triggers, and artifact
events. It MUST NOT own objective math, Qwen model internals, adapter target
taxonomy, or artifact schemas.

#### Scenario: New loss term added later

- **WHEN** a future loss term is added
- **THEN** it MUST be implemented in loss modules and invoked through the loss
  runner
- **AND** `SupervisedTrainer` MUST NOT become the place where objective math is
  hand-coded.

### Requirement: TrainRuntime Owns Backend Mechanics

`TrainRuntime` SHALL own device placement, distributed wrapping, rank guards,
backend prepare, backward, gradient clipping, optimizer-step helpers,
scheduler-step helpers, metric gathering, and safe save helpers. Data loading,
template rendering, Qwen encoding, packing policy, and loss semantics MUST
remain outside runtime ownership.

#### Scenario: Distributed artifact write

- **WHEN** a rank-safe artifact must be written
- **THEN** `TrainRuntime` MUST provide the rank guard or save helper
- **AND** the artifact manager MUST own the file schema and path.

### Requirement: Optimizer-Step Order

The trainer/runtime boundary SHALL use the canonical step order: build or fetch
packed sequence, move tensors, Qwen forward, construct loss context, compute
loss bundle, pre-backward finite check, backward, all-rank gradient/overflow
decision, gradient clipping when safe, optimizer step when safe, scheduler step
on planned-step clock, zero gradients, and emit metrics/artifacts.

#### Scenario: Unsafe step after backward

- **WHEN** post-backward global overflow status is unsafe
- **THEN** runtime MUST skip the optimizer update
- **AND** still emit the planned-step lifecycle event with update status.

### Requirement: Artifact Manager And Manifest

The artifact manager SHALL own run directory creation, `run_manifest.json`,
resolved configs, schedule receipts, subsystem receipts, metrics, checkpoints,
eval summaries, and aliases. Artifacts MUST be sufficient to explain the run's
config, data, model, adapter, embedding, optimizer, schedule, metric, and
checkpoint identities. `run_manifest.json` MUST record the absolute `run_dir`,
run-dir-relative internal artifact links, resolved config paths, resolution
provenance, `resolved_step_schedule.json`, optimizer group receipts,
checkpoints, eval outputs, metric files, backend status, warnings, and compact
subsystem receipt links. The minimum manifest top-level sections SHALL include
`run_id`, `run_name`, `run_dir`, `status`, `created_at`, `updated_at`,
`configs`, `resolution`, `runtime_identity`, `schedule`, `receipts`,
`metrics`, `checkpoints`, `eval`, `runtime`, `backend_status`, and `warnings`.
Random ordering seed sources and dataset identities used by train or
`eval.forward` MUST be linked from the manifest or resolved config.

#### Scenario: Run starts successfully

- **WHEN** a training run passes config resolution and creates its output
  directory
- **THEN** `run_manifest.json` MUST be written with run id, absolute run
  directory, artifact paths, resolved config fingerprint, resolution
  provenance, dataset identities, ordering seed source when applicable, and
  backend status before the first optimizer update.

### Requirement: Metric Event Shape

Metric events SHALL store split and metric name separately. Train events MUST
include protected weighted losses, top-level `acc_top1`, top-level `acc_top5`,
learning-rate group values, optimizer-update status, and warning/non-finite
status when present. `eval.forward` events MUST use split `eval.forward` and
the same metric-name vocabulary where applicable. Each metric event record
SHALL include at least `event_type`, `planned_step_id`, `split`, `name`,
`value`, `trigger_reasons`, `optimizer_update_status`, `finite_status`, and
`warning_status`, with unavailable values represented explicitly rather than
by omitting required keys.

#### Scenario: Best checkpoint selector

- **WHEN** best checkpoint selection uses `eval.forward/acc_top1:max`
- **THEN** that string MUST be treated as selector syntax
- **AND** stored metric events MUST still store split and name as separate
  fields.

#### Scenario: Update-skipped checkpoint has best metric

- **WHEN** an update-skipped or unsafe non-finite checkpoint has a better
  `eval.forward` metric than a safe checkpoint
- **THEN** default best-checkpoint selection MUST exclude the update-skipped or
  unsafe checkpoint
- **AND** any override MUST be explicit and recorded in checkpoint metadata.

### Requirement: Eval Forward Artifacts

`eval.forward` SHALL be a minimal packed forward-evaluation loop using the same
render, encode, pack, Qwen forward, model-output, loss-context, loss, and metric
stack as training, but without backward or optimizer updates. Each scheduled
forward-eval run MUST write a compact summary at
`eval/forward/step-<planned_step_id>.json` with unpadded step ids. Each
summary SHALL include at least `planned_step_id`, `split`, `trigger_reasons`,
`example_count`, `pack_count`, `loss_summary`, `metric_summary`, and
`artifact_links`. `eval.forward` MUST use an explicit eval data source or an
explicit smoke-fixture eval binding. Train JSONL MUST NOT be implicitly reused,
randomly split, or silently sampled for eval.

#### Scenario: Eval at scheduled smoke step

- **WHEN** planned step 4 triggers `eval.forward`
- **THEN** the summary path MUST be `eval/forward/step-4.json`
- **AND** the summary MUST record the triggering planned step and trigger
  reason.

#### Scenario: Eval data path omitted

- **WHEN** scheduled `eval.forward` is configured without `data.eval_path` and
  without an explicit smoke-fixture eval binding
- **THEN** config or schedule validation MUST fail or disable the scheduled eval
  before training starts
- **AND** it MUST NOT fall back to the training JSONL implicitly.

### Requirement: Checkpoint Writer

The checkpoint writer SHALL own checkpoint schema, adapter payloads,
special-token embedding deltas, metadata, and aliases. V1 checkpoints MUST save
adapter weights when enabled, selected special-token embedding deltas when
enabled, processor identity, resolved config fingerprints, schedule identity,
metric status, and trainable-surface metadata. V1 MUST NOT promise optimizer,
scheduler, scaler, dataloader, iterator, or RNG resume. Checkpoint metadata
SHALL include at least `checkpoint_id`, `planned_step_id`, `checkpoint_path`,
`adapter`, `special_token_embeddings`, `processor_identity`,
`resolved_config_fingerprint`, `schedule_identity`, `metric_status`,
`trainable_surface`, and `optimizer_update_status`.

#### Scenario: Final checkpoint written

- **WHEN** a run completes its resolved planned steps
- **THEN** `checkpoints/checkpoint-final.json` MUST exist
- **AND** it MUST point to the final state of this run even if the final
  planned step had warnings or skipped optimizer update status.

### Requirement: Checkpoint Naming

Checkpoint step ids in file and directory names SHALL be unpadded planned-step
ids. The final alias MUST be named `checkpoint-final.json`. Best accuracy alias
MAY be named `best_acc_top1.json` when best-checkpoint selection is enabled.

#### Scenario: Checkpoint at planned step five

- **WHEN** a checkpoint is written at planned step 5
- **THEN** the checkpoint path MUST use `step-5` or an equivalent unpadded
  planned-step id
- **AND** MUST NOT use `step-000005`.

### Requirement: Warning And Bad-Example Policy

Recoverable bad examples or warnings SHALL be recorded or warned without
changing the precomputed planned-step schedule. Unsafe non-finite scalar or
gradient state MUST prevent an optimizer update, but the planned-step id and
scheduled events MUST remain static.

#### Scenario: Bad example encountered before a planned step

- **WHEN** a recoverable bad example is skipped or warned during stream
  construction
- **THEN** the warning MUST be recorded with bounded context
- **AND** the resolved planned-step schedule MUST NOT be recomputed mid-run.
