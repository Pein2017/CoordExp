## ADDED Requirements

### Requirement: Rank-Zero Run Record

Each training launch SHALL create exactly one durable run directory owned by
Accelerate rank zero. The minimum normal run tree SHALL contain `run.json`,
`resolved_config.json`, `logging.jsonl`, and checkpoint payloads/aliases under
`checkpoints/` when those events are configured. Non-main ranks MUST NOT create
suffixed copies of the run tree or independently persist config, logging,
checkpoint, or terminal-result artifacts.

`run.json` SHALL remain compact and contain run identity, lifecycle status and
timestamps, resolved-config path/fingerprint, runtime world-size summary,
resolved maximum steps, completed/consumed counts, final optimizer-update and
finite status, terminal failure summary when present, checkpoint event counts
when present, and a compact immutable binding to the actual train/eval
materialization consumed. Final and best checkpoint selection MUST be owned
only by their aliases and MUST NOT be duplicated in `run.json`. The
materialization binding MUST include the
packing-cache format version and semantic fingerprint plus either the complete
semantic-determinant digest or equivalent bounded data-content, ordering,
augmentation, and seed identities. A cache path MAY be recorded for operator
convenience but MUST NOT be the sole binding. `run.json` MUST NOT contain an
unbounded list of step results, metric events, receipts, eval events,
checkpoints, cache chunks, or presentation records.

#### Scenario: Eight-rank launch starts

- **WHEN** training starts with Accelerate world size eight
- **THEN** exactly one run directory MUST be created
- **AND** only rank zero MUST own durable run-file writes.

#### Scenario: Run completes

- **WHEN** training reaches its resolved terminal planned step
- **THEN** `run.json` MUST record completed status and compact terminal counters
- **AND** MUST NOT embed the complete step history.

#### Scenario: Run fails

- **WHEN** training raises after the run directory is initialized
- **THEN** rank zero MUST atomically record failed status, terminal planned
  step when known, and a bounded error summary in `run.json`.

#### Scenario: Cache root is later removed

- **WHEN** the cache payload used by a completed run is deleted or rebuilt
- **THEN** `run.json` MUST still identify the consumed materialization by
  format version and semantic fingerprint/determinant identity
- **AND** MUST not depend on the cache path remaining readable.

### Requirement: Wide-Step Logging Stream

Rank zero SHALL append one compact JSON object to `logging.jsonl` for every
completed train step and every completed eval invocation. A train row MUST use
`split: "train"`; an eval row MUST use `split: "eval"`; both MUST use the
planned training step as `step`. Each row SHALL store the complete scalar
logging mapping for that event together instead of writing one record per
metric. Train rows MUST include weighted configured losses, top-level
`acc_top1`, top-level `acc_top5`, actual learning-rate values,
optimizer-update status, and finite status where those values are available.
Eval rows MUST include eval counts and the corresponding eval loss/metric
mapping.

The stream SHALL be a direct single-writer append. Before serialization, every
raw NaN or Inf scalar MUST be represented as JSON `null` and its field name
MUST appear in `non_finite_fields`; `finite_status` and
`optimizer_update_status` MUST remain explicit. The writer MUST reject any raw
non-finite number that remains after normalization, but MUST NOT rescan prior
rows, build a durable event index, fsync every row, create per-rank streams, or
require rotation, compression, sampling, retention, or database machinery in
this version.

#### Scenario: Planned train step completes safely

- **WHEN** planned step 42 completes with an applied optimizer update
- **THEN** `logging.jsonl` MUST receive exactly one `train` row with `step: 42`
- **AND** all logging metrics for that step MUST be fields in that row.

#### Scenario: Eval runs at planned step

- **WHEN** scheduled eval completes at planned step 42
- **THEN** `logging.jsonl` MUST receive exactly one `eval` row with `step: 42`
- **AND** that row MUST be the canonical durable scalar record for the eval.

#### Scenario: Optimizer update is skipped

- **WHEN** an all-rank finite gate skips the optimizer update for one planned
  step
- **THEN** that step MUST still receive one train logging row
- **AND** the row MUST state the skipped update and non-finite/unsafe status
- **AND** any non-finite scalar MUST be `null` and named in
  `non_finite_fields`.

### Requirement: Bounded Training Result

`SupervisedTrainer` SHALL retain only the current planned-step state, compact
terminal counters, scheduled-event counts, and latest logging/status values.
After logging and scheduled handlers consume a completed step, tensors,
micro-step forward receipts, and detailed step-result objects from that step
MUST become eligible for release. The returned training result MUST NOT contain
the complete sequence of planned-step results.

#### Scenario: Long run completes

- **WHEN** a run completes many planned steps
- **THEN** the returned trainer result size MUST not grow with serialized
  copies of every completed step
- **AND** durable step history MUST be represented only by the compact
  rank-zero logging stream and scheduled artifacts.

### Requirement: Minimal Inference Checkpoint Payloads

Scheduled checkpoint saving SHALL materialize only learned payloads required by
the inference engine: a standard PEFT/DoRA adapter directory when adapter
training is enabled and a compact selected-token embedding-delta directory
when that trainable surface is enabled. Checkpoints MUST NOT save base-model
weights or promise optimizer, scheduler, scaler, RNG, dataloader, iterator, or
sampler resume state. New checkpoints MUST NOT require `checkpoint.json`,
`checkpoint_handoff.json`, readiness manifests, setup receipts, or duplicated
identity graphs to be loadable.

All ranks SHALL enter checkpoint-save operations in the same order when
Accelerate collectives require it, but only rank zero SHALL materialize the
durable checkpoint directory. Supported distributed saving SHALL be limited to
replicated DDP: all ranks enter a pre-save barrier; rank zero unwraps the model
and writes only the configured adapter into staging using PEFT safe
serialization with embedding-layer saving disabled, then writes the optional
compact selected embedding delta. The adapter safetensor MUST contain required
LoRA A/B and DoRA magnitude-vector state and MUST NOT contain full embedding,
LM-head, or base-model tensors. Rank zero MUST atomically commit the step
directory only after validation and broadcast a bounded success/error
descriptor to every rank. Every rank MUST continue or raise the same named
checkpoint-save error from that collective. Failed staging state MUST be
removed, and aliases MUST update only after successful commit.

#### Scenario: Adapter-only checkpoint is saved

- **WHEN** adapter training reaches a scheduled checkpoint step without
  selected-token embedding training
- **THEN** the step directory MUST contain the standard adapter payload needed
  by the inference adapter loader
- **AND** MUST use `adapter_model.safetensors`
- **AND** MUST NOT contain a copy of base-model, full-embedding, or LM-head
  weights.

#### Scenario: Adapter and selected embeddings are saved

- **WHEN** both adapter and selected-token embeddings are trainable
- **THEN** the step directory MUST contain both inference-loadable payload
  directories
- **AND** inference MUST be able to compose them from explicit config paths.

#### Scenario: Rank-zero checkpoint save fails

- **WHEN** rank zero fails while saving or validating a distributed checkpoint
- **THEN** every rank MUST observe the same checkpoint-save failure without
  hanging
- **AND** no final/best alias MUST reference the incomplete step
- **AND** no committed partial checkpoint directory may remain.

#### Scenario: Existing checkpoint is used

- **WHEN** an older CoordExp-Swift checkpoint contains a standard adapter and
  optional compatible selected-token embedding payload plus extra historical
  metadata
- **THEN** inference MUST load the configured payload paths
- **AND** MUST NOT require the historical metadata to be regenerated.

## MODIFIED Requirements

### Requirement: SupervisedTrainer Owns Loop Only

`SupervisedTrainer` SHALL own the supervised execution loop: step iteration,
calls into pack streams, Qwen forward, loss runner, runtime backward, optimizer
boundary, scheduler boundary, eval triggers, checkpoint triggers, and compact
completed-step observations. It MUST NOT own objective math, Qwen model
internals, adapter target taxonomy, file paths, logging schemas, checkpoint
payload schemas, or historical step retention.

#### Scenario: New loss term added later

- **WHEN** a future loss term is added
- **THEN** it MUST be implemented in loss modules and invoked through the loss
  runner
- **AND** `SupervisedTrainer` MUST NOT become the place where objective math is
  hand-coded.

#### Scenario: Completed step observation consumed

- **WHEN** logging and scheduled handlers finish consuming a completed step
- **THEN** the trainer MUST release detailed step-local state
- **AND** MUST retain only bounded terminal/latest state.

#### Scenario: Trainer observation interface is inspected

- **WHEN** callers wire training logging
- **THEN** the trainer MUST expose one typed completed-step callback plus direct
  scheduled handlers
- **AND** MUST NOT expose event-name dispatch, subscriptions, or micro-step/gate
  event callbacks.

### Requirement: TrainRuntime Owns Backend Mechanics

`TrainRuntime` SHALL be the concrete Accelerate execution owner for device
placement, distributed preparation, rank/world identity, accumulation,
backward, all-rank finite decisions, gradient clipping, optimizer-step helpers,
scheduler-step helpers, metric/denominator gathering, barriers, and rank-safe
save mechanics. Data loading, template rendering, Qwen encoding, packing
policy, loss semantics, run-file schemas, and checkpoint payload layout MUST
remain outside runtime ownership. The LR scheduler MUST remain CoordExp-owned,
MUST NOT be wrapped in backend scheduler semantics through
`accelerator.prepare(...)`, and MUST advance exactly once per completed
planned-step boundary.

#### Scenario: Rank-safe artifact write

- **WHEN** a distributed checkpoint payload requires Accelerate-safe saving
- **THEN** all ranks MUST follow the same runtime save/barrier order
- **AND** only rank zero MUST materialize the durable files.

#### Scenario: Accelerate runtime setup

- **WHEN** runtime prepares training objects
- **THEN** it MUST prepare the model and optimizer through Accelerate
- **AND** MUST keep scheduler ownership in CoordExp runtime.

### Requirement: Eval Forward Artifacts

`eval.forward` SHALL remain a minimal packed forward-evaluation loop using the
same render, encode, pack, Qwen forward, model-output, loss-context, loss, and
metric stack as training, but without backward or optimizer updates. Each
scheduled forward-eval run MUST append one wide `eval` row to `logging.jsonl`
containing `step`, `split`, trigger reasons, example count, pack count, loss
summary, and metric summary. That row SHALL be the sole durable scalar schema
for the eval invocation; normal training MUST NOT write a second eval-summary
metric file. Eval MUST use an explicit eval data source or explicit
smoke-fixture binding; train JSONL MUST NOT be implicitly reused, randomly
split, or silently sampled for eval.

#### Scenario: Eval at scheduled smoke step

- **WHEN** planned step 4 triggers eval
- **THEN** one `eval` row with `step: 4` MUST be appended to `logging.jsonl`
- **AND** no duplicate eval metric summary file MUST be required.

#### Scenario: Eval data path omitted

- **WHEN** scheduled eval is configured without an explicit eval source or
  smoke-fixture binding
- **THEN** config or schedule validation MUST fail or disable scheduled eval
  before training starts
- **AND** it MUST NOT fall back to the training JSONL implicitly.

### Requirement: Checkpoint Naming

Checkpoint step-directory ids SHALL be unpadded planned-step ids. The final
alias MUST be named `checkpoints/final.json`; the optional best-accuracy alias
MUST be named `checkpoints/best.json`. Each alias SHALL contain only the
selected planned step, run-relative checkpoint directory, and selector/value
when applicable. By default an eval result associated with an update-skipped,
unsafe, or non-finite planned step MUST NOT advance `best.json`, even when its
selector value is better. Any later override MUST be explicit and the alias
MUST record eligibility/override status compactly.

#### Scenario: Checkpoint at planned step five

- **WHEN** a checkpoint is written at planned step 5
- **THEN** its directory MUST be `checkpoints/step-5/`
- **AND** MUST NOT use a zero-padded step id.

#### Scenario: Final alias is written

- **WHEN** final checkpoint saving completes
- **THEN** `checkpoints/final.json` MUST point to that run-relative step
  directory
- **AND** MUST not duplicate the checkpoint payload metadata.

#### Scenario: Unsafe step has a better eval value

- **WHEN** an update-skipped, unsafe, or non-finite step reports a better eval
  selector value than the current eligible checkpoint
- **THEN** `best.json` MUST remain on the prior eligible checkpoint by default
- **AND** the unsafe step MUST remain visible in `logging.jsonl` without being
  promoted.

### Requirement: Warning And Bad-Example Policy

Recoverable bad examples or warnings SHALL be reported without changing the
precomputed planned-step schedule. Unsafe non-finite scalar or gradient state
MUST prevent an optimizer update, but the planned-step id and scheduled events
MUST remain static. Step-associated warnings and unsafe status SHALL be
represented in the corresponding train logging row; bounded terminal or setup
warnings MUST be represented by bounded counters grouped by warning code in
`run.json`. Cache-materialization warnings MAY instead be owned by the cache
manifest when `run.json` retains the materialization fingerprint. The run tree
MUST NOT grow an unbounded warning-context or bad-example receipt family.

#### Scenario: Bad example encountered before a planned step

- **WHEN** a recoverable bad example is skipped or warned during stream
  construction
- **THEN** a bounded warning summary MUST remain observable
- **AND** the resolved planned-step schedule MUST NOT be recomputed mid-run.

## REMOVED Requirements

### Requirement: Artifact Manager And Manifest

**Reason**: The manifest/receipt registry rewrites global state for every
subsystem artifact and duplicates run evidence across ranks. It is replaced by
one compact rank-zero run record and fixed run-tree conventions.

**Migration**: Consumers of new runs use `run.json`, `resolved_config.json`,
`logging.jsonl`, and checkpoint aliases directly.

### Requirement: Metric Event Shape

**Reason**: Per-metric event records duplicate step context and require
unnecessary event identity/deduplication machinery.

**Migration**: Read one wide `train` or `eval` row per completed step from
`logging.jsonl`.

### Requirement: Checkpoint Writer

**Reason**: The old writer persists metadata, receipts, handoff identity, and
unsupported resume-state declarations beyond what inference needs.

**Migration**: New checkpoints contain standard adapter and optional selected
embedding payloads plus small final/best aliases. Existing learned payloads
remain loadable by explicit path.
