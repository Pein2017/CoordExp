# coordexp-swift-training-artifacts Specification

## Purpose
TBD - created by archiving change rebuild-coordexp-swift-training-infra. Update Purpose after archive.
## Requirements
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

#### Scenario: Distributed artifact write

- **WHEN** a distributed checkpoint payload requires Accelerate-safe saving
- **THEN** all ranks MUST follow the same runtime save/barrier order
- **AND** only rank zero MUST materialize the durable files.

#### Scenario: Accelerate runtime setup

- **WHEN** runtime prepares training objects
- **THEN** it MUST prepare the model and optimizer through Accelerate
- **AND** MUST keep scheduler ownership in CoordExp runtime.

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

### Requirement: Eval Forward Artifacts

`eval.forward` SHALL remain a minimal packed forward-evaluation loop using the
same render, encode, pack, Qwen forward, model-output, loss-context, loss, and
metric stack as training, but without backward or optimizer updates. Eval
packs MAY be partitioned across distributed ranks through a deterministic
disjoint covering assignment; when the eval pack count is smaller than the
world size, evaluation MUST fall back to the replicated form. The active
reduction mode MUST be explicit in the internal reduction payload or call
path and MUST NOT be inferred ambiguously: disjoint-shard mode reduces by
summed rank-local counts and statistics, while the replicated fallback MUST
keep current replicated semantics — identical rank values reduced once,
never summed as disjoint contributions — so no count or total is multiplied
by the world size. When sharded, every durable scalar in the canonical eval
row MUST be produced from a defined exact sufficient statistic and reducer:

- segment-balanced loss terms and total: full-eval-set global denominators
  gathered before any rank finalizes, combined with summed partial segment
  numerators (equivalently, world-size-scaled local contributions under the
  existing all-rank mean reduction, whose identity with the global sum MUST
  be proven by test);
- `acc_top1` and `acc_top5`: summed rank-local integer correct counts divided
  by summed rank-local supervised-atom counts, never plain or
  global-count-weighted means;
- token-weighted diagnostics: summed products of rank-local value and
  rank-local selected count, divided by summed rank-local selected counts;
- `example_count`, `pack_count`, and rank-local count fields: global sums
  over the disjoint shards;
- fields derived from the shared global denominator (per-term segment counts
  and globally merged atom/segment counts): emitted once from that shared
  denominator and MUST NOT be additionally rank-summed;
- finite flags and non-finite handling: unchanged fail/normalize contract,
  with any rank-local non-finite contribution propagating to the
  corresponding global value.

A compact explicit eval reduction payload MAY own these sufficient
statistics internally; a generic reduction framework MUST NOT be introduced,
and internal statistics need not become durable logging fields. Equivalence
acceptance against the fully replicated evaluator on the same checkpoint and
eval set MUST cover the entire canonical eval row — exact integer counts,
exact top-k accuracies, loss and diagnostic scalars within declared
floating-point tolerance — not only loss and top-k fields. Each scheduled
forward-eval run MUST append one wide `eval` row to `logging.jsonl` containing
`step`, `split`, trigger reasons, example count, pack count, loss summary, and
metric summary. That row SHALL be the sole durable scalar schema for the eval
invocation; normal training MUST NOT write a second eval-summary metric file.
Eval MUST use an explicit eval data source or explicit smoke-fixture binding;
train JSONL MUST NOT be implicitly reused, randomly split, or silently sampled
for eval.

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

#### Scenario: Sharded eval matches replicated eval

- **WHEN** eval packs are partitioned across ranks for one scheduled eval
- **THEN** the written `eval` row MUST match the replicated evaluator across
  its entire scalar surface: global counts and `example_count`/`pack_count`
  as exact sums, `acc_top1`/`acc_top5` exactly equal from summed integer
  statistics, loss and token-weighted diagnostic scalars within the declared
  floating-point tolerance, and global-denominator-derived fields emitted
  once
- **AND** best-checkpoint selection driven by `acc_top1` MUST be unchanged.

#### Scenario: Replicated fallback keeps replicated reduction

- **WHEN** a multi-rank eval has fewer eval packs than ranks
- **THEN** evaluation MUST run in the replicated form with current replicated
  reduction semantics
- **AND** counts, `example_count`, and `pack_count` MUST NOT be multiplied by
  the world size
- **AND** the entire eval row MUST be identical to the pre-change replicated
  evaluator
- **AND** the active reduction mode MUST be explicit rather than inferred.

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
when present, a compact immutable binding to the actual train/eval
materialization consumed, and the compact resolved forward-input-provider
mode. Final and best checkpoint selection MUST be owned only by their aliases
and MUST NOT be duplicated in `run.json`. The materialization binding MUST
include the packing-cache format version and semantic fingerprint plus either
the complete semantic-determinant digest or equivalent bounded data-content,
ordering, augmentation, and seed identities. A cache path MAY be recorded for
operator convenience but MUST NOT be the sole binding. The forward-input-
provider mode MUST be bound at most once and MUST NOT change after binding.
`run.json` MUST NOT contain an unbounded list of step results, metric events,
receipts, eval events, checkpoints, cache chunks, or presentation records.

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

#### Scenario: Forward-input-provider mode recorded once

- **WHEN** a training run resolves its forward-input-provider mode at
  assembly time
- **THEN** `run.json` MUST record that compact mode value exactly once
- **AND** a later attempt to bind a different mode value for the same run
  MUST fail rather than silently overwrite the recorded mode.

### Requirement: Wide-Step Logging Stream

Rank zero SHALL append one compact JSON object to `logging.jsonl` for every
completed train step and every completed eval invocation. A train row MUST use
`split: "train"`; an eval row MUST use `split: "eval"`; both MUST use the
planned training step as `step`. Each row SHALL store the complete scalar
logging mapping for that event together instead of writing one record per
metric.

Train and forward-eval rows MUST use explicit `loss/<term>/raw` and
`loss/<term>/weighted` fields for every computed loss term, plus matching
term-count and finite-status fields. `loss/total` MUST be the sum of weighted
objective terms. The zero-weight protected gate ablation MUST retain its raw,
weighted-zero, count, and finite fields. A disabled optional auxiliary term
MUST have no raw, weighted, denominator, count, or finite field. Train rows
MUST also include top-level `acc_top1`, top-level `acc_top5`, actual
learning-rate values, optimizer-update status, and finite status where those
values are available.

Train rows MUST include low-overhead timing scalars measured inside the
planned-step compute/optimizer boundary — from the start of the step's first
micro-step handling through gradient zeroing — excluding the completed-step
handler and scheduled eval/checkpoint handlers: `step_duration_seconds` for
that boundary's wall time, `input_build_seconds` for forward-input construction
time, and `input_wait_seconds` for time spent waiting on prepared inputs.
Distributed reduction for these timing fields MUST be the all-rank maximum,
because the slowest rank owns the distributed critical path; a mean MAY
additionally be emitted only under a name that explicitly states it is a mean.
Timing values MUST travel through the existing metric collective without
introducing additional synchronization, and timing collection MUST NOT add
per-step synchronization beyond monotonic-clock reads. Eval rows MUST include
eval counts and the corresponding eval loss/metric mapping.

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
- **AND** all computed loss terms MUST have explicit raw and weighted fields in
  that row
- **AND** all other logging metrics for that step MUST be fields in that row.

#### Scenario: Eval runs at planned step

- **WHEN** scheduled eval completes at planned step 42
- **THEN** `logging.jsonl` MUST receive exactly one `eval` row with `step: 42`
- **AND** that row MUST use the same computed-term raw/weighted and
  disabled-term omission rules as training
- **AND** that row MUST be the canonical durable scalar record for the eval.

#### Scenario: Gate ablation row

- **WHEN** a step uses the named zero-weight token-gate ablation
- **THEN** its row MUST contain the gate raw diagnostic, weighted value `0`,
  selected count, and finite status
- **AND** MUST distinguish that computed diagnostic from optimized terms.

#### Scenario: Optional auxiliary omitted

- **WHEN** a step's resolved coordinate Gaussian/RPS auxiliary has weight `0`
- **THEN** its row MUST contain no coordinate Gaussian/RPS raw, weighted,
  denominator, count, or finite field.

#### Scenario: Optimizer update is skipped

- **WHEN** an all-rank objective finite gate skips the optimizer update for one
  planned step
- **THEN** that step MUST still receive one train logging row
- **AND** the row MUST state the skipped update and non-finite/unsafe status
- **AND** any non-finite computed scalar MUST be `null` and named in
  `non_finite_fields`.

#### Scenario: Timing fields observed

- **WHEN** a planned train step completes under normal production settings
  with world size greater than one
- **THEN** its train row MUST contain `step_duration_seconds`,
  `input_build_seconds`, and `input_wait_seconds` reduced as the all-rank
  maximum
- **AND** the measured window MUST exclude completed-step and scheduled
  eval/checkpoint handler time.

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

Scheduled checkpoint saving SHALL always materialize an independently loadable
minimal inference payload: a standard PEFT/DoRA adapter directory when adapter
training is enabled, a compact selected-token embedding-delta directory when
that trainable surface is enabled, and a self-authenticating manifest binding
those learned files. The inference payload MUST NOT save base-model weights and
MUST NOT require optimizer, scheduler, scaler, RNG, dataloader, iterator, or
sampler state to be loadable. New inference payloads MUST NOT require
`checkpoint.json`, `checkpoint_handoff.json`, setup receipts, duplicated
identity graphs, or an exact-training-state sibling.

Exact training state, when explicitly enabled, MUST be published as a typed
`training_state/` sibling rather than added to or inferred from the inference
payload. When disabled, no exact-state sibling or identity may be written.
Inference loaders MUST ignore the sibling and consume only explicit adapter and
selected-token payload paths covered by the inference manifest.

All ranks SHALL enter checkpoint-save operations in the same order when
distributed collectives require it, but only rank zero SHALL materialize the
durable inference payload. Supported distributed saving SHALL be limited to
replicated DDP: all ranks enter a pre-save barrier; rank zero unwraps the model
and writes only the configured adapter into staging using safe serialization
with embedding-layer saving disabled, then writes the optional compact selected
embedding delta and inference manifest. The adapter safetensor MUST contain
required LoRA A/B and DoRA magnitude-vector state and MUST NOT contain full
embedding, LM-head, or base-model tensors. Rank zero MUST atomically commit the
inference payload only after validation and broadcast a bounded success/error
descriptor to every rank. Every rank MUST continue or raise the same named
checkpoint-save error. Failed inference staging MUST be removed. Final and best
aliases MUST update only after all payloads required by the selected checkpoint
mode have committed successfully.

#### Scenario: Adapter-only checkpoint is saved

- **WHEN** adapter training reaches a scheduled checkpoint step without
  selected-token embedding training
- **THEN** the step directory MUST contain the standard adapter payload and
  manifest needed by the inference adapter loader
- **AND** MUST use `adapter_model.safetensors`
- **AND** MUST NOT contain a copy of base-model, full-embedding, or LM-head
  weights.

#### Scenario: Adapter and selected embeddings are saved

- **WHEN** both adapter and selected-token embeddings are trainable
- **THEN** the inference manifest MUST cover both payload directories
- **AND** inference MUST be able to compose them from explicit config paths.

#### Scenario: Exact state is disabled

- **WHEN** a scheduled checkpoint is saved with exact training state disabled
- **THEN** the inference payload and aliases MUST retain their normal behavior
- **AND** no `training_state/` sibling or exact-state identity may be written.

#### Scenario: Exact state is enabled

- **WHEN** a scheduled checkpoint is saved with exact training state enabled
- **THEN** the inference payload MUST commit independently before the typed
  sibling is published
- **AND** final or best aliases and the completed checkpoint event MUST update
  only after the exact-state publication succeeds.

#### Scenario: Exact-state publication fails on one rank

- **WHEN** the inference payload commits but one required rank fails during
  exact-state publication
- **THEN** every live rank MUST observe the same bounded checkpoint failure
  without hanging
- **AND** no final or best alias and no completed exact checkpoint event may
  reference that step
- **AND** the inference payload MUST remain typed as inference-only rather than
  being treated as partial exact state.

#### Scenario: Rank-zero inference payload save fails

- **WHEN** rank zero fails while saving or validating the inference payload
- **THEN** every rank MUST observe the same checkpoint-save failure without
  hanging
- **AND** no final or best alias MUST reference the incomplete step
- **AND** no committed partial inference checkpoint directory may remain.

#### Scenario: Existing checkpoint is used

- **WHEN** an older CoordExp-Swift checkpoint contains a standard adapter and
  optional compatible selected-token embedding payload plus extra historical
  metadata
- **THEN** inference MUST load the configured payload paths
- **AND** MUST ignore unrelated training-state or historical metadata
- **AND** MUST NOT require that metadata to be regenerated.

### Requirement: Run Artifacts Record Executed Environment Provenance

Every training run SHALL record enough non-secret provenance to distinguish
the code and critical dependency state that actually executed. The run record
MUST identify the repository commit, whether relevant tracked or untracked
changes were present, a stable digest or explicit unavailable status for that
local state, and the resolved versions plus available source or binary
identities for critical training dependencies. Provenance collection MUST NOT
copy credentials or secret environment values into artifacts.

#### Scenario: A dirty checkout starts training

- **WHEN** relevant tracked or untracked changes are present
- **THEN** the run record MUST mark the executed tree dirty and retain a stable
  non-secret identity for the local state
- **AND** the repository commit alone MUST NOT be presented as complete
  execution provenance.

#### Scenario: A dependency identity is unavailable

- **WHEN** a critical source or binary identity cannot be obtained safely
- **THEN** the run record MUST preserve the dependency version and an explicit
  unavailable status with a bounded reason
- **AND** it MUST NOT substitute an assumed identity.

