## Context

The current training path already owns the scientific seams that must remain
stable: typed config resolution, source-order and geometry behavior,
template/token alignment, packed Qwen forward, loss normalization, finite
gates, optimizer/scheduler order, scheduled eval, and adapter checkpoints. The
problem is concentrated in infrastructure around those seams.

`run_training_pipeline()` currently initializes an artifact manager before it
has one distributed owner, then assembles setup receipts, per-step event
streams, checkpoint metadata, eval handlers, and a trainer that retains every
`PlannedStepResult`. `RunArtifactManager.append_metric_event()` rereads the
complete stream before each append, and most artifact mutations rewrite and
fsync the complete manifest. Distributed launches create rank-local run trees
containing substantially duplicated metrics, step histories, receipts, and
checkpoint payloads. One inspected 484-step, eight-rank run family occupied
about 554 MiB across eight run directories; about 424 MiB was checkpoint data
and about 124 MiB was duplicated training-result, metric, forward-receipt, and
diagnostic data.

The user-approved target is not an observability platform. It is a lightweight
research runtime that can answer: what configuration ran, what happened at
each train/eval step, and which adapter payload should inference load.

The following contracts remain research-visible and unchanged unless a delta
spec says otherwise:

- data rows, images, geometry, and object ordering;
- template, token, packing, position, and forward semantics;
- losses, planned-step normalization, all-rank finite decisions, optimizer and
  scheduler order;
- effective batch size, planned-step schedule, eval source, eval/checkpoint
  cadence, and metric meaning;
- existing inference loading of standard adapter and optional selected-token
  embedding-delta payloads.

## Goals / Non-Goals

**Goals:**

- Make Accelerate the single concrete execution owner for one-process and
  distributed training.
- Create one rank-zero-owned durable run tree with one wide-row
  `logging.jsonl` stream.
- Keep trainer memory and artifact volume proportional to current step state,
  scheduled eval/checkpoints, and learned payloads rather than total internal
  events multiplied by world size.
- Preserve active production experiment semantics while deleting obsolete
  config and runtime surfaces.
- Make packing caches rebuildable implementation state with strict semantic
  invalidation.
- Save only adapter and optional selected-token embedding data needed by the
  inference loader.
- Deepen ownership boundaries without introducing factories, plugin systems,
  backend registries, databases, retention engines, or general event buses.

**Non-Goals:**

- Exact optimizer, scheduler, scaler, RNG, dataloader, iterator, or sampler
  resume.
- DeepSpeed execution, configuration validation, or reserved backend status.
- A stable telemetry API, database, compression/rotation framework, adaptive
  sampling, anomaly retention, or remote artifact store.
- Rewriting historical run directories or old config archives.
- Changing model forward, data construction, objective math, metric meaning,
  eval meaning, or training budget.
- Redesigning inference/evaluation artifacts outside removal of automatic
  training-checkpoint handoff discovery.

## Decisions

### 1. One concrete Accelerate runtime owns execution mechanics

**Concept and owner.** `TrainRuntime` remains the owner of device placement,
Accelerate preparation, accumulation, collectives, all-rank finite decisions,
backward, clipping, optimizer/scheduler stepping, barriers, and rank-safe model
save mechanics. It becomes a concrete Accelerate runtime rather than a backend
selection facade.

**Caller knowledge.** The pipeline needs only runtime rank/world identity,
`is_main_process`, prepared model/optimizer access, step mechanics, metric
reduction, barriers, and safe payload-save operations. It does not need a
backend name, plugin object, DeepSpeed status matrix, or separate single-rank
implementation.

**Selected interface direction.** Construct one runtime from the resolved
Accelerate settings. One-process execution is world size one through the same
code. The LR scheduler remains CoordExp-owned and advances on the planned-step
clock. Distributed save methods are called by every rank when collectives are
required, but only rank zero materializes durable files.

After `Accelerator` construction and before model preparation, runtime MUST
accept only `DistributedType.NO` for one process and ordinary replicated
multi-GPU DDP for distributed execution. FSDP, DeepSpeed, tensor parallel, and
other wrappers selected by launcher files or environment variables fail
explicitly; "Accelerate-only" does not imply support for every Accelerate
plugin. Rank, world size, device, and main-process identity come from the
constructed `Accelerator`, not a parallel environment-variable authority.

**Alternatives considered.**

1. Keep a backend protocol with `Single`, `Accelerate`, and reserved DeepSpeed
   adapters. Rejected because only one implementation is supported and the
   interface exposes hypothetical variation.
2. Call `Accelerator` directly throughout the pipeline and trainer. Rejected
   because distributed ordering and finite-gate invariants would become caller
   knowledge.
3. Keep one concrete `TrainRuntime` around Accelerate. Selected because it
   hides real execution complexity behind a small, testable seam without
   claiming a backend framework.

### 2. The pipeline owns lifecycle; the trainer owns only planned-step execution

The pipeline remains explicit orchestration, but its phases become locally
named functions or small owned records rather than one growing function:

1. resolve and validate config;
2. initialize one Accelerate process context to establish rank, world size,
   device, and main-process ownership before run-directory creation;
3. let rank zero resolve collision policy, broadcast the selected run identity
   and path to all ranks, and create the rank-zero writer/initial run record;
4. assemble model, adapters/embedding deltas, cache/schedule, losses, and
   optimizer/scheduler using the established world size;
5. bind those training objects to the concrete `TrainRuntime`, then assemble
   eval and checkpoint handlers;
6. execute `SupervisedTrainer`;
7. finalize the compact run record or terminal failure.

This is not a dependency-injection framework. Assembly functions accept their
real dependencies explicitly and return cohesive setup records where several
values share one lifecycle. The pipeline may know which concrete modules are
assembled; it must not know their internal schema or distributed mechanics.

`SupervisedTrainer` continues to own micro-step/planned-step order. It invokes
one typed `on_completed_step(observation)` callback and direct scheduled
handlers. There are no event names, subscriptions, routing registry, or
micro-step/gate event callbacks. The trainer does not own file paths, JSON
schemas, checkpoint payload layout, or historical step storage.

### 3. A concrete run writer replaces the artifact ledger

**Concept and owner.** A single concrete run writer under `src/artifacts/`
owns the supported training run tree:

```text
run/
├── run.json
├── resolved_config.json
├── logging.jsonl
└── checkpoints/
    ├── step-<n>/
    │   ├── adapter/
    │   └── special_token_embeddings/   # optional
    ├── final.json
    └── best.json
```

Only rank zero constructs the writer. Non-main ranks receive no writer and do
not create alternate run directories. Rank zero resolves collision policy once
and broadcasts the selected run id/path before any rank builds run-associated
handlers. The pipeline centralizes optional writer ownership once when wiring
sinks and handlers rather than scattering rank guards through artifact
modules.

The writer presents a small concrete interface equivalent to:

- initialize `run.json` and `resolved_config.json`;
- append one validated logging row;
- update a small checkpoint alias;
- finalize terminal run state.

Internal implementation hides atomic JSON replacement for lifecycle documents
and aliases. `logging.jsonl` is a direct single-writer append; it does not scan
or index prior rows. Logging flushes each completed row so a process failure
loses at most an incomplete final line. There is no per-row `fsync`, rotation,
compression, retention, or repair protocol in this version.

**Run state.** `run.json` contains compact identity and lifecycle data: run id,
status/timestamps, config fingerprint/path, runtime rank/world summary,
resolved maximum steps, completed/consumed counts, final update/finite status,
terminal error summary when present, bounded warning counters, and bounded
train/eval materialization identities (cache format version, semantic
fingerprint, and determinant digest). The identity survives deletion of cache
payloads without copying cache chunks or per-presentation metadata. Final and
best checkpoint selection belongs only to their atomic aliases; `run.json`
does not duplicate selector paths or values. It is updated at initialization,
materialization binding, lifecycle changes, and completion/failure, not for
every metric.

**Logging rows.** Each completed train step produces one wide JSON object with
at least `step`, `split: train`, weighted loss/metric values, learning rates,
optimizer-update status, and finite status. Each completed eval invocation
produces one wide row with `step`, `split: eval`, trigger/eval counts, and eval
metrics. Metric producers own names and meaning; the writer validates JSON
serializability but does not interpret objectives. Before serialization, raw
NaN/Inf scalar values are represented as JSON `null` and their field names are
listed in `non_finite_fields`; finite/update status remains explicit so an
unsafe step cannot disappear from the planned-step clock.

`ForwardEvalRunner` is part of this replacement seam: it returns one compact
eval observation containing counts, triggers, and the wide scalar mapping and
does not import or call the run writer or metric-event types. The pipeline's
rank-zero eval handler appends that one canonical logging row after required
all-rank reduction. There is no second eval-summary metric schema. This keeps
eval math testable without artifact side effects and prevents the old manager
interface from surviving through eval.

**Alternatives considered.**

1. Simplify `RunArtifactManager` while preserving manifest registration and
   event types. Rejected because the old abstraction's interface is the
   receipt/registry model being removed.
2. Write files procedurally from the pipeline and handlers. Rejected because
   path, atomicity, and row-shape knowledge would spread across callers.
3. Introduce a generic event bus with pluggable sinks. Rejected because there
   is one required durable sink and no verified sink variation.
4. Use one concrete run writer. Selected because deletion of the writer would
   force real file-contract complexity back into callers, which demonstrates
   useful module depth.

### 4. Step state is ephemeral after its consumers finish

`SupervisedTrainingResult` no longer contains `step_results`. The trainer keeps
only counters, scheduled-event counts, latest logging values/status, and the
current step while its sink and scheduled handlers execute. A step observation
may contain the reduced scalar mapping and statuses needed by logging and the
scheduled checkpoint selector, but not live tensors or per-microstep Qwen
receipts.

Debug proof of packed-forward behavior is an explicitly invoked smoke/probe
output, not a normal training artifact. Existing unit and smoke tests assert
packed positions, FA2 varlen inputs, image grids, and loss mapping at their
own interfaces instead of forcing every production step to persist those
proofs.

### 5. Setup evidence is compact state, not a receipt directory

Trainable-surface, adapter target, selected-token, optimizer-group,
augmentation, pack, and schedule facts remain inspectable through the resolved
config, cache manifest, initial/final `run.json` summaries, logging rows, and
targeted test/probe artifacts. They are not independently registered receipt
files in every run.

This change distinguishes semantic validation from durable serialization:
setup code must still validate target coverage, optimizer group uniqueness,
augmentation determinism, packed alignment, and memory/attention preconditions
even when it no longer writes a receipt for each validation result.

### 6. Packing cache format is current-version-only

The cache owner continues to compute a semantic fingerprint from all inputs
that can change packed tensors or supervision, including data content,
template/order/augmentation settings, tokenizer/processor identity, packing
budget, seed inputs, and relevant source identities. Worker count remains
operational and must not change semantic identity or output order.

The reader accepts an explicit expected semantic fingerprint and accepts only
the current cache version, matching fingerprint, complete manifest, contiguous
chunk plan, declared counts, verified chunk hashes, and valid current payload
files. Direct rank/all-step load APIs MUST carry or derive that expected
fingerprint; they may not bypass identity validation merely because a manifest
is structurally complete. Any mismatch is a cache miss followed by rebuild. No
legacy optional fields, version migrations, or old payload decoders remain.
Cache files live under the cache root and are not copied into the durable run
tree. `run.json` retains only the cache format version and semantic
fingerprint/determinant digest so the consumed materialization remains
identifiable after cache cleanup.

### 7. Checkpoints are inference payloads, not handoff dossiers

Checkpoint saving writes a standard PEFT/DoRA adapter directory and, when the
configured trainable surface includes selected tokens, the compact embedding
delta directory required by `load_inference_embedding_delta`. It never writes
base-model weights. A step checkpoint directory needs no second identity graph
or per-step metadata document beyond the standard payload metadata required by
those loaders.

For the supported replicated DDP/one-process runtime, all ranks enter a
pre-save barrier, rank zero unwraps the model and calls PEFT saving with safe
serialization, the configured adapter name only, and
`save_embedding_layers=False`, then writes the optional compact embedding
delta into a staging directory. Rank zero validates the adapter safetensor has
nonempty LoRA A/B and DoRA magnitude-vector state and no full embedding,
LM-head, or base-model tensors, then atomically commits the step directory.
Rank zero catches save/validation failures and broadcasts one bounded success
or error descriptor to every rank. That failure-aware collective is the
post-save synchronization point: every rank either continues after success or
raises the same checkpoint-save contract error, so peers cannot hang behind an
unreached unconditional barrier. Failed staging directories are removed and
aliases update only after a successful commit.

`final.json` and optional `best.json` are small atomic aliases containing the
selected planned step and run-relative checkpoint directory. Best selection
uses the reduced eval metric already observed during the run, but an
update-skipped, unsafe, or non-finite planned step is ineligible by default
even when its eval value is better. A future override would have to be explicit
and recorded in the compact alias; none is introduced by this change.

Inference remains explicit: the inference config supplies base model,
`adapter.path`, and optional `embedding_delta.path`. It validates the actual
configured payloads through their loaders. Runtime no longer searches for a
neighboring `checkpoint_handoff.json`, changes composition mode based on that
file, or invokes readiness gates. Removal includes downstream inference
identity/provenance/merge fields and tests that exist only to propagate that
handoff object; actual loaded base/adapter/delta identity remains in inference
provenance. Old checkpoint directories remain loadable because their standard
adapter/delta payloads are unchanged; extra historical metadata is ignored.

The standard adapter boundary intentionally validates the configured base
identifier/model class, PEFT type, `use_dora`, target modules, tensor shapes,
and required LoRA/DoRA keys; it does not promise immutable base-config or
tokenizer hashes that standard PEFT metadata does not carry. The optional
selected-token delta retains its existing stronger base-config/tokenizer hash
checks. This is the user-approved inference-loader boundary, not a replacement
handoff dossier.

### 8. Config migration is strict and in place

The authored runtime schema removes backend selection, DeepSpeed settings, and
the separate single-runtime concept. Actual Accelerate settings remain only
where they affect supported execution. Logging cadence fields are removed
because every completed train step is logged; eval and checkpoint cadence stay
on the planned-step schedule.

Active production and smoke YAMLs are updated directly. Migration compares old
and new resolved mappings after removing only an explicit allowlist of approved
infrastructure deletions: backend/DeepSpeed selection, training-logging
cadence, and old artifact/debug receipt controls. Every other resolved value
must match, including seeds, data/order/augmentation, template, packing/cache
semantics, precision/attention, adapter seed/source, selected-token source,
losses, optimizer/scheduler, effective batch, epochs/max steps, eval cadence,
and checkpoint cadence. Unknown removed fields fail strict validation.
Historical configs are not loaded or rewritten.

Only `resolved_config.json` is written. It remains self-contained and retains
resolution provenance and path origins needed to interpret active values; a
duplicate resolved YAML rendering is removed.

### 9. Verification crosses the new interfaces

Tests assert behavior through the concrete runtime, trainer, cache, run writer,
checkpoint loaders, and public config entrypoint. They do not preserve old
call choreography or deleted receipt schemas.

The required integrated evidence is:

- targeted unit/contract tests for config, runtime, trainer, writer, cache, and
  checkpoint payloads;
- one-process Accelerate smoke through the public training entrypoint;
- multi-rank Accelerate smoke proving one run directory and rank-zero-only
  durable writes;
- expected train/eval rows and finite/update status in `logging.jsonl`;
- existing and new adapter payload loading through the inference engine;
- file-count and non-checkpoint-byte checks showing the removed multiplicative
  artifact pattern is absent;
- independent fixed-point audit before docs promotion or implementation
  approval.

## Risks / Trade-offs

- **[A process can leave a truncated final JSONL line]** → The single-writer
  reader ignores or reports one incomplete terminal line; lifecycle JSON stays
  atomic. Do not add a transactional stream to eliminate this bounded failure.
- **[Removing per-step receipts reduces post-hoc internal debugging]** → Keep
  correctness at module tests and explicit smoke/probe outputs; add diagnostic
  capture only in a future evidence-driven change.
- **[Every-step wide logging can still grow on very long runs]** → Each step is
  one compact rank-zero row with scalars only. Measure real size before adding
  sampling or rotation machinery.
- **[Rank-zero save failure can deadlock peer ranks]** → Use one pre-save
  barrier followed by a failure-aware status collective that makes every rank
  continue or raise together; inject a rank-zero save failure in a two-rank
  subprocess gate.
- **[Removing automatic handoff discovery weakens implicit mismatch checks]** →
  Inference config becomes the explicit composition authority; adapter and
  embedding loaders retain direct payload compatibility checks.
- **[Cache rebuilds cost one materialization after format changes]** → Cache is
  derived acceleration state; a clear invalidation diagnostic and deterministic
  rebuild are preferable to permanent compatibility code.
- **[Broad deletion can accidentally change scientific behavior]** → Compare
  active production config semantics, retain focused golden tests for data/
  packing/loss/schedule behavior, and require real one-/multi-rank smokes.
- **[Removing DeepSpeed reduces an apparent extension point]** → Reintroduce a
  backend seam only with demonstrated resource pressure, a separate OpenSpec,
  and real systems evidence.

## Migration Plan

1. Add replacement-facing tests, establish the Accelerate process context,
   broadcast run identity, collapse runtime branches, and migrate backend
   config fields before changing artifact ownership.
2. Atomically replace trainer events, eval side effects, run writing, and
   checkpoint/handoff handling, then delete every old production artifact
   caller in the same Wave so no dual writer or compatibility facade survives.
3. Increment cache format, enforce expected-fingerprint validation at every
   load API, and replace legacy-compatible reads with rebuild behavior.
4. Run one-process and multi-rank success/failure smokes, inference payload
   round trips, disk/file assertions, and independent audit. Update accepted
   docs only after the implementation gate passes.
5. If a Wave fails, revert that Wave's commits and retain the previous stable
   contract. Historical run trees and learned payloads are not modified by the
   migration.

## Review Resolutions

- Round 1 accepted fixes for rank-zero collision/path broadcast, cache identity
  binding, safe best-checkpoint eligibility, eval/artifact coupling, direct
  cache fingerprint validation, non-finite serialization, config semantic
  comparison, and warning ownership.
- Round 2 collapsed the implementation graph to four dependency-ordered Waves,
  made checkpoint aliases the sole selector owner, removed duplicate eval
  scalar files, narrowed adapter identity to the actual loader guarantee, and
  pinned supported Accelerate/PEFT behavior.
- Final boundary review added explicit inference payload-path deltas and a
  failure-aware distributed checkpoint commit protocol. These corrections do
  not change the approved research meaning or reintroduce heavy artifact
  machinery.

## Open Questions

None before implementation. File/class naming and exact helper decomposition
are reversible implementation decisions; any newly discovered change to model,
data, loss, metric, eval, checkpoint payload, or experiment semantics returns
to a user decision instead of being absorbed into this refactor.
