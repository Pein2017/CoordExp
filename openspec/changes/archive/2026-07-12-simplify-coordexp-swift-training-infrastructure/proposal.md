## Why

CoordExp-Swift's training path is functional, but its infrastructure currently
persists the same step and checkpoint evidence through rank-local run trees,
per-metric streams, per-step receipts, full-history training results, and
duplicated handoff metadata. This change makes the supported path a lightweight
Accelerate-only research runtime whose durable output is proportional to the
experiment evidence actually needed for comparison and inference.

## What Changes

- **BREAKING** Use Accelerate as the only training runtime for both one-process
  and distributed launches; remove the separate `single` runtime and all
  DeepSpeed schema, config, plugin, status, smoke, and execution surfaces.
- **BREAKING** Create one rank-zero-owned run directory per launch containing a
  compact `run.json`, one self-contained `resolved_config.json`, one wide-row
  `logging.jsonl`, and inference-loadable checkpoints. Remove rank-local
  run-directory copies, separate eval-summary metric files, `metrics/`, receipt
  registries, per-step Qwen receipt files, default progress streams, and the
  full step-history `training_result.json`.
- Emit one `logging.jsonl` row for every completed train step and one row for
  every completed eval invocation, using the planned training step as the
  shared clock and recording skipped/non-finite update status explicitly.
- Bound trainer result ownership to terminal counters and the latest logging
  state instead of retaining every step result in memory.
- **BREAKING** Treat packing-cache files as disposable acceleration state:
  validate current version, semantic fingerprint, completeness, and ordering,
  but reject and rebuild old payload versions instead of supporting legacy
  manifests or payload readers.
- **BREAKING** Save only checkpoint payloads required by inference: the standard
  PEFT/DoRA adapter and, when configured, the selected-token embedding delta.
  Remove new `checkpoint_handoff.json` generation, automatic neighboring
  handoff discovery, readiness gates, and duplicated checkpoint identity
  receipts. Existing adapter and embedding-delta directories remain directly
  loadable through explicit inference config paths.
- Migrate checked-in `configs/coordexp_swift/prod/` and smoke profiles directly
  to the new strict schema while preserving active production experiment
  semantics. Do not add compatibility aliases for removed runtime or artifact
  fields.
- Preserve data, geometry, template, token alignment, packed-forward, loss,
  optimizer, planned-step, eval, and checkpoint-cadence semantics. Exact
  optimizer/scheduler/scaler/RNG/dataloader/iterator training-state resume
  remains out of scope.
- Verify the redesign with one-process and multi-rank Accelerate smokes, a
  representative existing adapter load, a newly written checkpoint load, and
  file-count/disk-size assertions for the new run tree.

## Capabilities

### New Capabilities

None. This change deliberately simplifies the existing CoordExp-Swift
training capabilities instead of introducing a new framework or telemetry
subsystem.

### Modified Capabilities

- `coordexp-swift-config-runtime`: Make Accelerate the sole runtime, write one
  resolved JSON config, and remove DeepSpeed/separate-single and durable-receipt
  requirements while preserving active production experiment semantics.
- `coordexp-swift-training-artifacts`: Replace the manifest/receipt/metric-event
  estate with a single rank-zero run record, wide-row `logging.jsonl`, bounded
  trainer results, and minimal checkpoint aliases.
- `coordexp-swift-pack-cache-semantic-identity`: Make cache payload versions
  disposable and rebuild-only while preserving strict semantic invalidation,
  completeness, and ordering checks.
- `coordexp-swift-checkpoint-handoff-readiness`: Remove the handoff manifest and
  readiness-gate capability; explicit inference configuration plus standard
  adapter and optional embedding-delta payloads become the supported loading
  boundary.
- `coordexp-swift-infer-config-runtime`: Replace `checkpoint-final` metadata
  resolution with explicit adapter and optional embedding-delta paths and the
  exact identity guarantees of their real loaders.
- `coordexp-swift-infer-benchmark-smoke`: Point adapter-enabled smoke directly
  at adapter and optional embedding-delta payloads instead of checkpoint-final
  metadata.
- `coordexp-swift-adapters-embeddings-optim`: Preserve trainable-surface and
  optimizer semantics without requiring a durable subsystem receipt graph.
- `coordexp-swift-packing-forward`: Keep packed-forward correctness and
  optional targeted debug proof while removing normal-run per-step forward
  receipts.
- `coordexp-swift-supervision-losses`: Preserve all-rank loss and finite-gate
  semantics while reporting compact step diagnostics through `logging.jsonl`
  rather than durable rank-local receipt families.
- `coordexp-swift-geometry-augmentation`: Preserve deterministic augmentation
  and ordering evidence in compact setup/cache provenance without per-example
  durable receipt expansion.
- `coordexp-swift-vertical-smoke`: Replace the receipt-heavy and DeepSpeed
  smoke requirements with one-process and multi-rank Accelerate artifact,
  logging, checkpoint-load, and bounded-output gates.

## Impact

- Primary source areas: `src/training/`, `src/runtime/`, `src/artifacts/`,
  `src/config/`, `src/inference/`, and `src/training/pack_cache.py`.
- Config areas: active `configs/coordexp_swift/prod/` and
  `configs/coordexp_swift/smoke/`; `configs/coordexp_swift/deepspeed/` is
  removed.
- Verification areas: `tests/training/`, `tests/runtime/`, `tests/artifacts/`,
  `tests/config/`, `tests/inference/`, and focused real Accelerate smokes.
- Operator docs are updated only after implementation and verification so
  `docs/` remains the accepted architecture snapshot rather than active
  project state.
- Historical artifacts and configs are not rewritten. Old adapter and optional
  embedding-delta payloads remain usable by explicit path; old cache and
  observability formats are not compatibility targets.
