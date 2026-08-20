## Why

CoordExp-Swift already preserves one authoritative scalar row for every
completed planned training step and eval invocation, but operators still lack
a coherent live view of progress, ETA, optimization health, throughput, and
resource pressure. This change adds that operational visibility while keeping
the strict `logging.jsonl` stream—not a presentation sink—as the durable source
of truth.

## What Changes

- Sequence this change after `reconcile-coordexp-swift-training-contracts`,
  `decompose-coordexp-swift-training-orchestration`, and
  `standardize-coordexp-swift-supervised-losses` are synced and archived. It
  consumes their final execution owners and explicit
  `loss/<term>/{raw,weighted}` schema rather than creating parallel aliases.
- Pin the post-decomposition implementation owners: typed reduction in
  `src/runtime/metrics.py`, canonical row construction in
  `src/training/reporting.py`, JSONL-first derived-sink publication in
  `src/artifacts/observation_publisher.py`, and lifecycle wiring in
  `src/training/session.py`. The implementation MUST NOT move these concerns
  back into the facade or create a second owner.
- **BREAKING**: add required, no-default `observability.steps` configuration.
  It MUST be a positive planned-step interval and controls only rank-zero
  console and TensorBoard presentation; it MUST NOT suppress, sample, or change
  the canonical one-row-per-completed-step `logging.jsonl` contract.
- Keep `step` as a required key in every canonical row and presentation update,
  and present progress as the current planned step over the resolved total.
  ETA is an explicitly approximate, non-authoritative display derived from
  observed progress rather than a persisted scheduling promise.
- Extend train observations with semantically explicit optimization metrics:
  the learning rate applied by the impending optimizer update, the pre-clip
  gradient norm reduced by rank maximum, and raw plus weighted loss terms with
  their denominators. Every distributed metric receives an explicit typed
  reducer instead of relying on field-name conventions or implicit averaging.
- Make fp16 optimizer-boundary truth distributed rather than rank-local. Every
  real fp16 wrapper call converges `{all_skipped, none_skipped, mixed}` before
  scheduler or successful row handling. Uniform finite application and uniform
  GradScaler skip remain completed planned-step boundaries; a mixed or action-
  contradictory result publishes one bounded terminal row and fails the run
  before scheduler, eval, checkpoint, exact-resume, selector, or success-final
  publication. Pre-wrapper mixed/unsupported fp16 state follows the same
  terminal-receipt rule without calling the wrapper; because this decision is
  after exactly-once unscale, it preserves known untouched parameter/optimizer
  truth while marking the unfinalized composite GradScaler state unsafe.
- Consume loss telemetry only from the prerequisite result: for each actually
  computed term, preserve raw, configured weight, and weighted value. Do not
  reconstruct loss mathematics or emit fields for an omitted zero-weight
  optional term.
- Extend operational observations with throughput, input construction and
  wait time, host-to-device transfer time, bounded stage timing, current and
  peak GPU memory, retry counts, and OOM status. The existing completed-step
  and non-finite/update-status semantics remain protected.
- Add rank-zero console and TensorBoard projections at the configured
  presentation interval. Every eval observation is mirrored when eval runs.
  TensorBoard files live under the run directory and are derived only after
  the corresponding canonical JSONL row is published; a sink failure warns,
  disables that sink, and does not corrupt or fail an otherwise valid run.
- Keep MFU, TFLOPS, energy, and detailed all-rank traces probe-only until their
  measurement boundaries and calibration are validated.
- Keep RL/rollout metrics and rollout-specific logging semantics deferred.
  This change does not add an event bus, metrics database, log rotation,
  per-rank durable streams, W&B integration, or another scalar authority.
- Treat the locally installed `ms`-environment Transformers `Trainer` as a
  patched behavioral comparator only; this change neither depends on it nor
  adopts its architecture as authority.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-swift-config-runtime`: introduce the required presentation-only
  `observability.steps` field while preserving canonical every-step JSONL
  logging and the planned-step clock.
- `coordexp-swift-training-artifacts`: extend the canonical train/eval scalar
  observations and define rank-zero console/TensorBoard projections, metric
  reduction semantics, optimizer-boundary terminal receipts, sink ordering, and
  sink-failure isolation.
- `coordexp-swift-supervision-losses`: make the existing post-backward
  finite/overflow decision and fp16 wrapper result an all-rank contract, while
  distinguishing recoverable synchronized skips from terminal divergence.

## Impact

- Affected public surface: strict training YAML/resolved-config schema; every
  supported production and smoke training config must author
  `observability.steps` explicitly.
- Affected runtime surfaces: completed-step and eval observations, distributed
  metric reduction, optimizer-boundary instrumentation, loss diagnostics,
  rank-zero run writing, and terminal sink cleanup.
- Affected artifacts: additive observability fields on the prerequisite loss
  row schema, plus TensorBoard event files beneath the existing run directory.
  `logging.jsonl` remains the sole authoritative durable scalar stream.
- Affected verification and operator material: config fixtures, train/eval
  logging tests, multi-rank reduction tests, sink-failure tests, production
  vertical smoke coverage, example configs, and observability documentation.
- No new training backend, external metrics service, exact-resume promise, or
  change to data, packing, forward, loss mathematics, normal eval scheduling,
  or checkpoint selection is introduced. The only lifecycle refinement is
  fail-closed handling of a distributed optimizer outcome that has no coherent
  global update truth; it cannot be admitted as a completed boundary.
- Every GPU-backed acceptance action requires fresh user authorization and a
  predeclared quantitative execution bound; this proposal is not launch
  authority.
