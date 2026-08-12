## Context

See [proposal.md](proposal.md) for motivation. After the prerequisite
decomposition, `CompletedStepReporter` consumes one
`CompletedStepObservation`, reduces typed metrics, and appends through
`RunWriter`. This preserves a useful canonical stream, but three remaining
contract gaps block the requested behavior:

- reducer meaning is partly inferred from key names and partly selected by an
  eval mode branch;
- learning rates are currently obtained from the post-update scheduler receipt,
  so they can describe the next update rather than the update just applied;
- row construction, publication, and derived-sink failure semantics are not yet
  split across their final narrow owners.

The prerequisite loss change exposes finalized per-term telemetry and global
planned-step denominators. The gradient finite gate already observes a rank-local pre-clip
norm and produces an all-rank maximum. These are the calculation authorities;
the observability path must reuse them rather than recalculate objective or
gradient semantics. `RunWriter.append_logging_row` remains the sole canonical
scalar append. Exact-resume config projection and the Wave 7 comparator already
separate semantic results from wall-clock/resource observations and must be
extended for the new presentation-only surface.

This design is a sequential successor, not a parallel rewrite. It begins only
after the contract-reconciliation, orchestration-decomposition, and supervised-
loss changes are synced and archived. In particular, the loss predecessor owns
the breaking migration to `loss/<term>/raw` and `loss/<term>/weighted`; this
change consumes those names and does not restore the removed ambiguous alias.
Before implementation, its `Wide-Step Logging Stream` delta is rebased against
the complete stable requirement produced by syncing the supervised-loss
change. The rebase must retain every post-loss paragraph and scenario, then add
only this change's observation and presentation behavior.

The final owner map is fixed: `src/runtime/metrics.py` owns metric value types
and collective reduction; `src/training/reporting.py` owns canonical train/eval
row construction; `src/artifacts/observation_publisher.py` owns JSONL-first
publication plus console/TensorBoard lifecycle; and `src/training/session.py`
wires those owners. No facade or generic coordinator becomes a fifth owner.

## Goals / Non-Goals

**Goals:**

- Make every distributed scalar carry explicit reduction and availability
  meaning from producer to collective.
- Capture optimization values at the boundary where they are true, especially
  applied LR and pre-clip norm, without another gradient collective.
- Give rank zero one small publisher that enforces JSONL-first ordering and
  owns console/TensorBoard lifecycle and failure isolation.
- Keep normal observation overhead bounded to existing collectives, monotonic
  clock reads, allocator queries, and presentation work only at the configured
  interval.
- Keep exact-resume comparison strict for model/optimizer/loss outputs while
  excluding presentation cadence, ETA, timing, and resource noise.

**Non-Goals:**

- A generic metrics registry, event bus, callback subscription system, tracing
  framework, metrics database, W&B adapter, or per-rank log product.
- Changing loss mathematics, optimizer/scheduler policy, eval schedule,
  checkpoint selection, or exact-resume state payloads.
- Claiming calibrated MFU, TFLOPS, energy, or GPU-kernel-stage duration from
  host enqueue timing.
- Reproducing `transformers.Trainer`; the patched local implementation is only
  a behavior comparator for operator ergonomics.

## Decisions

### 1. Use typed metric samples, not key-name dispatch or registration

Add a narrow runtime value model with two sample forms:

- a scalar sample carries `name`, finite value or explicit unavailability,
  and one reducer from `SUM`, `MAX`, `IDENTICAL`, or `BOOL_ALL`;
- a ratio sample carries `name`, an exact integer or float numerator, an exact
  positive denominator, and sum-before-divide semantics.

A `MetricBatch` also carries planned step, split, and whether a conditional
field is required on the active backend. Rank reports must agree on metric
names, sample form, reducer, and required/conditional status before values are
reduced. There is no mutable registry and no fallback reducer. Train and eval
builders construct the closed batch directly from their typed observations;
adding a field therefore requires choosing its reducer at the call site and a
test for asymmetric ranks.

This replaces the current fixed name tables and implicit plain-mean fallback in
`src/runtime/train_runtime.py`. It also replaces eval's key-suffix inference:
replicated eval produces `IDENTICAL` samples, while disjoint eval produces
`SUM`/ratio samples from its sufficient statistics. Per-rank reports may remain
ephemeral inside reduction and bounded lifecycle accounting, but normal JSONL
rows do not serialize `per_rank_measurement`.

Alternatives rejected:

- Expanding key-name tables: small initially, but still lets an unclassified
  scientific metric silently become a mean.
- A global plugin registry: adds registration order, mutation, and discovery
  problems without a current extension requirement.
- Gathering arbitrary dictionaries and reducing on rank zero: preserves the
  ambiguity and makes producer tests unable to prove reducer intent.

### 2. Consume semantic loss telemetry without reconstructing it

`src/training/reporting.py` consumes the prerequisite completed loss telemetry;
it does not consume local backward tensors or reconstruct a loss from
`segment_mean_numerator`, denominators, or `backend_gradient_scale`. For every
term that the loss predecessor actually computed, the reporter preserves raw
value, configured weight, and weighted value and may attach already-produced
counts/denominators. A zero-weight protected gate remains computed and is
preserved; an omitted zero-weight optional term remains absent as a complete
field family.

The prerequisite loss contract already owns `loss/<term>/raw` and
`loss/<term>/weighted` and removes the ambiguous `loss/<term>` alias.
The reporter preserves those semantic names and adds only the
configured weight, denominator scope, eligible segments, selected atoms, and
skipped segments needed to interpret them. Accuracy continues to use summed
integer correct/atom statistics. Work-rate fields use summed physical-token,
supervised-atom, and pack counts divided by the rank-max
`step_duration_seconds`; the observation builder obtains physical-token and
pack counts before step-local tensors are released.

Alternative rejected: derive raw loss again from internal sufficient
statistics. That would create a second semantic owner and permit reporting and
objective finalization to diverge.

### 3. Capture LR and gradient norm at their authoritative boundaries

The runtime owns one `AppliedUpdateReceipt` type for all optimizer outcomes and
the trainer only carries that receipt into reporting. The receipt exposes the
planned-step clock, whether the optimizer wrapper was called, nullable global
application/skip truth, mutation state, and whether the boundary is terminal.
The post-backward gate owns one closed normal `optimizer_boundary_action` with
exactly three values: `apply`, `scaler_skip`, and `not_attempted`. The action is
reduced before any rank enters the wrapper, so all ranks take the same branch.
A mixed/unsupported fp16 pre-wrapper state converges one terminal unsafe receipt
instead of growing the normal-action enum or raising rank-locally.
The same runtime type provides explicit `terminal_not_attempted(...)` and
`terminal_post_wrapper(...)` constructors; reporter/session code never invents
receipt booleans from an exception.

Supported `not_attempted` is used for pre-backward rejection and retained
bf16/non-scaler post-backward rejection. It constructs
`AppliedUpdateReceipt.not_attempted(planned_step_id, group_count, reason)` with
`attempted: false`, `applied: false`, `step_was_skipped: false`, and one JSON-null
LR slot per configured group. `scaler_skip` is narrower: after exactly-once
fp16 unscale, every rank must report an active scaler and a non-finite current
unscaled gradient, with no unrelated report failure. This branch skips
clipping but calls the optimizer wrapper exactly once on every rank only to let
GradScaler suppress the underlying optimizer mutation and update its scale.
A mixed-rank overflow candidate or unrelated unsafe fp16 state after unscale is
terminal before the wrapper and records known
`attempted=false/applied=false/step_was_skipped=false` truth. Although the
underlying optimizer and parameters are untouched, exactly-once unscale has
already moved GradScaler's per-optimizer state to `UNSCALED` and populated its
`found_inf` record. Because that scaler state is unfinalized and may differ by
rank, `terminal_not_attempted(...)` records the existing bounded composite
`mutation_state=divergent_or_unknown`, never `unchanged`.

For every real fp16 wrapper call, immediate rank-local
`accelerator.optimizer_step_was_skipped` values are authoritative inputs, but no
rank may interpret or raise from one alone. Before scheduler or successful row
handling, runtime uses one bounded correctness consensus to classify them as
`all_skipped`, `none_skipped`, or `mixed`. `apply` accepts only
`none_skipped`; `scaler_skip` accepts only `all_skipped`. A mixed result records
`attempted=true`, nullable global `applied/step_was_skipped`, null LRs, and
`mutation_state=divergent_or_unknown`. Unanimous contradictory outcomes retain
known truth: `apply + all_skipped` records applied false/skipped true, while
`scaler_skip + none_skipped` records applied true/skipped false and corrupted or
unsafe mutation state together with the identical pre-call LRs actually used.
All three contradictions are terminal. Pre-call overflow
prediction must not read a previous step's skip flag. The runtime samples each
optimizer param-group LR immediately before the call, but publishes those values
as applied LR only when every rank is known to have applied that value: accepted
`apply + none_skipped`, or the terminal corrupted `scaler_skip + none_skipped`
outcome. A scheduled, merely attempted, skipped, or rank-divergent value is never
labeled applied.

The scheduler advances exactly once for completed `apply`, `scaler_skip`, and
supported `not_attempted` planned-step boundaries. Eval and checkpoint triggers
also use that clock. `optimizer_step_count` retains its current meaning as every
completed optimizer-wrapper invocation, including accepted scaler skip and any
post-wrapper terminal outcome; pre-wrapper `not_attempted` or terminal unsafe
does not increment it. A terminal boundary retains the current planned-step id
but does not increment completed-step or scheduler-step counts or dispatch eval,
checkpoint, exact-resume, selector, final-success, or later-step handlers. The
receipt and `optimizer_update_status`, not a new durable applied counter, own
whether that step actually applied an update.

Every terminal boundary clears gradients and first attempts to
publish/converge exactly one terminal unsafe train row. The row carries the
truthful nullable receipt fields, bounded terminal reason, mutation state, and
current counters. Only then does every rank converge failed run finalization.
If row publication also fails, finalization retains the optimizer-boundary code
as primary. Clearing gradients is cleanup, never a claim that possible rank-
selective parameter or scaler mutation was repaired.

Before gradient finiteness, norm, or clipping is inspected under fp16,
`TrainRuntime.post_backward` invokes
`accelerator.unscale_gradients(optimizer)` exactly once. It then remains the
sole owner of current-gradient finiteness and pre-clip norm calculation. The
`apply` branch follows with a non-unscaling clipping primitive over the already-
unscaled gradients (for example `torch.nn.utils.clip_grad_norm_`). The
`scaler_skip` branch does not clip a known non-finite gradient; it proceeds only
to the wrapper finalization described above. No branch may call
`accelerator.clip_grad_norm_`, because that API may unscale again. The reporter
copies the runtime's already reduced `max_grad_norm` diagnostic into the
completed observation before any clip. The observation field is therefore the
maximum of rank-local finite pre-clip norms, or explicitly non-finite/unavailable
on an unsafe branch, and does not initiate a second gradient collective. Every
fp16 wrapper branch uses one bounded post-call boolean consensus. It is an
optimizer-correctness collective required before scheduler progression, not a
metric/observability collective.

Alternative rejected: sample LR after `scheduler.step` or recompute a norm in
the logging handler. Both create off-by-one or duplicate-collective failure
modes.

### 4. Measure only scopes that can be named honestly

Keep the existing monotonic CPU build, input-wait, and planned-step wall
timers. Add exact work counts to the completed observation. For provider-owned
host-to-device copies, bracket the real copy with CUDA events and resolve the
elapsed time only after the forward has necessarily consumed the input; do not
add a synchronization solely to make the timer ready. CPU runs and paths where
completion cannot be established mark `input_h2d_seconds` unavailable. Only
the retained synchronous and explicit experimental overlapped providers may
produce such a receipt; host enqueue duration is not relabeled as H2D
execution.

Extend the existing resource collector with one process-local CUDA allocator
sample at the completed-step boundary:

- current allocated and reserved bytes;
- process-lifetime peak allocated and reserved bytes;
- deltas since the prior completed observation for allocator retries and OOMs.

Bytes use rank maximum; counter deltas use rank sum. CUDA-inapplicable fields
are omitted and listed in `unavailable_fields`. The normal path does not reset
PyTorch peak statistics because that would mutate measurement state shared by
other resource receipts. More detailed stage timers remain probe-only until a
specific scope can be measured without changing the step boundary.

Alternative rejected: call `torch.cuda.synchronize()` around every stage. It
would improve timer readability by changing the execution being measured.

### 5. Put row construction and presentation behind one deep publisher

`src/training/reporting.py` accepts a reduced train or eval observation and
returns the strict canonical row. `src/artifacts/observation_publisher.py` owns:

1. the existing all-rank shared status choreography around rank-zero
   `RunWriter.append_logging_row`;
2. rank-zero console formatting;
3. a lazily created `torch.utils.tensorboard.SummaryWriter` rooted at
   `<run_dir>/tensorboard`;
4. terminal close and a one-way TensorBoard-disabled latch.

The publisher has one direct method for a completed train/eval row. It is not
an event bus: no event strings, subscribers, registry, or dynamic sinks. JSONL
publication and its distributed success handshake complete first. Only then
does rank zero present a train row when its step is on cadence or terminal, and
every eval row when it runs.

TensorBoard tags are deterministic `train/<canonical-key>` and
`eval/<canonical-key>` projections of finite numeric row fields, using the
required row `step` as `global_step`. Strings, nulls, nested diagnostic values,
and approximate ETA are not synthesized into scalar tags. TensorBoard import/
initialization, `add_scalar`, `flush`, and `close` are each inside the same
one-way failure contract. The first failure records at most one bounded warning
through the run writer plus one stderr warning when possible, best-effort
closes/discards the writer, and latches the sink off; a close failure cannot
recurse or emit another warning. It is never broadcast as a training failure
and never changes the canonical row.

`unavailable_fields` and `non_finite_fields` are sorted, unique field-name
lists with at most 256 names and at most 256 UTF-8 bytes per name. Overflow is
represented by the corresponding bounded `*_truncated_count`, never by an
unbounded list or arbitrary exception text.

The console presenter prints a compact step/total line with the most useful
available loss, LR, pre-clip norm, throughput, memory, and status fields. ETA is
computed in memory from segment-local monotonic elapsed time and completed
planned-step progress, labeled approximate, and discarded on process exit or
resume.

Alternatives rejected:

- Put all behavior in `RunWriter`: this would mix strict artifact
  serialization with interval policy, terminal UI, TensorBoard dependency, and
  mutable sink health.
- Let the trainer call console and TensorBoard directly: this makes the loop
  own artifact schemas and failure policy.
- Write TensorBoard first: a later JSONL failure could leave a derived event
  that has no canonical observation.

### 6. Make presentation config required but non-semantic for continuation

Add required `ObservabilityConfig(steps: int > 0)` to `TrainConfig`, with no
default factory. Update every supported training YAML under
`configs/coordexp_swift/prod/` and `configs/coordexp_swift/smoke/`; inference
and historical config roots are untouched. No alias or migration fallback is
accepted.

Exclude the `observability` block alongside `run` and `resume` in exact-resume
semantic projections and in the three-run input-attestation projection. Update
the Wave 7 log comparator so new timing/resource fields are classified as
excluded observations while loss, accuracy, applied LR, pre-clip norm, finite
status, and optimizer-update status remain strict comparison fields.
TensorBoard files, console text, presentation cadence, and ETA are not inputs
to exact-resume comparison.

Alternative rejected: include presentation cadence in resume identity. It
would reject an otherwise exact continuation for a setting that cannot affect
forward, backward, optimizer, scheduler, data order, or RNG state.

### 7. Verify semantics with small executed probes before a production smoke

The minimum decision-bearing probes are:

- a CPU optimizer/scheduler probe that proves the logged LR was sampled before
  the applied update and not after scheduler advance;
- genuine CUDA fp16 finite and overflow arms through real Accelerate: both prove
  the sole unscale/finite/norm authority; the finite arm then performs the one
  non-unscaling clip and proves an applied update with pre-call LR, while the
  all-rank-confirmed overflow arm proves `scaler_skip`, performs no clip, calls
  the wrapper once only for scaler finalization, observes post-call
  `step_was_skipped`, publishes no applied LR, counts the completed wrapper
  invocation, and confirms planned-step scheduler progression without claiming
  an underlying optimizer update;
- injected two-rank fp16 outcome fixtures prove `mixed`, `apply+all_skipped`,
  and `scaler_skip+none_skipped` converge one truthful terminal row before
  failed finalization, never advance scheduler/scheduled handlers, and never
  raise rank-locally before consensus;
- a real two-process Gloo probe with intentionally asymmetric rank-local norms,
  counts, ratios, and timings that proves schema agreement and MAX/SUM/ratio
  reducers without a CUDA dependency;
- a TensorBoard event-reader test that recovers tags, values, and global step
  from a temporary run directory;
- injected initialization/write/close failures proving JSONL-first durability,
  one warning, sink disablement, and continued later rows;
- the existing exact-resume comparator with presentation-only config drift and
  new observational fields;
- one production-shaped distributed vertical smoke after unit/integration
  gates, checking one shared run tree and no rank-local TensorBoard files.

The local `ms` Transformers trainer may inform expected console ergonomics, but
no test imports or snapshots its patched source.

## Risks / Trade-offs

- **[Risk] Typed samples touch both train and the already subtle replicated vs
  disjoint eval reduction.** → Preserve current eval sufficient statistics,
  migrate one reducer family at a time, and require full-row replicated/sharded
  equivalence tests before removing the old path.
- **[Risk] Loss reporting can accidentally become a second semantic owner or
  include DDP backend compensation.** → Consume the prerequisite finalized
  raw/weight/weighted telemetry without re-derivation and compare complete
  single-rank and multi-rank rows.
- **[Risk] TensorBoard's background writer can surface errors later than
  `add_scalar`.** → Catch initialization, add, flush, and close separately;
  force flush in failure-injection tests; keep JSONL publication independent.
- **[Risk] fp16 wrapper calls can look like applied updates even when GradScaler
  skips them.** → Treat the post-call `step_was_skipped` flag as authoritative,
  keep attempt/application truth in the single runtime receipt and status,
  preserve the existing wrapper-invocation counter, allow wrapper finalization
  only for an all-rank-confirmed scaler-overflow candidate, and forbid an fp16
  correctness claim until genuine CUDA finite and overflow arms pass.
- **[Risk] A mixed-rank fp16 overflow candidate can make some wrappers skip while
  others mutate parameters, or leave GradScaler state unfinalized.** → Reduce
  the closed boundary action before the call, terminate on mixed or unrelated
  unsafe fp16 state after unscale, and require post-call boolean consensus for
  every fp16 wrapper invocation before planned-step progression. Preserve known
  unanimous contradictory truth; only a mixed result uses nullable global
  application state. For a pre-wrapper terminal, preserve known no-wrapper/no-
  parameter-mutation truth while marking the already-unscaled, unfinalized
  composite scaler state `divergent_or_unknown` rather than `unchanged`.
- **[Risk] CUDA event timing may be unavailable or tempt a synchronization.** →
  Omit and declare unavailable unless completion is already established; never
  emit zero or force a per-step synchronization for observability.
- **[Risk] Allocator counters are process-lifetime counters.** → Log explicit
  deltas from a retained prior snapshot and process-lifetime peak byte fields;
  do not imply that the peak is step-local.
- **[Risk] Making config required invalidates fixtures and unattended launch
  configs.** → Inventory and resolve every active train YAML in CI, update
  fixture factories explicitly, and reject rather than default old configs.
- **[Risk] Console formatting or sink writes add rank-zero latency.** → Perform
  them only at the configured interval/eval/terminal boundary, use the
  TensorBoard writer's bounded asynchronous queue, and include sink overhead
  outside `step_duration_seconds`.

## Migration Plan

1. Verify the three prerequisite changes are synced and archived, record their
   exact commits, rebase this delta against the complete post-loss stable
   `Wide-Step Logging Stream`, and characterize the four fixed owners.
2. Introduce the required typed config and update all active train configs and
   fixture factories in the same slice; verify every active config resolves.
3. Introduce typed metric samples/reduction in `src/runtime/metrics.py` behind
   unit and two-rank Gloo
   probes, then migrate train and eval producers while retaining current row
   keys.
4. Correct pre-wrapper and post-call update receipts/LR, preserve existing
   counter semantics, and enforce the closed `apply` / `scaler_skip` /
   `not_attempted` boundary, exactly-once unscale, action-specific clipping, and
   all-fp16-wrapper outcome consensus with terminal-row finalization;
   add
   work/timing/resource fields while consuming, never reconstructing, the
   prerequisite raw/weight/weighted loss schema.
5. Introduce row construction in `src/training/reporting.py`, JSONL-first
   publication in `src/artifacts/observation_publisher.py`, and lifecycle wiring
   in `src/training/session.py`; delete obsolete reducer tables/pass-through
   helpers without moving behavior into the facade.
6. Update exact-resume projections/comparator, operator docs, and artifact
   inventory; run focused suites, strict OpenSpec validation, a production-
   shaped vertical smoke, and independent audit.

Freeze an exact command manifest before implementation, recording cwd,
environment, command, config, world size/devices, artifact roots, expected
evidence, and quantitative limits. Every GPU-backed action requires fresh user
authorization plus declared bounds for planned steps, model forwards, wall
time, peak GPU memory, cache/materialization passes, and artifact bytes. A
bound violation stops the action. The genuine CUDA fp16 finite and overflow
arms are both mandatory before claiming fp16 correctness; CPU/Gloo evidence
cannot replace them.

Rollback requires reverting schema/code/config changes together. Existing
`logging.jsonl` remains readable because additions preserve prior field types;
run-local `tensorboard/` directories are derived artifacts and may remain after
rollback without becoming authority.
