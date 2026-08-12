## MODIFIED Requirements

> Application precondition: before implementation, this complete modified
> requirement MUST be rebased against the stable `Wide-Step Logging Stream`
> produced by syncing `standardize-coordexp-swift-supervised-losses`; every
> post-loss paragraph and scenario is retained, and this delta remains additive.

### Requirement: Optimizer-Step Order

The trainer/runtime boundary SHALL use the canonical step order: build or fetch
packed sequence, move tensors, Qwen forward, construct loss context, compute
loss bundle, pre-backward finite check, backward, exactly-once fp16 unscale when
applicable, all-rank current-gradient/action decision, action-specific clipping
and wrapper call, all-rank fp16 post-wrapper outcome consensus when a wrapper
was called, update receipt, scheduler step for a completed boundary, zero
gradients, and metrics/artifacts. No rank may enter the wrapper before the
pre-call action converges or raise from its local post-call skip flag before the
post-call consensus converges.

`apply + none_skipped`, `scaler_skip + all_skipped`, and a supported
`not_attempted` decision are completed planned-step boundaries. The first uses
the finite already-unscaled gradients, applies one non-unscaling clip when
configured, and calls the wrapper. The second does not clip and calls the
wrapper only to finalize GradScaler's recorded skip. The third does not call the
wrapper; supported cases include pre-backward scalar rejection and retained
bf16/non-scaler post-backward rejection. All three clear gradients, advance the
scheduler exactly once, emit one completed train row, and dispatch scheduled
handlers by the original planned-step id. The wrapper-invocation counter
increments only for the first two.

A mixed/unsupported fp16 pre-wrapper state converges a terminal unsafe receipt
after exactly-once unscale without calling the wrapper. Its underlying optimizer
and parameters are untouched, but GradScaler's per-optimizer stage and
`found_inf` record are already changed and may be unfinalized or differ across
ranks; its composite `mutation_state` MUST therefore be
`divergent_or_unknown`, never `unchanged`. A post-wrapper `mixed` result, `apply +
all_skipped`, or `scaler_skip + none_skipped` converges a terminal receipt after
every rank's wrapper returned. Every terminal branch clears gradients and
publishes/converges one terminal train row before common failed finalization,
but does not advance scheduler, completed-step count, eval, checkpoint,
exact-resume, selector, final-success, or later-step handlers. A mixed
post-wrapper result records nullable global application/skip truth; unanimous
contradictory results retain their known booleans. Gradient clearing is cleanup
and MUST NOT be represented as repairing rank-selective parameter or scaler
mutation.

#### Scenario: Recoverable unsafe step after backward

- **WHEN** a retained bf16/non-scaler post-backward gate chooses supported
  `not_attempted`
- **THEN** runtime MUST skip the optimizer wrapper and clear gradients
- **AND** still complete the scheduler and planned-step lifecycle event with
  synchronized update status.

#### Scenario: Uniform fp16 scaler overflow

- **WHEN** every rank selects `scaler_skip` and post-call consensus is
  `all_skipped`
- **THEN** no underlying optimizer mutation is applied
- **AND** runtime MUST complete the scheduler and planned-step lifecycle event
  with synchronized skipped status.

#### Scenario: Terminal optimizer-boundary inconsistency

- **WHEN** pre-wrapper fp16 state is mixed/unsupported or the post-wrapper
  consensus contradicts the selected action
- **THEN** runtime MUST produce one synchronized terminal receipt and train row
- **AND** MUST clear gradients and fail commonly without scheduler or scheduled-
  handler progression.

### Requirement: Wide-Step Logging Stream

Rank zero SHALL append one compact JSON object to `logging.jsonl` for every
completed planned optimizer-step boundary and every completed eval invocation.
This includes recoverable completed boundaries whose optimizer update is
skipped. In addition, a distributed optimizer boundary that becomes terminal
before completion MUST append exactly one terminal train row for its current
planned-step id before failed run finalization. That exception does not make the
boundary completed or advance its schedule. A train row MUST use
`split: "train"`; an eval row MUST use `split: "eval"`; both MUST contain the
required planned training `step`. Each row SHALL store the complete scalar
logging mapping for that observation together instead of writing one record per
metric.

Train rows MUST include the weighted total loss; for every actually computed loss
term, its raw value, configured weight, weighted value, denominator scope,
eligible-segment count, selected-atom count, and skipped-segment count; top-level
`acc_top1` and `acc_top5`; optimizer-update and finite status; and optimizer
group learning rates. Every computed term MUST retain the prerequisite
`loss/<term>/raw` and `loss/<term>/weighted` fields; the removed ambiguous
`loss/<term>` alias MUST NOT be restored. Additive observability fields SHALL
expose the corresponding configured weight and denominator inputs. An
`lr/group_<index>` value MUST be sampled
immediately before the optimizer-step call and, when the update is applied,
MUST be the exact learning rate used for that update rather than the value
produced by the subsequent scheduler advance. When an update is skipped, each
configured optimizer-group LR field MUST be JSON `null`, named in
`unavailable_fields` as not applied, and remain distinguishable through
`optimizer_update_status`; a scheduled or would-have-been value MUST NOT be
labeled as applied.

Loss telemetry MUST be consumed from the completed loss result owned by the
supervised-loss contract. Reporting MUST NOT reconstruct raw or weighted loss
from backend-scaled backward tensors or independently reduce loss sufficient
statistics. A computed zero-weight protected gate MUST retain its raw, weight,
weighted-zero, count, denominator, and finite fields; a zero-weight optional
term omitted by the loss owner MUST have none of those fields.

One runtime-owned `AppliedUpdateReceipt` MUST represent both pre-wrapper and
post-wrapper outcomes. The post-backward all-rank decision MUST expose one
closed `optimizer_boundary_action` with exactly `apply`, `scaler_skip`, and
`not_attempted`; every rank MUST take the same action before any rank enters the
optimizer wrapper. A mixed or unsupported fp16 pre-wrapper state MUST instead
produce the same terminal unsafe decision on every rank without being added as
a fourth normal action. When the action is `not_attempted`,
`AppliedUpdateReceipt.not_attempted(planned_step_id, group_count, reason)` MUST
produce `attempted: false`, `applied: false`, `step_was_skipped: false`, and one
JSON-null LR per configured optimizer group.

For fp16, `scaler_skip` MUST be selected only after exactly-once unscale when
every rank reports an active GradScaler and a non-finite current unscaled
gradient, with no unrelated report failure. It MUST skip clipping and call the
optimizer wrapper exactly once on every rank only so GradScaler can suppress
the underlying optimizer mutation and update its scale.

Every real fp16 wrapper call, including `apply`, MUST then converge the immediate
post-wrapper skip booleans across all ranks before scheduler or nonterminal row
handling. The consensus result is exactly `all_skipped`, `none_skipped`, or
`mixed`. `apply` accepts only `none_skipped`; `scaler_skip` accepts only
`all_skipped`. An accepted scaler skip MUST expose null applied-LR fields with
explicit unavailability. `mixed` MUST produce a terminal receipt with
`attempted: true`, `applied: null`, `step_was_skipped: null`, null LRs, and
`mutation_state: divergent_or_unknown`. A unanimous result that contradicts the
expected action is also terminal but MUST preserve its known truth:
`apply + all_skipped` records `applied: false` and `step_was_skipped: true`,
while `scaler_skip + none_skipped` records `applied: true` and
`step_was_skipped: false` plus the pre-call LRs that were in fact applied. A
mixed-rank scaler-overflow candidate or unrelated unsafe fp16 state before the
wrapper MUST record `attempted: false`, `applied: false`,
`step_was_skipped: false`, null LRs, and
`mutation_state: divergent_or_unknown`. This state truthfully distinguishes no
underlying optimizer/parameter mutation from the already-unscaled, unfinalized
GradScaler state. Pre-call logic MUST NOT use a previous step's skip flag as
current overflow truth, and no rank may raise from its local post-call flag
before the shared consensus.

Under fp16, `accelerator.unscale_gradients(optimizer)` MUST occur exactly once,
followed by finite inspection and pre-clip norm calculation. The `apply` branch
MUST then use a non-unscaling clip primitive over the already-unscaled
gradients. The `scaler_skip` branch MUST NOT clip a known non-finite gradient
before its wrapper finalization. No branch may call
`accelerator.clip_grad_norm_`. Scheduler, eval, and checkpoint progression MUST
remain on the planned-step clock both when the finite gate chooses
`not_attempted` and when an all-rank-confirmed fp16 overflow completes
`scaler_skip`.
`optimizer_step_count` MUST retain its existing meaning: every completed
optimizer-wrapper invocation, including accepted `scaler_skip` and any post-
wrapper terminal outcome, increments it; pre-wrapper `not_attempted` or terminal
unsafe does not. `scheduler_step_count` MUST count the once-per-completed-
planned-step scheduler advances and MUST NOT increment for a terminal boundary.
The receipt and
`optimizer_update_status` MUST own actual application; exact-resume MUST retain
those meanings and this change MUST NOT silently reinterpret a counter or add a
redundant durable applied-update counter. All fp16 wrapper calls MUST use one
bounded post-wrapper boolean consensus for optimizer correctness. This is not a
metric/observability collective and MUST complete before scheduler or row
handling.

A pre-wrapper terminal unsafe receipt or post-wrapper terminal receipt MUST
clear gradients, retain the current planned-step id, and publish/converge one
terminal unsafe train row before common failed finalization. That row MUST
include the action or terminal reason, `attempted`, truthful nullable `applied`
and `step_was_skipped`, `mutation_state`, finite/update status, null/unavailable
LRs unless every rank is known to have applied the same pre-call values, and the
current optimizer/scheduler counters. Such a boundary MUST NOT
advance scheduler, completed-step count, eval, checkpoint, exact-resume,
selector, final-success, or later-step handlers. If terminal-row publication
itself fails, all ranks MUST still converge failed finalization and preserve the
primary optimizer-boundary failure code ahead of the publication failure.

The train row MUST expose the local pre-clip gradient norm reduced as the
all-rank maximum under `grad_norm/pre_clip_rank_max`; post-clip or rank-mean
values MUST NOT be substituted. It MUST expose global work-rate measurements
derived from exact summed work counts and the all-rank maximum step duration,
including physical-token, supervised-atom, and pack throughput when their
counts are available.

Train rows MUST also include low-overhead timing scalars measured inside the
planned-step compute/optimizer boundary—from the start of the step's first
micro-step handling through gradient zeroing—excluding the completed-step
handler and scheduled eval/checkpoint handlers: `step_duration_seconds` for
that boundary's wall time, `input_build_seconds` for CPU forward-input
construction, `input_h2d_seconds` for host-to-device transfer when the backend
can measure completion accurately, and `input_wait_seconds` for time spent
waiting on prepared inputs. Additional bounded stage timings MAY be emitted
only when their measurement scope is named and their completion is measured
accurately; host enqueue time MUST NOT be labeled as GPU execution time.
Distributed reduction for timing fields MUST be the all-rank maximum because
the slowest rank owns the distributed critical path. A mean MAY additionally
be emitted only under a name that explicitly states it is a mean. Timing values
MUST travel through the existing metric collective without adding a per-step
barrier solely for observation.

On CUDA, train rows MUST expose current and peak allocated and reserved GPU
memory reduced as all-rank maxima, plus per-step allocator retry and OOM counter
deltas reduced as all-rank sums. On a backend where a metric cannot be measured
accurately, the row MUST omit the scalar and identify it in a bounded
`unavailable_fields` list rather than emit a fabricated zero. An OOM that
terminates before a planned step completes remains a terminal run failure and
MUST NOT fabricate a completed-step row.

Every durable scalar MUST have one exact declared reducer before the metric
collective runs. Required reducer semantics include: exact summed integer
counts; ratio from summed numerator and denominator; all-rank maximum for
critical-path timing, pre-clip gradient norm, and memory high-water values;
all-rank sum for counter deltas; identical-across-ranks validation for applied
learning rates and shared schedule values; and explicit finite/boolean
propagation. A metric without a declared reducer MUST fail before durable
publication instead of falling back to an implicit mean or guessing from its
key. Loss reduction MUST use its declared sufficient statistics and global
denominator rather than averaging already-normalized rank-local losses.
Boolean conjunction MUST use the explicitly named `BOOL_ALL` reducer; the
ambiguous reducer name `ALL` MUST NOT be accepted.

This change's timing, resource, optimization, and availability fields are
additive to the prerequisite row schema: no field present after the supervised-
loss migration may be silently renamed, removed, or retyped. Eval rows MUST include eval
counts and the corresponding eval loss/metric mapping under the same explicit
reducer discipline. Normal production rows MUST contain aggregated values only;
detailed per-rank metric traces, MFU, TFLOPS, and energy estimates MUST remain
disabled unless an explicit calibrated probe requests them.

The stream SHALL be a direct single-writer append. Before serialization, every
raw NaN or Inf scalar MUST be represented as JSON `null` and its field name
MUST appear in `non_finite_fields`; `finite_status` and
`optimizer_update_status` MUST remain explicit. The writer MUST reject any raw
non-finite number that remains after normalization, but MUST NOT rescan prior
rows, build a durable event index, fsync every row, create per-rank streams, or
require rotation, compression, sampling, retention, database, or external
tracking machinery in this version.

`unavailable_fields` and `non_finite_fields` MUST each be sorted and unique,
MUST contain at most 256 canonical field names, and MUST reject a field name
whose UTF-8 encoding exceeds 256 bytes. If more distinct names are observed,
the retained list MUST remain bounded and the omitted count MUST be recorded in
`unavailable_fields_truncated_count` or `non_finite_fields_truncated_count`.
Arbitrary exception text MUST NOT be stored in either list.

#### Scenario: Planned train step completes safely

- **WHEN** planned step 42 completes with an applied optimizer update
- **THEN** `logging.jsonl` MUST receive exactly one `train` row with required
  `step: 42`
- **AND** all canonical logging metrics for that step MUST be fields in that
  row, independent of `observability.steps`.

#### Scenario: Eval runs at planned step

- **WHEN** scheduled eval completes at planned step 42
- **THEN** `logging.jsonl` MUST receive exactly one `eval` row with required
  `step: 42`
- **AND** that row MUST be the canonical durable scalar record for the eval.

#### Scenario: Pre-wrapper rejection skips the optimizer update

- **WHEN** an all-rank decision chooses `not_attempted` for one planned step
- **THEN** that step MUST still receive one train logging row
- **AND** the row MUST state the skipped update and non-finite/unsafe status
- **AND** optimizer-group LR fields MUST be `null` and explicitly unavailable
  rather than reporting a value that was not applied
- **AND** any non-finite scalar MUST be `null` and named in
  `non_finite_fields`.
- **AND** the runtime receipt MUST be the `not_attempted` form with
  `attempted: false`, `applied: false`, `step_was_skipped: false`, and null LRs
- **AND** `optimizer_step_count` MUST NOT increment while the planned-step
  scheduler progression continues normally.

#### Scenario: Applied learning rate is observed

- **WHEN** a CPU training probe applies one optimizer update and advances its
  scheduler afterward
- **THEN** every `lr/group_<index>` in that train row MUST equal the optimizer
  group value sampled before the update
- **AND** it MUST NOT report the value first produced by the scheduler advance.

#### Scenario: Genuine fp16 scaler overflow skips an attempted update

- **WHEN** every rank reports an active GradScaler and a non-finite current
  unscaled gradient after exactly one unscale, with no unrelated report failure
- **THEN** the global boundary action MUST be `scaler_skip`
- **AND** no gradient clipping call may occur on that branch
- **AND** the real CUDA fp16 optimizer wrapper MUST be called exactly once on
  every rank solely to finalize the scaler-owned skip
- **AND** its immediate post-call truth MUST report `step_was_skipped: true` on
  every rank
- **AND** the row MUST distinguish wrapper attempt from actual update and MUST
  report the update as skipped
- **AND** applied-LR fields MUST be JSON `null` and explicitly unavailable
- **AND** `optimizer_step_count` MUST increment once for the completed wrapper
  invocation even though the update was not applied
- **AND** scheduler, eval, checkpoint, and planned-step counters MUST retain
  their stable planned-step policy
- **AND** executed evidence MUST prove exactly one
  `accelerator.unscale_gradients(optimizer)` preceded gradient finite and norm
  authority, with no `torch.nn.utils.clip_grad_norm_` or
  `accelerator.clip_grad_norm_` call on the overflow branch.

#### Scenario: fp16 overflow candidates disagree before the wrapper

- **WHEN** ranks do not agree that the current exactly-once-unscaled gradients
  form a scaler-overflow candidate
- **THEN** runtime MUST converge one pre-wrapper terminal unsafe decision before
  any rank calls the optimizer wrapper
- **AND** the receipt MUST report `attempted: false`, `applied: false`,
  `step_was_skipped: false`, null LRs,
  `mutation_state: divergent_or_unknown`, and a bounded terminal reason
- **AND** it MUST report that the underlying optimizer and parameters were not
  mutated while the already-unscaled GradScaler state is unfinalized; it MUST
  NOT report the composite mutation state as `unchanged`
- **AND** gradients MUST be cleared and one terminal row MUST converge before
  common failed finalization
- **AND** it MUST NOT claim a completed synchronized skip or advance scheduled
  handlers.

#### Scenario: fp16 post-call skip flags disagree

- **WHEN** the fp16 post-wrapper boolean consensus observes any rank
  that skipped and any rank that did not
- **THEN** the terminal receipt MUST report `attempted: true`, `applied: null`,
  `step_was_skipped: null`, null LRs, and
  `mutation_state: divergent_or_unknown`
- **AND** gradients MUST be cleared and one terminal row MUST converge before
  common failed finalization
- **AND** runtime MUST NOT advance scheduler or any scheduled handler or
  continue from the potentially divergent state.

#### Scenario: fp16 apply action is unanimously skipped

- **WHEN** every rank selected `apply` but post-wrapper consensus is
  `all_skipped`
- **THEN** the terminal receipt MUST preserve the known truth as
  `attempted: true`, `applied: false`, and `step_was_skipped: true`
- **AND** it MUST use null LRs, clear gradients, publish one terminal row, and
  fail commonly without scheduler or scheduled-handler progression.

#### Scenario: fp16 scaler-skip action is unanimously applied

- **WHEN** every rank selected `scaler_skip` but post-wrapper consensus is
  `none_skipped`
- **THEN** the terminal receipt MUST preserve the known truth as
  `attempted: true`, `applied: true`, and `step_was_skipped: false`
- **AND** it MUST preserve the identical pre-call optimizer-group LRs that were
  applied while marking the mutation state as corrupted or unsafe
- **AND** it MUST clear gradients, publish one terminal row, and fail commonly
  without scheduler or scheduled-handler progression.

#### Scenario: Genuine fp16 finite update is applied

- **WHEN** real CUDA Accelerate fp16 completes a finite optimizer-wrapper call
- **THEN** the receipt MUST report `attempted: true`, `applied: true`, and
  `step_was_skipped: false`, with the pre-call optimizer-group LR values
- **AND** `optimizer_step_count` MUST increment once
- **AND** executed evidence MUST prove exactly one
  `accelerator.unscale_gradients(optimizer)` followed by gradient finite and
  pre-clip norm calculation and a non-unscaling clip primitive
- **AND** the path MUST NOT call `accelerator.clip_grad_norm_`.

#### Scenario: Terminal optimizer boundary row fails to publish

- **WHEN** a terminal optimizer-boundary row cannot be appended or converged
- **THEN** every rank MUST still converge failed run finalization
- **AND** the bounded primary optimizer-boundary failure code MUST remain the
  terminal cause ahead of the row-publication failure
- **AND** no success, eval, checkpoint, exact-resume, or selector artifact may
  publish.

#### Scenario: Terminal optimizer boundary finalizes the run

- **WHEN** a terminal optimizer-boundary row is successfully published and
  converged for planned step 42
- **THEN** rank zero MUST atomically finalize `run.json` with failed status,
  terminal planned step 42, the bounded optimizer-boundary failure code, and the
  last prior completed-step/scheduler counters
- **AND** every rank MUST converge that failed outcome before teardown
- **AND** planned step 42 MUST NOT be counted as completed.

#### Scenario: Asymmetric rank metrics are reduced

- **WHEN** two ranks report different local pre-clip gradient norms, timings,
  counts, and ratio sufficient statistics
- **THEN** `grad_norm/pre_clip_rank_max` and timing fields MUST equal the larger
  rank-local values
- **AND** counts and ratios MUST be produced by their declared sum and
  numerator/denominator reducers
- **AND** no value may use an implicit mean.

#### Scenario: Timing fields observed

- **WHEN** a planned train step completes under normal production settings
  with world size greater than one
- **THEN** its train row MUST contain `step_duration_seconds`,
  `input_build_seconds`, and `input_wait_seconds` reduced as the all-rank
  maximum
- **AND** `input_h2d_seconds` or an additional stage field MUST appear only
  when its stated scope is measured accurately
- **AND** the measured step window MUST exclude completed-step and scheduled
  eval/checkpoint handler time.

#### Scenario: CUDA allocator state is observed

- **WHEN** a completed CUDA training step has valid allocator measurements
- **THEN** its train row MUST contain current and peak allocated/reserved byte
  values reduced by rank maximum
- **AND** retry and OOM deltas MUST be reduced by rank sum
- **AND** the canonical row MUST NOT contain a default per-rank trace.

#### Scenario: Unknown reducer is requested

- **WHEN** a new metric is presented to the distributed observation boundary
  without an exact reducer declaration
- **THEN** the step MUST fail before the row is appended
- **AND** the runtime MUST NOT infer mean, sum, maximum, or ratio semantics from
  the metric name.

#### Scenario: Diagnostic field-name set exceeds its bound

- **WHEN** more than 256 distinct unavailable or non-finite field names are
  presented for one completed observation
- **THEN** the canonical list MUST remain sorted, unique, and bounded to 256
  names
- **AND** the corresponding truncation-count field MUST report the omitted
  distinct-name count without retaining arbitrary error text.

## ADDED Requirements

### Requirement: Rank-Zero Presentation Sinks

Console and TensorBoard SHALL be rank-zero derived presentations of canonical
train/eval observations, not independent metric authorities. For a train
observation, presentation SHALL occur at positive planned-step multiples of
resolved `observability.steps`, at the resolved terminal planned step, and after
successful publication of an off-cadence terminal optimizer-boundary row; each
completed eval invocation SHALL be presented when it runs. Every presentation
payload MUST contain `step` and resolved total planned steps. Console output MUST
render current step over total; an ETA MAY be shown only as an approximate
derived estimate and MUST NOT be written into exact-resume state or used as
scheduling evidence.

Typed metric reduction SHALL be owned by `src/runtime/metrics.py`; canonical
row construction by `src/training/reporting.py`; JSONL-first publication and
derived-sink lifecycle by `src/artifacts/observation_publisher.py`; and their
composition by `src/training/session.py`. The training facade MUST NOT become a
second row, reducer, or sink owner, and no generic coordinator owner may be
added.

The corresponding canonical `logging.jsonl` row MUST be successfully published
before either derived sink consumes it. TensorBoard event files MUST be written
under `tensorboard/` in the same run directory and MUST mirror only finite
scalar values plus the step identity from the canonical row. TensorBoard
initialization or write failure MUST produce a bounded warning, disable that
sink for the rest of the run, and MUST NOT remove, rewrite, or invalidate an
already published JSONL row. No non-main rank may write console progress or
TensorBoard event files.

TensorBoard import, initialization, `add_scalar`, `flush`, and `close` failures
MUST all use one one-way disabled latch. The first failure MAY emit at most one
bounded run warning with a stable code and one best-effort stderr warning; a
failure during best-effort close MUST NOT recurse, emit another warning, or
change canonical publication status. Once disabled, no later sink method may
be called during that run.

#### Scenario: Presentation interval is reached

- **WHEN** rank zero publishes a train row whose planned `step` is a positive
  multiple of `observability.steps`
- **THEN** console and enabled TensorBoard receive a derived update containing
  that `step` and the resolved total
- **AND** the JSONL row MUST already be readable before either derived update.

#### Scenario: Eval completes between train presentation steps

- **WHEN** a scheduled eval completes at a step that is not a multiple of
  `observability.steps`
- **THEN** its canonical eval row MUST be appended and mirrored to each enabled
  presentation sink when the eval runs.

#### Scenario: Terminal step is off cadence

- **WHEN** the final planned train step is not a multiple of
  `observability.steps`
- **THEN** the final canonical train row MUST still be presented with exact
  `step` and total progress.

#### Scenario: TensorBoard event is consumed

- **WHEN** a finite canonical train or eval row is mirrored to TensorBoard
- **THEN** a standard TensorBoard event reader MUST recover the expected tag,
  scalar value, and planned `step` from the run-local event files.

#### Scenario: TensorBoard sink fails

- **WHEN** TensorBoard initialization or event publication raises after the
  canonical JSONL row is published
- **THEN** the run MUST retain that canonical row, emit one bounded warning,
  and disable further TensorBoard writes
- **AND** training, eval, checkpoint, and JSONL publication MUST continue if
  their own contracts remain valid.

#### Scenario: TensorBoard flush or close fails after a successful add

- **WHEN** `flush` or `close` raises after the canonical JSONL row is durable
- **THEN** the publisher MUST retain the row, latch TensorBoard disabled, and
  emit no more than the single bounded sink warning for the run
- **AND** cleanup failure MUST NOT recurse or block later canonical JSONL,
  eval, checkpoint, or terminal publication.

#### Scenario: Non-main rank observes a completed step

- **WHEN** a non-main rank completes the same distributed planned step
- **THEN** it MUST participate in required metric collectives
- **AND** it MUST NOT print the shared progress update or create a TensorBoard
  event file.
