## MODIFIED Requirements

### Requirement: Cadence Config And Resolved Step Schedule

Cadence config SHALL use the canonical authored fields
`checkpoint.every_fraction`, `checkpoint.steps`, `checkpoint.save_final`,
`eval.forward.every_fraction`, `eval.forward.steps`, and
`observability.steps`. Every completed planned optimizer-step boundary MUST
produce one train row in `logging.jsonl`; therefore the public schema MUST NOT
expose a cadence that samples or suppresses canonical training logging.
`observability.steps` MUST be authored explicitly in every supported training
config, MUST have no schema default, and MUST be a positive integer measured on
the planned-step clock. It controls only rank-zero console and TensorBoard
presentation of already canonical observations. The schema MUST reject aliases
such as `save_steps`, `eval_steps`, `logging_steps`, a separate `global_step`,
or the removed `training.logging` cadence block.

Runtime MAY materialize its resolved schedule in memory, but MUST NOT require a
separate durable schedule receipt when the same resolved maximum steps and
scheduled eval/checkpoint events are represented by `run.json`, train/eval
logging rows, and checkpoint aliases. Presentation cadence and derived ETA
state MUST NOT enter training-semantic or exact-resume equality; changing only
`observability.steps` between an admitted parent and continuation MUST NOT make
otherwise identical training state incompatible.

#### Scenario: Fractional cadence resolved

- **WHEN** checkpoint or eval cadence uses `every_fraction`
- **THEN** fractional milestones MUST use
  `ceil(fraction * resolved_max_steps)` style planned-step materialization,
  repeated and clamped to `[1, resolved_max_steps]`
- **AND** collisions with explicit steps or final events MUST be de-duplicated.

#### Scenario: Every train step is logged

- **WHEN** one planned training step completes
- **THEN** rank zero MUST append exactly one train row for that step to
  `logging.jsonl`
- **AND** `observability.steps` MUST NOT suppress or sample that row.

#### Scenario: Presentation interval omitted

- **WHEN** a training config omits `observability.steps`
- **THEN** strict config validation MUST fail before model, optimizer, dataset,
  cache, or run-directory mutation
- **AND** no default interval may be inferred.

#### Scenario: Presentation interval is invalid

- **WHEN** a training config authors a non-integer or non-positive
  `observability.steps`
- **THEN** strict config validation MUST fail before schedule resolution.

#### Scenario: Supported active config is resolved

- **WHEN** any supported training config under `configs/coordexp_swift/prod/`
  or `configs/coordexp_swift/smoke/` is resolved
- **THEN** its authored inheritance chain MUST resolve exactly one explicit
  positive `observability.steps` value
- **AND** the resolved config artifact MUST contain that value.

#### Scenario: Presentation cadence changes across exact continuation

- **WHEN** an exact-resume continuation differs from its admitted parent only
  in `observability.steps`, run identity, resume path, or derived presentation
  state such as approximate ETA
- **THEN** training-semantic resume compatibility MUST remain equal
- **AND** no presentation record or ETA estimate may be restored as model,
  optimizer, scheduler, data-order, or RNG state.

#### Scenario: Resolved schedule artifact written

- **WHEN** runtime resolves the planned-step schedule
- **THEN** it MUST retain the schedule in memory for train, eval, checkpoint,
  final, and presentation dispatch
- **AND** it MUST NOT write a separate `resolved_step_schedule.json` artifact.

#### Scenario: Legacy cadence alias authored

- **WHEN** a config authors `save_steps`, `eval_steps`, `logging_steps`,
  `global_step`, or `training.logging`
- **THEN** strict config validation MUST fail before schedule resolution.

### Requirement: Planned Step Schedule

The resolved planned-step schedule SHALL be computed before training begins
from the dataloader, world size, `training.effective_batch_size`, `epochs`, and
optional debug `max_steps`. If `max_steps` is set, it MUST take priority over
epoch-derived length. Eval, checkpoint, logging, and final events MUST use the
planned-step clock, not a successful-update counter.

An applied update, an all-rank-confirmed fp16 GradScaler skip, and a supported
`not_attempted` skip are completed planned-step boundaries. Each MUST advance
the scheduler exactly once and dispatch eval/checkpoint/final events using the
original planned-step id. A pre-wrapper terminal unsafe decision or a post-
wrapper distributed outcome that contradicts its expected action is not a
completed planned-step boundary. It MUST retain the current planned-step id for
one terminal train row and failed run finalization, but MUST NOT increment the
completed-step or scheduler-step count or dispatch eval, checkpoint,
exact-resume, best-selector, final-success, or later-step handlers. Terminal
failure MUST NOT recompute or renumber the resolved schedule. In particular, an
fp16 terminal decided after exactly-once unscale remains non-completed even
though GradScaler's per-optimizer state is already `UNSCALED` and unfinalized;
the schedule MUST NOT reinterpret that scaler mutation as a completed update.

#### Scenario: Debug max steps configured

- **WHEN** a config sets `max_steps: 5`
- **THEN** the resolved maximum planned steps MUST be 5 regardless of the
  epoch-derived value
- **AND** final checkpoint and final metrics MUST be scheduled at planned step
  5.

#### Scenario: Unsafe optimizer update skipped

- **WHEN** a planned step is marked unsafe before optimizer update
- **THEN** schedule events for that planned step MUST still use the original
  planned-step id
- **AND** artifacts MUST record that the optimizer update was not applied.

#### Scenario: Recoverable optimizer update is skipped

- **WHEN** a planned step reaches an all-rank-confirmed `scaler_skip` or a
  supported `not_attempted` outcome
- **THEN** schedule events for that completed boundary MUST use the original
  planned-step id
- **AND** the scheduler MUST advance exactly once
- **AND** artifacts MUST record that the optimizer update was not applied.

#### Scenario: Optimizer boundary is terminally inconsistent

- **WHEN** all ranks converge a pre-wrapper terminal unsafe decision or an
  action-contradictory post-wrapper outcome for planned step 42
- **THEN** the terminal row and failed run record MUST retain `step: 42`
- **AND** an after-unscale fp16 terminal MUST remain non-completed even though
  its receipt truthfully marks the composite scaler mutation state unsafe
- **AND** completed-step and scheduler-step counters MUST NOT advance
- **AND** scheduled eval, checkpoint, exact-resume, selector, final-success, and
  later-step handlers MUST NOT run.
