## MODIFIED Requirements

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
