## MODIFIED Requirements

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
metric. Train rows MUST include weighted configured losses, top-level
`acc_top1`, top-level `acc_top5`, actual learning-rate values,
optimizer-update status, and finite status where those values are available.
Train rows MUST also include low-overhead timing scalars measured inside the
planned-step compute/optimizer boundary — from the start of the step's first
micro-step handling through gradient zeroing — excluding the completed-step
handler and scheduled eval/checkpoint handlers: `step_duration_seconds` for
that boundary's wall time, `input_build_seconds` for forward-input
construction time, and `input_wait_seconds` for time spent waiting on
prepared inputs. Distributed reduction for these timing fields MUST be the
all-rank maximum, because the slowest rank owns the distributed critical
path; a mean MAY additionally be emitted only under a name that explicitly
states it is a mean. Timing values MUST travel through the existing metric
collective without introducing additional synchronization, and timing
collection MUST NOT add per-step synchronization beyond monotonic-clock
reads. Timing fields are additive schema only: no existing field may be
renamed, removed, or retyped. Eval rows MUST include eval counts and the
corresponding eval loss/metric mapping.

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

#### Scenario: Timing fields observed

- **WHEN** a planned train step completes under normal production settings
  with world size greater than one
- **THEN** its train row MUST contain `step_duration_seconds`,
  `input_build_seconds`, and `input_wait_seconds` reduced as the all-rank
  maximum
- **AND** the measured window MUST exclude completed-step and scheduled
  eval/checkpoint handler time
- **AND** existing consumers reading previously defined fields MUST be
  unaffected.
