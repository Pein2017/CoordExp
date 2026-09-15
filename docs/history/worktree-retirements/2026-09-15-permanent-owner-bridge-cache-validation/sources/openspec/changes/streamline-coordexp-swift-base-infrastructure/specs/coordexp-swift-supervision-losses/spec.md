## ADDED Requirements

### Requirement: Loss Runner Streaming Protocol Is Required At Construction

Training and eval construction SHALL require the paired `LossRunner` to
implement the full streaming protocol — `prepare_planned_step`,
`compute_micro_step`, and `finalize_planned_step` — and MUST fail closed at
construction time, before any planned step begins or any micro-step forward
runs, when any of the three is absent. This check MUST be unconditional: it
applies to every `SupervisedTrainer` construction (with or without a
`forward_input_provider`) and every `ForwardEvalRunner` construction (in
every reduction mode, `replicated` or `disjoint_shard`), not only specific
provider/mode combinations. Whole-planned-step batch loss computation (a
single call that returns a complete `LossBundle` for all contexts at once,
without the prepare/compute-micro-step/finalize sequence) is unsupported and
removed from both the trainer and the eval-forward runner; a loss runner
that implements only a batch-style call and not the streaming trio MUST be
rejected, never silently routed to a batch code path.

#### Scenario: Non-streaming loss runner rejected at trainer construction

- **WHEN** `SupervisedTrainer` is constructed with a `loss_runner` missing
  `prepare_planned_step`, `compute_micro_step`, or `finalize_planned_step`
- **THEN** construction MUST raise a contract error identifying the missing
  streaming protocol before any planned step begins
- **AND** no micro-step forward, loss computation, or gradient computation
  may occur for that trainer instance.

#### Scenario: Non-streaming loss runner rejected at eval construction

- **WHEN** `ForwardEvalRunner` is constructed with a `loss_runner` missing
  any member of the streaming trio
- **THEN** construction MUST raise a contract error identifying the missing
  streaming protocol, regardless of the configured reduction mode
- **AND** no eval forward pass may occur for that runner instance.

#### Scenario: Batch-only loss runner is rejected, not routed to a batch path

- **WHEN** a `loss_runner` implements a whole-planned-step batch `compute`
  call but not `prepare_planned_step`/`compute_micro_step`/`finalize_planned_step`
- **THEN** both trainer and eval construction MUST reject it as non-streaming
- **AND** MUST NOT fall back to computing the planned step's loss in one
  batch call, since no batch code path exists to fall back to.

## MODIFIED Requirements

### Requirement: Loss Bundle Metrics

`LossRunner` SHALL return a `LossBundle` containing total weighted loss,
weighted per-term loss metrics, top-level `acc_top1`, top-level `acc_top5`,
selected-count diagnostics, and finite-status diagnostics. Stored per-term loss
metrics MUST be weighted values. After all-rank reduction, the complete scalar
mapping for one planned step SHALL be written together in that step's wide
`train` row in `logging.jsonl`. Cross-rank reduction of `acc_top1` and
`acc_top5` MUST be derived from exact rank-local sufficient statistics — the
integer top-1 correct count, the integer top-5 correct count, and the
rank-local supervised-atom count — summed across ranks before the ratio is
formed, so the reduced value equals `sum_r(correct_r) / sum_r(atoms_r)`.
These sufficient statistics MAY travel only in the internal reduction payload
and need not become durable logging fields. The reduction MUST NOT weight
per-rank accuracies by an already-global atom count (the planned-step metric
`count/supervised_atoms` is globally merged and therefore not a valid
per-rank weight) and MUST NOT reconstruct integer counts from rounded
floating-point ratios. At world size one the reduced value MUST equal the
rank-local value. Non-finite values MUST retain their field names with JSON
`null` values and MUST be listed in `non_finite_fields` rather than making the
planned step unwriteable.

#### Scenario: Train metric event emitted

- **WHEN** a train step completes
- **THEN** its single logging row MUST include weighted protected-loss metrics
  and top-level `acc_top1` and `acc_top5`
- **AND** top-1/top-5 names MUST NOT be nested under a base-CE namespace.

#### Scenario: Unequal rank-local atom counts

- **WHEN** distributed ranks contribute unequal supervised-atom counts to one
  planned step
- **THEN** the reduced `acc_top1` and `acc_top5` MUST equal the pooled ratio
  of summed rank-local integer correct counts over summed rank-local atom
  counts
- **AND** MUST NOT be the plain mean of per-rank accuracies or a mean
  weighted by a globally merged atom count.
