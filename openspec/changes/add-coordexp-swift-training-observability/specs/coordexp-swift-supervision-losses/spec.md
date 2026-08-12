## MODIFIED Requirements

### Requirement: Non-Finite Loss And Gradient Gates

Training SHALL separate scalar loss finite checks before backward from
current-gradient/overflow checks after backward. Unsafe non-finite state MUST
prevent an unacknowledged corrupted optimizer update. Recoverable bad examples
or synchronized skips MAY be reported without changing the planned-step
schedule. In distributed execution, the scalar finite check MUST produce one
reduced all-rank decision before any rank calls backward. Under fp16,
post-backward handling MUST unscale exactly once before each rank reports
current gradient finiteness, pre-clip norm, active-GradScaler status, and whether
the current unscale recorded non-finite gradients. Pre-call logic MUST NOT use a
previous wrapper call's skip flag as current overflow evidence.

The all-rank report reducer MUST either produce the same closed normal action on
every rank—`apply`, `scaler_skip`, or supported `not_attempted`—or the same
terminal unsafe decision. `scaler_skip` is admitted only when every rank has an
active GradScaler, exactly-once unscale completed, and the current unscaled
gradient is non-finite, with no unrelated report failure. It performs no clip
and calls the wrapper once on every rank only to let GradScaler suppress the
underlying optimizer mutation and update its state. Mixed scaler candidacy or
an unrelated unsafe fp16 state after unscale is terminal before the wrapper;
the normal-action enum MUST NOT grow a fourth terminal action. The underlying
optimizer and parameters remain untouched, but exactly-once unscale has already
changed GradScaler's per-optimizer stage and `found_inf` record. The terminal
receipt MUST therefore use `mutation_state: divergent_or_unknown` for this
composite, potentially rank-divergent and unfinalized state, never `unchanged`.

Every real fp16 wrapper call MUST converge its immediate rank-local skip
booleans as exactly `all_skipped`, `none_skipped`, or `mixed` before scheduler,
logging success, or any later collective. `apply` accepts only `none_skipped`;
`scaler_skip` accepts only `all_skipped`. `mixed` and unanimous outcomes that
contradict the expected action are terminal distributed decisions, never rank-
local raises. The resulting update and finite status MUST be represented once
in the rank-zero train row for that planned step; normal training MUST NOT write
duplicate rank-local gate receipts.

An applied action, uniform scaler skip, or supported `not_attempted` skip is a
completed planned-step boundary: gradients are cleared, scheduler and scheduled
handlers retain the stable planned-step policy, and rank zero writes one row. A
terminal pre-wrapper or post-wrapper outcome also clears gradients and first
publishes/converges exactly one terminal unsafe row, but does not advance
scheduler, completed-step count, eval, checkpoint, exact-resume, selector,
final-success, or later-step handlers. Failed finalization MUST be all-rank
converged. Gradient clearing MUST NOT be claimed to repair possibly divergent
parameters or scaler state.

#### Scenario: Non-finite scalar loss

- **WHEN** total loss is NaN or Inf before backward
- **THEN** runtime MUST reduce the unsafe scalar status across all ranks before
  any rank calls backward
- **AND** all ranks MUST skip backward and optimizer update for that planned
  step through supported `not_attempted`
- **AND** runtime MUST clear accumulated gradients and maintain the completed-
  boundary planned-step scheduler/update policy
- **AND** rank zero MUST log the synchronized unsafe/update status once
- **AND** non-finite scalar fields MUST be represented as JSON `null` and named
  in `non_finite_fields`.

#### Scenario: Uniform distributed fp16 gradient overflow

- **WHEN** every rank's exactly-once-unscaled current-gradient report satisfies
  the scaler-overflow predicate
- **THEN** all ranks MUST select `scaler_skip`, perform no clip, and call the
  wrapper once for scaler finalization
- **AND** post-call consensus MUST be `all_skipped`
- **AND** rank zero MUST log that synchronized completed skip once.

#### Scenario: Supported non-scaler gradient rejection

- **WHEN** the retained bf16/non-scaler post-backward gate rejects the current
  gradients without a distributed contract inconsistency
- **THEN** every rank MUST select supported `not_attempted`, clear gradients,
  and use the same completed-boundary scheduler/update policy
- **AND** rank zero MUST log the synchronized skipped decision once.

#### Scenario: Pre-wrapper fp16 reports are inconsistent

- **WHEN** ranks disagree on active-scaler/current-overflow candidacy or any rank
  reports an unrelated unsafe fp16 state after unscale
- **THEN** every rank MUST converge the same terminal unsafe decision before a
  wrapper call
- **AND** rank zero MUST publish one terminal row with known
  `attempted: false`, `applied: false`, `step_was_skipped: false`, and
  `mutation_state: divergent_or_unknown`
- **AND** the row MUST distinguish untouched parameters/underlying optimizer
  from the already-unscaled, unfinalized GradScaler state and MUST NOT label the
  composite state `unchanged`
- **AND** the run MUST fail commonly without completed-boundary progression.

#### Scenario: Post-wrapper fp16 outcomes are mixed

- **WHEN** some ranks report immediate post-call skip and others do not
- **THEN** every rank MUST converge one terminal divergent outcome before any
  scheduler or later collective
- **AND** rank zero MUST publish one terminal row with `attempted: true`,
  nullable global `applied` and `step_was_skipped`, and
  `mutation_state: divergent_or_unknown`
- **AND** the run MUST fail commonly without claiming that gradient clearing
  repaired the state.

#### Scenario: Post-wrapper fp16 outcome unanimously contradicts the action

- **WHEN** `apply` converges `all_skipped` or `scaler_skip` converges
  `none_skipped`
- **THEN** every rank MUST converge one terminal contract failure
- **AND** the terminal row MUST preserve the known attempted/applied/skipped
  booleans rather than replacing them with unknown values
- **AND** `scaler_skip + none_skipped` MUST preserve the identical pre-call LRs
  applied on every rank, while `apply + all_skipped` uses null applied-LR fields
- **AND** the run MUST fail commonly without completed-boundary progression.

#### Scenario: Zero eligible protected atoms on one rank

- **WHEN** a protected loss has zero eligible atoms on one rank
- **THEN** runtime MUST include the rank-local eligible count in the same
  planned-step all-rank denominator/finite decision before any rank raises or
  calls backward
- **AND** the global decision MUST avoid distributed deadlock.
