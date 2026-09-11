## Purpose

Defines a bounded experiment-local controller that converts K-native owner
support into safer Human-13 clean-greedy coverage by repairing current natural
states and rolling back updates that fail a direct behavior-preservation gate.

## ADDED Requirements

### Requirement: Dynamic Frontier Ledger

Every iteration SHALL begin from one accepted original-prompt HF clean-greedy
panel decode and SHALL derive a content-addressed frontier ledger containing the
exact generated tokens, parsed rows, owner matching, chronological class-
agnostic `IoU>0.95` duplicate events, protected owners, current covered owners,
and currently uncovered K-hit owners.  Candidate aliases MUST be complete
metric-valid native rows from the sealed K support bank; K-miss owners MUST
remain gradient-neutral.

The ledger MUST NOT infer exhaustiveness from a finite K union, introduce a
positive terminal target, or turn an unmatched row into a trusted positive.

#### Scenario: Accepted decode advances the frontier
- **WHEN** a post-update clean-greedy decode passes the behavior gate
- **THEN** its exact trajectory and matching SHALL become the sole natural-state source for the next iteration
- **AND** no second optimizer update SHALL be derived from the preceding ledger.

#### Scenario: No trusted uncovered candidate remains
- **WHEN** the current accepted decode has no uncovered owner with a metric-valid native alias
- **THEN** the iteration SHALL stop or refresh the declared K support bank
- **AND** SHALL NOT train `STOP` as proof that the image is complete.

### Requirement: HF Decision Surface and Cross-Surface Evidence

Candidate eligibility, first-bottleneck identity, target-versus-competitor
margin, and final behavior acceptance SHALL be decided on HF fp32/SDPA with
physical batch one and the bound processor/generation policy.  Packed
BF16/FlashAttention-2 scores MAY prefilter candidates, but MUST NOT be the sole
scientific decision surface.

For every trained candidate, the controller SHALL record aligned packed and HF
scores at each decision site and candidate-specific drift.  A margin reserve
MUST be derived from that bound evidence or a predeclared zero-reserve
diagnostic; a global historical maximum drift MUST NOT silently become a
universal margin hyperparameter.

#### Scenario: Packed and HF ranking disagree
- **WHEN** the packed prefilter and HF scorer choose different targets or first bottlenecks
- **THEN** the HF decision SHALL control candidate selection and supervision
- **AND** the disagreement SHALL remain visible in the iteration receipt.

### Requirement: Composable Candidate Selection

For each eligible owner, the controller SHALL keep the native alias with the
smallest nonnegative greedy barrier, defined as the sum of positive target-
versus-global-competitor deficits along the complete teacher-forced row.  It
SHALL shortlist two to four distinct owners when available.

Each shortlisted row SHALL be forced after the exact natural prefix and then
released to unconstrained greedy continuation.  Selection SHALL be
lexicographic: protected owners remain jointly matchable, unique owner coverage
increases, continuation terminates naturally before its cap, and only then do
duplicate, malformed, length, and barrier costs break ties.  The continuation
cap SHALL be at least the larger of twice the Source row count and Source token
count plus 512; a cap hit SHALL be harm, not a partial success.

#### Scenario: High-likelihood row destroys a protected owner
- **WHEN** one candidate has a lower teacher-forced barrier but its released continuation makes a protected owner unmatchable
- **THEN** it SHALL rank below a candidate that preserves the protected set and gains coverage.

### Requirement: Full-Row and First-Bottleneck Safe Treatments

`O-Full-Safe` and `O-First-Safe` SHALL share the same frontier, selected row,
dynamic duplicate evidence, rectangle-valid gate, optimizer dose, and behavior
gate.  `O-Full-Safe` SHALL supervise the selected native row body.
`O-First-Safe` SHALL apply a target-versus-global-competitor margin only at the
earliest selected-row token that is not strictly greedy-feasible on the HF
surface.  It MUST NOT supervise the remaining suffix merely because it appears
in the chosen alias.

If the current natural trajectory contains a duplicate branch and a selected
frontier exists, the earliest duplicate-versus-selected differentiating token
MAY receive pairwise rejection.  Shared row-openers and shared description
tokens MUST NOT be indiscriminately unlearned.  Rectangle gates SHALL accept
any coordinate token yielding positive extent.  No loss SHALL redirect toward
`STOP` merely because the K support bank is finite.

#### Scenario: Selected row has one greedy blocker
- **WHEN** all selected-row tokens except one are strict HF argmax decisions
- **THEN** `O-First-Safe` SHALL train exactly that one blocker plus shared safety terms
- **AND** `O-Full-Safe` SHALL remain the dose-matched full-row contrast.

### Requirement: Transactional Clean-Greedy Gate

Before each optimizer mutation the controller SHALL snapshot the trainable
parameters, AdamW state, scheduler state, update counter, and relevant CPU/CUDA
RNG state.  After one fixed-dose update it SHALL run exactly one fresh
original-prompt clean-greedy HF panel decode.  The update SHALL be accepted only
when all protected owners remain jointly coverable by an admissible one-to-one
matching, panel unique-owner coverage does not decrease, no continuation cap is
hit, malformed rows do not increase, row count is at most `1.2` times the prior
accepted count, and duplicates increase by at most two.

On rejection, all snapshotted mutable training state SHALL be restored before
another update.  A post-rollback clean-greedy decode SHALL reproduce the prior
accepted owner set; otherwise the run SHALL stop.  Rejected model bytes MUST
NOT be published as an accepted checkpoint.

#### Scenario: Owner exchange hides behind equal recall
- **WHEN** a proposed update gains one owner but loses one protected owner
- **THEN** the update SHALL be rejected even if pooled unique-owner count is unchanged.

#### Scenario: Rollback drill is forced
- **WHEN** the vertical slice marks an otherwise finite proposal for deliberate rejection
- **THEN** parameters and optimizer state SHALL restore exactly
- **AND** the subsequent clean-greedy owner set SHALL equal the pre-update accepted set.

### Requirement: Bounded Execution and Claim Boundary

The vertical slice SHALL execute one real `O-First-Safe` proposal through HF
scoring, forced-row continuation, packed forward/backward, AdamW, post-update
clean greedy, checkpoint handling, analyzer projection, and a deliberate
rollback drill before the full pilot.

The pilot SHALL start both arms from byte-identical Source payloads and fresh
optimizer states, attempt at most eight updates per arm, mutate at most once per
iteration, and preserve immutable iteration receipts.  Final decision-owning
outputs SHALL use unconstrained original-prompt HF fp32/SDPA batch-one greedy
decode with repetition penalty `1.0`.  Results SHALL be labeled closed-loop
same-panel control evidence only.

#### Scenario: Vertical mechanics succeed
- **WHEN** the real slice completes all forward, update, decode, rollback, and readback steps
- **THEN** the full pilot MAY start under fresh roots
- **AND** the slice SHALL make no model-quality or transfer claim.

#### Scenario: Bounded pilot completes
- **WHEN** both arms reach their stop condition or eight attempted updates
- **THEN** the analyzer SHALL report protected gains/losses, G/H/M identities, duplicates, malformed rows, caps, rows, tokens, accepted/rejected updates, candidate costs, and runtime
- **AND** no validation, production-safety, full-set, or architecture claim SHALL be inferred.
