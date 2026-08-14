## Purpose

Define a bounded Human-13 research capability that turns fresh RP-aware K16
trajectories into one-update proposals, audits greedy transfer and owner
retention under two repetition-penalty surfaces, and restores Source exactly.

## ADDED Requirements

### Requirement: Sealed RP-aware policy evidence
The screen SHALL acquire fresh trajectories under each repetition-penalty
contract declared by the owning research unit.  Every score-bearing generated
token MUST bind its complete conditioning history, chosen token, processed
policy log probability, decode processors in application order, sampler
settings, model identity, and request identity.  Sampling and replay MUST pass
the declared no-update numeric-parity gate before the evidence can enter a
scientific proposal.

#### Scenario: Matched sampling and replay
- **WHEN** a generated token is replayed under the same sealed policy contract
- **THEN** its processed chosen-token log probability agrees within the sealed
  tolerance and its evidence is admitted

#### Scenario: RP or history mismatch
- **WHEN** the RP value, processor order, prompt history, generated history,
  chosen token, or processed likelihood fails the parity contract
- **THEN** the affected run stops before applying a model-quality proposal

#### Scenario: Historical trajectory reuse
- **WHEN** a historical trajectory lacks the fresh score-bearing policy
  evidence required by this screen
- **THEN** it MAY supply frozen support metadata but MUST NOT supply
  score-function credit

#### Scenario: Decode cap reached
- **WHEN** a fresh trajectory reaches the declared generated-token cap before
  the sealed natural stop token
- **THEN** the cap and retained trace are reported as harm and MUST NOT be
  relabeled as natural termination

### Requirement: Detached trusted-owner trajectory credit
The screen SHALL project every admitted K16 group into a deterministic signed
row ledger using the frozen trusted-owner universe, exogenous owner weights,
one-to-one matching, first-hit credit, and the burden rules declared by the
owning research unit.  Legacy K-miss owners MUST remain neutral.  Row
advantages and all matching or selection decisions MUST be detached from the
model gradient.

#### Scenario: First trusted-owner hit
- **WHEN** a row first matches a trusted owner in an admitted trajectory
- **THEN** the row receives its declared marginal coverage credit exactly once

#### Scenario: Duplicate or invalid burden
- **WHEN** a row is classified by the sealed precedence rule as a duplicate,
  invalid row, trusted-owner repeat, or non-M unmatched row
- **THEN** it receives the declared signed burden without also receiving owner
  credit

#### Scenario: Legacy K-miss row
- **WHEN** a row matches a frozen legacy K-miss owner
- **THEN** its own chosen-token log probabilities are masked from trajectory
  credit while its tokens remain causal history for later trusted rows

#### Scenario: Terminal STOP
- **WHEN** a trajectory stops while trusted owner mass remains
- **THEN** STOP retains a non-positive advantage and MUST NOT receive positive
  imitation as an exhaustiveness target

#### Scenario: Tied trajectory group
- **WHEN** all admitted trajectories have identical row returns at a compared
  position
- **THEN** their leave-one-out trajectory advantages at that position are zero
  rather than fabricated by normalization

#### Scenario: Equivalent packing partitions
- **WHEN** the same logical image and trajectory ledger is split across
  different no-padding packs or gradient-accumulation boundaries
- **THEN** one sealed global numerator and denominator produce the same loss
  and parameter gradient within the declared numeric tolerance

### Requirement: Sparse greedy compiler
The compiler SHALL use only the frozen metric-valid native alias bank and the
sealed Source clean-greedy boundary for its RP contract.  It MUST compare a
normalized valid-token score with the realized bad child under RP-processed
greedy logits without sampling-temperature division, keep selectors detached,
and expose its loss separately from trajectory credit.  It MUST NOT infer
recursive alias composability or enlarge the valid set from fresh sampled rows.

#### Scenario: Premature bad child exists
- **WHEN** trusted owners remain and the sealed Source boundary chooses the
  declared bad child
- **THEN** the compiler emits the predeclared valid-versus-bad margin term and
  its complete alias and logit evidence

#### Scenario: No compiler boundary
- **WHEN** no admissible premature Source boundary exists for an image
- **THEN** the compiler contributes zero for that image and records why it was
  absent

#### Scenario: Unknown token outside frozen aliases
- **WHEN** a token is not represented by the frozen metric-valid alias bank
- **THEN** the compiler MUST NOT label it a trusted valid continuation solely
  because it appeared in a fresh trajectory

### Requirement: Exact proposal preservation
Every active arm SHALL begin from the sealed Source model and fresh optimizer
state.  The screen MUST reconstruct the parameter delta produced by the exact
configured optimizer proposal before behavioral evaluation.  A disjoint
qualification stage MAY replace the default learning rate once through the
owning unit's sealed mechanics-only dose rule, after which one global learning
rate MUST be shared by both RP contracts, every arm, and every matrix seed.
The preservation arm SHALL project that delta against the frozen owner-wise
witness constraints and trust radius declared by the research unit, and SHALL
apply and audit the projected delta itself.  The owner-wise witness bank,
its certification, and the qualification dose mechanics MUST be measured
exactly as the owning research unit declares them, on the sealed Source parser
spans and the HF fp32/SDPA batch-one surface, and MUST NOT be replaced by an
unsealed finite-difference policy or a new loss term.

#### Scenario: Qualification dose selection
- **WHEN** the default learning rate does not satisfy the declared mechanical
  floor or ceiling on both RP contracts
- **THEN** only the predeclared fixed dose ray may be evaluated, owner outcomes
  remain unavailable to selection, and exactly one content-bound global
  learning rate is frozen before matrix materialization

#### Scenario: Adaptive or contract-specific dose
- **WHEN** a runtime attempts to choose a learning rate from gradient norm,
  owner gain/loss, a matrix outcome, or a particular RP, arm, or seed
- **THEN** execution fails closed instead of admitting the proposal

#### Scenario: Unprojected proposal
- **WHEN** the trajectory or trajectory-plus-compiler arm is prepared
- **THEN** the artifact binds the exact gradient, optimizer configuration,
  proposed parameter delta, norm covariates, and update dose without post-hoc
  norm equalization

#### Scenario: Feasible preserved proposal
- **WHEN** the preservation constraints have a finite solution within the
  declared tolerance and trust radius
- **THEN** the projected delta is applied and its predicted and realized
  witness changes are receipted

#### Scenario: Infeasible or numerically uncertified projection
- **WHEN** the preservation solver cannot certify feasibility, finite values,
  or the exact applied delta
- **THEN** the proposal fails before behavioral interpretation and MUST NOT
  fall back to the unprojected update

#### Scenario: Frozen owner-wise witness bank
- **WHEN** the witness bank is materialized before acquisition
- **THEN** it holds one constraint per Source-emitted trusted owner and RP
  membership, selects the weakest full-vocabulary RP-processed margin inside
  that owner's sealed parser span, freezes the chosen and competitor tokens,
  and keeps legacy-M owners audit-only without a Jacobian

#### Scenario: Certified dose mechanics
- **WHEN** a qualification dose receipt is produced
- **THEN** its Jacobian-vector error is the maximum absolute difference between
  the frozen Jacobians applied to the actually applied projected delta and the
  re-maximized realized margin change, compared against the sealed first-order
  tolerance, and its decision-margin statistics use the deduplicated
  compiler/witness site union with each site scored on its own RP surface

#### Scenario: Unmeasurable witness surface
- **WHEN** the margin, Jacobian, or realized probe cannot be measured on the
  declared surface, disagrees with the frozen parameter layout, or is
  non-finite
- **THEN** the cell fails closed instead of publishing a synthesized mechanic

#### Scenario: Finite realized witness degradation
- **WHEN** a first-order-feasible projected delta has a finite realized witness
  degradation after application
- **THEN** the degradation is recorded and the cell continues to both
  behavioral audits rather than being removed from the matrix

### Requirement: Independent nested matrix and dual-RP audit
The matrix runner SHALL execute the exact nested arms, RP contracts, and seed
groups declared by the owning research unit.  Arms sharing an RP and seed
group MUST reuse byte-identical trajectory evidence, while every proposal MUST
use an independent Source model and fresh optimizer.  Each proposal SHALL be
audited once under clean greedy at every declared evaluation RP and then
restored; this screen MUST NOT publish or continue from an accepted model.

#### Scenario: Shared acquisition contrast
- **WHEN** the three nested arms for one RP and seed group are materialized
- **THEN** their acquisition, matching, owner weights, row ledgers, and
  advantages are byte-identical and only the declared arm additions differ

#### Scenario: Qualification vertical
- **WHEN** production-shaped qualification proposals are executed before the
  fixed matrix
- **THEN** they use disjoint sealed seeds, are excluded from matrix analysis,
  and can change only the single global learning rate through the predeclared
  mechanics-only dose rule, never from owner outcomes

#### Scenario: Dual-surface proposal audit
- **WHEN** one private proposal has been applied
- **THEN** the runner records trusted gains, named baseline losses, legacy
  owner identities, and output burdens under both clean-greedy RP surfaces

#### Scenario: Proposal completes normally
- **WHEN** both behavioral audits and artifact writes finish
- **THEN** the full model, optimizer, scheduler, counters, and random state are
  restored to their pre-proposal identity

#### Scenario: Proposal fails or rollback differs
- **WHEN** execution, evaluation, persistence, or rollback violates its sealed
  contract
- **THEN** a durable failure receipt is written and the affected run stops
  without adaptive retry or scientific success

### Requirement: Immutable bounded evidence and claim scope
Every artifact SHALL bind the Source checkpoint, manifest, trusted partition,
alias bank, policy contract, seeds, loss configuration, qualification dose
receipt, selected global learning rate, optimizer proposal, evaluation
surfaces, transaction hashes, and code/config identities.  The analyzer MUST
keep gains and named baseline losses separate and MUST reject an incomplete or
mixed matrix.  Results MUST remain labeled as same-panel, one-update mechanism
evidence.

#### Scenario: Complete bounded matrix
- **WHEN** every declared proposal has valid acquisition, update, dual-audit,
  and rollback evidence
- **THEN** the analyzer may report paired arm and RP comparisons within the
  owning research unit's claim boundary

#### Scenario: Missing or mixed evidence
- **WHEN** a cell is absent, duplicated, adaptively changed, or bound to a
  different Source, partition, policy, seed group, or update contract
- **THEN** the analyzer fails closed instead of pooling it into the matrix

#### Scenario: Incidental K-miss recovery
- **WHEN** a proposal happens to recover a legacy K-miss owner
- **THEN** the analyzer reports it separately and MUST NOT treat it as training
  credit, proof of support expansion, or a primary success
