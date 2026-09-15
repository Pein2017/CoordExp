## ADDED Requirements

### Requirement: Permanent owner-bridge loss bundle

Stage 1 SHALL optimize `L = L_AR + L_atom + 0.5 L_route + 0.1 L_use`, subject to the resolved ramps. `L_AR` MUST remain the canonical full-wrapper teacher-forced token cross entropy, including the ordinary COCO terminal token inherited from the S training contract. That terminal CE is native grammar supervision and MUST NOT be reinterpreted as a trusted direct null-routing label; `L_route` final-null supervision remains masked for `ordinary_partial`. Aggregate availability MUST be detached from atom-key, objectness, and router-null parameters before the admission map, so `L_AR` may train the soft admission map but cannot bypass the partial-label route mask. Bridge losses MUST be reduced with explicit eligible-count numerators and denominators across micro-steps and ranks; a rank with zero local eligible items MUST contribute a zero numerator and zero denominator without changing the global meaning.

`L_atom` SHALL supervise globally matched atom objectness and complete boxes, using L1 and generalized-IoU geometry terms. Trusted exhaustive examples MAY supervise unmatched-background objectness; every ordinary COCO unmatched atom MUST be masked in the first production leaf.

#### Scenario: Ordinary example has an unmatched atom
- **WHEN** any atom is unmatched in an ordinary COCO example
- **THEN** it MUST contribute neither positive nor negative objectness loss
- **AND** it MUST be excluded from route normalization rather than treated as background

#### Scenario: Ordinary terminal CE reaches admission
- **WHEN** full-wrapper CE supervises the terminal token of an `ordinary_partial` example
- **THEN** gradients MAY update the admission map and native trainable path
- **AND** gradients from that CE path MUST NOT update atom keys, objectness, or router-null parameters through aggregate availability

### Requirement: Normalized uncovered-set routing loss

For each non-final teacher-forced boundary, `L_route` SHALL be the negative log of the router probability mass assigned to the known-uncovered matched-owner set after normalizing over the trustworthy candidate domain. The domain MUST include known matched atoms and null, MUST exclude unknown ordinary-example unmatched atoms, and MAY include trusted exhaustive background atoms as negatives. Covered known owners remain in the domain, so moving mass from covered to uncovered owners lowers the loss. The loss MUST NOT assign a target order within the uncovered set.

At a trusted exhaustive final boundary, `L_route` SHALL supervise null probability. At an ordinary COCO final annotated boundary, final null/stop routing supervision MUST be masked.

#### Scenario: Uncovered-set cardinality changes
- **WHEN** two boundaries contain different numbers of uncovered owners
- **THEN** each loss MUST use normalized probability mass rather than an unnormalized logit sum whose scale changes with set size

#### Scenario: Sorted row is not router target
- **WHEN** the next teacher row is owner A but the router assigns most uncovered mass to owner B
- **THEN** routing loss MUST accept B as uncovered-set mass
- **AND** autoregressive row loss MUST still consume A's matched value rather than B's predicted value

### Requirement: Same-image causal owner-use loss

`L_use` SHALL compare two known-uncovered matched owners from the same image and the same teacher-forced boundary prefix by evaluating four one-row continuations: each owner row once with its correct value and once with the competitor value. It MUST separately measure average description-token likelihood and geometry-token likelihood, exclude wrapper/control tokens, and reward the diagonal owner-to-row pairing over the swap in both regions. Each selected value and each selected row MUST appear once on each side of the paired comparison so a generic activation-energy or row-continuation pulse cannot satisfy the objective. The four branches MUST share identical image and boundary-prefix hidden history; a wrong value in one branch MUST NOT alter the prefix used to score another branch. All four branches MUST preserve gradients through the row adapter, selected owner values, and shared atomizer; detaching or recomputing owner values under no-grad is non-conforming.

Pair sampling SHALL draw half of eligible pairs from same-class or spatially close competitors when available and half from deterministic random matched-owner pairs. Wrong-image values, global token permutations, and bridge-zero interventions MUST remain held-out diagnostics rather than training examples in Stage 1.

#### Scenario: Value carries only a continuation pulse
- **WHEN** correct and swapped values increase row likelihood equally
- **THEN** the causal owner-use margin MUST provide no identity-use credit

#### Scenario: Description binds but geometry does not
- **WHEN** the correct value improves description likelihood but not the matching coordinates relative to the swap
- **THEN** description and geometry owner-use metrics MUST expose the disagreement separately

#### Scenario: Earlier swapped row would contaminate a later prefix
- **WHEN** the two candidate owners occupy different positions in the canonical full trajectory
- **THEN** `L_use` MUST branch both candidate rows from one selected boundary prefix
- **AND** it MUST NOT score the later row after a counterfactually corrupted earlier-row hidden history

### Requirement: Rank-invariant collective choreography for bridge losses

Bridge loss execution SHALL emit an identical ordered sequence of distributed collectives on every rank of a planned step, independent of how many local micro-steps carry data and how many owner-use pairs are locally eligible.

Every physical accumulation slot on every rank MUST execute exactly one data-parallel-wrapped anchor forward and exactly one combined backward. For a real slot the anchor is the ordinary teacher-forced packed forward. For a shadow or no-data slot the anchor MUST be a deterministic differentiable dummy forward through the same wrapped model, whose output feeds the connected-zero surface so every trainable parameter receives a defined zero-valued gradient contribution. A slot MUST NOT construct its loss from parameters alone without a wrapped forward, and it MUST NOT execute more than one wrapped forward.

Every auxiliary `L_use` branch forward MUST execute through the unwrapped model, using the same underlying `Parameter` objects as the anchor forward. The local branch count MAY differ across ranks. Gradients from all branches MUST still reach the DoRA, selected-embedding, and bridge parameters through the slot's single combined backward. Existing eligible-count numerators, global denominators, and world-size scaling MUST remain unchanged, including the zero-numerator/zero-denominator contribution of a rank with no local eligible items.

Any standalone control collective used for rank reporting or finite gating MUST use a bounded timeout that is strictly below the effective NCCL watchdog timeout. That bound is a module-private static constant, not a configurable schema surface, and MUST NOT be exposed as one; it is enforced statically by a test that reads the constant and asserts the inequality. Extending that control timeout MUST NOT be treated as a repair for a divergent collective sequence.

This requirement changes execution shape only. It MUST NOT change pair sampling, loss weights, the loss formula, ramps, data order, annotation trust, effective batch size, cache identity, or any inference path.

#### Scenario: Rank has no eligible data for a slot
- **WHEN** one rank's accumulation slot is shadow or carries no data while another rank's slot at the same index is real
- **THEN** both ranks MUST execute exactly one wrapped anchor forward and one combined backward for that slot
- **AND** both ranks MUST enqueue the same ordered collectives for that slot

#### Scenario: Ranks have different eligible owner-use pair counts
- **WHEN** two ranks select different numbers of eligible owner-use branches in the same planned step
- **THEN** neither rank's collective count may change with its branch count
- **AND** the resulting global `L_use` numerator, denominator, and world-size scaling MUST equal the values produced before branch execution moved to the unwrapped model

#### Scenario: Control timeout would outlive the watchdog
- **WHEN** the static rank-report control timeout constant is set at or above the effective NCCL watchdog timeout
- **THEN** the static enforcement test MUST fail rather than allow the value to ship
- **AND** a divergent collective sequence MUST surface as a bounded failure rather than a prolonged stall
