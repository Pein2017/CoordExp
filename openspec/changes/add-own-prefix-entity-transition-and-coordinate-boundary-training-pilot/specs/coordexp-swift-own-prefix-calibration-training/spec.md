## ADDED Requirements

### Requirement: Immutable Checkpoint-Bound State Bank

The system SHALL load rollout-calibration examples from an immutable state-bank
manifest bound to one source checkpoint, tokenizer and special-token identity,
prompt identity, processor controls, and image-grouped split assignment. Each
record MUST preserve exact executed prompt and prefix token identifiers and
hashes, candidate token identifiers, generation-policy provenance, physical-
entity references, separate entity and geometry review status, and selected
causal intervals. Unknown or ambiguous status on one review axis MUST produce
zero direct gradient only for that corresponding entity or geometry objective;
it MUST NOT disable a separately trusted objective axis.

#### Scenario: State bank belongs to another checkpoint

- **WHEN** a training config references a state bank whose source-checkpoint
  identity differs from the configured source adapter
- **THEN** validation MUST fail before model forward or optimizer setup.

#### Scenario: Entity is real but geometry is not trusted

- **WHEN** a state-bank candidate is entity-positive and geometry-unknown
- **AND** physical ownership is resolved before the first coordinate token
- **THEN** it MAY participate in entity-transition loss
- **AND** it MUST be excluded from coordinate-boundary loss.

#### Scenario: Candidate-row ownership requires coordinate tokens

- **WHEN** a scored positive or harmful candidate path requires one or more
  coordinate tokens to resolve its physical owner
- **THEN** the geometry of that exact candidate path MUST be reviewed as
  trusted before it participates in entity-transition loss
- **AND** a reviewer-corrected box MUST NOT silently replace the sampled path
- **AND** an independently trusted correction MAY instead define a separate
  first-wrong-coordinate objective.

#### Scenario: Entity ownership is unknown but geometry is independently eligible

- **WHEN** entity ownership is unknown and an independently reviewed
  same-owner geometry correction satisfies the coordinate eligibility contract
- **THEN** the record MUST receive zero entity-transition gradient
- **AND** it MAY participate in coordinate-boundary loss.

#### Scenario: Both review axes are unknown

- **WHEN** both entity and geometry review status are unknown
- **THEN** the record MUST receive zero direct research gradient.

#### Scenario: Image-grouped split is inspected

- **WHEN** sibling candidates, alternate sampling seeds, or counterfactuals
  share one image identity
- **THEN** the manifest MUST assign all of them to the same split group.

### Requirement: Exact Image and Token Replay

The training path SHALL recompute the full Qwen3 Vision-Language model
(Qwen3-VL) image and language forward from the current parameters using the
stored exact token identifiers. It MUST
NOT reconstruct a stored prefix by decoding it to text and tokenizing it again,
and it MUST NOT use frozen hidden states or key-value tensors as training
inputs. Token, image, prompt, processor, and special-token parity MUST be
verified before an event contributes gradient.

#### Scenario: Exact prefix would retokenize differently

- **WHEN** human-readable evidence text tokenizes differently from the stored
  exact prefix token identifiers
- **THEN** training MUST use the stored identifiers
- **AND** MUST record or report the text mismatch without replacing the exact
  prefix.

#### Scenario: Prefix hash is invalid

- **WHEN** stored token identifiers do not match their declared hash
- **THEN** state-bank loading MUST fail before forward computation.

### Requirement: Atomic Candidate Event Packing

Every valid and harmful candidate for one calibration event SHALL be encoded as
an isolated causal segment and SHALL remain in the same planned optimizer step.
Companion typed metadata MUST map event identity, candidate role, nullable
physical owner, coverage status, selected logits positions, nullable owner-
resolution interval, coordinate eligibility, and intended token type without
changing ordinary `TokenAtom` target semantics.

#### Scenario: Candidate group would cross a planned-step boundary

- **WHEN** packing would place part of one calibration event in a different
  planned optimizer step
- **THEN** the planner MUST keep the event atomic or fail with a visible
  capacity diagnostic
- **AND** the loss MUST NOT compute from a partial candidate group.

### Requirement: Entity-Transition Preference Objective

The entity-transition objective SHALL compare one actual harmful greedy branch
against one or more verified, sampled, uncovered physical-entity paths at the
same exact image and prefix. Each entity-owned path score MUST cover the
next-row decision through the shortest stored token prefix that resolves
physical ownership. Multiple token aliases for one physical entity MUST be
reduced to one owner score by taking the maximum admitted alias-path score.
Multiple valid owner scores MUST be combined by a configured smooth maximum
rather than by treating one canonical next owner as the sole target. A harmful
premature terminal candidate MUST have a null physical owner and null owner-
resolution interval; its path score MUST be exactly the single terminal-token
log probability at the shared boundary.

Only a physical duplicate or a verified premature terminal action with a
same-prefix sampled rescue MAY be the harmful branch in the initial profile. A
valid new greedy entity, official-annotation mismatch, unsupported-but-
ambiguous entity, or entity reachable only under another prefix MUST NOT be
used as a negative.

Every scored nonterminal positive or harmful path whose owner-resolution
interval contains an actual coordinate token MUST have trusted geometry for
that exact path. This rule is independent of whether the candidate is enabled
for the coordinate-boundary objective. A corrected reference box does not make
an untrusted sampled row eligible for complete candidate-row scoring.

#### Scenario: Several uncovered entities are valid

- **WHEN** two or more verified uncovered physical entities occur in the same
  exact-prefix sample support
- **THEN** the loss MUST aggregate them as multiple valid positives
- **AND** MUST NOT force one geometry-sorted owner to be the unique positive.

#### Scenario: One owner has duplicate token aliases

- **WHEN** two or more admitted candidate paths resolve to the same physical
  owner
- **THEN** the objective MUST use only their maximum path score as that owner's
  score
- **AND** adding an exact duplicate alias MUST NOT change owner weight.

#### Scenario: Greedy branch is a valid new entity

- **WHEN** the actual greedy branch resolves to a verified uncovered physical
  entity
- **THEN** that branch MUST be committed for continued state collection
- **AND** MUST NOT be admitted as a harmful negative.

#### Scenario: Premature terminal lacks same-prefix rescue

- **WHEN** terminal output occurs but no verified uncovered candidate is
  sampled at the same exact prefix
- **THEN** the event MUST be diagnostic-only and MUST receive no
  entity-transition gradient.

#### Scenario: Verified premature terminal is harmful

- **WHEN** terminal output is the admitted harmful branch and a verified
  same-prefix uncovered rescue exists
- **THEN** the harmful candidate MUST use null owner metadata and the single
  terminal token as its complete score interval.

### Requirement: First-Wrong-Coordinate Objective

The coordinate-boundary objective SHALL require one uniquely trusted physical
owner, trusted corrected geometry, and a reviewed discrete acceptable-token
set. It MUST supervise only the first generated coordinate outside that set in
the initial profile, using the exact earlier generated tokens as context. The
objective MUST compare aggregate acceptable coordinate-token logits with the
actual wrong-token logit in 32-bit floating point. It MUST NOT create Gaussian
coordinate targets. The acceptable set MUST be nonempty, contain unique integer
coordinate tokens in `[0, 999]`, and use only horizontal tolerance for `x1` or
`x2` and only vertical tolerance for `y1` or `y2`.

#### Scenario: First coordinate is accepted and second is wrong

- **WHEN** `x1` lies in its accepted set and `y1` is the first coordinate
  outside its accepted set
- **THEN** the direct coordinate objective MUST select `y1`
- **AND** MUST NOT directly supervise `x1`, `x2`, or `y2` for that event.

### Requirement: Prefix Coverage Scope

The StateBank SHALL record the number of prior object rows and one typed prefix
coverage status: `empty`, `resolved`, or `unresolved`. Entity-transition events
MUST use `resolved` coverage and MUST prove one trusted physical owner for each
prior row. A coordinate-only event MAY use `unresolved` coverage only when its
exact prefix tokens are preserved, its current-row owner and geometry are
independently trusted, all candidates are entity-ineligible, and candidate
coverage is `unknown`. Such an event MUST NOT support a novelty, commitment,
covered-set redistribution, or coverage-aware enumeration claim.

#### Scenario: Coordinate boundary is trusted but prior ownership is unresolved

- **WHEN** a nonempty exact prefix contains at least one prior row without a
  unique trusted physical owner
- **AND** the current-row owner, geometry, earlier accepted coordinates, and
  first wrong coordinate are independently reviewed
- **THEN** the event MAY be admitted for coordinate-only or matched gate-only
  training with `prefix_coverage_status=unresolved`
- **AND** it MUST have zero prefix-owner proofs, disabled entity-transition
  eligibility, entity-ineligible candidates, and `coverage_status=unknown`.

#### Scenario: Entity transition uses unresolved prefix coverage

- **WHEN** an event with unresolved prefix coverage enables an entity-
  transition candidate or objective
- **THEN** StateBank validation MUST fail before loss computation.

#### Scenario: Row mixes physical owners

- **WHEN** a row's geometry cannot be uniquely assigned to one physical owner
- **THEN** coordinate supervision MUST be disabled unless an independently
  reviewed same-owner corrected target is present.

#### Scenario: Acceptable coordinate set crosses its axis or range

- **WHEN** an acceptable set is empty, contains a duplicate or non-integer
  coordinate, leaves `[0, 999]`, or uses vertical tolerance for an `x` boundary
  or horizontal tolerance for a `y` boundary
- **THEN** state-bank validation MUST fail before loss computation.

### Requirement: Intended Token-Type Gate on Selected Research Sites

Every candidate token position selected by a rollout-calibration objective
SHALL have exactly one intended token type and SHALL receive the configured
positive-weight token-type gate. The intended type MUST come from the valid
semantic phase rather than being inferred blindly from a harmful token. At a
shared boundary where premature terminal output is harmful, the intended type
MUST be object-row schema rather than terminal output.
Gate-site identity MUST be causal segment plus logits position. All declarations
for one site MUST agree on the intended token type; conflicting declarations
MUST fail validation. Repeated identical declarations MUST be counted once.
Distinct sites MUST be averaged within each event and then averaged over
eligible events in the complete planned optimizer step.

#### Scenario: Harmful terminal branch is selected

- **WHEN** the entity-transition loss compares a valid next-row path against a
  premature terminal token at the shared prefix boundary
- **THEN** the boundary token-type gate MUST allow the object-row schema group
- **AND** MUST NOT simultaneously gate the boundary as terminal output.

#### Scenario: Selected site lacks an intended type

- **WHEN** calibration metadata selects a gradient-bearing token position but
  does not declare one unambiguous intended type
- **THEN** event validation MUST fail before loss computation.

#### Scenario: One site is selected by both research objectives

- **WHEN** transition and coordinate metadata refer to the same causal segment
  and logits position with the same intended token type
- **THEN** the token-type gate MUST count that site once in the event mean.

#### Scenario: One site declares conflicting intended types

- **WHEN** two metadata records refer to the same causal segment and logits
  position but declare different intended token types
- **THEN** event validation MUST fail before loss computation.

### Requirement: Rollout-Only Research Profiles

The strict training configuration SHALL expose transition-only,
coordinate-boundary-only, coordinate-boundary-gate-only, and joint rollout-
calibration profiles. These
profiles MUST consume only the configured frozen rollout state bank and MUST
NOT mix canonical supervised-fine-tuning examples, full-row base
cross-entropy, Kullback-Leibler divergence anchoring, or online bank refresh.
The joint profile MUST normalize entity-transition and coordinate-boundary
terms independently before applying their configured weights.

The coordinate-boundary-gate-only profile MUST consume the same coordinate
events and selected sites as the coordinate-boundary-only profile, set both
research-objective weights to zero, and retain only the configured rollout-site
token-type gate. It is a matched training control, not an unchanged-source
baseline.

#### Scenario: Canonical supervised data is mixed into a calibration run

- **WHEN** a rollout-calibration config also declares a canonical
  supervised-fine-tuning data mixture or positive full-row base-cross-entropy
  weight
- **THEN** strict config validation MUST fail before model loading.

#### Scenario: Joint arm has no eligible geometry events

- **WHEN** a complete planned optimizer step for the joint profile contains
  transition events but zero eligible geometry events
- **THEN** the step MUST fail or be skipped according to an explicit visible
  incomplete-objective policy
- **AND** it MUST NOT silently renormalize the joint arm into transition-only.

#### Scenario: Coordinate gate-only control activates a research objective

- **WHEN** the coordinate-boundary-gate-only profile has a nonzero entity-
  transition or coordinate-boundary objective weight
- **THEN** strict config validation MUST fail before model loading.

### Requirement: Calibration Metrics and Run Evidence

Rank zero SHALL bind the state-bank identity and research profile in existing
resolved config and run artifacts. Each completed training step MUST expose raw
and weighted research losses, per-term global eligible-event denominators,
positive-versus-harmful margins, coordinate acceptable-mass margins,
token-type legal mass, ignored and unknown counts, and finite status. No
non-main rank may create a parallel durable run tree.

#### Scenario: Training step completes

- **WHEN** a rollout-calibration optimizer step is applied
- **THEN** the wide training log row MUST contain all enabled research-term
  values and eligibility counts
- **AND** the state-bank manifest identity MUST remain traceable from the run
  record.

### Requirement: Calibration Smoke Ladder

The implementation SHALL provide a deterministic one-transition-plus-one-
geometry smoke and an 8-to-16-state smoke before the formal screen. Before
implementation worktrees are forked, one immutable lead-published smoke
fixture manifest MUST bind the shared source checkpoint, state-bank checksum,
exact event identifiers, optimizer and loss values, and random seeds. The first
smoke MUST verify exact replay, eligibility masks, intended token types,
32-bit floating-point loss math, finite gradients, and the direction of every
declared margin after a tiny optimizer update. The second smoke MUST execute
ordinary free-row inference from the produced checkpoint and verify that no
canonical supervised-fine-tuning replay was consumed.

#### Scenario: One-event update moves a margin backward

- **WHEN** the first smoke applies an optimizer update and any enabled target
  margin moves in the opposite direction from its declared objective
- **THEN** the smoke MUST fail
- **AND** the formal training screen MUST remain blocked.

#### Scenario: Implementation uses a different smoke event

- **WHEN** an implementation smoke substitutes a checkpoint, state bank,
  event, optimizer value, loss value, or seed that differs from the shared
  fixture manifest
- **THEN** its smoke result MUST be rejected as non-comparable.

### Requirement: Inference Architecture Remains Unchanged

A checkpoint produced by rollout-calibration training SHALL be loadable by the
existing Weight-Decomposed Low-Rank Adaptation inference composition and SHALL
run through ordinary greedy Qwen3-VL decoding without a state bank, new model
head, external detector, object slot, object query, persistent ledger, or
custom inference-time controller.

#### Scenario: Calibrated checkpoint is evaluated

- **WHEN** post-training greedy inference loads the calibrated adapter
- **THEN** it MUST use the existing inference request, model composition,
  forward, parsing, scoring, and artifact paths
- **AND** no rollout state-bank record may be required at inference time.
