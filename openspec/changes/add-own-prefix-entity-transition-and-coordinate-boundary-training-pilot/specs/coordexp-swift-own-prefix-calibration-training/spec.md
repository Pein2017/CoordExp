## ADDED Requirements

### Requirement: Immutable Checkpoint-Bound State Bank

The system SHALL load rollout-calibration examples from an immutable state-bank
manifest bound to one source checkpoint, tokenizer and special-token identity,
prompt identity, processor controls, and image-grouped split assignment. Each
record MUST preserve exact executed prompt and prefix token identifiers and
hashes, candidate token identifiers, generation-policy provenance, physical-
entity references, separate entity and geometry review status, and selected
causal intervals. Entity-transition records MUST additionally preserve the
fixed row budget, fixed generated-token budget, target-owner retention,
verified-owner-set delta, and confirmed downstream-harm verdict used for
admission. Unknown or ambiguous status on one review axis MUST produce
zero direct gradient only for that corresponding entity or geometry objective;
it MUST NOT disable a separately trusted objective axis.

#### Scenario: Strict replay state bank belongs to another checkpoint

- **WHEN** a training config references a state bank whose source-checkpoint
  identity differs from the configured source adapter
- **AND** explicit off-policy StateBank replay is disabled
- **THEN** validation MUST fail before model forward or optimizer setup.

### Requirement: Explicit Compatible Off-Policy StateBank Replay

The rollout-calibration configuration SHALL keep strict checkpoint equality as
its default. A separately declared off-policy replay mode MAY warm-start
training from a checkpoint different from the immutable StateBank trajectory
source. In that mode the system MUST preserve and report both identities. It
MAY differ only in the language-adapter fingerprint and selected-token
embedding-delta fingerprint. Base configuration, tokenizer, token identity,
special-token identity, and processor identity MUST match exactly. The bank's
record-level generation provenance MUST continue to name the bank trajectory
source and MUST NOT be rewritten as the training warm-start.

#### Scenario: Compatible later checkpoint replays an old StateBank

- **WHEN** explicit off-policy replay is enabled
- **AND** the bank trajectory source and training warm-start differ only in
  adapter and selected-token embedding-delta fingerprints
- **THEN** training MAY consume the immutable old StateBank
- **AND** rank-zero run evidence MUST record both complete checkpoint
  identities and `off_policy_state_bank_replay=true`.

#### Scenario: Off-policy replay crosses a model or token boundary

- **WHEN** explicit off-policy replay is enabled but base configuration,
  tokenizer, token identity, special-token identity, or processor identity
  differs
- **THEN** validation MUST fail before model forward or optimizer setup.

#### Scenario: Off-policy replay is not explicitly enabled

- **WHEN** the warm-start checkpoint differs from the StateBank trajectory
  source and off-policy replay is absent or false
- **THEN** the existing strict checkpoint-mismatch failure MUST remain in
  effect.

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
against one or more verified, sampled, uncovered physical-entity rows at the
same exact image and prefix. For each positive row, the objective MUST locate
the first token at which the positive and harmful paths diverge. The branch
margin MUST compare the positive token and harmful token at that shared causal
history in 32-bit floating point. It MUST NOT compare summed path
log-probabilities over unequal token horizons.

Multiple token aliases for one physical entity MUST be reduced to the strongest
admitted alias. When multiple distinct valid owners remain, the branch term
MUST ask that at least one valid owner beat the harmful token rather than force
one geometry-sorted owner as the sole target. A harmful premature terminal
candidate MUST have a null physical owner and null owner-resolution interval;
its harmful branch token MUST be exactly the single terminal token at the
shared boundary.

The selected positive row SHALL also receive coherent continuation supervision
from the first divergence through row closure. Schema-and-description sites
and trusted coordinate sites MUST be mean-normalized as separate non-empty
groups before being combined, so description length and coordinate count do
not silently change event weight. Geometry-unknown coordinate sites MUST
receive zero exact-token continuation gradient. Historical prefix tokens and
unrelated generated rows MUST receive no continuation gradient.

Only a physical duplicate or a verified premature terminal action with a
same-prefix sampled rescue MAY be the harmful branch in the initial profile. A
valid new greedy entity, official-annotation mismatch, unsupported-but-
ambiguous entity, or entity reachable only under another prefix MUST NOT be
used as a negative.

Every coordinate token selected for coherent continuation MUST have trusted
geometry for that exact sampled path. A corrected reference box does not make
an untrusted sampled coordinate token eligible. If physical ownership becomes
distinguishable only through untrusted coordinate tokens, the candidate MUST
remain diagnostic-only for the initial entity-transition profile.

Every gradient-eligible entity-transition event MUST also contain fixed-budget
counterfactual admission evidence. The positive row MUST retain its intended
owner under the same row and token budgets as native greedy decoding, and the
counterfactual MUST NOT add a confirmed duplicate, malformed row, or verified
unsupported entity. Unknown suffix rows MUST remain neutral. Trajectory-level
quality MAY decide admission but MUST NOT silently become a continuous event
weight in this profile.

#### Scenario: Several uncovered entities are valid

- **WHEN** two or more verified uncovered physical entities occur in the same
  exact-prefix sample support
- **THEN** the loss MUST retain them as multiple valid positives grouped by
  physical owner
- **AND** MUST NOT force one geometry-sorted owner to be the unique positive.

#### Scenario: One owner has duplicate token aliases

- **WHEN** two or more admitted candidate paths resolve to the same physical
  owner
- **THEN** the objective MUST use only the strongest admitted alias as that
  owner's branch candidate
- **AND** adding an exact duplicate alias MUST NOT change owner weight.

#### Scenario: Positive row and harmful branch have unequal lengths

- **WHEN** a complete positive row is compared with a one-token premature
  terminal branch
- **THEN** the preference margin MUST compare only their first divergent
  tokens under the shared history
- **AND** positive-row continuation MUST be mean-normalized separately from
  that branch margin.

#### Scenario: Positive row has trusted entity but untrusted geometry

- **WHEN** physical ownership is resolved before coordinate tokens and exact
  sampled geometry is not trusted
- **THEN** trusted schema and description continuation sites MAY receive
  gradient
- **AND** coordinate continuation sites MUST receive zero exact-token gradient.

#### Scenario: Ownership requires an untrusted coordinate

- **WHEN** the first owner-resolving divergence occurs at a coordinate whose
  geometry is not trusted
- **THEN** the candidate MUST be diagnostic-only in the initial profile.

#### Scenario: Counterfactual only reorders owners

- **WHEN** the candidate row appears locally but does not improve the admitted
  fixed-budget owner set or reduce one confirmed native harm
- **THEN** the event MUST receive zero entity-transition gradient.

#### Scenario: Counterfactual receives extra capacity

- **WHEN** the candidate arm uses a larger row or generated-token budget than
  the native arm
- **THEN** the event MUST fail admission before training.

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
coverage status: `empty`, `resolved`, `target_scoped_noncoverage`, or
`unresolved`. Duplicate-negative entity-transition events MUST use `resolved`
coverage and MUST prove one trusted physical owner for each prior row. A
premature-terminal entity-transition event MAY instead use
`target_scoped_noncoverage` when it proves one positive target is absent from
every prior row while leaving all unrelated prior-row owners unknown. A
coordinate-only event MAY use `unresolved` coverage only when its
exact prefix tokens are preserved, its current-row owner and geometry are
independently trusted, all candidates are entity-ineligible, and candidate
coverage is `unknown`. Such an event MUST NOT support a novelty, commitment,
covered-set redistribution, or coverage-aware enumeration claim.

A target-scoped non-coverage proof MUST name the positive target owner, record
one exclusion receipt for every prior prediction, bind the geometric tests and
thresholds used, and reject any plausible same-target association. It MAY
support only “prefer this verified missed target row over premature STOP.” It
MUST NOT support complete covered-set, duplicate-redistribution, or downstream
owner-set claims.

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

#### Scenario: Target-scoped non-coverage is used against premature terminal

- **WHEN** a verified positive target is conservatively excluded from every
  prior prediction and the actual harmful branch is premature terminal
- **THEN** the event MAY enable entity-transition training with
  `prefix_coverage_status=target_scoped_noncoverage`
- **AND** all unrelated prior-row owners MUST remain unknown.

#### Scenario: Target-scoped non-coverage is used against a duplicate

- **WHEN** a duplicate harmful branch or a complete covered-set claim uses
  `prefix_coverage_status=target_scoped_noncoverage`
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
coordinate-boundary-only, coordinate-boundary-gate-only, joint, and
positive-path-imitation-only, and
sampled-path-and-source-route-imitation-only rollout-calibration profiles. These
profiles MUST consume only the configured frozen rollout state bank and MUST
NOT mix canonical supervised-fine-tuning examples, full-row base
cross-entropy, Kullback-Leibler divergence anchoring, or rollout collection
inside the trainer.
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

#### Scenario: Offline trajectory refresh is staged between training runs

- **WHEN** a completed checkpoint is used by ordinary inference to construct a
  new immutable StateBank before a later training process starts
- **THEN** the later run MAY consume that refreshed bank
- **AND** this MUST NOT be represented as online StateBank refresh inside one
  trainer process.

### Requirement: Positive Sampled-Path Row Imitation

The positive-path-imitation-only profile SHALL consume StateBank events with
exactly one positive candidate and no harmful candidate. Historical prefix
tokens and unrelated generated rows MUST remain conditioning context with zero
imitation gradient. Each eligible event MUST declare a positive complete-row
site mask, an image-balanced event weight, and whether each selected coordinate
site has trusted geometry. Entity-transition and coordinate-boundary
eligibility MUST be disabled for this profile.

The complete-row loss MUST be computed in 32-bit floating point. It SHALL first
average negative log probability over selected schema-and-description sites and
selected trusted-coordinate sites as separate non-empty groups, then average
the available group means, then multiply by the declared image-balanced event
weight. The token-type gate SHALL use the same selected sites and SHALL receive
the same event weight. The profile MUST NOT apply gradient to terminal tokens,
untrusted coordinate sites, prefix tokens, unresolved rows, duplicate rows, or
malformed rows.

#### Scenario: One sampled history leads to several verified rows

- **WHEN** several verified first-occurrence rows from one admitted sampled
  route occur no later than its last added-owner row
- **THEN** each eligible row MAY become one positive-path event with its exact
  sampled prefix
- **AND** the sum of declared unscaled row weights for that image MUST equal
  one before any global mean-one rescaling.

#### Scenario: Prefix contains an unresolved row

- **WHEN** an eligible current row follows one or more unresolved sampled rows
- **THEN** those rows MAY remain in the exact prefix as zero-gradient context
- **AND** the run evidence MUST expose the unresolved-prefix-row count for that
  event.

#### Scenario: Coordinate geometry is not trusted

- **WHEN** a positive row has trusted physical ownership but one or more exact
  sampled coordinate sites are not trusted
- **THEN** those coordinate sites MUST receive zero imitation and gate gradient
- **AND** any separately trusted schema-and-description sites MAY remain
  eligible only when physical ownership is resolved without the untrusted
  coordinates.

#### Scenario: Positive-path event declares a harmful candidate

- **WHEN** a positive-path-imitation-only event contains a harmful candidate or
  enables entity-transition or coordinate-boundary eligibility
- **THEN** StateBank or profile validation MUST fail before model forward.

### Requirement: Sampled-Path and Source-Route Complete-Row Imitation

The sampled-path-and-source-route-imitation-only profile SHALL reuse the
positive complete-row imitation loss and selected-site token-type gate for two
separately identified event families. A sampled-path event MUST retain sampled
generation provenance and positive-path eligibility. A Source-route
preservation event MUST retain greedy generation provenance and Source-route
eligibility. The two eligibility flags MUST be mutually exclusive for one
event, and the profile MUST reject every other rollout-calibration event
family.

Each admitted event MUST contain exactly one trusted positive candidate, no
harmful candidate, one complete-row owner interval, and no terminal site. The
candidate owner MUST be uncovered in its own exact route prefix. The existing
complete-row loss grouping, trusted-coordinate omission, 32-bit floating-point
math, image-balanced event weight, and token-type gate semantics SHALL remain
unchanged. StateBank and run evidence MUST report sampled-path and Source-route
event counts separately.

The historical positive-path-imitation-only profile MUST continue to admit
only sampled positive-path events. Adding the mixed profile MUST NOT broaden
the meaning of an existing bank or run.

#### Scenario: Exact greedy Source row is admitted for preservation

- **WHEN** an event is Source-route-imitation eligible and its single trusted
  positive row was generated by greedy decoding from the bound Source
  checkpoint
- **THEN** its canonical generation provenance MUST declare greedy mode
- **AND** the mixed profile MUST apply the same complete-row imitation and
  token-type-gate math used for a sampled positive row.

#### Scenario: Source row is mislabeled as sampled

- **WHEN** a Source-route-imitation event declares sampled generation mode
- **THEN** StateBank validation MUST fail before model forward
- **AND** free-form review metadata MUST NOT override the canonical generation
  provenance.

#### Scenario: Sampled route is mislabeled as greedy

- **WHEN** a positive-path-imitation event declares greedy generation mode
- **THEN** StateBank validation MUST fail before model forward.

#### Scenario: Event enables both route families

- **WHEN** one event enables both positive-path imitation and Source-route
  imitation
- **THEN** StateBank validation MUST fail before model forward.

#### Scenario: Historical sampled-only profile reads a Source event

- **WHEN** the positive-path-imitation-only profile receives a Source-route-
  eligible event
- **THEN** that event MUST NOT be admitted into its training exposure.

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

#### Scenario: Off-policy training step completes

- **WHEN** a training step consumes a StateBank generated by a compatible
  checkpoint other than its warm-start checkpoint
- **THEN** the run record MUST make both checkpoint identities and the
  off-policy declaration traceable.

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
