## ADDED Requirements

### Requirement: Stage-2 correction uses grammar-valid self-prefix residual sets

The system SHALL build residual-set correction samples only from grammar-valid
compact-full self-prefixes whose emitted-object and active-object state can be
computed exactly.

Normative behavior:

- A valid correction prefix MUST contain completed emitted objects plus at most
  one active object prefix with a known grammar state.
- Malformed schema, unparseable object boundaries, broken coordinate arity,
  prompt/template mismatches, unknown token-role states, and prefixes whose
  active object cannot be uniquely located MUST be dropped or routed to a future
  format-repair path, not trained through residual-set correction.
- At an object-continuation boundary with non-empty `R_t`, the active candidate
  set SHALL initialize as `A_t = R_t`.
- At a valid prefix, the builder SHALL define:

```text
E_t = completed, grammar-valid, uniquely matched emitted labeled/UL objects
R_t = G*_k \ E_t
A_t = active compatible candidates for the current partial object prefix
```

#### Scenario: Malformed prefix is excluded

- **WHEN** a rollout prefix cannot be parsed into completed objects plus at most
  one known active object state
- **THEN** the builder does not emit residual-set correction atoms for that
  prefix
- **AND** the skip reason is counted in diagnostics.

#### Scenario: Valid object-boundary prefix yields residual state

- **WHEN** a grammar-valid prefix ends at an object boundary
- **THEN** the builder computes `E_t`, `R_t`, and valid next actions from the
  objects that remain in the rollout-local universe `G*_k`.

### Requirement: Completed emitted objects use desc-gated one-to-one matching

The system SHALL require canonical description equality before a completed
emitted object can match a labeled GT or rollout-local UL object.

Normative behavior:

- `matchable(pred, target)` requires canonical description id equality and
  geometry threshold pass.
- Matching SHALL be deterministic one-to-one over matchable pairs using geometry
  score and stable tie-breaks.
- Geometry-only matching MUST NOT remove labeled GT from `R_t`.
- During a prefix walk, completed objects SHALL be classified in rollout order
  against unconsumed labeled GT ids and unconsumed rollout-local UL ids; later
  completed objects MUST NOT retroactively reassign earlier emitted objects.
- Completed-object classification SHALL use this mutually exclusive precedence:
  labeled GT match, then `ul_promoted_local` match, then
  `repeated_object_boundary`, then `fp_boundary`.

#### Scenario: Wrong-description high-IoU prediction is not emitted GT

- **WHEN** a completed predicted object has high IoU with a GT object but a
  different canonical description id
- **THEN** that prediction does not match the GT object
- **AND** the GT object remains in the residual set.

#### Scenario: Later duplicate does not steal an earlier match

- **WHEN** a rollout emits two same-description objects that can overlap the
  same labeled or UL target id
- **THEN** the earlier completed object keeps its consumed target id when it is
  matched during prefix walk
- **AND** the later object is classified as `repeated_object_boundary` rather
  than retroactively replacing the earlier match.

### Requirement: Residual state machine emits transition-validated token actions

The system SHALL implement valid next-token enumeration through a residual-state
machine that emits token-level actions containing their next state.

Normative behavior:

- `ResidualStateMachine.valid_actions(state)` SHALL return `ValidAction` records
  with `token_id`, token role, transition-validated `next_state`, candidate
  subset after the action, and optional provenance metadata.
- Returned `ValidAction` records SHALL be unique by `token_id` for a state.
- When multiple candidates share the same next token, the action
  `candidate_subset_after` SHALL be the union of all candidates compatible with
  that token.
- A `ValidAction` SHALL set `selected_object_id` only when the token transition
  makes the candidate subset singleton.
- `valid_token_ids` for a supervision atom SHALL be derived from the returned
  action set.
- Corrected roll-in SHALL sample from `ValidAction`, not from a separately
  computed bare token-id set.
- A corrected roll-in action whose next state is invalid or empty is an
  invariant violation, not a normal training example.

#### Scenario: Valid action carries next state

- **WHEN** the state machine returns a valid coordinate action
- **THEN** that action includes the candidate subset that remains after
  consuming the coordinate token
- **AND** the corrected roll-in uses the action next state directly.

#### Scenario: Shared token does not commit branch early

- **WHEN** two active candidates share the same next description or coordinate
  token
- **THEN** the state machine emits one valid action for that token
- **AND** the action next state keeps both compatible candidates active.

#### Scenario: Invalid action is guarded

- **WHEN** a corrected roll-in action would produce an impossible or empty next
  state
- **THEN** strict mode raises an invariant error
- **AND** non-strict smoke/debug mode may drop the sample while emitting
  `invalid_builder_state` diagnostics.

### Requirement: Active candidate commitment controls marginal versus hard CE

The system SHALL use valid-set marginal only while the current active candidate
set is ambiguous and SHALL fall back to hard selected-path CE once the active
branch is committed.

Normative behavior:

- If `|A_t| > 1`, the token position uses residual-set valid-set marginal over
  the valid actions.
- If `|A_t| == 1`, the token position uses selected-object hard teacher-forcing
  CE for the committed candidate.
- If `|A_t| == 0` while consuming raw rollout tokens, the builder SHALL anchor a
  transition-failure event immediately before the token that made the active set
  empty.
- First-version coordinate transitions SHALL use exact `<|coord_k|>` token
  equality for candidate filtering. Any coordinate-neighborhood or transition
  tolerance behavior MUST be introduced through an explicit future ablation
  contract that changes both valid-action construction and diagnostics.

#### Scenario: x1 commits repeated same-description objects

- **WHEN** multiple remaining objects share the same description and differ at
  `x1`
- **THEN** the `x1` position uses a valid coordinate set
- **AND** the subsequent `y1/x2/y2` positions use hard CE if the selected `x1`
  action narrows `A_t` to one candidate.

#### Scenario: y1 remains ambiguous when x1 is shared

- **WHEN** multiple remaining objects share description and `x1` but differ at
  `y1`
- **THEN** `x1` does not commit the branch
- **AND** `y1` remains a valid-set marginal branch point.

#### Scenario: Coordinate transition is exact by default

- **WHEN** the active candidate set contains only objects with `x1=<|coord_120|>`
- **AND** the raw rollout emits `<|coord_121|>` at `x1`
- **THEN** the candidate set becomes empty for the raw transition
- **AND** the correction anchors before that coordinate token.

### Requirement: Residual-set atom weights preserve labeled and UL provenance

The system SHALL define scalar atom weights for residual-set valid-action
marginals so labeled GT and UL-promoted targets remain provenance-separated
without changing the core valid-action support.

Normative behavior:

- Core residual-set marginal support SHALL be the unweighted union of valid
  token actions for the current state.
- Each candidate target reachable through a valid action SHALL carry a
  provenance confidence weight: labeled GT `1.0`; `ul_promoted_local`
  `lambda_ul_promoted` (default `0.5`).
- A marginal atom's scalar `loss_weight` SHALL be the maximum provenance
  confidence weight among candidate targets reachable by its valid actions.
- If valid support contains any labeled-GT candidate, the marginal atom weight
  is `1.0`; if valid support contains only UL-promoted candidates, the marginal
  atom weight is `lambda_ul_promoted`.
- If one token action is shared by labeled and UL candidates, the token remains
  one valid action and inherits the maximum provenance confidence weight for
  atom weighting and diagnostics.
- After corrected roll-in commits to a singleton UL-promoted object, subsequent
  hard CE atoms SHALL use `lambda_ul_promoted`.
- Diagnostics SHALL report labeled-valid mass, UL-valid mass, and shared-token
  valid mass separately where applicable.

#### Scenario: Mixed labeled and UL support keeps full correction strength

- **WHEN** an ambiguous residual-set position has both labeled-GT and
  UL-promoted candidates in its valid-action support
- **THEN** the valid-token support remains the union of all valid actions
- **AND** the atom scalar loss weight is `1.0`
- **AND** labeled and UL valid masses are reported separately.

#### Scenario: UL-only support is downweighted

- **WHEN** an ambiguous residual-set position has only UL-promoted candidates in
  its valid-action support
- **THEN** the valid-token support remains the union of UL valid actions
- **AND** the atom scalar loss weight is `lambda_ul_promoted`.

### Requirement: Correction events anchor at the earliest actionable decision

The system SHALL select the earliest actionable correction event in sequence
order for a rollout by default.

Normative behavior:

- Transition failures SHALL anchor immediately before the invalid token.
- Premature STOP SHALL anchor at the prefix immediately before the STOP token;
  the atom `target_position` is the STOP decision slot and
  `logit_position = target_position - 1`.
- FP and repeated-object entries SHALL anchor immediately before the object
  starts.
- If multiple provenance tags describe the same anchor, the builder SHALL emit
  one correction event with combined provenance, not competing correction
  samples.

#### Scenario: Wrong coordinate anchors at coordinate slot

- **WHEN** a raw rollout emits a coordinate token that makes `A_t` empty
- **THEN** the correction event anchors before that coordinate token
- **AND** the target support is the valid coordinate action set at that slot.

#### Scenario: Repeated object anchors before object start

- **WHEN** a raw rollout emits a same-description object that repeats an already
  emitted labeled/UL object
- **THEN** the correction event anchors before that repeated object starts
- **AND** the target support is the remaining-object continuation set or STOP.

### Requirement: Corrected teacher-forced sequences use next-token alignment

The system SHALL compile correction events into teacher-forced atoms with causal
next-token alignment.

Normative behavior:

- For every emitted atom, `logit_position + 1 == target_position`.
- The selected target token MUST appear in the corrected input sequence at
  `target_position`.
- Raw bad tokens or wrong continuations SHALL be provenance/artifact evidence,
  not selected targets.
- Position validation SHALL check bounds, prompt/assistant boundary safety, and
  selected-token equality.

#### Scenario: Raw bad token is not the selected target

- **WHEN** a rollout emits an invalid `x1` token and the corrected roll-in
  selects a valid `x1` token
- **THEN** the atom selected token is the corrected valid token
- **AND** the raw invalid token is stored only as provenance.

#### Scenario: STOP target uses preceding logits

- **WHEN** the corrected target at an object boundary is STOP
- **THEN** `target_position` points to the `<|im_end|>` slot
- **AND** `logit_position` points to the previous causal token.

### Requirement: Corrected roll-in samples valid branches deterministically

The system SHALL use deterministic seeded sampling for ambiguous corrected
roll-in branches by default.

Normative behavior:

- Default roll-in policy SHALL be `random_valid_branch`.
- Default base seed SHALL be `17`.
- Default resampling policy SHALL be `fixed_event`, so the same
  sample/rollout/event id chooses the same branch across runs.
- A stable-first policy MAY exist only as a debug fallback and MUST NOT be the
  default training policy.

#### Scenario: Same event samples the same corrected branch

- **WHEN** the same correction event is built twice with the same base seed and
  identifiers
- **THEN** the selected corrected roll-in branch is identical.

### Requirement: Coordinate corrections supervise bbox-tail local spans

The system SHALL treat coordinate correction as a local bbox-tail span from the
anchor, not as one-token-only correction or full corrected suffix supervision.

Normative behavior:

- Anchor before `x1` supervises `x1,y1,x2,y2`.
- Anchor before `y1` supervises `y1,x2,y2`.
- Anchor before `x2` supervises `x2,y2`.
- Anchor before `y2` supervises `y2`.
- The span SHALL NOT extend into the next object unless an explicit future
  ablation mode is selected.

#### Scenario: y1 failure repairs bbox tail

- **WHEN** the first coordinate failure occurs at `y1`
- **THEN** the correction span emits atoms for `y1,x2,y2`
- **AND** it does not emit atoms for a later object.

### Requirement: STOP is a mutually exclusive token-level action

The system SHALL represent STOP as a token-level action and SHALL keep STOP and
object continuation mutually exclusive in the core valid-action set.

Normative behavior:

- If `R_t` is empty at an object boundary, valid actions are exactly STOP.
- If `R_t` is non-empty at an object boundary, valid actions are object
  continuations and STOP is absent.
- STOP SHALL use `<|im_end|>` and `TokenRole.STOP`.
- `<|endoftext|>` MUST NOT be used as EOS/STOP semantics.
- Continuation-vs-STOP margin defaults to `0.0` and MUST NOT change the core
  valid-action set.

#### Scenario: Premature STOP is corrected to continuation

- **WHEN** a raw rollout emits `<|im_end|>` while `R_t` is non-empty
- **THEN** the correction event target support excludes STOP
- **AND** it includes valid object-continuation actions.

#### Scenario: True stop has only STOP action

- **WHEN** `R_t` is empty at an object boundary
- **THEN** valid actions contain the `<|im_end|>` STOP action
- **AND** no object-continuation action is valid.

### Requirement: UL consensus promotes rollout-local positives with audit trail

The system SHALL support strict K-valid consensus mining of unlabeled objects
and SHALL keep promoted UL positives rollout-local and provenance-separated from
labeled GT.

Normative behavior:

- Consensus denominator SHALL be `K_valid`, the number of generated,
  parseable, grammar-valid, eligible rollouts.
- Promotion requires `K_valid >= min_ul_valid_rollouts`,
  `support_rollouts == K_valid`, and `support_ratio == 1.0`.
- Default `min_ul_valid_rollouts` SHALL be `3`.
- The first implementation SHALL validate `ul_consensus_ratio == 1.0`; weaker
  consensus ratios require a later explicit contract update.
- Each rollout SHALL contribute at most one vote per local same-description
  unmatched cluster after per-rollout pre-dedup.
- Per-rollout pre-dedup SHALL keep the earliest object in rollout order as the
  representative and suppress other same-rollout same-description near-duplicate
  members from UL voting and local UL-positive targets.
- Cross-rollout promotion SHALL require same canonical description id and
  complete-link all-pairs geometry pass.
- The complete-link geometry gate SHALL be configurable with explicit names for
  IoU minimum, center-distance scale, area-ratio maximum, and aspect-ratio
  maximum; authored configs and artifacts SHALL record the effective values.
- A discovered cluster that has same-description high-overlap with a consumed
  labeled GT object, consumed UL object, or already-emitted object in the same
  rollout group SHALL be quarantined or rejected before promotion. This is a
  promotion-safety monitor/gate, not a duplicate-specific training loss.
- A promoted UL object SHALL extend only the rollout-local universe `G*_k` and
  MUST NOT silently mutate dataset GT.
- Default UL loss weight SHALL be `0.5`.

#### Scenario: Consensus-promoted UL becomes rollout-local positive

- **WHEN** every valid rollout contributes one same-description unmatched object
  cluster member and complete-link geometry passes
- **THEN** the cluster is promoted as `ul_promoted_local`
- **AND** each rollout uses its own promoted member bbox as the local positive
  target.

#### Scenario: Same-rollout duplicate members do not vote twice

- **WHEN** one rollout emits multiple same-description unmatched boxes that form
  a local near-duplicate cluster
- **THEN** only the earliest object in rollout order may represent that rollout
  for UL voting and local UL-positive supervision
- **AND** suppressed members are duplicate-like diagnostics only.

#### Scenario: Cross-rollout duplicate burst is quarantined

- **WHEN** every valid rollout emits an unmatched same-description box that is
  close to a consumed labeled/UL/emitted object
- **THEN** the cluster is not promoted as `ul_promoted_local`
- **AND** the cluster artifact records the consumed-target overlap evidence and
  decision reason.

#### Scenario: Invalid rollouts do not lower consensus ratio

- **WHEN** some requested rollouts are invalid or ineligible
- **THEN** those rollouts are counted by skip reason
- **AND** the consensus ratio denominator is `K_valid`, not requested K
- **AND** promotion is blocked if `K_valid < min_ul_valid_rollouts`.

### Requirement: UL metrics and artifacts are reviewable and separated

The system SHALL report UL mining metrics separately from labeled GT metrics and
SHALL materialize detailed UL cluster evidence only when artifact dumping is
enabled.

Normative behavior:

- Metrics SHALL distinguish labeled TP, UL-promoted TP-like, labeled remaining,
  UL remaining, and mixed remaining cases.
- Residual-set UL metric keys SHALL use the stable prefix
  `stage2_ab/channel_b/residual_set/ul/`.
- Detailed cluster evidence SHALL be written under the Stage-2
  monitor/debug/smoke artifact root at relative path `ul_clusters.jsonl` only
  when artifact dumping is enabled.
- `ul_clusters.jsonl` SHALL include promoted, rejected, and quarantined
  discovered clusters.
- Each UL cluster artifact row SHALL include image id, description id/text,
  support rollout ids, support ratio, member boxes, pairwise geometry stats,
  overlap stats to labeled GT/emitted objects, decision, reason, and anchor or
  boundary metadata when available.
- `ul_promoted_local` SHALL NOT count as labeled recall evidence.

#### Scenario: UL artifact row is reviewable

- **WHEN** monitor dumps are enabled and a UL cluster is discovered
- **THEN** the Stage-2 artifact root contains `ul_clusters.jsonl` with one row
  for that cluster
- **AND** the row contains member boxes and decision evidence sufficient for
  later visualization/review.

#### Scenario: UL metrics do not inflate labeled recall

- **WHEN** a UL-promoted local object is used for training
- **THEN** labeled recall metrics remain based on dataset GT only
- **AND** UL usage is reported under separate UL metric keys.
