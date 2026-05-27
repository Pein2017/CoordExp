## ADDED Requirements

### Requirement: Stage-2 residual correction consumes offline prepared rollout attempts

The system SHALL train the residual-set correction objective from offline
prepared rollout attempts, not from online generation inside the training loop.

Normative behavior:

- New prepared rollout records MUST include `response_token_ids`, raw decoded
  text, decode metadata, generation config hash, and sample/image provenance.
- `response_token_ids` are required for new data; missing token ids MUST drop
  the sample with diagnostics unless an explicit legacy fallback mode is
  selected.
- Raw decoded text MAY be used for diagnostics, span recovery, and human review,
  but MUST NOT be the canonical training prefix in strict new-data mode.
- Prepared records are not required to store rollout-time absolute assistant
  spans; training assembly MUST recompute assistant span and logit positions in
  the active tokenizer/processor environment.
- Default first-version generation SHOULD use `K=4`. Temperature schedules may
  be homogeneous or explicitly mixed; after generation all attempts have equal
  semantic status.
- The residual-state dynamic valid set is the canonical Stage-2 trie /
  multiple-positive object. `candidate0`, anchor rollout, greedy rollout, first
  rollout, primary segment, or any single repaired candidate MUST NOT define the
  coordinate system for multiple-positive supervision.

#### Scenario: Missing response token ids drop strict samples

- **WHEN** strict new-data mode reads a prepared rollout record without
  `response_token_ids`
- **THEN** the record is not used for correction training
- **AND** a missing-token-id diagnostic is emitted.

#### Scenario: Rollout-time absolute span is ignored

- **WHEN** a prepared rollout record contains token ids and raw text
- **THEN** training assembly locates the assistant response span in the current
  tokenizer/processor context
- **AND** it validates logit positions without trusting rollout-time absolute
  spans.

### Requirement: Rollout attempts are equal self-prefix samples with exact deduplication only

The system SHALL treat K rollout attempts as K equal self-prefix contexts after
exact duplicate removal.

Normative behavior:

- Canonical terms are `rollout_attempt` and `rollout_id`; anchor/explorer terms
  MUST NOT be used by the new runtime contract.
- Exact duplicate attempts MAY be removed within one sample bundle using
  `response_token_ids` as the key, or exact raw text only in explicit legacy
  fallback mode.
- Approximate duplicates MUST remain separate training samples.
- UL consensus support MUST count distinct rollout ids after exact attempt
  deduplication.
- Default UL promotion MUST require full consensus from every valid retained
  rollout id and `min_ul_valid_rollouts=expected_num_rollouts=4`; if exact
  duplicate removal or invalid attempts leaves fewer than four valid ids, no UL
  pseudo-positive is promoted by default.
- Decode-mode-sliced diagnostics SHOULD report clean-success, invalid,
  dirty-correction, and promoted-UL support rates.

#### Scenario: Near-duplicate boxes remain separate attempts

- **WHEN** two rollout attempts produce similar boxes but different token
  sequences
- **THEN** both attempts remain eligible self-prefix samples
- **AND** they may provide distinct UL support if they pass all UL gates.

### Requirement: Template boundary adapter owns schema rendering and suffix slicing

The system SHALL use a Stage-1-compatible template boundary adapter for
constructed suffix rendering, tokenization, and slicing.

Normative behavior:

- Stage-2 builders MUST NOT manually concatenate schema strings such as
  newline, `<|object_ref_start|>`, `<|box_start|>`, or `<|im_end|>` outside the
  adapter.
- The adapter SHALL resolve or mirror the existing Stage-1 assistant template /
  serialization policy.
- The adapter SHALL be aligned with the existing Stage-1 surfaces:
  `src/detection/template.py::get_detection_template`,
  `RenderedAssistantSequence.object_entries`, separator/terminal spans, and
  `src/detection/tokenization.py::tokenize_rendered_detection_conversation`.
- The adapter SHALL expose object spans, separator spans, terminal/stop spans,
  token ids, and suffix-start slicing.
- If the resolved Stage-1 template renders newline or another separator at a
  position, that token is deterministic schema/control supervision, not an
  optional grammar alternative.
- The adapter MUST validate that prefix plus suffix does not duplicate
  deterministic schema tokens already present in the kept raw prefix.

#### Scenario: Prefix already contains separator

- **WHEN** the kept raw prefix already contains the deterministic Stage-1
  separator token for a boundary
- **THEN** the constructed suffix starts after that separator
- **AND** the adapter does not emit a duplicate separator.

### Requirement: Supervision atoms use canonical logit positions

The system SHALL represent residual-set supervision through shared
`SupervisionAtom` records with canonical `logit_position`.

Normative behavior:

- The conceptual residual-set atom fields are: `logit_position`, token type,
  role, `valid_token_ids`, selected token id, weight, target kind, and
  provenance.
- The current shared IR mapping MUST use
  `src/training/teacher_forcing/ir.py::SupervisionAtom` fields:
  - `batch_index`
  - `logit_position`
  - `target_position`
  - `allowed_token_roles`
  - `selected_token_role`
  - `valid_token_ids`
  - `selected_token_id`
  - `latent_valid_token_ids`
  - `coverage_target_weights`
  - `loss_tags`
  - `loss_weight`
  - `coord_role`
  - `provenance`
- In v1, `target_position` and `selected_token_id` are required because the
  shared IR validator checks selected-token equality against `input_ids`.
- Validation MUST require `logit_position + 1 == target_position`.
- Loss modules MUST consume `logit_position`; implementations MUST NOT create a
  separate peer field named `position_index`.
- Raw rollout bad tokens are provenance only and MUST NOT become selected
  targets.
- Conceptual token type / role / target kind MUST map explicitly to
  `allowed_token_roles`, `selected_token_role`, `coord_role`, `loss_tags`, and
  `provenance`; implementations MUST NOT invent a second incompatible atom
  schema.

#### Scenario: Atom validates next-token alignment

- **WHEN** an atom has both `logit_position` and `target_position`
- **THEN** validation requires the target token slot to be exactly one position
  after the supervised logits slot
- **AND** the target position does not cross prompt/assistant boundaries.

### Requirement: Residual-set loss combines type exclusivity and valid-set marginal

The system SHALL apply global token-type exclusivity and inner valid-set
marginal likelihood for residual-set atoms.

Normative behavior:

- Token-type exclusivity MUST be enabled by default for residual-set atoms.
- Default first-version coefficients SHALL be `lambda_type=1.0` and
  `lambda_inner=1.0`.
- Type loss owns global schema/text/coord mass supervision. Existing registry
  names may map schema/control to `struct` and free text to `desc`; the
  implementation MUST make this mapping explicit.
- Inner valid-set loss uses `-log sum p(valid_token_ids)` over the atom's valid
  set; hard CE, deterministic schema tokens, newline when template-rendered, and
  EOS are singleton valid-set cases.
- Malformed or ambiguous segments MUST be dropped/resynced/masked rather than
  weakening type supervision.

#### Scenario: Wrong token type is penalized

- **WHEN** an atom has token type `coord`
- **THEN** probability mass assigned outside the coordinate token group
  contributes to type loss
- **AND** the inner valid-set objective remains scoped to the atom's valid
  coordinate tokens.

### Requirement: Valid-set construction is action-based and transition-validated

The system SHALL derive valid token sets from token-level residual-state actions
whose next state is owned by the same state machine.

Normative behavior:

- The builder SHALL expose `ValidAction` records containing token id, token
  role, next state, candidate subset after the action, optional selected object
  id, optional object provenance, and optional coordinate slot.
- `SupervisionAtom.valid_token_ids` SHALL equal the set of token ids returned by
  `valid_actions(state)`.
- When corrected roll-in needs a selected token, the builder MUST select from
  `ValidAction` records using the deterministic seed policy, not from a bare
  token-id list.
- The selected action's `next_state` MUST drive all later suffix/atom
  construction.
- Once the active candidate set narrows to one labeled GT or rollout-local
  promoted-UL object, subsequent object-internal tokens SHALL use singleton
  selected-object supervision.
- Strict mode MUST fail, and non-strict smoke/debug mode MAY drop the sample, if
  a selected action has an invalid or empty next state.

#### Scenario: x1 action commits later bbox tokens

- **WHEN** two same-description objects diverge at `x1`
- **THEN** the `x1` atom uses valid-set marginal over valid `x1` actions
- **AND** the selected `x1` action's next state drives subsequent atoms
- **AND** later `y1/x2/y2` atoms are singleton hard-path atoms when that action
  commits to one object.

### Requirement: Multiple atoms in one rollout sequence use per-sequence weighted mean

The system SHALL allow one retained rollout attempt to contain multiple
correction atoms and SHALL normalize their loss by atom weights within the
sequence.

Normative behavior:

- One retained rollout attempt becomes one training sequence.
- All eligible non-conflicting atoms in that sequence are active by default.
- Atoms with the same `logit_position` and identical target merge provenance.
- Same-position conflicting targets MUST be diagnosed and resolved
  deterministically.
- Sequence loss SHALL be:

```text
sum_i atom_weight_i * atom_loss_i / max(eps, sum_i atom_weight_i)
```

- Batch loss SHALL mean over retained rollout-attempt sequences.
- Clean/dirty/UL/EOS provenance MUST be diagnostics only and MUST NOT introduce
  a second bucket-normalization layer.

#### Scenario: Dirty sequence with many atoms does not dominate by count

- **WHEN** one rollout sequence has many correction atoms and another has one
  atom
- **THEN** each sequence contributes one mean-like sequence loss to the batch
- **AND** the many-atom sequence is not multiplied merely by atom count.

### Requirement: Dirty prefix recovery masks unreliable tokens but keeps reliable context

The system SHALL support dirty-context recovery when boundaries and logit
positions are reliable.

Normative behavior:

- Rollout prefix tokens are masked by default and are not imitated.
- Complete invalid-geometry rows MAY remain dirty context, but they do not
  update emitted/remaining state and do not receive coordinate repair atoms.
- Structurally malformed middle spans MAY remain masked context only when a
  later reliable resynchronization boundary exists.
- No atoms or type loss may be created inside a malformed span.
- If no reliable boundary/logit position can be located, the sample MUST cut
  back to the last stable boundary or be dropped.
- Trailing incomplete object spans MUST be removed from their
  `<|object_ref_start|>` and the prefix must revert to the last stable boundary.

#### Scenario: Malformed span with reliable resync remains context

- **WHEN** a malformed object row is followed by a reliable object boundary
- **THEN** the malformed tokens may remain in the input as masked dirty context
- **AND** later reliable correction atoms may be trained.

### Requirement: Bbox geometry is a commitment gate, not a repair target

The system SHALL NOT perform raw-rollout coordinate repair in Stage-2 residual
correction v1.

Normative behavior:

- The system MUST NOT emit `bbox_tail_from_anchor` correction events.
- The system MUST NOT sort/clamp invalid boxes, use nearest-GT coordinate
  repair, apply coordinate-neighborhood tolerance, or train low-IoU coordinate
  refinement from raw rollout coordinates.
- A row commits only when it is legal positive-area, has exact normalized
  description match, and passes IoU `>= 0.75` against a remaining GT or promoted
  UL object.
- Rows are scanned in rollout order.
- Duplicate burst classification happens before commitment matching.
- If multiple remaining labeled GT candidates pass the commit threshold, the
  matcher SHALL choose exactly one by deterministic tie-break: IoU descending,
  center distance ascending, stable object id ascending.
- Only if no labeled GT candidate passes, the matcher applies the same
  deterministic tie-break to rollout-local promoted-UL candidates.
- The matcher removes exactly the selected id from remaining; all other
  candidates remain.
- Otherwise the row is uncommitted dirty context and the semantic remaining set
  is unchanged.

#### Scenario: Invalid x order does not become coordinate repair

- **WHEN** a rollout row has `x2 <= x1`
- **THEN** the row is uncommitted
- **AND** no coordinate repair atom is emitted for that row
- **AND** later boundary targets are derived from the unchanged remaining set.

### Requirement: Constructed suffix uses deterministic random remaining order

The system SHALL construct teacher-forced continuation suffixes from all
remaining supervision objects in deterministic random order.

Normative behavior:

- The base seed SHALL be `17`.
- The derived seed SHOULD include sample id, rollout id, and suffix-start
  boundary identity.
- GT and promoted UL objects SHALL be shuffled together; object source affects
  weight, not order.
- The suffix order is fixed for the built correction sequence in v1 and MUST
  NOT be resampled per epoch or optimizer step.
- Clean-success rollouts are skipped by default.
- Optional clean GT SFT stabilizer MAY be implemented, but default mix is `0`
  and promoted UL MUST NOT enter the clean stream in v1.

#### Scenario: Clean success is skipped

- **WHEN** a rollout commits all objects, has no dirty/uncommitted rows, and
  stops correctly
- **THEN** the correction builder emits no training sequence for that attempt
- **AND** clean-success diagnostics are counted.

### Requirement: UL mining uses strict consensus with near-GT gray-zone rejection

The system SHALL promote unlabeled objects only from strict cross-rollout
consensus over legal unmatched non-duplicate proposals.

Normative behavior:

- UL candidates MUST be legal positive-area bbox rows with normalized
  description in the dataset desc vocabulary.
- Invalid rows, malformed rows, duplicate bursts, spatial wrong-description
  conflicts, and OOV descriptions MUST NOT become UL candidates.
- A cluster must contain same normalized desc proposals from distinct rollout
  ids after exact attempt deduplication.
- Default cluster IoU threshold SHALL be `0.9`.
- Default `K_valid` minimum SHALL equal `expected_num_rollouts`, default `4`.
- Default support from distinct rollout ids SHALL equal `K_valid`; with default
  `ul_consensus_ratio=1.0`, every valid retained rollout id must support the
  cluster.
- If exact duplicate removal or invalid attempts leaves fewer than
  `expected_num_rollouts` valid retained rollout ids, no UL pseudo-positive is
  promoted by default.
- Default support/K_valid ratio SHALL be `1.0`.
- Promoted UL weight SHALL default to `0.5`.
- Same-desc IoU `>= 0.75` with GT rejects as GT conflict.
- Same-desc IoU in `[0.30, 0.75)` with GT rejects as near-GT gray zone and is
  review-only.
- No hard per-sample promoted-UL cap is applied in v1; high UL counts are
  diagnostics only.
- Cross-rollout consensus is an admission test only.
- For retained rollout attempt `k`, a promoted UL object SHALL use rollout
  `k`'s own promoted member description and bbox as the rollout-local
  supervision object.
- Cluster representative or medoid bboxes are review/artifact metadata only and
  MUST NOT replace rollout-local member bboxes for loss targets.
- A completed emitted object in rollout `k` may commit to and remove only its
  rollout-local promoted-UL id.
- UL ids, metrics, and losses MUST remain provenance-separated from labeled GT.

#### Scenario: Near-GT localization bias is not promoted

- **WHEN** a same-description UL candidate cluster has IoU `0.55` with a GT box
- **THEN** the cluster is not promoted
- **AND** the decision is recorded as near-GT gray-zone review.

#### Scenario: UL medoid is not the training bbox

- **WHEN** a promoted UL cluster has three member boxes and a medoid
  representative
- **THEN** rollout attempt `k` trains on its own promoted member bbox
- **AND** the medoid may appear only as review metadata in the UL artifact.

#### Scenario: Multiple promoted UL objects are not capped

- **WHEN** a sample has many UL clusters that pass all gates
- **THEN** all passing clusters enter the sample-local supervision set
- **AND** high UL/GT ratio diagnostics are emitted.

### Requirement: UL review artifacts are lightweight and replayable

The system SHALL write reviewable UL cluster records without default image
rendering.

Normative behavior:

- UL review records SHALL be written to the canonical relative path
  `monitor_dumps/ul_clusters.jsonl` when monitor/debug/smoke artifact dumping is
  enabled.
- Bbox fields MUST use canonical norm1000 xyxy coordinates.
- Rows MUST include enough image provenance to render later visualizations,
  including image/sample ids, dimensions when available, and image path when
  available.
- PNG visualization MUST NOT be generated by default.

#### Scenario: UL artifact supports later visualization

- **WHEN** a cluster is promoted or rejected for review
- **THEN** the JSONL row contains norm1000 member boxes, representative box,
  rollout ids, decision, reason, and image provenance.

### Requirement: Duplicate bursts are diagnostic uncommitted rows

The system SHALL treat duplicate bursts as uncommitted dirty rows without a live
duplicate-specific loss.

Normative behavior:

- Duplicate burst is same normalized desc, legal bbox, same rollout attempt,
  and IoU `>= 0.95` with an earlier legal prediction.
- Duplicate burst detection happens before GT/UL matching.
- Duplicate rows do not update emitted or remaining state.
- Duplicate rows cannot provide UL support.
- The live objective MUST NOT emit `loss_duplicate_burst_unlikelihood`.

#### Scenario: Duplicate burst cannot vote for UL

- **WHEN** two same-rollout predictions overlap each other with IoU `>= 0.95`
- **THEN** later duplicate rows are excluded from UL candidate collection.

### Requirement: Spatial wrong-description conflicts are low-weight review-only corrections

The system SHALL treat high-IoU wrong-description rows as possible label
conflicts, not as committed objects or UL candidates.

Normative behavior:

- A legal row with desc-agnostic IoU `>= 0.75` to a remaining GT/promoted UL but
  different normalized desc is `spatial_wrong_desc_conflict`.
- When text span and divergence position are reliable, it MUST create an
  earliest-divergence low-weight description correction atom; otherwise it MUST
  diagnose the no-atom reason.
- Its weight multiplier SHALL default to `label_conflict_weight=0.25`.
- It remains uncommitted and does not update emitted or remaining state.
- It is never a UL candidate and cannot be promoted by consensus.
- V1 does not create a standalone `label_conflict_review.jsonl`; compact
  monitor diagnostics and capped examples are sufficient.

#### Scenario: High-IoU wrong-desc row remains remaining

- **WHEN** a row predicts `chair` over a high-IoU `couch` GT
- **THEN** the row gets a low-weight desc correction when divergence is reliable
- **AND** the `couch` GT remains in the remaining supervision set
- **AND** the row is not a UL candidate.
