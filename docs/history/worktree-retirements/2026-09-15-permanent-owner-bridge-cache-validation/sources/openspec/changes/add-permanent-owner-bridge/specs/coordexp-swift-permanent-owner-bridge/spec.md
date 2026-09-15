## Purpose

Define a permanent, owner-addressable interface that turns late visual evidence into a row-local latent and lets native autoregressive history learn which labeled owners remain available.

## ADDED Requirements

### Requirement: Permanent owner-bridge composition

A bridge-enabled checkpoint SHALL include the owner bridge in both training and inference computation. The native language-model residual path, native image-token path, native `<|object_ref_start|>` decision, and native stop decision MUST remain available. Setting every bridge contribution to zero MUST recover the corresponding native base-plus-adapter computation without requiring a different token schema.

#### Scenario: Bridge checkpoint used for inference
- **WHEN** inference loads a checkpoint trained with the permanent owner bridge
- **THEN** it MUST load and execute the matching bridge payload
- **AND** it MUST fail before generation rather than silently omit that payload

#### Scenario: Zero bridge contribution
- **WHEN** bridge contributions are diagnostically forced to zero
- **THEN** the forward path MUST retain native visual attention, language residuals, tokenizer semantics, and row grammar

### Requirement: Static local owner atoms

For the Qwen3-VL-2B Stage 1 profile, the system SHALL derive exactly four locally anchored atoms from every merged visual carrier after zero-based language block 20. Each atom MUST share one latent trunk and expose an objectness value, one complete carrier-relative box, one 128-dimensional normalized routing key, and one 512-dimensional normalized owner value. The atom inventory MUST be computed once per image-conditioned sequence and remain static while rows decode.

The four slot identities MAY break symmetry but MUST NOT impose a fixed quadrant-to-owner meaning. The bridge MUST NOT add a closed-set class head or treat atom fields as four independent owner identities.

#### Scenario: Several owners share one visual carrier
- **WHEN** multiple labeled owners are assigned near the same merged visual carrier
- **THEN** distinct local slots MUST be available for injective owner assignment
- **AND** their keys and values MUST remain separately addressable

#### Scenario: Row history changes
- **WHEN** another object row is appended to the autoregressive history
- **THEN** static atom objectness, boxes, keys, and values for that image MUST remain unchanged

### Requirement: Global injective owner assignment

Stage 1 SHALL match labeled physical owners to the full image's carrier-by-slot atom inventory with one global injective assignment and a soft locality prior. Every labeled owner MUST receive exactly one atom, and one atom MUST match at most one labeled owner. Assignment MUST consider box geometry and objectness; locality MAY narrow candidates or break ties but MUST NOT create independent per-carrier duplicates of one owner. If the local candidate graph cannot support a complete injective assignment, matching MUST expand to the full inventory; if the full inventory is insufficient or no complete finite assignment exists, the example MUST fail before loss computation rather than run a teacher row without an owner value.

For exhaustively labeled or human-refined examples, unmatched atoms MAY receive background objectness supervision and the exhausted boundary MAY supervise null. For ordinary COCO examples, every unmatched atom MUST remain unknown-neutral in the first production leaf, regardless of overlap with a labeled owner. Cross-image persistent owner identifiers MUST NOT be introduced.

#### Scenario: Adjacent carriers propose the same owner
- **WHEN** several atoms around adjacent carriers fit one labeled box
- **THEN** the global assignment MUST select at most one of them as that owner's positive atom
- **AND** every remaining atom MUST stay unknown-neutral when the example is `ordinary_partial`

#### Scenario: Ordinary COCO annotation is exhausted
- **WHEN** all annotated rows in an ordinary COCO example have appeared
- **THEN** unmatched atoms and the final null state MUST NOT be treated as proof that the visual scene contains no further owners

#### Scenario: Complete owner assignment is impossible
- **WHEN** a labeled example has more owners than atoms or no finite complete injective assignment after full-inventory fallback
- **THEN** training MUST fail that example before row latching or loss computation
- **AND** it MUST NOT substitute a zero, arbitrary, or repeated owner value

### Requirement: Order-agnostic set routing

At each assistant-start or completed-row boundary, the router SHALL read the post-final-RMSNorm boundary representation and produce one normalized distribution over eligible atoms plus a learned internal null state. The router MUST score owner-specific keys and MAY use objectness only as a bounded quality prior; it MUST NOT use a hard covered-set mask or persistent external ledger.

During Stage 1, routing supervision SHALL reward total normalized probability mass on any known-uncovered matched owner at a non-final boundary. Covered matched owners and null remain competing alternatives, but no individual uncovered owner is designated as the mandatory next row. Only trusted exhaustive examples MAY supervise null after all known owners are covered.

#### Scenario: Two uncovered owners remain
- **WHEN** a teacher-forced prefix has covered some owners and two known owners remain uncovered
- **THEN** routing loss MUST depend on the sum of probability assigned to both uncovered owners
- **AND** it MUST NOT penalize the router merely for preferring either uncovered owner

#### Scenario: A row becomes covered
- **WHEN** the next teacher-forced boundary follows a completed owner row
- **THEN** that owner's atom remains in the router distribution but moves from the uncovered set to the covered set
- **AND** the router is trained to reallocate normalized mass toward the remaining uncovered set

### Requirement: Soft admission and row-local hard commitment

Boundary admission SHALL use only the router's aggregate non-null versus null availability, not a soft mixture of owner values. Its residual contribution MUST be capped at three percent of the boundary hidden state's root-mean-square magnitude. Admission MAY influence the native opener/stop logits but MUST NOT force either token.

Once the native decoder emits `<|object_ref_start|>`, inference SHALL latch the highest-scoring non-null atom from that boundary and hold its owner value through the matching `<|box_end|>`. The latch MUST be local to one row and one batch sequence; it MUST clear after the row-closing boundary has been evaluated. If native decoding emits an opener while null was top-ranked, the runtime MUST still latch the highest-scoring non-null atom and record the disagreement.

Teacher-forced Stage 1 rows SHALL use their globally matched gold-owner atom as the active row value while the router is trained independently by set mass. A predicted atom that names another valid uncovered owner MUST NOT be paired with the current teacher row.

#### Scenario: Native decoder chooses to continue
- **WHEN** `<|object_ref_start|>` is emitted at a boundary
- **THEN** exactly one non-null owner value MUST be latched for that row
- **AND** the runtime MUST NOT change that owner before `<|box_end|>`

#### Scenario: Native decoder chooses to stop
- **WHEN** the native decoder emits its legal terminal token instead of an object opener
- **THEN** no row owner MUST be latched
- **AND** the bridge MUST NOT synthesize an additional row

### Requirement: Late row write and final boundary read

The Qwen3-VL-2B Stage 1 architecture profile SHALL write the latched owner into every row token after zero-based block 20, beginning with `<|object_ref_start|>` and ending with `<|box_end|>` inclusive. The write MUST be a zero-initialized owner-conditioned interaction, retain the native residual, and cap its per-token update at ten percent of that hidden state's root-mean-square magnitude.

Blocks 21 through 27 SHALL therefore form owner-conditioned row hidden states and KV entries. The router for the next boundary MUST read after the final RMS normalization, so the `<|box_end|>` boundary can observe the current row's upper-layer owner-conditioned history before its latch is cleared. Write depth and read depth MUST be selected by the named architecture profile and MUST NOT be independently user-configurable in Stage 1.

#### Scenario: Completed row informs the next query
- **WHEN** a latched owner is written through the row-closing token
- **THEN** the final boundary representation MUST be computed from upper-layer KV state that includes those owner-conditioned row writes
- **AND** the next routing distribution MUST be derived from that final representation

#### Scenario: Bridge starts from a production checkpoint
- **WHEN** the zero-initialized row-write output is first enabled
- **THEN** its initial update MUST be exactly zero before the configured ramp begins
- **AND** the native checkpoint computation MUST remain intact

### Requirement: Stage 1 teacher-forced curriculum

Stage 1 SHALL consist only of full teacher-forced trajectories. It MUST perform four complete data presentations in this order: authored `geo_sorted`, deterministic `random-1`, authored `geo_sorted`, deterministic `random-2`. The two random presentations MUST use different reproducible permutations for eligible multi-owner examples. All row tokens remain teacher forced in all four presentations.

Set routing in Stage 1 MUST derive covered and uncovered sets from the gold prefix and stable source-owner identities. Self-generated prefixes, rollout buffers, replayed atom choices, actor/learner scheduling, hard without-replacement masks, and reinforcement learning MUST NOT be part of this change.

#### Scenario: Second random presentation
- **WHEN** an eligible example reaches `random-2`
- **THEN** its row order MUST be generated from the declared run seed, example identity, and presentation identity
- **AND** it MUST differ from `random-1` whenever more than one distinct permutation exists

#### Scenario: Stage 1 completes
- **WHEN** all four teacher-forced presentations finish safely
- **THEN** the system MUST publish a bridge-complete checkpoint usable by supported HF inference
- **AND** publication MUST NOT depend on a natural-rollout quality threshold
