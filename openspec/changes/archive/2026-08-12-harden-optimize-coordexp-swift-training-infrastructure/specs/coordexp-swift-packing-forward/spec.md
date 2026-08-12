## ADDED Requirements

### Requirement: FlashAttention Execution Proof Covers Every Text Layer

A packed forward proof MUST attest the backend and explicit cumulative
boundaries used by every executed Qwen text-attention layer. The proof SHALL
distinguish text-layer calls from vision-attention or unrelated attention calls,
SHALL compare the observed layer identities and call count with the resolved
model topology, and SHALL fail if any expected text layer is missing, duplicated,
or uses a different backend or boundary vector.

#### Scenario: Every expected text layer uses the configured backend

- **WHEN** a packed micro-step executes through all resolved Qwen text layers
- **THEN** the proof records one matching text-attention event per expected
  layer with the exact explicit cumulative boundaries and accepts the forward

#### Scenario: Only one matching attention call is observed

- **WHEN** one matching varlen call is observed but one or more expected text
  layers are unobserved or use an unverified path
- **THEN** the proof fails and names the missing or mismatched layer identities

#### Scenario: Vision attention also executes

- **WHEN** the forward includes vision-attention calls in addition to Qwen text
  attention
- **THEN** those calls are labeled separately and cannot satisfy or inflate the
  expected text-layer proof count

### Requirement: Packed Numerical Parity Remains The Acceptance Oracle

Backend execution proof MUST be paired with a versioned packed-versus-separate
numerical comparison on the same real model implementation. The release oracle
SHALL cover supervised logits, loss/objective semantics, and a boundary-only
negative control. The receipt SHALL also cover backward gradients for the exact
trainable parameter set as a retained diagnostic, without treating cross-shape
BF16 gradient equality as proof of segment isolation.
Wave 2 v3 SHALL preserve BF16-autocast Qwen attention/linear execution and the
production graph-connected FP32 output/loss seam; only serialized comparison
observations SHALL be detached.
Because the packed and separate arms are distinct BF16 forwards, every
mandatory per-term `raw_loss`, `weighted_loss`, `segment_mean_numerator`, and
`token_weighted_diagnostic` value SHALL be classified as BF16-derived,
compared in FP32 at the frozen BF16 real-model tolerance, and recorded with its
source and comparison dtypes. Term inventory, weights, exact denominator
semantics, and every per-term field SHALL remain mandatory even when total loss
passes.

The model-free v3 plan SHALL bind a structural inventory of exactly 196 LoRA-A,
196 LoRA-B, 196 DoRA magnitude, and one shared special-token delta parameter,
together with strict owner/suffix classification rules. After CPU model,
adapter, and delta installation but before GPU setup, exact installed names,
shapes, storage dtypes, gradient dtypes, expected presence, and BF16 compute
provenance SHALL be derived from the live model and authoritative adapter/delta
receipts and bound into the attempt-start marker and terminal receipt. That
concrete 589-row inventory SHALL be revalidated for both packed repetitions and
the separate reference. Every expected gradient SHALL be present and finite,
no extra gradient SHALL be accepted, and the complete surface SHALL have a
nonzero aggregate signal. Individual exact-zero tensors remain valid. Coverage
is a mandatory gate independent of the cross-shape numerical diagnostic. The
diagnostic result MUST be preserved, but after the immutable v3 result it SHALL
NOT veto the scoped forward/loss/FA2 release claim.

The current v3 reader SHALL retain live resolved runtime-config fingerprint
`da2a010eaacc6970c616e39a790372db43e6089357b9501d3b40f60f157fb5a9`
and SHALL authenticate immutable parent fingerprint
`de02f2664890109e1fbcf41b8f8d0fe1c4a226729e1d320b8cf5cae5b9b5463d`
only through
`coordexp-swift-wave2-config-compatibility-projection-v2`. That projection MUST
remove exactly `training.forward_input_provider_mode: synchronous`, the
`source_order_next_fit` packing policy, `window_size: null`, `lookahead: null`,
`seed: 0`, `worker_count: 1`, `fragment_item_budget: 1024`,
`fragment_byte_budget: 4194304`, `cursor_byte_budget: 65536`,
`max_packs_per_fragment: null`, and the complete
`{checkpoint_dir: null, mode: disabled}` resume object. It MUST
reject a missing, extra, reordered, or changed path/value row and either digest
drift without weakening generic whole-plan equality. The live resolved provider
mode MUST be reasserted as `synchronous` immediately before attempt-marker
publication and GPU setup, and the plan, marker, and terminal receipt MUST
retain the current config identity and runtime attestation. The executed
immutable v3 plan and failed receipt remain bound to their historical v1
projection and SHALL NOT be regenerated or reinterpreted.

Every BF16-compute-derived compared value, including FP32-stored trainable
gradients, SHALL retain the predeclared elementwise FP32
`torch.allclose(rtol=5e-3, atol=5e-3)`. Storage dtype SHALL NOT select a stricter
cross-forward tolerance. Cosine, relative-L2, adaptive near-zero floors, or a
post-result threshold change SHALL NOT retroactively change the immutable v3
result. Two identical packed arms SHALL each be compared with the production-
streaming separate arm and their diagnostic results retained. Their global FP32
maximum absolute gradient difference SHALL be at most `2.5e-3` for the
same-packed measurability control; a larger repeat difference or incomplete/
non-finite coverage is terminal `unmeasurable` and SHALL NOT select one packed
arm. A cross-shape gradient diagnostic failure with a measurable repeat does not
override byte-identical forward evidence or widen any band.
The diagnostic process SHALL require `FLASH_ATTENTION_DETERMINISTIC=1` before
plan preparation and runtime/model setup, and SHALL authenticate that value in
the plan and receipt. This probe-only determinism control SHALL NOT alter the
production default or dependency/backend disposition.
V3 SHALL NOT recompute or publish a parallel legacy storage-dtype comparison;
the immutable v2 failed receipt remains readable historical evidence and
cannot affect v3 acceptance.

For the frozen one-packed-context versus two-separate-context contrast,
denominator equality SHALL compare the complete term inventory plus
`term_name`, `denominator_scope`, `eligible_segment_count`,
`selected_atom_count`, and `skipped_segment_count` exactly for every term.
`context_count` SHALL be recorded and validated against the declared arm shape
(`1` for the packed arm and `2` for the shared separate arm), but SHALL NOT be
treated as a cross-arm semantic-equality field. Both arms SHALL additionally
bind the same term weights, normalizer/formula version, and resulting planned-
step denominator. This projection MUST pass before Accelerator construction,
model loading, or any comparison-arm forward.

#### Scenario: Packed and separate execution are equivalent

- **WHEN** identical segments are run through the accepted packed path and the
  separate-segment reference with matched model state and stochastic controls
- **THEN** supervised outputs, every mandatory BF16-derived per-term scalar,
  total loss, denominator semantics, and semantic-atom identities agree within
  the declared tolerances
- **AND** the complete finite trainable-gradient comparison is retained as a
  cross-shape diagnostic rather than silently omitted

#### Scenario: Cross-shape backward numerics fail after forward parity passes

- **WHEN** the immutable v3 receipt shows byte-identical supervised logits,
  accepted loss/denominator semantics, a measurable identical-packed repeat, a
  decisive boundary negative, and a complete all-layer FA2 proof, while the
  packed-versus-streaming gradient diagnostic fails its frozen band
- **THEN** the receipt remains an immutable terminal failure of that diagnostic
  and is not rescored, retried, or threshold-adjusted
- **AND** Wave 2 may release only the scoped forward/loss/boundary/FA2 claim
  under the recorded 2026-08-10 user decision
- **AND** exact cross-shape Jacobian equivalence is not claimed

#### Scenario: Separate execution reproduces production backward cadence

- **WHEN** the two one-segment reference micro-steps execute
- **THEN** gradients are cleared once before the arm and each forward/loss is
  followed immediately by its own ordered `accelerator.backward` call
- **AND** no summed differentiable two-forward graph is retained

#### Scenario: The same packed arm is repeated

- **WHEN** packed-primary and packed-repeat restore identical model, RNG,
  input, and train-mode state
- **THEN** each packed arm clears gradients exactly once immediately before its
  forward and performs one backward without an intervening clear
- **AND** their exact 589-row detached FP32 gradient inventories have global
  maximum absolute difference at most `2.5e-3`
- **AND** both packed arms independently satisfy the frozen packed-versus-
  separate comparison rather than choosing the better result

#### Scenario: Arm container counts differ but denominator semantics agree

- **WHEN** the packed arm contains one two-segment context and the shared
  separate arm contains two one-segment contexts
- **THEN** `context_count` is accepted only as the declared `1` versus `2`
  structural difference while every semantic denominator field, term weight,
  normalizer/formula version, and resulting planned-step denominator matches
  exactly

#### Scenario: A semantic denominator field drifts

- **WHEN** any term inventory, scope, selected-atom count, eligible/skipped
  segment count, term weight, normalizer/formula version, or resulting
  denominator differs across arms
- **THEN** the pre-arm denominator gate fails before Accelerator construction,
  model loading, or comparison-arm execution

#### Scenario: Segment isolation is intentionally broken

- **WHEN** the parity harness changes only the explicit FA2 segment-boundary
  vector from `[0,1436,2822]` to `[0,2822]` while model state, input IDs,
  images, position IDs, supervision, masks, loss wiring, and every other field
  remain unchanged
- **THEN** supervised logits or total loss leaves the frozen BF16 tolerance,
  demonstrating that the oracle detects cross-segment leakage
- **AND** a gradient-only mismatch does not satisfy this negative-control gate

### Requirement: Packing Policy Is Explicit Semantic Identity

Every packing policy MUST be an explicit strict-config value with a stable
algorithm identifier. Source-order next-fit SHALL remain the compatibility
default. One encoded image example SHALL be the atomic planner item. Under a
resolved sorted policy, packing MUST preserve the deterministic object/row
order inside that image and MUST NOT split or rewrite those rows. A policy MAY
reorder whole image examples across packs or batches and MAY change cross-image
co-presentation. Any policy that can alter membership, inter-image order, or
co-presentation MUST bind its parameters, deterministic seed, and algorithm
version into the cache fingerprint and run receipt. A requested policy MUST NOT
silently fall back to another policy.

#### Scenario: Compatibility configuration is used

- **WHEN** no experimental packing policy is selected
- **THEN** source-order next-fit is used and its algorithm identity is recorded
  without changing the accepted source-order semantics

#### Scenario: Fixed-window binpacking is selected

- **WHEN** a bounded binpacking policy is selected
- **THEN** its window size, tie-breaking rule, seed, and algorithm version are
  fingerprinted and the exact resulting pack membership and within-pack order
  are reproducible from the receipt

#### Scenario: Whole images are reordered under a sorted policy

- **WHEN** a packing policy reorders encoded image examples across packs or
  batches
- **THEN** every image appears exactly once in the declared epoch or stream
- **AND** the sorted object/row order inside each image is identical before and
  after packing
- **AND** the inter-image order and co-presentation change is deterministic and
  present in the cache identity and run receipt

#### Scenario: An unsupported policy is requested

- **WHEN** configuration names an unknown, retired, or semantically
  incompatible packing policy
- **THEN** validation fails before cache preparation or training begins

### Requirement: Online Packing Is Bounded And Replayable

An online packing implementation MUST use an explicit finite lookahead and
bounded host-memory budget, MUST NOT cache GPU vision features or trainable
hidden states, and MUST emit enough policy and cursor state to replay or resume
the exact presentation sequence. If exact replay is unavailable, the policy
MUST be marked non-resumable and rejected when exact-resume mode is requested.

#### Scenario: Online packing stays within its declared bounds

- **WHEN** a production-shaped stream is packed online
- **THEN** observed lookahead and host-memory usage do not exceed the configured
  bounds and every input example appears exactly once in the planned epoch

#### Scenario: Exact resume is requested with a non-replayable policy

- **WHEN** exact-resume mode is enabled but the selected online policy cannot
  restore its cursor, pending window, and deterministic decisions
- **THEN** launch validation fails before training state is mutated
