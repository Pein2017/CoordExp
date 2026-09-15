## ADDED Requirements

### Requirement: Owner-aligned packed metadata

Each packed segment used by Stage 1 SHALL carry compact, segment-local owner metadata sufficient to reconstruct annotation trust class, stable source-owner identity, matched box target, row token bounds, description token bounds, geometry token bounds, assistant-start boundary, and every row-closing boundary. This metadata MUST remain aligned after causal shifting and MUST NOT refer across packed segments.

#### Scenario: Two images share one packed sequence
- **WHEN** two examples are packed into one training row
- **THEN** each boundary, owner row, and visual carrier MUST resolve only within its own segment
- **AND** no atom assignment, route set, or causal swap may cross the segment boundary

### Requirement: Permanent owner-bridge forward visibility

Bridge-enabled packed forward SHALL preserve the canonical Qwen `input_ids` path and Transformers-owned visual replacement while exposing the named architecture profile's post-block-20 visual-carrier states, row-token write positions, and post-final-RMSNorm boundary states. Row writes MUST affect only the selected segment and token positions, and the next boundary read MUST be able to observe upper-layer owner-conditioned KV state from completed row tokens.

When activation checkpointing recomputes an intercepted block during backward, the recomputation MUST restore the exact immutable typed bridge context used by the original logical forward, including segment mapping, branch identity, row positions, active owner value, schedule caps, and inventory identity. It MUST apply the identical bridge transformation without double-counting diagnostics or lifecycle side effects. Checkpointing on and off MUST agree within declared dtype tolerances on logits, scalar losses, and bridge/DoRA/selected-embedding gradients for ordinary TF and all four owner-use branches.

Disabled bridge mode MUST retain the existing packed-forward result contract. Bridge mode MUST fail closed if installed model structure cannot prove the profile's block count, hidden width, final normalization, image-token mapping, or per-segment positions.

#### Scenario: Same-layer query would miss upper-layer writes
- **WHEN** a bridge profile attempts to route the next row from the same post-block-20 seam used for row writes
- **THEN** forward validation MUST reject that profile because the query cannot observe blocks 21-27 owner-conditioned KV state

#### Scenario: Bridge disabled
- **WHEN** no permanent owner-bridge profile is configured
- **THEN** packed forward MUST preserve the existing logits, hidden-state request semantics, and segment isolation contract

#### Scenario: Checkpointed owner-use branch is recomputed
- **WHEN** non-reentrant activation checkpointing reruns an intercepted layer for any correct or swapped owner-use branch
- **THEN** it MUST receive the same branch-local bridge context as the original forward
- **AND** diagnostic counts MUST describe one logical branch rather than both physical executions

### Requirement: Truncated and packed owner-use branch execution

Owner-use branch execution SHALL support a performance lane that truncates each branch to its segment-local causal prefix up to the selected boundary plus its candidate row, and packs independent branches into one forward. The lane MUST be an exact rewrite of per-branch full-pack execution, not an approximation.

Truncation MUST preserve the candidate row's description and geometry target logits, MRoPE position ids, the segment's image-token mapping and visual carrier states, and the bridge context bound to that branch. Truncated branch logits, `L_use`, and gradients MUST agree with untruncated per-branch execution within declared dtype tolerance.

Packed branches MUST be isolated by FA2 `cu_seq_lens` boundaries at every branch start so attention never crosses a branch. Packed branches MUST NOT share prefix hidden state, KV state, or attention with one another; they share only model parameters and the image's atom inventory. The lane MUST publish a receipt recording the realized `cu_seq_lens_q` and `cu_seq_lens_k` boundaries and the packed-versus-isolated equivalence outcome.

The lane MUST NOT change pair selection, the description/geometry margin definition, the `L_use` objective, or any loss weight. Any equivalence mismatch MUST fail the lane rather than be accepted as a performance trade.

#### Scenario: Branch is truncated to its causal prefix
- **WHEN** a branch executes only its segment-local prefix plus candidate row instead of the full pack
- **THEN** its scored description and geometry logits MUST match full-pack execution within declared tolerance
- **AND** its gradients into the row adapter, selected owner values, and shared atomizer MUST match as well

#### Scenario: Two branches are packed into one forward
- **WHEN** independent owner-use branches are concatenated into a single packed forward
- **THEN** `cu_seq_lens` boundaries MUST prevent any attention from one branch's tokens to another branch's prefix or row
- **AND** packed results MUST equal per-branch isolated results on branch logits, `L_use`, and gradients

#### Scenario: Branch boundary is missing from the packed receipt
- **WHEN** the FA2 receipt's `cu_seq_lens` boundaries do not enclose every packed branch exactly once
- **THEN** the packed lane MUST fail closed rather than execute a contaminated forward
