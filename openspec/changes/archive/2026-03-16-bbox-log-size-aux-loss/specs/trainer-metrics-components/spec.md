# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only **atomic objective contributions** under canonical `loss/<provenance>/<atom>` keys and MUST NOT emit raw component loss keys by default.

Definitions:
- An "objective atom" is a post-weighting contribution used in the trainer's total loss.
- For multi-term modules (notably bbox geometry + coord regularization), objective atoms are emitted per sub-term (no pre-summed aggregates).
- "Provenance" encodes which forward/context produced the objective for Stage-2 AB.

Normative behavior:
- Stage-2 AB and rollout-aligned trainers MUST emit only the following objective keys (minimum set), and only when the effective weight is non-zero:
  - Channel-A:
    - `loss/A1_text/{struct_ce,desc_ce}` (GT-anchor forward; token CE objective atoms)
    - `loss/A1_coord/{bbox_log_wh,bbox_log_area,bbox_oversize}` (from `bbox_size_aux`; optional A1 anchor-forward size-aux atoms)
    - `loss/A2_text/struct_ce` (final self-context forward; optional struct/EOS CE stabilizer atom)
    - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}` (from `bbox_geo`; final self-context forward; geometry objective atoms)
    - `loss/A2_coord/{bbox_log_wh,bbox_log_area,bbox_oversize}` (from `bbox_size_aux`; final self-context forward; size-aux objective atoms)
    - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (final self-context forward; coord_reg objective atoms)
  - Channel-B (rollout context):
    - `train/optimization/{loss_structure_ce,loss_description_ce,loss_dead_anchor_suppression}` (rollout-context forward; token/UL objective atoms)
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}` (from `bbox_geo`; rollout-context forward; geometry objective atoms)
    - `loss/B_coord/{bbox_log_wh,bbox_log_area,bbox_oversize}` (from `bbox_size_aux`; rollout-context forward; size-aux objective atoms)
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (rollout-context forward; coord_reg objective atoms)
- Raw/duplicate loss keys MUST NOT be emitted by default (non-exhaustive):
  - Per-component registry metrics: `loss/token_ce`, `loss/struct_ce`, `loss/desc_ce`, `loss/geo`, `loss/bbox_size_aux`, `loss/coord_reg`, `loss/coord_gate`, `loss/text_gate`
  - Legacy objective suffixes: `loss/token_ce_obj`, `loss/bbox_geo_obj`, `loss/coord_reg_obj`
  - Trainer-specific aliases: `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail`

#### Scenario: Canonical-only objective keys
- **WHEN** a Stage-2 or rollout-aligned training step emits objective metrics
- **THEN** emitted keys include only canonical `loss/<provenance>/<atom>` keys for registry-defined objective modules
- **AND** raw component loss keys and legacy aliases are absent.

#### Scenario: Channel-B emits loss_dead_anchor_suppression as a canonical objective atom
- **WHEN** a Channel-B training step applies duplicate unlikelihood
- **THEN** the emitted objective key is `train/optimization/loss_dead_anchor_suppression`
- **AND** no raw alias key for duplicate-unlikelihood is emitted.

## ADDED Requirements

### Requirement: Stage-1 bbox size aux SHALL use canonical geometry atom names
The Stage-1 trainer MUST emit canonical geometry atom names for the
`bbox_size_aux` plugin.

When Stage-1 bbox size auxiliary supervision is enabled through the Stage-1 aux
plugin host, the trainer SHALL emit the same geometry atom names used by later
plugin-owned geometry objectives.

Normative behavior:

- Stage-1 single-forward emission MUST use:
  - `loss/geo/{bbox_log_wh,bbox_log_area,bbox_oversize}`
- Stage-1 MUST NOT invent a second Stage-1-only loss namespace for the same
  bbox-size plugin math,
- because Stage-1 has one GT forward, these keys remain unsplit by provenance.

#### Scenario: Stage-1 plugin-owned bbox size aux uses canonical loss atoms
- **GIVEN** Stage-1 bbox size auxiliary is enabled
- **WHEN** a training step emits geometry objective atoms
- **THEN** the emitted keys use `loss/geo/{bbox_log_wh,bbox_log_area,bbox_oversize}`
- **AND** the same atom names remain recognizable relative to Stage-2
  provenance-split geometry atoms.
