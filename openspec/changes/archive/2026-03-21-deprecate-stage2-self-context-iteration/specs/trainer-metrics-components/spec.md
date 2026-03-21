# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only atomic
objective contributions under canonical `loss/<provenance>/<atom>` keys and
MUST NOT emit raw component loss keys by default.

Definitions:
- An "objective atom" is a post-weighting contribution used in the trainer's
  total loss.
- For multi-term modules (notably bbox geometry + coord regularization),
  objective atoms are emitted per sub-term (no pre-summed aggregates).
- "Provenance" encodes which forward/context produced the objective for
  Stage-2 AB.

Normative behavior:
- Stage-2 AB and rollout-aligned trainers MUST emit only the following
  objective keys (minimum set), and only when the effective weight is non-zero:
  - Channel-A:
    - `loss/text/{struct_ce,desc_ce}` (GT-anchor forward; token CE objective
      atoms)
    - `loss/coord/{bbox_smoothl1,bbox_ciou,bbox_log_wh,bbox_oversize,coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
      when Channel-A bbox/coord supervision is active on the single-pass
      GT-anchor forward
  - Channel-B (rollout context):
    - `train/optimization/{loss_structure_ce,loss_description_ce,loss_dead_anchor_suppression}`
      (rollout-context forward; token/UL objective atoms)
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}` (from `bbox_geo`;
      rollout-context forward; geometry objective atoms)
    - `loss/B_coord/{bbox_log_wh,bbox_oversize}` (from `bbox_size_aux`;
      rollout-context forward; size-aux objective atoms)
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
      (rollout-context forward; `coord_reg` objective atoms)
- `loss/A1_*` objective atoms MUST NOT be emitted by active Stage-2
  two-channel training.
- `loss/A2_*` objective atoms MUST NOT be emitted by active Stage-2
  two-channel training.
- Raw/duplicate loss keys MUST NOT be emitted by default (non-exhaustive):
  - Per-component registry metrics: `loss/token_ce`, `loss/struct_ce`,
    `loss/desc_ce`, `loss/geo`, `loss/bbox_size_aux`, `loss/coord_reg`,
    `loss/coord_gate`, `loss/text_gate`
  - Legacy objective suffixes: `loss/token_ce_obj`, `loss/bbox_geo_obj`,
    `loss/coord_reg_obj`
  - Trainer-specific aliases: `loss/ce`, `loss/coord`, `loss/coord_prefix`,
    `loss/coord_tail`

#### Scenario: Canonical-only objective keys
- **WHEN** Stage-2 two-channel training emits objective metrics
- **THEN** Channel-A text metrics use `loss/text/*`
- **AND** Channel-A bbox/coord metrics use `loss/coord/*`
- **AND** the same run does not emit `loss/A1_*` or `loss/A2_*`.

### Requirement: Coord distribution diagnostics are provenance-split in Stage-2 two-channel
Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) MUST attribute coord-distribution diagnostics to the forward surface that produced the logits used for the coord-vocab slice.

Rationale:
- Channel-A now runs one GT-anchored teacher-forced forward.
- Stage-1 runs only one forward and reports coord distribution monitors without
  forward provenance.
- Provenance-splitting still matters for distinguishing Channel-A anchor
  behavior from Channel-B rollout behavior.

Normative behavior:
- Stage-2 two-channel MUST emit coord-vocab distribution monitors under:
  - `coord_diag/*`: computed from Channel-A GT-anchor logits.
  - `coord_diag/B/*`: computed from Channel-B logits (rollout-context forward).
- `coord_diag/A1/*` and `coord_diag/A2/*` MUST NOT be emitted.
- The set of per-provenance keys SHOULD include (non-exhaustive):
  - `coord_diag/<prov>/coord_tokens_total`
  - `coord_diag/<prov>/{acc_top5,p_gt_mean,margin_mean,expected_bin_mae,expected_bin_abs_err_p90,coord_vocab_mass_mean}`
- The bare `coord_diag/*` namespace is the canonical single-pass Channel-A
  coord group after self-context removal.

#### Scenario: A1/A2 coord diagnostics are absent after deprecation
- **WHEN** Stage-2 two-channel executes with coord diagnostics enabled
- **THEN** coord diagnostics are emitted only under supported provenance keys
- **AND** the same run does not emit `coord_diag/A1/*` or `coord_diag/A2/*`.

## ADDED Requirements

### Requirement: Loss-gradient monitor uses the normal single-pass coord group for Channel-A
When Stage-2 loss-gradient monitoring emits per-term coord diagnostics, it MUST
use the normal single-pass Channel-A coord group name rather than iterative
Channel-A provenance names.

Normative behavior:
- Channel-A coord monitor terms MUST use `coord/<atom>`.
- Channel-B coord monitor terms MAY continue to use `B_coord/<atom>`.
- `A1_coord/<atom>` and `A2_coord/<atom>` MUST NOT be emitted by the active
  loss-gradient monitor for Stage-2 two-channel training.

#### Scenario: Stage-2 gradmon no longer emits A1/A2 coord groups
- **WHEN** Stage-2 two-channel loss-gradient monitoring is enabled
- **THEN** emitted coord monitor term names may include `coord/<atom>` and
  `B_coord/<atom>`
- **AND** the same monitor payload does not emit `A1_coord/<atom>` or
  `A2_coord/<atom>`.
