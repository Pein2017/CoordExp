# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Metric namespace hierarchy is explicit and stable
The system SHALL keep the metric namespace hierarchy explicit so operators can
infer meaning from the key shape alone.

Normative behavior:
- `loss/<provenance>/<atom>` denotes a post-weighting objective atom.
- `coord_diag/<metric>` is reserved for Stage-1-style bare coord diagnostics.
- `coord_diag/<provenance>/<metric>` is reserved for provenance-split Stage-2 coord diagnostics.
- `stop_signal/<provenance>/<metric>` is reserved for adaptive stop-signal diagnostics:
  - `gt` denotes single-forward GT teacher forcing,
  - `A1` denotes Stage-2 Channel-A GT-anchor teacher forcing.
- `gradmon/<metric>` and `gradmon/<group>/<term>` are reserved for optional loss-gradient diagnostics.
- `rollout/*` and `eval/detection/*, eval/parsing/*, eval/description/*, eval/config/*, eval/runtime/*` remain distinct training-vs-eval families.
- Internal reducer-helper keys MUST remain underscore-prefixed and MUST NOT appear in the final logged payload.

#### Scenario: Stop-signal diagnostics use explicit provenance
- **WHEN** stop-signal damping metrics are emitted for Stage-1 or Stage-2 Channel-A
- **THEN** they are emitted under `stop_signal/gt/*` or `stop_signal/A1/*`
- **AND** ambiguous bare `stop_signal/*` keys are absent.

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only **atomic objective contributions** under canonical `loss/<provenance>/<atom>` keys and MUST NOT emit raw component loss keys by default.

Definitions:
- An "objective atom" is a post-weighting contribution used in the trainer's total loss.
- For multi-term modules (notably bbox geometry + coord regularization), objective atoms are emitted per sub-term (no pre-summed aggregates).
- "Provenance" encodes which forward/context produced the objective for Stage-2 AB.

Normative behavior:
- Stage-2 AB and rollout-aligned trainers MUST emit only the following objective keys (minimum set), and only when the effective weight is non-zero:
  - Channel-A:
    - `loss/A1_text/{struct_ce,desc_ce,stop_signal_ce}` (GT-anchor forward; token CE objective atoms, with `stop_signal_ce` optional and emitted only on eligible dense object-list targets when stop-signal damping is enabled)
    - `loss/A1_coord/{bbox_smoothl1,bbox_ciou,bbox_log_wh,bbox_oversize,coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` when `application.preset` routes Channel-A bbox/coord supervision to the anchor forward
    - `loss/A2_text/struct_ce` (final self-context forward; optional struct/EOS CE stabilizer atom)
    - `loss/A2_coord/{bbox_smoothl1,bbox_ciou,bbox_log_wh,bbox_oversize,coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` when `application.preset` routes Channel-A bbox/coord supervision to the final self-context forward
  - Channel-B (rollout context):
    - `train/optimization/{loss_structure_ce,loss_description_ce,loss_dead_anchor_suppression}` (rollout-context forward; token/UL objective atoms)
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}` (from `bbox_geo`; rollout-context forward; geometry objective atoms)
    - `loss/B_coord/{bbox_log_wh,bbox_oversize}` (from `bbox_size_aux`; rollout-context forward; size-aux objective atoms)
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (rollout-context forward; coord_reg objective atoms)
- Raw/duplicate loss keys MUST NOT be emitted by default (non-exhaustive):
  - Per-component registry metrics: `loss/token_ce`, `loss/struct_ce`, `loss/desc_ce`, `loss/stop_signal_ce`, `loss/geo`, `loss/bbox_size_aux`, `loss/coord_reg`, `loss/coord_gate`, `loss/text_gate`
  - Legacy objective suffixes: `loss/token_ce_obj`, `loss/bbox_geo_obj`, `loss/coord_reg_obj`
  - Trainer-specific aliases: `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail`

#### Scenario: Canonical-only objective keys
- **WHEN** a Stage-2 or rollout-aligned training step emits objective metrics
- **THEN** emitted keys include only canonical `loss/<provenance>/<atom>` keys for registry-defined objective modules
- **AND** raw component loss keys and legacy aliases are absent.

#### Scenario: Channel-A stop signal uses a canonical objective atom
- **WHEN** Channel-A GT-anchor token CE enables stop-signal damping on an
  eligible dense object-list target
- **THEN** the emitted objective key is `loss/A1_text/stop_signal_ce`
- **AND** no raw alias key for stop-signal damping is emitted.

## ADDED Requirements

### Requirement: Stage-1 stop-signal damping SHALL use canonical GT text atom names
The Stage-1 trainer MUST emit canonical GT-text atom names for stop-signal
damping when the experiment is enabled.

Normative behavior:

- Stage-1 single-forward emission MUST use:
  - `loss/gt_text/stop_signal_ce`
- Stage-1 MUST NOT invent a second Stage-1-only alias for the same
  stop-signal objective math,
- because Stage-1 has one GT forward, these keys remain unsplit beyond the
  canonical `gt_text` provenance.

#### Scenario: Stage-1 stop signal uses canonical GT text atom
- **GIVEN** Stage-1 stop-signal damping is enabled
- **WHEN** a training step emits stop-signal objective atoms
- **THEN** the emitted key is `loss/gt_text/stop_signal_ce`
- **AND** the same atom name remains recognizable relative to Stage-2
  `loss/A1_text/stop_signal_ce`.

### Requirement: Stop-signal diagnostics are explicit, provenance-aware, and sparse-emitted
The trainer metrics contract SHALL expose stop-signal diagnostics with stable
key names when the adaptive stop-signal experiment runs.

Normative behavior:

- when Stage-1 single-forward GT stop-signal damping executes on at least one
  eligible branch in the current finalized step, the trainer MUST emit:
  - `stop_signal/gt/eligible_seq_count`
  - `stop_signal/gt/branch_count`
  - `stop_signal/gt/weight_mean`
  - `stop_signal/gt/p_stop_mean`
  - `stop_signal/gt/p_cont_mean`
  - `stop_signal/gt/margin_mean`
- when Stage-2 Channel-A GT-anchor stop-signal damping executes on at least one
  eligible branch in the current finalized step, the trainer MUST emit:
  - `stop_signal/A1/eligible_seq_count`
  - `stop_signal/A1/branch_count`
  - `stop_signal/A1/weight_mean`
  - `stop_signal/A1/p_stop_mean`
  - `stop_signal/A1/p_cont_mean`
  - `stop_signal/A1/margin_mean`
- `eligible_seq_count` MUST count original sequences with at least one eligible
  semantic stop branch in the finalized step,
- `branch_count` MUST count eligible semantic stop-branch positions,
- `weight_mean` MUST be the mean effective stop-signal weight after applying
  bounded weighting and `curve_gamma`,
- `p_stop_mean` and `p_cont_mean` MUST be the mean pair-normalized stop and
  continue probabilities computed from the two-token branch logits after
  applying `branch_temperature`,
- `margin_mean` MUST be the mean pair-local logit margin
  `(stop_logit - continue_logit) / branch_temperature`, equivalently
  `log(p_stop / p_cont)`,
- `_count` keys MUST aggregate additively across micro-steps,
- non-count stop-signal gauges MUST be mean-like and MUST reflect only observed
  stop-signal branches,
- when the experiment is disabled or no eligible stop branches are present for
  the step, the trainer MUST omit `stop_signal/gt/*` and `stop_signal/A1/*`
  keys rather than emitting placeholder zeros.

#### Scenario: No eligible branches means no stop-signal metrics
- **WHEN** a finalized training step has no eligible semantic stop branches or
  stop-signal damping is disabled
- **THEN** the log payload omits `stop_signal/gt/*` and `stop_signal/A1/*` keys
- **AND** the step does not emit placeholder zero-valued stop-signal metrics.
