# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 AB supports text_gate via coord_reg module config
Stage-2 AB MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `stage2_ab.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution
  when coord-vocab mass appears at supervised text positions (subject to
  registry masking).

#### Scenario: Non-zero text_gate_weight is effective
- **WHEN** `coord_reg.config.text_gate_weight > 0`
- **AND** the model places substantial coord-vocab probability mass at
  supervised `type=struct|desc` positions
- **THEN** the emitted `text_gate` objective atom increases relative to the
  same run with `text_gate_weight = 0`:
  - Channel-A: `loss/coord/text_gate`
  - Channel-B: `loss/B_coord/text_gate`
- **AND** the increase is attributable to the `text_gate` sub-term inside
  `coord_reg`.

### Requirement: Stage-2 AB pipeline specs are explicit and complete (no implicit defaults)
Stage-2 AB pipeline module specs MUST be authored with explicit fields and
complete module configs to prevent silent drift from default injection.

Normative behavior:
- Each entry in `stage2_ab.pipeline.objective[]` and
  `stage2_ab.pipeline.diagnostics[]` MUST include:
  - `name`, `enabled`, `weight`, `channels`, `application`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `application` MUST be an explicitly authored mapping with exactly one key:
  - `preset`.
- `application.preset` MUST be valid for the referenced module:
  - `token_ce`: `anchor_text_only`, `rollout_text_only`
  - `loss_dead_anchor_suppression`: `rollout_only`
  - `bbox_geo`, `bbox_size_aux`, `coord_reg`:
    - `anchor_only`
- Presets that imply a deprecated final Channel-A self-context pass MUST be
  rejected with actionable migration guidance:
  - `token_ce.application.preset: anchor_text_plus_final_struct`
  - `bbox_geo.application.preset: anchor_if_single_iter_else_final`
  - `bbox_geo.application.preset: final_only`
  - `bbox_geo.application.preset: anchor_and_final`
  - and the equivalent `bbox_size_aux` / `coord_reg` preset values.
- `config` MUST include exactly the allowlisted keys for the referenced module.

#### Scenario: Deprecated final-pass preset fails fast
- **GIVEN** a Stage-2 AB config that authors
  `token_ce.application.preset: anchor_text_plus_final_struct`
- **WHEN** the config is loaded
- **THEN** validation fails fast
- **AND** the error explains the single-pass replacement
  `anchor_text_only`.

### Requirement: Stage-2 AB objective application is explicit and non-redundant
Stage-2 AB SHALL route Channel-A objective provenance through
`application.preset` instead of duplicating loss strengths across separate
`a1_*` config families.

Normative behavior:
- `bbox_geo`, `bbox_size_aux`, and `coord_reg` MUST express Channel-A routing
  through `stage2_ab.pipeline.objective[*].application.preset`.
- The canonical non-redundant Channel-A preset is now `anchor_only`:
  - Channel-A bbox/coord atoms MUST emit only under `loss/coord/*`.
- `token_ce.application.preset=anchor_text_only` MUST keep
  `loss/text/{struct_ce,desc_ce}` on the GT anchor path and MUST NOT emit
  `loss/A1_*` or `loss/A2_*`.
- Stage-2 AB module configs MUST NOT reintroduce duplicated Channel-A routing
  weights such as `a1_smoothl1_weight`, `a1_ciou_weight`,
  `a1_log_wh_weight`, `a1_oversize_penalty_weight`,
  `a1_soft_ce_weight`, or `a1_w1_weight`.
- Stage-2 AB module configs MUST NOT use final-pass/self_context aliases or
  weight families to recreate A2 routing.

#### Scenario: Anchor-only Channel-A routes bbox/coord atoms to the normal coord group
- **GIVEN** `bbox_geo.application.preset: anchor_only`
- **AND** `bbox_size_aux.application.preset: anchor_only`
- **AND** `coord_reg.application.preset: anchor_only`
- **WHEN** Channel-A executes
- **THEN** bbox/coord objective atoms emit under `loss/coord/*`
- **AND** the same step does not emit `loss/A1_*` or `loss/A2_*`.

### Requirement: Stage-2 AB canonical profiles resolve high-signal knobs after inheritance
Canonical Stage-2 AB profiles MUST resolve the following high-signal run and
ablation knobs after inheritance for each profile under
`configs/stage2_two_channel/prod/*.yaml` and
`configs/stage2_two_channel/smoke/*.yaml`.

Required resolved fields:
- `model.model`
- `training.run_name`
- `training.output_dir`
- `training.logging_dir`
- `training.learning_rate`
- `training.vit_lr`
- `training.aligner_lr`
- `training.effective_batch_size`
- `training.eval_strategy`
- `training.eval_steps`
- `training.save_strategy`
- `training.save_steps`
- `stage2_ab.schedule.b_ratio`

Rationale for resolved-profile explicitness:
- The LR trio (`training.learning_rate`, `training.vit_lr`,
  `training.aligner_lr`) is treated as MUST for canonical profiles to avoid
  hidden optimizer-group drift across ablations, even when these values are
  supplied by an intermediate parent.

Validation behavior:
- Canonical Stage-2 AB profile loading MUST fail fast if any required field is
  missing from the fully resolved profile.
- Error text MUST identify missing fields by full key path.

#### Scenario: Canonical profile with resolved high-signal fields is accepted
- **WHEN** a Stage-2 AB canonical profile resolves all required high-signal
  keys after inheritance
- **THEN** config loading succeeds and the profile is considered
  self-consistent.

### Requirement: Channel-A forward path is compatible with Qwen3-VL multimodal semantics
For Qwen3-VL (dense) models, each forward MUST provide exactly one of
`input_ids` or `inputs_embeds`.

Normative behavior:
- The canonical Stage-2 two-channel Channel-A path is a single teacher-forced
  forward and MUST NOT depend on iterative coord-slot embedding rewrites.
- Channel-A MAY use ordinary teacher-forced `input_ids` as its supported
  default path.
- If a specialized `inputs_embeds` path is ever used for another reason, it
  MUST preserve multimodal placeholder semantics and MUST NOT perturb the
  placeholder rows needed for model-internal feature insertion.
- The forward call MUST remain compatible with multimodal feature insertion
  (placeholder token count matches provided visual feature length).
- When using padding-free packing with Qwen3-VL, the trainer MUST pass the
  4-row `position_ids` format consistently for the Channel-A forward.

#### Scenario: Single-pass multimodal Channel-A remains compatible
- **WHEN** Stage-2 two-channel executes a multimodal Channel-A update
- **THEN** the Channel-A forward remains compatible with Qwen3-VL visual
  feature insertion
- **AND** it does not rely on iterative self-context forwards.

### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract
Stage-2 Two-Channel training SHALL implement loss naming and masking semantics
per the `teacher-forcing-unified-loss-registry` capability.

Normative behavior:
- Stage-2 Two-Channel MUST build token-type masks and object-subset masks
  according to the registry contexts:
  - Channel-A uses `context=gt` for CE and bbox/coord supervision.
  - Channel-B uses `context=rollout` with FP-neutral + EOS-enforced semantics.
- When the module pipeline is enabled, objective and diagnostics modules MUST
  emit metric keys consistent with the registry’s canonical component names.

#### Scenario: Channel-A uses GT registry context only
- **WHEN** Stage-2 Two-Channel executes a Channel-A update step
- **THEN** Channel-A loss computation uses `context=gt`
- **AND** the same step does not construct `context=self_context`.

### Requirement: Stage-2 Two-Channel removes self-context-era decode toggles from typed YAML config
Stage-2 Two-Channel SHALL reject geometry-decode toggles that were carried with
the deprecated self-context surface.

Normative behavior:
- Config MUST be expressed under the typed Stage-2 Two-Channel namespace
  (`stage2_ab.*`) and MUST be strict (unknown keys fail).
- `stage2_ab.coord_decode_mode` is deprecated and MUST fail fast if authored.
- `stage2_ab.coord_ctx_embed_mode` is deprecated and MUST fail fast if authored.
- Geometry decode follows the supported expectation-decode baseline without an
  authored Stage-2 override.

#### Scenario: Deprecated Stage-2 decode toggle fails fast
- **WHEN** `stage2_ab.coord_decode_mode` is authored in an active Stage-2 config
- **THEN** configuration parsing fails fast
- **AND** the error explains that Stage-2 geometry decode now uses the fixed
  expectation-decode baseline.

### Requirement: Hybrid objective preserves Channel-A anchoring and uses clean-prefix Channel-B supervision
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- Token CE anchor at the GT teacher-forced forward:
  - CE on non-coord tokens MUST be computed from the GT-context logits.
  - Coord tokens MUST NOT contribute to CE, to avoid double-supervision.
- Geometry + distribution regularizers from the same single-pass Channel-A
  logits:
  - Geometry losses and any distribution-level losses MUST be computed from the
    supported single-pass Channel-A logits, not from a deprecated final
    self-context pass.

Channel-B:
- Clean-prefix positive supervision:
  - the positive teacher-forced prefix MUST be canonical serialization of
    `accepted_objects_clean`,
  - matched clean prefix objects MUST receive structure-only CE,
  - generic unmatched clean extras MAY remain in the clean prefix as context
    but MUST remain neutral.

#### Scenario: Channel-A geometry no longer depends on a final iteration
- **WHEN** Stage-2 AB computes Channel-A geometry or coord regularization
- **THEN** the same single-pass Channel-A logits feed those terms
- **AND** no final self-context logits are required.

### Requirement: Stage-2 AB can add matched decoded-box size auxiliaries through `bbox_size_aux`
Stage-2 AB SHALL support optional decoded-box size auxiliaries on the existing
matched geometry path without changing bbox parameterization or decode format.

Normative behavior:
- when `bbox_size_aux.config.log_wh_weight > 0`, the trainer MUST add matched
  log-width/log-height supervision on canonicalized decoded boxes,
- when `bbox_size_aux.config.oversize_penalty_weight > 0`, the trainer MAY add
  the thresholded oversize penalty on decoded boxes for the same context,
- Channel-A and Channel-B applicability MUST remain controlled by the authored
  `channels` field on the `bbox_size_aux` module entry,
- Channel-A provenance MUST remain controlled by
  `bbox_size_aux.application.preset`, with `anchor_only` as the supported
  Channel-A preset,
- `bbox_size_aux` MUST remain separate from `bbox_geo` in the authored pipeline
  so the new size loss is an independently removable plugin module,
- `bbox_size_aux` MUST consume the current four bbox coord slots in the
  existing `xyxy` order rather than introducing a new bbox expression,
- the default canonical Stage-2 profile behavior SHOULD enable only the
  matched `log_wh` term at a small weight and keep `log_area` / `oversize` off.

#### Scenario: Channel-A matched geometry uses the normal coord-group log-size aux
- **GIVEN** a Stage-2 AB config with `bbox_size_aux.channels: [A]`
- **AND** `bbox_size_aux.application.preset: anchor_only`
- **AND** `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** Channel-A computes matched geometry loss from decoded boxes
- **THEN** `bbox_size_aux` contributes `bbox_log_wh` under `loss/coord/*`
- **AND** the same step does not emit legacy `A*` size-aux objective
  atoms.

### Requirement: Stage-2 AB objective includes coord soft-CE and W1 terms on supervised bbox slots
Stage-2 AB trainer MUST support Stage-1-style coord distribution penalties in
Stage-2 training:

- `soft_ce` and `w1` MUST be computed on coord distributions for
  Stage-2-supervised bbox coord slots only:
  - matched-prefix groups (`bbox_groups_prefix`),
  - and FN-injected groups (`bbox_groups_fn`).
- The coord distribution for each supervised coord slot MUST follow the same
  causal shift contract as other Stage-2 coord losses (coord token at position
  `p` uses logits at `p-1`).
- These terms MUST contribute to the Stage-2 coord regularization objective
  (`coord_reg` module), and MUST be surfaced in training logs as objective atoms
  under provenance keys:
  - Channel-A: `loss/coord/{coord_soft_ce,coord_w1}`
  - Channel-B (rollout-context): `loss/B_coord/{coord_soft_ce,coord_w1}`
- The trainer MUST NOT apply these terms to unsupervised FP-only coord slots.

Weighting/config contract:
- Stage-2 uses the declared pipeline config for coord distribution penalties:
  - `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.temperature`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.target_sigma`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.target_truncate`
- If `soft_ce_weight` and `w1_weight` are both `0`, Stage-2 soft-CE/W1
  contributions MUST be zero.

#### Scenario: Enabled coord soft-CE/W1 increases Stage-2 coord regularization
- **GIVEN** Stage-2 config has
  `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight > 0`
- **AND** `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight > 0`
- **AND** a batch has supervised bbox coord slots
- **WHEN** Stage-2 computes loss
- **THEN** `loss/B_coord/coord_soft_ce` and `loss/B_coord/coord_w1` are
  positive
- **AND** the Channel-A coord group, when active, uses `loss/coord/*` rather
  than any `loss/A1_*` or `loss/A2_*` key.

## REMOVED Requirements

### Requirement: Channel-A performs iterative ST/soft self-context via N× full-forwards (no rollout)
This requirement is removed. Stage-2 two-channel no longer supports iterative
Channel-A self-context as an authored training contract.

#### Scenario: (removed) n_softctx_iter controls Channel-A forward count
- **GIVEN** a config that authors `stage2_ab.n_softctx_iter`
- **WHEN** Stage-2 two-channel config validation runs
- **THEN** validation fails fast instead of interpreting the value.

### Requirement: Coord diagnostics are attributed to A1 vs A2 logits in Channel-A
This requirement is removed. Stage-2 two-channel no longer treats A2/final
self-context logits as a supported Channel-A provenance surface.

## ADDED Requirements

### Requirement: Deprecated self-context knobs fail fast in Stage-2 two-channel configs
Stage-2 two-channel MUST reject authored config knobs that exist only for the
deprecated self-context loop.

Normative behavior:
- The following keys MUST fail fast with actionable migration guidance when
  authored under `stage2_ab`:
  - `n_softctx_iter`
  - `softctx_grad_mode`
  - `softctx_temperature`
  - `coord_ctx_embed_mode`
- The error MUST explain that Channel-A now uses a single GT-anchored forward
  and MUST point users to the supported preset replacements.

#### Scenario: Deprecated self-context knob fails fast
- **GIVEN** a Stage-2 AB config with `stage2_ab.softctx_grad_mode: unroll`
- **WHEN** schema validation runs
- **THEN** validation fails
- **AND** the error explains that self-context iteration is deprecated and
  unsupported.

### Requirement: Coord diagnostics are attributed to the normal coord group and B provenance in Stage-2 two-channel
When Stage-2 two-channel emits coord-distribution diagnostics, it MUST
attribute them only to the supported forward surfaces that still exist.

Normative behavior:
- `coord_diag/*`: computed from the Channel-A GT-anchor logits.
- `coord_diag/B/*`: computed from Channel-B rollout-context logits.
- `coord_diag/A1/*` and `coord_diag/A2/*` MUST NOT be emitted.
- The trainer MUST NOT emit ambiguous bare `coord_diag/*` keys for these
  monitors beyond the normal single-pass Channel-A coord group.

#### Scenario: Channel-A diagnostics use the normal coord group
- **WHEN** Stage-2 AB runs with `coord_diag` enabled
- **THEN** emitted coord diagnostics may include `coord_diag/*` and
  `coord_diag/B/*` when the relevant channel runs
- **AND** the same run does not emit `coord_diag/A1/*` or `coord_diag/A2/*`.
