# stage1-detection-objectives Specification

## Purpose
Define the stable objective contract for Stage-1 compact
recursive detection surface, including objective variant ownership, recursive
CE semantics, detection-schema coord-soft-target overlays, runtime guardrails, and
diagnostic expectations.

## Requirements

### Requirement: Detection objective authoring is config-first and typed
The compact detection objective surface SHALL be authored through the
top-level `objective` section parsed by `DetectionTrainingConfig`.

Normative behavior:

- detection configs MUST use top-level `data`, `prompt`,
  `detection_template`, `token_rows`, `objective`, `packing`, `evaluation`, and
  `validation` sections,
- detection configs MUST NOT use legacy `custom.stage1_set_continuation`,
  `custom.trainer_variant: stage1_set_continuation`, or legacy
  `custom.coord_soft_ce_w1.*` to configure recursive detection
  objectives,
- `objective.id=sft` MUST pair only with SFT objective variants,
- `objective.id=teacher_forcing` MUST pair only with DetectionScene-backed
  detection teacher-forcing variants,
- objective variants MUST remain YAML/config driven and MUST NOT require new
  stable CLI flags.

#### Scenario: Legacy set-continuation authoring is rejected on the detection surface
- **WHEN** a compact detection config authors
  `custom.stage1_set_continuation` or
  `custom.trainer_variant: stage1_set_continuation`
- **THEN** config parsing fails fast
- **AND** the error explains that the detection surface owns its
  objective through top-level `objective`.

#### Scenario: Recursive CE variant is selected through `objective.variant`
- **GIVEN** a detection config with `objective.id: teacher_forcing`
- **WHEN** config parsing and runtime resolution run
- **THEN** the teacher-forcing objective mode is selected from `objective.variant`
- **AND** no extra CLI flag is needed.

### Requirement: Random-permutation ET-RMP-CE remains the production baseline/comparator
The compact recursive detection production baseline SHALL remain
`objective.variant: random_permutation_et_rmp_ce` until an explicitly promoted
successor is approved.

Normative behavior:

- the comparator config path MUST remain
  `configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml`
  unless a later contract supersedes it,
- this baseline MUST use `objective.id: teacher_forcing`,
- this baseline MUST use `detection_template.id: compact_full`,
- this baseline MUST use `data.object_ordering: random_permutation`,
- this baseline MUST keep `objective.state_weighting:
  uniform_permutation`,
- this baseline MUST keep `objective.normalization:
  semantic_image_bucket_balanced`,
- trie support and balance weights for this baseline MUST be authored as
  `objective.trie_support_weight` and `objective.trie_balance_weight`,
- the baseline MUST remain available as a comparator when new detection
  objective variants are introduced.

#### Scenario: Comparator config resolves to random-permutation ET-RMP-CE
- **WHEN** the comparator config
  `configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml`
  is parsed
- **THEN** it resolves to `objective.variant: random_permutation_et_rmp_ce`
- **AND** runtime teacher-forcing support and balance weights come from the
  top-level `objective.trie_*` fields.

### Requirement: Prefix-rollin ET-RMP-CE is a compact-full ablation surface
The compact recursive detection surface SHALL support
`objective.variant: prefix_rollin_et_rmp_ce` as a compact-full ablation and
diagnostic route, not as a production baseline by default.

Normative behavior:

- `prefix_rollin_et_rmp_ce` MUST require `detection_template.id: compact_full`,
- `prefix_rollin_et_rmp_ce` MUST sample a GT roll-in prefix length `K`
  uniformly over `[0, object_count]` for the current V1 route,
- roll-in prefix labels MUST be masked,
- suffix order MUST remain `same_sampled_permutation` for the V1 checked-in
  route,
- support and balance weights MUST be authored under `objective.target`,
- obsolete flat support/balance aliases MUST fail fast for this variant,
- E1/E2 boundary-pressure and EOS-loosening variants MUST remain historical
  evidence only, not live latest-detection training surfaces.

#### Scenario: Prefix-rollin rejects obsolete flat trie weights
- **WHEN** `objective.variant: prefix_rollin_et_rmp_ce` is authored with legacy
  flat trie-weight aliases such as `objective.trie_support_weight`
- **THEN** config validation fails fast
- **AND** the error points to the nested `objective.target` and
  `objective.boundary` surfaces.

#### Scenario: Separator-continue ablation changes only append-boundary pressure
- **GIVEN** the E2 separator-continue ablation config
- **WHEN** it is compared against the E1 prefix-rollin route
- **THEN** it preserves E1 target, type-gate, EOS, and roll-in semantics
- **AND** changes only the append-boundary continuation pressure needed to test
  newline-versus-`<|im_end|>` early stopping.

### Requirement: Recursive detection coord-soft-target overlays are objective-local
Recursive detection coord-soft-target overlays SHALL be authored through
`objective.coord_soft_ce`, not through legacy Stage-1 SFT coord-loss surfaces.

Normative behavior:

- `objective.coord_soft_ce` MUST be valid only for
  `objective.id: teacher_forcing` with ET-RMP-family objective
  variants,
- `iou_gibbs_v0` and `ciou_gibbs_v0` overlays MUST be treated as historical
  A5/A6 ablation candidates unless promoted by later evidence,
- `instance_trie_gaussian` overlays MUST use focused R95 policy fields:
  `gaussian_mixture_weight`, `gaussian_r95_axis_fraction`, and
  `gaussian_r95_cap_bins`,
- `instance_trie_gaussian` overlays MUST reject stale Gibbs or generic Gaussian
  keys such as `tau`, `weighting`, `replace_coord_hard_ce`, `target_sigma`, or
  `target_truncate`,
- coord-soft-target overlays MUST require a valid `token_rows` coord-geometry
  group with expected coord-token id bounds.

#### Scenario: Instance-trie Gaussian overlay rejects stale knobs
- **WHEN** `objective.coord_soft_ce.target_distribution:
  instance_trie_gaussian` includes stale Gibbs keys such as `tau`
- **THEN** config parsing fails fast
- **AND** the error identifies the unsupported `objective.coord_soft_ce` key.

#### Scenario: Coord-soft-target overlay uses token-row coord ids
- **GIVEN** a recursive detection config with `objective.coord_soft_ce`
  enabled
- **WHEN** runtime config is resolved
- **THEN** coord token ids are read from the `token_rows` coord-geometry group
- **AND** the overlay does not use legacy `custom.coord_soft_ce_w1` ids or
  weights.

### Requirement: Compact-full token-row adaptation covers geometry and structural rows
Compact-full recursive detection SHALL treat trainable token rows as a
detection contract rather than as a coord-only adapter contract.

Normative behavior:

- compact-full recursive detection MUST train the 1000 coord-token rows plus
  `<|object_ref_start|>` and `<|box_start|>` when token-row adaptation is
  enabled,
- the persisted module name `coord_offset_adapter` MAY remain for checkpoint
  compatibility,
- contract wording MUST describe this surface as token-row adaptation rather
  than coord-only adaptation,
- the `coord_geometry` token-row group MUST preserve the expected
  `<|coord_0|>` through `<|coord_999|>` token id range.

#### Scenario: Compact-full token rows include structural rows
- **WHEN** compact-full recursive detection enables trainable token rows
- **THEN** the trainable row set includes coord geometry rows
- **AND** it includes the compact structural rows needed by the template.

### Requirement: Recursive detection runtime rejects unsupported packing and cache paths
Compact recursive detection sidecars SHALL fail fast on runtime paths
that would invalidate target-position alignment.

Normative behavior:

- recursive detection sidecars MUST require right padding until sidecar
  offset rewriting is explicitly supported,
- `training.packing=true` MUST fail fast for recursive detection
  sidecars,
- `training.eval_packing=true` MUST fail fast for recursive detection
  sidecars,
- `packing.static_packing=true` and `packing.padding_free_packed=true` MUST fail
  fast for recursive detection sidecars,
- `training.encoded_sample_cache.enabled=true` MUST fail fast until sidecar
  cache fingerprints are implemented,
- runtime MUST reject `training.use_logits_to_keep=true` because full logits
  are required by the recursive detection CE objective.

#### Scenario: Packing is rejected before recursive sidecar training starts
- **GIVEN** a recursive detection config with `training.packing: true`
- **WHEN** runtime support validation runs
- **THEN** training fails fast
- **AND** the error explains that packed target-position offset rewriting is not
  implemented yet.

### Requirement: EOS and generation-token contracts are explicit
Compact recursive detection SHALL preserve the Qwen chat-template stop
contract across training and generation.

Normative behavior:

- training EOS supervision for prefix-rollin detection MUST target
  `<|im_end|>` only,
- text-level terminators such as `<|endoftext|>` or `<|end_of_text|>` MUST NOT
  be used as semantic training EOS targets for this surface,
- HF/Qwen generation MUST use `eos_token_id=id("<|im_end|>")`,
- HF/Qwen generation MUST use `pad_token_id=id("<|endoftext|>")`,
- vLLM inference MUST stop on `"<|im_end|>"` only,
- inference and rollout paths MUST preserve training-time geometry with
  `do_resize=false`,
- inference artifacts MUST record the Qwen chat generation contract when this
  surface is evaluated.

#### Scenario: Prefix-rollin EOS target is the assistant stop marker
- **GIVEN** `objective.variant: prefix_rollin_et_rmp_ce`
- **WHEN** training target labels are constructed
- **THEN** semantic EOS supervision targets `<|im_end|>`
- **AND** `<|endoftext|>` is not treated as the semantic EOS label.

### Requirement: Recursive detection CE diagnostics expose target mix, support, balance, boundary, EOS, and type-gate health
Recursive detection CE SHALL expose stable diagnostics for interpreting
multi-target supervision and continuation health.

Normative behavior:

- logs MUST include `loss/recursive_detection_ce` as the comparable
  objective-loss scalar for recursive detection CE variants,
- target-mix diagnostics SHOULD include:
  - `recursive_detection_ce/target_mix/targets_per_sample`,
  - `recursive_detection_ce/target_mix/eos_fraction`,
  - `recursive_detection_ce/target_mix/non_eos_fraction`,
  - `recursive_detection_ce/target_mix/trie_multi_positive_fraction`,
  - `recursive_detection_ce/target_mix/coord_fraction`,
  - `recursive_detection_ce/target_mix/desc_fraction`,
  - `recursive_detection_ce/target_mix/object_control_fraction`,
  - `recursive_detection_ce/target_mix/positive_children_per_trie_target`,
- support/balance diagnostics SHOULD include:
  - `recursive_detection_ce/trie_valid_mass`,
  - `recursive_detection_ce/support_loss`,
  - `recursive_detection_ce/balance_loss`,
  - `recursive_detection_ce/entry/valid_child_entropy`,
  - `recursive_detection_ce/entry/valid_child_kl_to_uniform`,
- continuation diagnostics SHOULD distinguish entry-after-separator and
  free-boundary continuation health,
- boundary diagnostics SHOULD include authored boundary weights when applicable,
- EOS diagnostics SHOULD expose trust weight and weighted/unweighted EOS loss
  when EOS weighting is configured,
- type-gate diagnostics SHOULD expose allowed mass, allowed tokens, and weight
  when the type gate is configured.

#### Scenario: Low aggregate loss is not sufficient evidence of healthy multi-target training
- **WHEN** a recursive detection run reports low aggregate loss
- **THEN** operators must also check target-mix diagnostics before interpreting
  the run as healthy multi-positive continuation training
- **AND** EOS-dominated rows must be distinguished from real continuation rows.

### Requirement: Objective-status language separates production baselines, ablations, diagnostics, and retired routes
Stage-1 objective contracts SHALL preserve explicit status language so
new modules do not accidentally inherit production claims.

Normative behavior:

- `random_permutation_et_rmp_ce` support2 MUST remain the production
  baseline/comparator until explicitly superseded,
- A5/A6 Gibbs coord-soft-target configs MUST remain historical or ablation
  candidates unless measured evidence promotes them,
- focused instance-trie Gaussian configs MUST remain ablation/smoke surfaces
  until production-scale validation evidence and promotion approval exist,
- prefix-rollin E1/E2 configs MUST remain ablation/diagnostic surfaces unless a
  later contract promotes them,
- retired set-continuation candidate-branch objective routes MUST NOT be used
  as current detection training surfaces.

#### Scenario: New objective work keeps comparator baselines available
- **WHEN** a new detection objective variant is introduced
- **THEN** the random-permutation ET-RMP-CE comparator remains runnable and
  documented
- **AND** the new variant must declare its own claim scope before benchmark
  interpretation.
