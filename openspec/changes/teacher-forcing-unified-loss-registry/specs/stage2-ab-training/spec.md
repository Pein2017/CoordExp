# stage2-ab-training Specification (Delta)

## ADDED Requirements

### Requirement: Stage-2 two-channel training supports a config-declared objective and diagnostics pipeline
When `custom.trainer_variant: stage2_two_channel`, the system SHALL support a
YAML-declared module pipeline for the Stage-2 two-channel training objective and diagnostics.

Normative behavior:
- If `stage2_ab.pipeline` is provided, the trainer MUST interpret it according to the
  `teacher-forcing-objective-pipeline` capability.
- If `stage2_ab.pipeline` is absent, the trainer MUST construct and use a default pipeline that preserves the current
  Stage-2 two-channel objective semantics, by resolving to the **Default Pipeline Manifest** defined below.
- Pipeline parsing and module resolution MUST be strict and MUST fail fast on unknown module names.
- The pipeline identity (module list + checksum) MUST be recorded in logs for reproducibility.

#### Default Pipeline Manifest (when `stage2_ab.pipeline` is omitted)

To eliminate ambiguity, the “pipeline omitted” behavior is defined as resolving to the following manifest.

Normative default objective modules (ordered):
1) `token_ce`
2) `bbox_geo`
3) `coord_reg`

Normative default diagnostics modules (ordered):
1) `coord_diag`

Normative default module weights (all objective modules):
- `weight = 1.0`

Normative default module configs (effective values):
- `token_ce`:
  - Channel-A (Expectation):
    - `desc_ce_weight = stage2_ab.desc_ce_weight` (default `1.0`)
  - Channel-B (Rollout):
    - `rollout_fn_desc_weight = stage2_ab.desc_ce_weight` (default `1.0`)
    - `rollout_matched_prefix_struct_weight = 1.0`
    - `rollout_drop_invalid_struct_ce_multiplier = stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` (default `1.0`)
- `bbox_geo`:
  - `smoothl1_weight = stage2_ab.bbox_smoothl1_weight` (default `1.0`)
  - `ciou_weight = stage2_ab.bbox_ciou_weight` (default `1.0`)
  - decode mode is controlled by `stage2_ab.coord_decode_mode` (default `exp`)
- `coord_reg`:
  - Coord-shape terms (enabled only when `custom.coord_soft_ce_w1.enabled: true`):
    - `soft_ce_weight = custom.coord_soft_ce_w1.soft_ce_weight` (default `1.0`)
    - `w1_weight = custom.coord_soft_ce_w1.w1_weight` (default `1.0`)
    - `temperature = custom.coord_soft_ce_w1.temperature` (default `1.0`)
    - `target_sigma = custom.coord_soft_ce_w1.target_sigma` (default `2.0`)
    - `target_truncate = custom.coord_soft_ce_w1.target_truncate` (default `null`)
  - Additional coord terms (Stage-2 Two-Channel typed knobs):
    - `coord_ce_weight = stage2_ab.coord_ce_weight` (default `0.0`)
    - `coord_el1_weight = stage2_ab.coord_el1_weight` (default `0.0`)
    - `coord_ehuber_weight = stage2_ab.coord_ehuber_weight` (default `0.0`)
    - `coord_huber_delta = stage2_ab.coord_huber_delta` (default `0.001`)
    - `coord_entropy_weight = stage2_ab.coord_entropy_weight` (default `0.0`)
    - `coord_gate_weight = stage2_ab.coord_gate_weight` (default `0.0`)
    - `text_gate_weight = stage2_ab.text_gate_weight` (default `0.0`)

Informative YAML expression (conceptual; values shown are symbolic):
```yaml
stage2_ab:
  # pipeline omitted => resolves to the following manifest:
  # pipeline:
  #   objective:
  #     - {name: token_ce, weight: 1.0}
  #     - {name: bbox_geo, weight: 1.0}
  #     - {name: coord_reg, weight: 1.0}
  #   diagnostics:
  #     - {name: coord_diag}
```

#### Scenario: Pipeline omitted uses current (full_idea-aligned) defaults
- **WHEN** a Stage-2 Two-Channel config does not define `stage2_ab.pipeline`
- **THEN** training starts successfully
- **AND** the Stage-2 Two-Channel objective matches the Default Pipeline Manifest (and is aligned to `full_idea`)
- **AND** canonical `loss/<component>` keys are emitted (per `teacher-forcing-unified-loss-registry`)

#### Scenario: Pipeline is applied when provided
- **WHEN** a Stage-2 Two-Channel config defines `stage2_ab.pipeline` with at least one objective module
- **THEN** the trainer uses that declared module pipeline
- **AND** the resolved module list and checksum are logged.

#### Scenario: Pipeline mode rejects duplicated flat objective knobs
- **WHEN** a Stage-2 Two-Channel config defines `stage2_ab.pipeline`
- **AND** the config also defines any objective-affecting flat knobs used by the default manifest (e.g.,
  `stage2_ab.desc_ce_weight`, `stage2_ab.bbox_smoothl1_weight`, `stage2_ab.bbox_ciou_weight`,
  `stage2_ab.coord_ce_weight`, `stage2_ab.coord_entropy_weight`, `stage2_ab.coord_gate_weight`,
  `stage2_ab.text_gate_weight`,
  `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`, or `custom.coord_soft_ce_w1.*`)
- **THEN** config validation fails fast with guidance to move those values into the declared module configs.

#### Scenario: Stage-2 Two-Channel rejects rollout-matching pipeline keys
- **WHEN** `custom.trainer_variant=stage2_two_channel`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast with guidance to use `stage2_ab.pipeline`.

#### Scenario: Unknown module name fails fast
- **WHEN** a Stage-2 Two-Channel config defines `stage2_ab.pipeline` referencing an unknown module `name`
- **THEN** training initialization fails fast with actionable diagnostics.

### Requirement: Trainer variant naming is clear and stable
To reduce public-facing confusion, the system SHALL support clear trainer variant naming for Stage-2 two-channel
training.

Normative behavior:
- The system MUST accept `custom.trainer_variant: stage2_two_channel` as the canonical trainer variant string.
- The system MUST reject `custom.trainer_variant: stage2_ab_training` (fail fast) with actionable guidance to use
  `stage2_two_channel`.

### Requirement: Stage-2 Two-Channel module names are stable and discoverable
Stage-2 Two-Channel SHALL provide a strict module registry for its pipeline modules, and the module names SHALL be stable so
YAML-declared experiments remain auditable.

Normative minimum module names (initial set; may be extended):
- Objective modules:
  - `token_ce` (masked/weighted CE per unified loss registry; includes EOS-enforced closure supervision)
  - `bbox_geo` (bbox SmoothL1 + CIoU on decoded boxes; FP-neutral in rollout context)
  - `coord_reg` (coord distribution regularizers: coord CE / softCE / W1 / entropy / gate / EL1/EHuber)
- Diagnostics modules:
  - `coord_diag` (coord distribution diagnostics; best-effort)

Normative behavior:
- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available Stage-2 Two-Channel module names.

### Requirement: Stage-2 Two-Channel module configs are strict and typed
Stage-2 Two-Channel SHALL validate module `config` payloads strictly so experiments are reproducible and fail fast on schema
drift.

Normative behavior:
- Module `config` mappings MUST be validated at trainer initialization (before training starts).
- Unknown keys in a module `config` MUST fail fast with actionable diagnostics listing allowed keys.

Normative config schemas (minimum set; may be extended):
- `token_ce.config`:
  - `desc_ce_weight: float` (default: `1.0`)
  - `rollout_fn_desc_weight: float` (default: `1.0`)
  - `rollout_matched_prefix_struct_weight: float` (default: `1.0`)
  - `rollout_drop_invalid_struct_ce_multiplier: float` (default: `1.0`)
- `bbox_geo.config`:
  - `smoothl1_weight: float` (default: `1.0`)
  - `ciou_weight: float` (default: `1.0`)
- `coord_reg.config`:
  - `coord_ce_weight: float` (default: `0.0`)
  - `coord_el1_weight: float` (default: `0.0`)
  - `coord_ehuber_weight: float` (default: `0.0`)
  - `coord_huber_delta: float` (default: `0.001`)
  - `coord_entropy_weight: float` (default: `0.0`)
  - `coord_gate_weight: float` (default: `0.0`)
  - `text_gate_weight: float` (default: `0.0`)
  - `soft_ce_weight: float` (default: `0.0`)
  - `w1_weight: float` (default: `0.0`)
  - `temperature: float` (default: `1.0`)
  - `target_sigma: float` (default: `2.0`)
  - `target_truncate: int|null` (default: `null`)
- `coord_diag.config`:
  - (no required keys; implementations MAY accept a strict subset of the above for convenience, but MUST document them)

Note:
- When `stage2_ab.pipeline` is omitted, the effective defaults for the Stage-2 Two-Channel objective are defined by the Default
  Pipeline Manifest above (which sources values from the typed Stage-2 Two-Channel flat schema keys and `custom.coord_soft_ce_w1`
  as applicable).


### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract
Stage-2 Two-Channel training SHALL implement loss naming and masking semantics per the `teacher-forcing-unified-loss-registry`
capability.

Normative behavior:
- Stage-2 Two-Channel MUST build token-type masks and object-subset masks according to the registry contexts:
  - Channel-A uses `context=gt` for CE anchoring and `context=self_context` for geometry.
  - Channel-B uses `context=rollout` with FP-neutral + EOS-enforced semantics.
- When the module pipeline is enabled, objective/diagnostics modules MUST emit metric keys consistent with the
  registry’s canonical component names.

#### Scenario: Channel-B remains FP-neutral and EOS-enforced
- **WHEN** Stage-2 Two-Channel runs a Channel-B update step
- **THEN** FP spans do not contribute to CE or geometry losses
- **AND** top-level closure `}` and `<|im_end|>` remain supervised.


### Requirement: Stage-2 Two-Channel exposes ST bridge knobs as typed YAML config
Stage-2 Two-Channel SHALL expose config knobs to enable Straight-Through (ST) behavior for:
- Channel-A coord-slot self-context embeddings, and
- geometry coord decode.

Normative behavior:
- Config MUST be expressed under the typed Stage-2 Two-Channel namespace (`stage2_ab.*`) and MUST be strict (unknown keys fail).
- When ST knobs are omitted, defaults MUST preserve current behavior (soft embeddings + expectation decode).
- When ST knobs are enabled, the forward/backward behavior MUST follow the ST semantics defined by
  `teacher-forcing-unified-loss-registry`.

Normative key names:
- `stage2_ab.coord_ctx_embed_mode: soft|st|hard` (default: `soft`)
- `stage2_ab.coord_decode_mode: exp|st` (default: `exp`)

Normative mapping / identity:
- The resolved values MUST feed a single internal enum for embedding and decode behavior.
- The resolved values MUST be included in the pipeline identity checksum (so ST-vs-soft differences are auditable even
  when the pipeline module list is unchanged).

#### Scenario: ST can be enabled for Channel-A without changing CE anchoring
- **WHEN** `stage2_ab.coord_ctx_embed_mode=st` is enabled for Stage-2 Channel-A
- **THEN** Channel-A CE remains anchored to `context=gt` logits
- **AND** only coord-slot context embeddings follow ST semantics.

### Requirement: Eval-step supports COCO mAP when enabled
Stage-2 two-channel training SHALL support computing COCO-style bbox mAP during `eval_step` (in addition to the
rollout matching evaluator metrics), so evaluation is consistent across Stage-2 trainers.

Normative behavior:
- COCO/mAP evaluation MUST be controlled by the typed config surface:
  - `rollout_matching.eval_detection.enabled: bool`
- For Stage-2 trainers, `rollout_matching.eval_detection.enabled` MUST default to `true` so COCO `mAP` is a standard
  eval metric like `rollout/f1`. Disabling it MUST be explicit via YAML (`enabled: false`).
- The eval-step COCO evaluation implementation MUST reuse the same detection evaluator modules used by offline
  evaluation (see `openspec/specs/detection-evaluator`), rather than re-implementing COCO preparation/scoring logic in
  trainer code.
- When `rollout_matching.eval_detection.enabled=true`, the trainer MUST attempt to compute COCO bbox AP/mAP for the
  eval-step rollouts and MUST emit, at minimum:
  - `rollout/mAP` (float; COCO `bbox_AP` = AP@[.50:.95])
- If COCO eval fails unexpectedly (missing dependencies, invalid records, etc.), the trainer MUST:
  - emit `rollout/mAP=0.0` (so the key is always present when enabled), and
  - surface the failure as a warning (not silent).

#### Scenario: Stage-2 two-channel eval reports mAP
- **GIVEN** `rollout_matching.eval_detection.enabled=true`
- **WHEN** `eval_step` runs
- **THEN** `rollout/mAP` is present in the eval metrics payload
