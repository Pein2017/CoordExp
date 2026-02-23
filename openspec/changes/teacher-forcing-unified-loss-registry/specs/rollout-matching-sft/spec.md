# rollout-matching-sft Specification (Delta)

## ADDED Requirements

### Requirement: Rollout-aligned Stage-2 supports a config-declared objective and diagnostics pipeline
When `custom.trainer_variant: stage2_rollout_aligned`, the system SHALL support a YAML-declared module pipeline for
rollout-aligned teacher-forcing objective and diagnostics.

Normative configuration shape:
- The pipeline MUST be declared under the rollout-matching namespace:
  - `rollout_matching.pipeline.objective`: ordered list of objective modules
  - `rollout_matching.pipeline.diagnostics`: ordered list of diagnostics modules
- Each module entry MUST follow the `teacher-forcing-objective-pipeline` capability contract.

Normative behavior:
- If `rollout_matching.pipeline` is provided, the trainer MUST interpret it according to the
  `teacher-forcing-objective-pipeline` capability.
- If `rollout_matching.pipeline` is absent, the trainer MUST construct and use a default pipeline that is coherent
  with the unified teacher-forcing objective semantics.
  - The default objective MUST include bbox geometry regression (`bbox_geo`) so the Stage-2 objective described in
    `docs/training/STAGE2_RUNBOOK.md` applies consistently across `stage2_rollout_aligned` and `stage2_two_channel`.
    Disabling geometry MUST be expressed explicitly via an authored pipeline (e.g., `bbox_geo.enabled=false` or
    `bbox_geo` weights set to `0`).
- Pipeline parsing and module resolution MUST be strict and MUST fail fast on unknown module names.
- The resolved module list and a stable pipeline checksum MUST be logged at trainer initialization.

#### Default Pipeline Manifest (when `rollout_matching.pipeline` is omitted)

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
- `token_ce` (rollout context):
  - `rollout_fn_desc_weight = 1.0`
  - `rollout_matched_prefix_struct_weight = 1.0`
- `bbox_geo`:
  - `smoothl1_weight = 1.0`
  - `ciou_weight = 1.0`
  - decode mode is controlled by `rollout_matching.coord_decode_mode` (default `exp`)
- `coord_reg` (enabled only when `custom.coord_soft_ce_w1.enabled: true`):
  - `coord_ce_weight = custom.coord_soft_ce_w1.ce_weight` (default `0.0`)
  - `soft_ce_weight = custom.coord_soft_ce_w1.soft_ce_weight` (default `1.0`)
  - `w1_weight = custom.coord_soft_ce_w1.w1_weight` (default `1.0`)
  - `coord_gate_weight = custom.coord_soft_ce_w1.gate_weight` (default `1.0`)
  - `text_gate_weight = 0.0` (default `0.0`; pipeline-only knob)
  - `temperature = custom.coord_soft_ce_w1.temperature` (default `1.0`)
  - `target_sigma = custom.coord_soft_ce_w1.target_sigma` (default `2.0`)
  - `target_truncate = custom.coord_soft_ce_w1.target_truncate` (default `null`)

Informative YAML expression (conceptual; values shown are symbolic):
```yaml
rollout_matching:
  # pipeline omitted => resolves to the following manifest:
  # pipeline:
  #   objective:
  #     - {name: token_ce, weight: 1.0}
  #     - {name: bbox_geo, weight: 1.0}
  #     - {name: coord_reg, weight: 1.0}
  #   diagnostics:
  #     - {name: coord_diag}
```

#### Scenario: Pipeline omitted uses latest defaults (coherent)
- **WHEN** a rollout-matching SFT config does not define `rollout_matching.pipeline`
- **THEN** training starts successfully
- **AND** the rollout-matching objective matches the Default Pipeline Manifest
- **AND** canonical `loss/<component>` keys are emitted (per `teacher-forcing-unified-loss-registry`)

#### Scenario: Pipeline is applied when provided
- **WHEN** a rollout-matching SFT config defines `rollout_matching.pipeline` with at least one objective module
- **THEN** the trainer uses that declared module pipeline
- **AND** the resolved module list and checksum are logged.

#### Scenario: Pipeline mode rejects duplicated flat objective knobs
- **WHEN** a rollout-matching SFT config defines `rollout_matching.pipeline`
- **AND** the config also defines any objective-affecting flat knobs used by the default manifest (e.g.,
  `custom.coord_soft_ce_w1.*`)
- **THEN** config validation fails fast with guidance to move those values into the declared module configs.

#### Scenario: Unknown module name fails fast
- **WHEN** a rollout-matching SFT config defines `rollout_matching.pipeline` referencing an unknown module `name`
- **THEN** training initialization fails fast with actionable diagnostics.

### Requirement: Rollout-aligned Stage-2 module names are stable and share a registry with two-channel Stage-2
Rollout-aligned Stage-2 SHALL provide a strict module registry for its pipeline modules. Module names SHALL be stable
so YAML-declared experiments remain auditable, and the module registry SHOULD be shared with the two-channel Stage-2
engine where possible to avoid duplicated implementations.

Normative minimum module names (initial set; may be extended):
- Objective modules:
  - `token_ce` (masked/weighted CE per unified loss registry; includes EOS-enforced closure supervision)
  - `coord_reg` (coord distribution regularizers; default includes softCE + W1 + gate behavior)
  - `bbox_geo` (bbox SmoothL1 + CIoU on decoded boxes under rollout context)
- Diagnostics modules:
  - `coord_diag` (coord distribution diagnostics; best-effort)

Normative behavior:
- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available rollout-matching module names.

### Requirement: Rollout-aligned Stage-2 module configs are strict and typed
Rollout-aligned Stage-2 SHALL validate module `config` payloads strictly so experiments are reproducible and fail fast
on schema drift.

Normative behavior:
- Module `config` mappings MUST be validated at trainer initialization (before training starts).
- Unknown keys in a module `config` MUST fail fast with actionable diagnostics listing allowed keys.

Normative config schemas (minimum set; may be extended):
- `token_ce.config`:
  - `rollout_fn_desc_weight: float` (default: `1.0`)
  - `rollout_matched_prefix_struct_weight: float` (default: `1.0`)
- `bbox_geo.config`:
  - `smoothl1_weight: float` (default: `1.0`)
  - `ciou_weight: float` (default: `1.0`)
- `coord_reg.config`:
  - `coord_ce_weight: float` (default: `0.0`)
  - `soft_ce_weight: float` (default: `0.0`)
  - `w1_weight: float` (default: `0.0`)
  - `coord_gate_weight: float` (default: `0.0`)
  - `text_gate_weight: float` (default: `0.0`)
  - `temperature: float` (default: `1.0`)
  - `target_sigma: float` (default: `2.0`)
  - `target_truncate: int|null` (default: `null`)
- `coord_diag.config`:
  - (no required keys; implementations MAY accept a strict subset of the above for convenience, but MUST document them)

Note:
- When `rollout_matching.pipeline` is omitted, the effective defaults for the rollout-aligned objective are defined by
  the Default Pipeline Manifest above (which sources values from `custom.coord_soft_ce_w1` as applicable).


### Requirement: Rollout-aligned Stage-2 adheres to the unified loss registry contract
Rollout-aligned Stage-2 SHALL implement loss naming and masking semantics per the `teacher-forcing-unified-loss-registry`
capability.

Normative behavior:
- Teacher-forced forward passes constructed from rollout prefix + mandatory FN append MUST be treated as
  `context=rollout` for the purpose of token-type partitioning and masking.
- The unified registry MUST be able to represent the rollout-matching supervision surface as a registry
  instantiation (including ablation variants such as “FN desc unsupervised” or “matched-prefix struct CE disabled”).
- When the module pipeline is enabled, objective/diagnostics modules MUST emit metric keys consistent with the
  registry’s canonical component names.

#### Scenario: Packing-safe registry masks do not leak across packed segments
- **GIVEN** rollout-matching teacher-forced sequences are packed into a single forward pass
- **WHEN** registry masks are built for `context=rollout`
- **THEN** per-segment boundaries are respected
- **AND** no supervised indices/masks cross segment boundaries.

### Requirement: Trainer variant naming is clear and stable
To reduce public-facing confusion, the system SHALL support clear trainer variant naming for rollout-aligned Stage-2.

Normative behavior:
- The system MUST accept `custom.trainer_variant: stage2_rollout_aligned` as the canonical trainer variant string.
- The system MUST reject `custom.trainer_variant: rollout_matching_sft` (fail fast) with actionable guidance to use
  `stage2_rollout_aligned`.

### Requirement: Rollout-aligned Stage-2 rollout-context semantics are coherent with the two-channel Rollout channel
Rollout-aligned Stage-2 SHALL apply the same rollout-context masking semantics as the two-channel Rollout channel by default
(`progress/full_idea.md`), so teacher-forcing objectives do not drift across code paths.

Normative behavior:
- Rollout-context token supervision MUST enforce:
  - matched prefix: `CE_struct=1`, `CE_desc=0`, `CE_coord=0`,
  - FP spans: fully masked (`0`) for CE and excluded from geometry/coord-dist losses,
  - FN injected: `CE_struct=1`, `CE_desc=1` by default (configurable weight), `CE_coord=0`,
  - closure/EOS: supervised (EOS-enforced).
- The system MUST expose typed configuration to ablate (at minimum):
  - FN `desc` supervision weight, and
  - matched-prefix struct CE weight,
  via YAML diffs (no trainer code edits).
- When these weights differ from defaults, the resolved effective values MUST be logged as part of pipeline identity so
  runs are auditable.

Recommended expression (module config on `token_ce`, strict + typed):
- `rollout_matching.pipeline.objective[].config.rollout_fn_desc_weight: float`
- `rollout_matching.pipeline.objective[].config.rollout_matched_prefix_struct_weight: float`

#### Scenario: FN `desc` supervision can be disabled via YAML (ablation)
- **WHEN** a rollout-matching SFT config sets `rollout_fn_desc_weight: 0`
- **THEN** FN-injected `desc` tokens do not contribute to token CE
- **AND** FP-neutral and EOS-enforced semantics remain intact.


### Requirement: Rollout-matching SFT exposes ST geometry decode mode as typed YAML config
Rollout-matching SFT SHALL expose a typed config knob to select geometry coord decode mode when geometry losses are
enabled (either directly in trainer code or via the pipeline).

Normative behavior:
- Config MUST be expressed under the typed rollout-matching namespace (`rollout_matching.*`) and MUST be strict
  (unknown keys fail fast).
- When the key is omitted, defaults MUST preserve current behavior.
- When enabled, ST decode MUST follow the ST semantics defined by `teacher-forcing-unified-loss-registry`.

Normative key name:
- `rollout_matching.coord_decode_mode: exp|st`

Normative mapping / identity:
- The resolved value MUST feed the same internal decode enum used by the two-channel Stage-2 engine
  (`stage2_ab.coord_decode_mode`).
- The resolved value MUST be included in the pipeline identity checksum so ST-vs-exp differences are auditable even when
  the module list is unchanged.

### Requirement: Eval-step supports COCO mAP when enabled
Rollout-aligned Stage-2 SHALL support computing COCO-style bbox mAP during `eval_step` (config-driven) and MUST log the
results under stable metric keys so training runs are paper-ready.

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

#### Scenario: Rollout-aligned eval reports mAP
- **GIVEN** `rollout_matching.eval_detection.enabled=true`
- **WHEN** `eval_step` runs
- **THEN** `rollout/mAP` is present in the eval metrics payload


### Requirement: Trainer-variant guardrails prevent ambiguous pipeline configuration
To avoid “config declared but ignored” ambiguity, the system SHALL enforce strict guardrails:

Normative behavior:
- If `custom.trainer_variant: stage2_two_channel`, the presence of `rollout_matching.pipeline` MUST fail fast with
  guidance to use `stage2_ab.pipeline`.
- If `custom.trainer_variant: stage2_rollout_aligned`, the presence of `stage2_ab.pipeline` MUST fail fast with guidance
  to use `rollout_matching.pipeline`.

#### Scenario: Stage-2 Two-Channel rejects rollout-matching pipeline keys
- **WHEN** `custom.trainer_variant=stage2_two_channel`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast with guidance to use `stage2_ab.pipeline`.

#### Scenario: Rollout-matching rejects stage2_ab pipeline keys
- **WHEN** `custom.trainer_variant=stage2_rollout_aligned`
- **AND** `stage2_ab.pipeline` is present
- **THEN** config validation fails fast with guidance to use `rollout_matching.pipeline`.
