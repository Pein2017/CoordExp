# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 two-channel training supports a config-declared objective and diagnostics pipeline
When `custom.trainer_variant: stage2_two_channel`, the system SHALL use an explicit YAML-declared objective/diagnostics pipeline for the canonical clean-prefix Channel-B contract.

Normative behavior:
- `stage2_ab.pipeline` MUST be present. There is no implicit default pipeline manifest for this contract.
- Canonical Stage-2 AB objective ordering for this contract is:
  1. `token_ce`
  2. `duplicate_ul`
  3. `bbox_geo`
  4. `coord_reg`
- Canonical Stage-2 AB diagnostics MAY include `coord_diag`.
- `duplicate_ul` MUST be present in canonical Stage-2 AB pipelines and MUST declare `channels: [B]`.
- `duplicate_ul` module `weight` is the only v1 scaling surface for duplicate UL.
- The old raw-prefix Channel-B contract is removed; there is no contract toggle or compatibility mode.

#### Scenario: Missing stage2_ab.pipeline fails fast
- **WHEN** a Stage-2 AB config sets `custom.trainer_variant: stage2_two_channel`
- **AND** `stage2_ab.pipeline` is absent
- **THEN** config loading fails fast before trainer init
- **AND** the error indicates `stage2_ab.pipeline` is required.

#### Scenario: Missing duplicate_ul in the canonical Channel-B pipeline fails fast
- **WHEN** a Stage-2 AB config declares `stage2_ab.pipeline.objective`
- **AND** the objective list omits `duplicate_ul`
- **THEN** config validation fails fast
- **AND** the error indicates the canonical clean-prefix Channel-B contract requires `duplicate_ul`.

### Requirement: Stage-2 Two-Channel module names are stable and discoverable
Stage-2 Two-Channel SHALL provide a strict module registry for its pipeline modules, and the module names SHALL be stable so YAML-declared experiments remain auditable.

Normative minimum objective module names for this contract:
- `token_ce`
- `duplicate_ul`
- `bbox_geo`
- `coord_reg`

Normative minimum diagnostics module names:
- `coord_diag`

#### Scenario: Unknown Stage-2 two-channel module names fail fast
- **WHEN** `stage2_ab.pipeline` references an objective module name not present in the Stage-2 registry
- **THEN** trainer initialization fails fast
- **AND** the error includes the unknown name and allowed module names.

### Requirement: Stage-2 Two-Channel module configs are strict and typed
Stage-2 Two-Channel SHALL validate module `config` payloads and `stage2_ab.channel_b` payloads strictly so experiments are reproducible and fail fast on schema drift.

Normative behavior:
- `duplicate_ul.config` MUST be an empty mapping in v1.
- `token_ce.config` no longer accepts any legacy invalid-structure amplification knob for Channel-B.
- `stage2_ab.channel_b` MUST accept only:
  - `duplicate_iou_threshold`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`
- Unknown keys in a module `config` or in `stage2_ab.channel_b` MUST fail fast with actionable diagnostics.

#### Scenario: Non-empty duplicate_ul.config fails fast
- **WHEN** `stage2_ab.pipeline.objective[*].name=duplicate_ul`
- **AND** its `config` mapping contains any key
- **THEN** configuration parsing fails fast
- **AND** the error indicates `duplicate_ul.config` must be empty for v1.

#### Scenario: Legacy invalid-structure multiplier placement fails fast
- **WHEN** a Stage-2 AB config sets `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`
- **THEN** configuration parsing fails fast
- **AND** the error indicates that legacy raw-prefix invalid-structure amplification is not part of the canonical clean-prefix contract.

### Requirement: Channel-B reuses rollout-matching infra (clean-prefix parse/match + mandatory FN append)
Channel-B MUST reuse rollout generation and matching infrastructure, but its positive supervision contract is now clean-prefix based rather than raw-prefix based.

Normative behavior:
- Rollout generation MUST remain configured under `rollout_matching`.
- Parsing MUST use bounded container salvage plus strict record acceptance.
- Matching MUST be deterministic and MUST operate on `accepted_objects_clean`, not on the raw parsed bbox list.
- The positive teacher-forced prefix MUST be canonical serialization of `accepted_objects_clean`.
- FN append MUST remain mandatory so all GT objects are present in the final teacher-forced target.

#### Scenario: Channel-B teacher-forced target uses the clean accepted prefix
- **GIVEN** Channel-B is selected and rollout generation succeeds
- **WHEN** the trainer builds the teacher-forced target
- **THEN** the positive prefix is canonical serialization of `accepted_objects_clean`
- **AND** later correct objects are teacher-forced on that clean prefix rather than the raw rollout prefix.

### Requirement: Channel-B invalid rollouts fall back deterministically (no silent skips)
When Channel-B is selected and a rollout response cannot be recovered into an append-ready `{"objects": [...]}` prefix, the trainer MUST:

- mark the rollout invalid for that sample,
- fall back to the canonical empty prefix `{"objects": [`,
- treat the rollout as containing zero valid predicted objects,
- append all GT objects as FN and continue training that sample.

#### Scenario: Invalid rollout falls back to the canonical empty objects prefix
- **GIVEN** Channel-B is selected for a sample
- **AND** the rollout response does not yield an append-ready `{"objects": [...]}` prefix
- **WHEN** the trainer parses the rollout for matching
- **THEN** it marks the rollout invalid for that sample
- **AND** it uses `{"objects": [` as the prefix and FN-appends all GT objects
- **AND** the sample is still included in teacher-forced training.

#### Scenario: Closure-resolution ambiguity keeps the sample on the FN-tail fallback path
- **GIVEN** Channel-B has already built a deterministic clean-prefix teacher-forced target
- **AND** explicit closure-marker bookkeeping cannot be resolved unambiguously for that target
- **WHEN** the trainer finalizes per-sample Channel-B supervision metadata
- **THEN** it keeps the sample in teacher-forced training
- **AND** it falls back to the normal FN-tail supervision path without dropping the sample.

## ADDED Requirements

### Requirement: Channel-B clean boundaries and duplicate bursts are canonical
The clean-prefix Channel-B contract SHALL define boundary-indexed duplicate bursts over the deduplicated clean sequence.

Normative behavior:
- Clean boundaries are indexed by insertion position in `accepted_objects_clean`:
  - boundary `0` before the first clean object,
  - boundary `N` after the last clean object, where `N = len(accepted_objects_clean)`.
- If `accepted_objects_clean` is empty, exactly one boundary exists: `0`.
- Duplicate bursts MUST attach to these canonical clean boundaries.

#### Scenario: Empty clean sequence still exposes one valid boundary
- **WHEN** sequential dedup yields `accepted_objects_clean = []`
- **THEN** duplicate bursts are still indexed against boundary `0`
- **AND** duplicate-ul target construction remains well-defined.

### Requirement: Generic unmatched clean extras remain neutral context
Accepted clean objects that are unmatched after Hungarian MAY remain in the clean prefix as context, but they MUST remain neutral with respect to supervision.

Normative behavior:
- Unmatched clean extras MUST NOT populate matched-prefix struct masks.
- Unmatched clean extras MUST NOT populate coord/bbox supervision groups.
- Unmatched clean extras MUST NOT create duplicate-ul positives.

#### Scenario: Unmatched clean extra stays in context but produces no positive supervision
- **WHEN** Channel-B retains an unmatched clean accepted object in the clean prefix
- **THEN** that object remains visible in the canonical teacher-forced prefix
- **AND** it contributes zero matched-prefix CE, zero bbox loss, zero coord loss, and zero duplicate-ul positives.
