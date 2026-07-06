## ADDED Requirements

### Requirement: Checkpoint Handoff Manifest Is Canonical For Production

Each production-eligible CoordExp-Swift checkpoint SHALL provide a
`checkpoint_handoff.json` manifest that is the canonical identity bridge from
training to production inference. The manifest MUST bind the base model path
and identity, tokenizer identity, processor identity, adapter payload identity
when present, selected special-token embedding delta identity when present,
trainable token set, prompt/template identity, checkpoint id, planned step id,
resolved config fingerprint, and intended inference config family.

#### Scenario: Production checkpoint written

- **WHEN** a production-eligible checkpoint is written
- **THEN** `checkpoint_handoff.json` MUST be written beside the checkpoint
  metadata
- **AND** the checkpoint metadata MUST link to that handoff manifest.

#### Scenario: Handoff identity incomplete

- **WHEN** a checkpoint lacks base model, adapter, selected-token embedding,
  tokenizer, processor, template, or intended inference-family identity that is
  required by its resolved config
- **THEN** the checkpoint MUST NOT be marked production handoff ready
- **AND** the diagnostic MUST name the missing identity field.

### Requirement: Production Inference Uses Handoff Identity

Production inference SHALL consume `checkpoint_handoff.json` by default when
loading a CoordExp-Swift checkpoint. It MUST reject mismatches between the
handoff manifest and the resolved inference config for base model, adapter,
selected-token embedding delta, tokenizer, processor, template, and intended
inference family. Explicit manual base/adapter/delta composition MAY be
allowed only in a marked research/dev mode and MUST be recorded as
noncanonical evidence.

#### Scenario: Matching handoff inference

- **WHEN** production inference is launched from a checkpoint handoff manifest
  and the resolved inference config matches the manifest identities
- **THEN** inference MAY proceed
- **AND** the run manifest MUST record the handoff manifest path and identity
  fingerprint.

#### Scenario: Adapter mismatch

- **WHEN** production inference resolves an adapter payload that differs from
  the adapter identity in `checkpoint_handoff.json`
- **THEN** inference MUST fail before model generation
- **AND** the diagnostic MUST identify the handoff adapter identity and the
  resolved adapter identity.

#### Scenario: Manual research composition

- **WHEN** inference is launched with explicit manual base, adapter, or
  selected-token embedding paths rather than a canonical handoff manifest
- **THEN** the run MUST be marked as research/dev composition
- **AND** the produced artifacts MUST NOT claim production handoff eligibility.

### Requirement: Readiness Validator Is Read-Only

CoordExp-Swift SHALL provide a read-only production-readiness validation
surface for run and checkpoint artifacts. The validator MUST inspect artifact
presence and identity consistency without modifying training checkpoints,
inference outputs, cached packs, or model payloads. It MUST return a pass/hold
decision with concrete missing or mismatched artifact handles.

#### Scenario: Complete ready checkpoint

- **WHEN** the validator inspects a checkpoint with complete handoff identity,
  resolved config evidence, runtime/scheduler receipts, pack-cache identity,
  FA2/MRoPE proof evidence when required, final checkpoint metadata, and
  accepted eval artifacts for the requested gate
- **THEN** it MUST return a production-ready decision for that gate.

#### Scenario: Missing pack-cache identity

- **WHEN** the validator inspects a run whose checkpoint exists but whose
  artifact tree lacks required pack-cache identity evidence
- **THEN** it MUST return hold
- **AND** the failure list MUST name the missing pack-cache identity evidence
  rather than emitting a generic readiness failure.

#### Scenario: Validator execution

- **WHEN** the readiness validator runs
- **THEN** it MUST NOT write, delete, rewrite, or repair model, checkpoint,
  cache, inference, or eval artifacts
- **AND** any optional report output MUST be clearly separate from the
  validated artifact tree.
