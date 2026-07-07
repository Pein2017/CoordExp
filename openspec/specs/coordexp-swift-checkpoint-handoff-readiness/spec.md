# coordexp-swift-checkpoint-handoff-readiness Specification

## Purpose
TBD - created by archiving change harden-coordexp-swift-cache-and-handoff-contracts. Update Purpose after archive.
## Requirements
### Requirement: Checkpoint Handoff Manifest Is Canonical For Handoff Identity

Each handoff-eligible CoordExp-Swift checkpoint SHALL provide a
`checkpoint_handoff.json` manifest that is the canonical identity bridge from
training to inference. The manifest MUST bind the base model path
and identity, tokenizer identity, processor identity, adapter payload identity
when present, selected special-token embedding delta identity when present,
trainable token set, prompt/template identity, checkpoint id, planned step id,
resolved config fingerprint, and intended inference config family.

#### Scenario: Handoff checkpoint written

- **WHEN** a handoff-eligible checkpoint is written
- **THEN** `checkpoint_handoff.json` MUST be written beside the checkpoint
  metadata
- **AND** the checkpoint metadata MUST link to that handoff manifest.

#### Scenario: Handoff identity incomplete

- **WHEN** a checkpoint lacks base model, adapter, selected-token embedding,
  tokenizer, processor, template, or intended inference-family identity that is
  required by its resolved config
- **THEN** the checkpoint MUST NOT pass the `handoff` gate
- **AND** the diagnostic MUST name the missing identity field.

### Requirement: Inference Provenance Distinguishes Handoff From Manual Composition

Inference SHALL record whether model composition came from a validated
`checkpoint_handoff.json` identity or from explicit manual base/adapter/delta
paths. When a neighboring handoff manifest is discovered for configured
checkpoint payload paths, runtime MUST validate it before generation and record
`composition_mode: canonical_handoff`. Explicit manual base/adapter/delta
composition MAY remain allowed, but it MUST be recorded as noncanonical
`composition_mode: research_manual` evidence.

#### Scenario: Matching handoff inference

- **WHEN** inference is launched with configured checkpoint adapter or selected
  special-token embedding payload paths and a neighboring handoff manifest is
  discovered whose identities match those payload paths
- **THEN** inference MAY proceed
- **AND** the run manifest MUST record the handoff manifest path and identity
  fingerprint
- **AND** runtime MUST reject mismatches in base model path, base config hash,
  tokenizer hash, processor identity, template identity, or intended inference
  config family before generation.

#### Scenario: Adapter mismatch

- **WHEN** handoff-backed inference resolves an adapter payload that differs
  from the adapter identity in `checkpoint_handoff.json`
- **THEN** inference MUST fail before model generation
- **AND** the diagnostic MUST identify the handoff adapter identity and the
  resolved adapter identity.

#### Scenario: Manual research composition

- **WHEN** inference is launched with explicit manual base, adapter, or
  selected-token embedding paths rather than a canonical handoff manifest
- **THEN** the run MUST be marked as `research_manual` composition
- **AND** the produced artifacts MUST NOT claim handoff readiness.

### Requirement: Readiness Validator Is Read-Only

CoordExp-Swift SHALL provide a read-only handoff readiness validation surface
for run and checkpoint artifacts. The validator MUST inspect artifact presence
and identity consistency without modifying training checkpoints, inference
outputs, cached packs, or model payloads. It MUST return a pass/hold decision
for the requested `handoff` or `eval` gate with concrete missing or mismatched
artifact handles. The future `production` gate MAY be accepted as a validator
argument in V1, but it MUST return hold with `production_gate_unimplemented`
until a later change defines production readiness.

#### Scenario: Complete ready checkpoint

- **WHEN** the validator inspects a checkpoint with complete handoff payload
  identity for the `handoff` gate
- **THEN** it MUST return `pass`.

#### Scenario: Eval gate without accepted eval roots

- **WHEN** the validator inspects a checkpoint with complete handoff payload
  identity but no accepted eval artifact roots for the `eval` gate
- **THEN** it MUST return hold
- **AND** the failure list MUST name `accepted_eval_artifact_roots`.

#### Scenario: Production gate requested before implementation

- **WHEN** the validator is called with `gate: production` in V1
- **THEN** it MUST return hold
- **AND** the failure list MUST name `production_gate_unimplemented`.

#### Scenario: Validator execution

- **WHEN** the readiness validator runs
- **THEN** it MUST NOT write, delete, rewrite, or repair model, checkpoint,
  cache, inference, or eval artifacts
- **AND** any optional report output MUST be clearly separate from the
  validated artifact tree.
