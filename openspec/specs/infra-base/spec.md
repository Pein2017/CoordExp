# Infra-base contract

## Purpose

Define the maintained CoordExp infrastructure surface for configuration-first
single-node training, fail-closed data and artifact handling, dynamic HF
inference, qualified vLLM execution, and direct detection evaluation.

## Requirements

### Requirement: Configuration-first local training

The system SHALL expose training through `python -m src.train --config <path>`
and support code-shaped single-node launches with one through four GPU
processes. A current configuration SHALL NOT hardcode an eight-rank topology.

#### Scenario: A local training launch is configured

- **WHEN** an operator loads a maintained training configuration
- **THEN** the configuration supports a single-node one-through-four-process launch without an eight-rank requirement

### Requirement: Fail-closed pack cache

The system SHALL bind a cached packed micro-step to all semantic determinants
of its content. It SHALL reject stale, corrupt, incomplete, and mismatched
cache state.

#### Scenario: Cached packing state does not match its consumer

- **WHEN** a cache is stale, corrupt, incomplete, or has a mismatched semantic identity
- **THEN** the cache is rejected before it can supply a training micro-step

### Requirement: Bounded exact resume

Exact resume SHALL be opt-in, admitted only at an optimizer boundary, require
the same world size, and fail closed when identities or runtime state do not
match.

#### Scenario: An exact continuation is requested

- **WHEN** a run requests exact resume from a completed optimizer boundary
- **THEN** restoration proceeds only when the checkpoint identity, runtime state, and world size match

### Requirement: COCO and LVIS provenance

The supported public-data routes SHALL be COCO and LVIS. Preparation and
materialization SHALL preserve JSONL validation and provenance sufficient to
reproduce the declared processed view.

#### Scenario: A maintained public-data view is prepared

- **WHEN** a COCO or LVIS processed view is validated or regenerated
- **THEN** its JSONL artifacts and provenance identify the declared source and transformation contract

### Requirement: Separate checkpoint consumers

An inference payload SHALL be independently self-authenticated. Exact-resume
training state SHALL remain separate and SHALL NOT be inferred from an
inference payload.

#### Scenario: A checkpoint is consumed for inference

- **WHEN** inference loads a checkpoint payload
- **THEN** it validates the inference payload independently and does not treat it as exact-resume training state

### Requirement: Qualified inference backends

The system SHALL expose dynamic HF and vLLM inference. Dynamic HF SHALL remain
the authoritative adapter-plus-embedding-delta execution semantics. A composed
vLLM production path SHALL execute qualification and fail closed when its
declared requirements are not met.

#### Scenario: A composed model is requested through vLLM

- **WHEN** a vLLM configuration resolves a composed execution model
- **THEN** production execution is denied unless the complete matching qualification set has been admitted

### Requirement: Direct evaluation artifacts

The direct evaluator SHALL consume the scored inference artifact family and
emit metrics plus conversion artifacts. Visualization SHALL consume run
artifacts without substituting for evaluation.

#### Scenario: Scored inference artifacts are evaluated

- **WHEN** the direct evaluator consumes a completed scored artifact family
- **THEN** it emits metrics and conversion artifacts whose eligibility and lineage remain bound to that inference run

### Requirement: Evidence-scoped acceptance

Static interface evidence SHALL be distinguished from runtime evidence. A
two-GPU Qwen3-VL 2B plus COCO vertical witness SHALL exist before making a
two-GPU runtime claim.

#### Scenario: A two-GPU runtime claim is reported

- **WHEN** the system is described as having executed on two GPUs
- **THEN** the claim cites a matching Qwen3-VL 2B plus COCO vertical witness and does not imply untested topologies or model quality
