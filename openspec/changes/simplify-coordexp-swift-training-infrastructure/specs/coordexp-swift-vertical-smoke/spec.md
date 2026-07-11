## ADDED Requirements

### Requirement: Multi-Rank Accelerate Ownership Smoke

Acceptance SHALL include a real multi-rank Accelerate smoke through the public
training entrypoint. The smoke MUST prove that all ranks participate in the
same training, finite-gate, scheduled-event, barrier, and checkpoint-save
order while only rank zero creates durable run artifacts.

#### Scenario: Multi-rank smoke completes

- **WHEN** the accepted multi-rank Accelerate smoke completes
- **THEN** exactly one run directory and one `logging.jsonl` MUST exist for the
  logical launch
- **AND** checkpoint payloads and final/best aliases MUST be materialized only
  once
- **AND** no rank-suffixed duplicate run trees may exist.

#### Scenario: Multi-rank checkpoint is loaded

- **WHEN** the smoke writes its final adapter checkpoint
- **THEN** the inference engine MUST load its configured adapter and optional
  selected-token embedding payloads successfully.

### Requirement: Smoke Enforces Bounded Artifact Shape

The accepted one-process and multi-rank smokes SHALL assert the expected
normal-run file inventory and measure non-checkpoint artifact bytes. The
inventory MUST reject `metrics/`, `receipts/`, per-step Qwen receipt files,
rank-local progress streams, full-history `training_result.json`,
`checkpoint_handoff.json`, and rank-suffixed run directories.

#### Scenario: Removed artifact family reappears

- **WHEN** a smoke produces any removed normal-run artifact family
- **THEN** artifact-contract verification MUST fail
- **AND** the smoke MUST NOT be accepted merely because optimization
  completed.

#### Scenario: Logging inventory is inspected

- **WHEN** the smoke completes five train steps and two eval invocations
- **THEN** `logging.jsonl` MUST contain exactly five `train` rows and two
  `eval` rows
- **AND** every row MUST use its planned training step as `step`.

## MODIFIED Requirements

### Requirement: Five-Step Vertical Smoke

The one-process acceptance smoke SHALL execute a real Qwen3-VL training path
through Accelerate world size one with `packing.global_max_length`, fixture
data with sample limiting, resolved maximum planned steps of 5, protected
default losses, backward, optimizer boundary, and static planned-step schedule.
It MUST include two scheduled eval runs, normally from explicit
`eval.forward.steps: [2, 4]`. The smoke config MUST use
`training.effective_batch_size: 2` so planned-step accumulation and
`segment_balanced` denominators are exercised. If local resources cannot
support that setting, lowering it requires an explicit user-approved smoke
scope change and an alternate test that proves uneven planned-step denominator
behavior.

#### Scenario: Smoke completes training

- **WHEN** the five-step smoke completes
- **THEN** `run.json` MUST show five completed planned steps
- **AND** `logging.jsonl` MUST contain five train rows and two eval rows
- **AND** a final inference-loadable adapter checkpoint and
  `checkpoints/final.json` MUST exist
- **AND** one explicit bounded packed-forward proof MUST show at least two
  isolated `PackedSegment`s.

### Requirement: Smoke Artifact Acceptance

The vertical smoke SHALL be accepted only when the normal run tree contains
one self-contained `resolved_config.json`, compact `run.json`, one
`logging.jsonl`, minimal inference-loadable checkpoint payloads, and final/best
aliases when configured. Unit tests and an
explicit bounded packed-forward proof MUST cover trainable-surface validation,
optimizer-group coverage, per-segment MRoPE reset points, FA2 cumulative
sequence splits, same-segment causal loss mapping, and `segment_balanced`
behavior on unequal eligible-atom counts. Those proof details MUST NOT be
required as per-step production artifacts.

#### Scenario: Smoke lacks required normal-run file

- **WHEN** the five-step smoke omits `run.json`, `resolved_config.json`,
  `logging.jsonl`, or its final adapter payload/alias
- **THEN** the vertical smoke MUST be considered incomplete.

#### Scenario: Unit and probe evidence covers packed semantics

- **WHEN** the artifact inventory is minimal
- **THEN** focused tests and one explicit packed-forward proof MUST still
  establish packed position, FA2 isolation, causal mapping, and loss-normalizer
  correctness
- **AND** artifact minimality MUST NOT weaken those semantic gates.

## REMOVED Requirements

### Requirement: DeepSpeed Smoke Boundary

**Reason**: DeepSpeed schema, setup validation, status labels, and execution
support are removed. A smoke boundary for an unsupported backend creates
maintenance work without supported behavior.

**Migration**: Use the required one-process and multi-rank Accelerate smokes.
Any future DeepSpeed support requires a separate OpenSpec and real systems
evidence.
