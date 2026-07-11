# coordexp-swift-config-runtime Specification

## Purpose
TBD - created by archiving change rebuild-coordexp-swift-training-infra. Update Purpose after archive.
## Requirements
### Requirement: Strict Resolved Config

Training configuration SHALL be loaded through a strict schema that rejects
unknown authored fields. Config inheritance MAY be used for normal training
configs, but every run MUST save the final resolved config as self-contained
YAML and JSON under the run artifact directory. The resolved config artifacts
MUST include compact resolution provenance: entry config path, inherited parent
paths, source content fingerprints, and path-origin metadata for path fields.
Inheritance MUST use a single top-level `extends` parent per YAML file, with
parent files resolved first and child dictionaries deep-merged over parents.
Lists MUST replace parent lists rather than append. Explicit YAML `null` MUST
only be accepted for optional fields and MUST NOT delete inherited keys.
Paths for data, fixtures, cache, and references MUST resolve relative to the
YAML file that declares them, while `run.artifact_root` MAY remain literal or
cwd-relative by operator choice. Cycles MUST fail with the full inheritance
chain.

#### Scenario: Unknown authored field

- **WHEN** a training config contains a field that is not in the strict schema
- **THEN** config loading MUST fail before model, adapter, optimizer, or
  dataset mutation begins.

#### Scenario: Inherited production config

- **WHEN** a production training config extends one or more base configs
- **THEN** the run artifacts MUST include the final resolved YAML and JSON
- **AND** the resolved artifacts MUST be sufficient to understand the run
  without reopening the inherited files.
- **AND** the resolved artifacts MUST include source fingerprints and path
  origins for inherited config files and path-valued fields.

#### Scenario: Unsupported train order configured

- **WHEN** a V1 training config sets `data.train_order` to any value other
  than `source_order`
- **THEN** config validation MUST fail before cache fingerprinting or packing
- **AND** the unsupported value MUST NOT change cache identity without changing
  example order behavior.

#### Scenario: Child config overrides inherited values

- **WHEN** a child config extends a parent config
- **THEN** parent dictionaries MUST resolve before child dictionaries
- **AND** child scalar or dictionary values MUST override parent values by key
- **AND** child list values MUST replace the parent list exactly.

#### Scenario: Inheritance cycle authored

- **WHEN** two or more config files form an `extends` cycle
- **THEN** config loading MUST fail before schema construction
- **AND** the diagnostic MUST include the full cycle of config paths.

### Requirement: Adapter Seed Mode Config Hierarchy

Adapter config SHALL expose a stable seed-mode hierarchy under
`adapter.type: dora`. If `adapter.seed_mode` is omitted, V1 MAY infer
`initialize_new` when `adapter.path` is absent and `load_existing` when
`adapter.path` is present for backward compatibility with earlier Swift
configs. If `adapter.seed_mode` is explicit, `initialize_new` MUST reject
adapter paths and source paths, `load_existing` MUST require `adapter.path` and
reject source paths, and `warm_start_expand_dora` MUST require
`adapter.source_adapter_path` plus `adapter.repaired_embedding_payload_path`
while rejecting `adapter.path`. The configured `adapter.target_towers` SHALL
remain the authoritative set of required targets in all seed modes.

#### Scenario: Fresh adapter seed mode configured

- **WHEN** `adapter.seed_mode: initialize_new` is configured
- **THEN** config validation MUST reject `adapter.path`,
  `adapter.source_adapter_path`, and `adapter.repaired_embedding_payload_path`
- **AND** adapter setup MUST create the configured targets from fresh DoRA
  initialization.

#### Scenario: Existing adapter seed mode configured

- **WHEN** `adapter.seed_mode: load_existing` is configured
- **THEN** config validation MUST require `adapter.path`
- **AND** config validation MUST reject warm-start source paths.

#### Scenario: Expand adapter seed mode configured

- **WHEN** `adapter.seed_mode: warm_start_expand_dora` is configured
- **THEN** config validation MUST require `adapter.source_adapter_path`
- **AND** config validation MUST require
  `adapter.repaired_embedding_payload_path`
- **AND** config validation MUST reject `adapter.path`
- **AND** `adapter.target_towers` MUST define the required target set.

### Requirement: Run Identity And Artifact Root

Each training run SHALL resolve a run identity and artifact root before model
or optimizer mutation. The artifact root MUST be the parent storage location
for run outputs, while the output directory MUST be the concrete run directory
created under that root.

#### Scenario: Run directory collision

- **WHEN** the resolved output directory already exists
- **THEN** runtime MUST follow the configured collision policy
- **AND** the chosen output directory MUST be recorded in the run manifest
  before training-side mutation begins.

### Requirement: Planned Step Schedule

The resolved planned-step schedule SHALL be computed before training begins
from the dataloader, world size, `training.effective_batch_size`, `epochs`, and
optional debug `max_steps`. If `max_steps` is set, it MUST take priority over
epoch-derived length. Eval, checkpoint, logging, and final events MUST use the
planned-step clock, not a successful-update counter.

#### Scenario: Debug max steps configured

- **WHEN** a config sets `max_steps: 5`
- **THEN** the resolved maximum planned steps MUST be 5 regardless of the
  epoch-derived value
- **AND** final checkpoint and final metrics MUST be scheduled at planned step
  5.

#### Scenario: Unsafe optimizer update skipped

- **WHEN** a planned step is marked unsafe before optimizer update
- **THEN** schedule events for that planned step MUST still use the original
  planned-step id
- **AND** artifacts MUST record that the optimizer update was not applied.

### Requirement: Cadence Config And Resolved Step Schedule

Cadence config SHALL use the canonical authored fields
`checkpoint.every_fraction`, `checkpoint.steps`, `checkpoint.save_final`,
`eval.forward.every_fraction`, `eval.forward.steps`,
`training.logging.every_fraction`, and `training.logging.steps`. V1 MUST reject
parallel aliases such as `save_steps`, `eval_steps`, `logging_steps`, and a
separate `global_step` concept. Every run MUST materialize
`resolved_step_schedule.json` before training starts.

#### Scenario: Fractional cadence resolved

- **WHEN** a cadence uses `every_fraction`
- **THEN** fractional milestones MUST use
  `ceil(fraction * resolved_max_steps)` style planned-step materialization,
  repeated and clamped to `[1, resolved_max_steps]`
- **AND** collisions with explicit steps or final events MUST be de-duplicated.

#### Scenario: Resolved schedule artifact written

- **WHEN** `resolved_step_schedule.json` is written
- **THEN** it MUST contain separate event lists for `eval.forward`,
  `checkpoint`, `training.logging`, and `final`
- **AND** every event MUST record `planned_step_id`, `event`,
  `trigger_reasons`, `source_config_path`, `deduped_from`, and `required`.

#### Scenario: Legacy cadence alias authored

- **WHEN** a config authors `save_steps`, `eval_steps`, `logging_steps`, or
  `global_step`
- **THEN** strict config validation MUST fail before schedule resolution.

### Requirement: Effective Batch Runtime Derivation

Public training configs SHALL expose global packed-sequence
`training.effective_batch_size`. Runtime MUST derive the per-rank accumulation
count from effective batch size and world size. The derived value MUST be
recorded in runtime receipts and MUST NOT be authored as a public config knob.
Runtime MUST fail rather than round, pad ranks, drop ranks, or mutate the
effective batch when the division is not exact.

#### Scenario: Two ranks with effective batch size four

- **WHEN** world size is 2 and `training.effective_batch_size` is 4
- **THEN** runtime MUST derive two rank-local micro-steps per planned optimizer
  step
- **AND** the derived count MUST be present in runtime receipts.

#### Scenario: Effective batch not divisible by world size

- **WHEN** world size is 2 and `training.effective_batch_size` is 3
- **THEN** runtime setup MUST fail before training begins
- **AND** it MUST NOT silently round or change the effective batch.

#### Scenario: Effective batch smaller than world size

- **WHEN** world size is 2 and `training.effective_batch_size` is 1
- **THEN** runtime setup MUST fail before training begins
- **AND** it MUST NOT drop a rank or create uneven rank ownership.

#### Scenario: Backend accumulation conflict

- **WHEN** an Accelerate or DeepSpeed backend config independently specifies an
  accumulation value that conflicts with the runtime-derived value
- **THEN** runtime setup MUST fail before model training begins.

#### Scenario: Incomplete final window

- **WHEN** the packed stream tail cannot form a complete planned optimizer-step
  window
- **THEN** epoch-led run-length resolution MUST deterministically continue into
  the next epoch/order stream just enough to complete the final effective-batch
  window
- **AND** training MUST NOT silently discard final packs
- **AND** training MUST NOT create a smaller partial final optimizer update
- **AND** runtime receipts MUST record the bounded tail-fill pack count.

### Requirement: Packed Qwen Runtime Controls

Packed Qwen3-VL supervised training SHALL make attention backend, compute
dtype, sequence-length budget, and logits-memory budget explicit before model
mutation. Unless an explicit debug/parity profile disables it, packed training
MUST request `attn_implementation: flash_attention_2` and a compute dtype
accepted by FlashAttention, such as bf16 or fp16. Runtime setup MUST fail if a
packed training config resolves to sdpa/eager attention, fp32 FlashAttention,
or a worst-case full-sequence logits memory estimate above the resolved budget.
Configs MAY expose an explicit FA2 branch-proof policy so smoke/debug profiles
can capture branch-level evidence while production profiles can disable
hot-path proof instrumentation after prior representative proof. The resolved
policy MUST be preserved in the resolved config artifact.
The worst-case estimate MUST use the resolved `packing.global_max_length`,
tokenizer vocab size, and model logits dtype, and the estimate MUST be recorded
in a setup or forward receipt. Implementations MAY materialize only selected
supervised rows, but MUST NOT use selected-row materialization to bypass the
preflight budget guard without an approved debug/parity profile.

#### Scenario: Packed training resolves to sdpa

- **WHEN** a packed supervised training config resolves to sdpa or eager
  attention without an approved debug/parity profile
- **THEN** runtime setup MUST fail before model mutation
- **AND** the diagnostic MUST name the resolved attention implementation.

#### Scenario: Worst-case logits estimate exceeds budget

- **WHEN** `packing.global_max_length`, vocab size, and logits dtype imply a
  worst-case full-sequence logits tensor larger than the resolved logits-memory
  budget
- **THEN** runtime setup MUST fail before training begins
- **AND** the diagnostic MUST report the estimated bytes, configured budget,
  sequence length, vocab size, and dtype.

### Requirement: Backend Status Labels

Runtime backend support SHALL use explicit status labels for DeepSpeed:
`schema_accepted`, `conflict_validation_implemented`,
`systems_smoke_verified`, and `production_supported`. V1 MUST NOT claim
DeepSpeed production support until the separate systems smoke verifies
prepare, backward, clipping, optimizer stepping, checkpoint save/load, and
rank-safe artifacts through `TrainRuntime`.

#### Scenario: V1 vertical smoke with DeepSpeed config

- **WHEN** the V1 vertical smoke validates DeepSpeed setup conflicts without
  running DeepSpeed execution
- **THEN** the artifact status MAY report `schema_accepted` and
  `conflict_validation_implemented`
- **AND** it MUST NOT report `production_supported`.

### Requirement: Config-First Entry Surface

The V1 training entry SHALL be role-named and config-first. `python -m
src.train --config <path>` MUST be the primary training invocation shape.
Stable training behavior MUST be expressed in config/schema rather than broad
CLI flags.

#### Scenario: Dry config trace requested

- **WHEN** a user asks to inspect resolved config behavior without training
- **THEN** the trace path MUST stop before model mutation, optimizer
  construction, checkpoint writing, or training metric emission.
