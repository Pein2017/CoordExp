# coordexp-swift-config-runtime Specification

## Purpose
TBD - created by archiving change rebuild-coordexp-swift-training-infra. Update Purpose after archive.
## Requirements
### Requirement: Strict Resolved Config

Training configuration SHALL be loaded through a strict schema that rejects
unknown authored fields. Config inheritance MAY be used for normal training
configs, but every run MUST save one final self-contained JSON config at
`resolved_config.json` under the rank-zero run directory. The resolved config
MUST include compact resolution provenance: entry config path, inherited
parent paths, source content fingerprints, and path-origin metadata for path
fields. Inheritance MUST use a single top-level `extends` parent per YAML file,
with parent files resolved first and child dictionaries deep-merged over
parents. Lists MUST replace parent lists rather than append. Explicit YAML
`null` MUST only be accepted for optional fields and MUST NOT delete inherited
keys. Paths for data, fixtures, cache, and references MUST resolve relative to
the YAML file that declares them, while `run.artifact_root` MAY remain literal
or cwd-relative by operator choice. Cycles MUST fail with the full inheritance
chain. A second resolved YAML artifact MUST NOT be required.

#### Scenario: Unknown authored field

- **WHEN** a training config contains a field that is not in the strict schema
- **THEN** config loading MUST fail before model, adapter, optimizer, or
  dataset mutation begins.

#### Scenario: Inherited production config

- **WHEN** a production training config extends one or more base configs
- **THEN** the run artifacts MUST include one self-contained
  `resolved_config.json`
- **AND** that artifact MUST be sufficient to understand the run without
  reopening the inherited files
- **AND** it MUST include source fingerprints and path origins for inherited
  config files and path-valued fields.

#### Scenario: Unsupported train order configured

- **WHEN** a training config sets `data.train_order` to any value other than
  `source_order`
- **THEN** config validation MUST fail before cache fingerprinting or packing
- **AND** the unsupported value MUST NOT change cache identity without changing
  example order behavior.

#### Scenario: Child config overrides inherited values

- **WHEN** a child config overrides inherited values
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

Each training launch SHALL resolve one run identity and artifact root before
model or optimizer mutation. Accelerate rank zero MUST select the concrete run
directory according to the configured collision policy, and every rank MUST
use the same selected identity/path for collective checkpoint ordering and
run-associated diagnostics. Only rank zero SHALL create or write that
directory. `run.json` MUST record the selected run identity, concrete run
directory, artifact root, and collision outcome during initialization.

#### Scenario: Run directory collision

- **WHEN** the requested output directory already exists
- **THEN** rank zero MUST follow the configured collision policy
- **AND** the selected output directory and collision outcome MUST be recorded
  in `run.json` before training-side mutation begins.

#### Scenario: Distributed run directory selected

- **WHEN** a multi-rank Accelerate launch resolves its run directory
- **THEN** rank zero MUST select one concrete path and make that identity
  available consistently to every rank
- **AND** non-main ranks MUST NOT create rank-suffixed alternatives.

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
`eval.forward.every_fraction`, and `eval.forward.steps`. Every completed
training step MUST produce a train row in `logging.jsonl`; therefore the public
schema MUST NOT expose a separate training-logging cadence. The schema MUST
reject aliases such as `save_steps`, `eval_steps`, `logging_steps`, a separate
`global_step`, or the removed `training.logging` cadence block. Runtime MAY
materialize its resolved schedule in memory, but MUST NOT require a separate
durable schedule receipt when the same resolved maximum steps and scheduled
eval/checkpoint events are represented by `run.json`, train/eval logging rows,
and checkpoint aliases.

#### Scenario: Fractional cadence resolved

- **WHEN** checkpoint or eval cadence uses `every_fraction`
- **THEN** fractional milestones MUST use
  `ceil(fraction * resolved_max_steps)` style planned-step materialization,
  repeated and clamped to `[1, resolved_max_steps]`
- **AND** collisions with explicit steps or final events MUST be de-duplicated.

#### Scenario: Every train step is logged

- **WHEN** one planned training step completes
- **THEN** rank zero MUST append exactly one train row for that step to
  `logging.jsonl`
- **AND** no authored logging cadence may suppress that row.

#### Scenario: Resolved schedule artifact written

- **WHEN** runtime resolves the planned-step schedule
- **THEN** it MUST retain the schedule in memory for train, eval, checkpoint,
  and final dispatch
- **AND** it MUST NOT write a separate `resolved_step_schedule.json` artifact.

#### Scenario: Legacy cadence alias authored

- **WHEN** a config authors `save_steps`, `eval_steps`, `logging_steps`,
  `global_step`, or `training.logging`
- **THEN** strict config validation MUST fail before schedule resolution.

### Requirement: Effective Batch Runtime Derivation

Public training configs SHALL expose global packed-sequence
`training.effective_batch_size`. Runtime MUST derive the per-rank accumulation
count from effective batch size and Accelerate world size. The derived value
MUST be available in the compact run setup state and MUST NOT be authored as a
public config knob. Runtime MUST fail rather than round, pad ranks, drop ranks,
or mutate the effective batch when the division is not exact.

#### Scenario: Two ranks with effective batch size four

- **WHEN** Accelerate world size is 2 and `training.effective_batch_size` is 4
- **THEN** runtime MUST derive two rank-local micro-steps per planned optimizer
  step
- **AND** the derived value MUST be inspectable from the run setup state.

#### Scenario: Effective batch not divisible by world size

- **WHEN** world size is 2 and `training.effective_batch_size` is 3
- **THEN** runtime setup MUST fail before training begins
- **AND** it MUST NOT silently round or change the effective batch.

#### Scenario: Effective batch smaller than world size

- **WHEN** world size is 2 and `training.effective_batch_size` is 1
- **THEN** runtime setup MUST fail before training begins
- **AND** it MUST NOT drop a rank or create uneven rank ownership.

#### Scenario: Backend accumulation conflict

- **WHEN** an externally configured Accelerate accumulation value conflicts
  with the CoordExp-derived rank-local micro-step count
- **THEN** runtime setup MUST fail before model training begins
- **AND** the public training schema MUST NOT expose a second accumulation
  owner.

#### Scenario: Incomplete final window

- **WHEN** the packed stream tail cannot form a complete planned optimizer-step
  window
- **THEN** epoch-led run-length resolution MUST deterministically continue into
  the next epoch/order stream just enough to complete the final
  effective-batch window
- **AND** training MUST NOT silently discard final packs
- **AND** training MUST NOT create a smaller partial final optimizer update
- **AND** compact run state MUST expose the bounded tail-fill pack count.

### Requirement: Packed Qwen Runtime Controls

Packed Qwen3-VL supervised training SHALL make attention implementation,
compute dtype, sequence-length budget, and logits-memory budget explicit before
model mutation. Unless an explicit debug/parity profile disables it, packed
training MUST request `attn_implementation: flash_attention_2` and a compute
dtype accepted by FlashAttention, such as bf16 or fp16. Runtime setup MUST fail
if a packed training config resolves to sdpa/eager attention, fp32
FlashAttention, or a worst-case full-sequence logits memory estimate above the
resolved budget. Explicit smoke/probe configuration MAY capture branch-level
evidence, while production profiles MUST be able to disable hot-path proof
instrumentation. When `model.fa2_branch_proof` is not set, it MUST resolve to
one representative first-micro-step capture rather than per-forward capture;
explicit `every_forward` MUST remain available for debugging and explicit
`disabled` for profiling. The resolved controls and preflight estimate MUST be
inspectable through the resolved config or explicit smoke/probe output; normal
training MUST NOT emit a per-step forward receipt for this purpose.

#### Scenario: Packed training resolves to sdpa

- **WHEN** a packed supervised training config resolves to sdpa or eager
  attention without an approved debug/parity profile
- **THEN** runtime setup MUST fail before model mutation
- **AND** the diagnostic MUST name the resolved attention implementation.

#### Scenario: Worst-case logits estimate exceeds budget

- **WHEN** `packing.global_max_length`, vocab size, and logits dtype imply a
  worst-case full-sequence logits tensor larger than the resolved budget
- **THEN** runtime setup MUST fail before training begins
- **AND** the diagnostic MUST report the estimated bytes, configured budget,
  sequence length, vocab size, and dtype.

#### Scenario: Production proof capture disabled

- **WHEN** production config disables packed-forward branch-proof capture
- **THEN** the runtime MUST preserve the same packed inputs and validation
- **AND** MUST NOT emit per-step proof receipts.

#### Scenario: Proof cadence omitted from config

- **WHEN** a training config does not set `model.fa2_branch_proof`
- **THEN** the resolved config MUST record `first_micro_step`
- **AND** exactly one representative FA2 branch proof MUST be captured and
  validated for the run
- **AND** all later forwards MUST retain explicit varlen inputs and runtime
  validation without capture instrumentation.

### Requirement: Config-First Entry Surface

The V1 training entry SHALL be role-named and config-first. `python -m
src.train --config <path>` MUST be the primary training invocation shape.
Stable training behavior MUST be expressed in config/schema rather than broad
CLI flags.

#### Scenario: Dry config trace requested

- **WHEN** a user asks to inspect resolved config behavior without training
- **THEN** the trace path MUST stop before model mutation, optimizer
  construction, checkpoint writing, or training metric emission.

### Requirement: Accelerate Is The Sole Training Runtime

CoordExp-Swift training SHALL execute through Accelerate for both world size
one and distributed launches. The authored training schema MUST NOT expose a
backend selector, a separate single-process runtime, DeepSpeed configuration,
or reserved DeepSpeed status fields. One-process execution MUST use the same
runtime ownership and planned-step semantics as distributed execution without
requiring an external multi-process launcher. After `Accelerator`
construction, runtime MUST accept only world-size-one `DistributedType.NO` and
ordinary replicated multi-GPU DDP. It MUST fail before model preparation when
launcher or environment configuration selects FSDP, DeepSpeed, tensor
parallel, or another unsupported wrapper.

#### Scenario: One-process training launch

- **WHEN** a valid training config is launched in one process
- **THEN** the runtime MUST initialize Accelerate with world size one
- **AND** MUST execute through the same `TrainRuntime` contract used by a
  distributed launch.

#### Scenario: Removed backend field is authored

- **WHEN** a config authors `runtime.backend`, `runtime.deepspeed`, or another
  removed backend-selection field
- **THEN** strict config validation MUST fail before model or optimizer
  mutation
- **AND** the field MUST NOT be silently ignored or translated.

#### Scenario: External launcher selects unsupported wrapper

- **WHEN** an Accelerate launcher file or environment variable selects FSDP,
  DeepSpeed, tensor parallel, or another unsupported distributed type
- **THEN** runtime MUST fail before model preparation
- **AND** the diagnostic MUST name the observed distributed type.

### Requirement: Exact Resume Configuration Is Strict And Opt-In

Training configuration SHALL expose a strict `resume` object whose supported
modes are `disabled` and `exact_same_world_size`. The compatibility default
MUST be `disabled`. `resume.checkpoint_dir` MUST be absent or null when
disabled. Exact mode with a null checkpoint path is the publish-only
control/parent branch that creates a committed exact boundary; when exact
continuation authors a checkpoint path, that path MUST resolve from the YAML
file that authors it. Unknown modes, unknown fields, or incompatible mode/path
combinations MUST fail during config resolution before cache, model, optimizer,
or accelerator mutation.

#### Scenario: Resume configuration is omitted

- **WHEN** a current training config does not author `resume`
- **THEN** the resolved config MUST record `resume.mode: disabled` and a null
  checkpoint path
- **AND** runtime MUST not publish exact training state.

#### Scenario: Exact resume path is authored

- **WHEN** a config selects `exact_same_world_size` and authors a checkpoint
  path
- **THEN** the path MUST resolve relative to the YAML file that declared it
- **AND** the resolved mode and absolute path MUST be preserved in
  `resolved_config.json`.

#### Scenario: Resume fields are incompatible

- **WHEN** a disabled config supplies a checkpoint path
- **THEN** strict validation MUST fail before training-side mutation.

#### Scenario: Exact state publication is selected without continuation

- **WHEN** a config selects `exact_same_world_size` with an absent or null
  checkpoint path
- **THEN** the resolved config MUST preserve exact mode with a null checkpoint
  path
- **AND** runtime MUST publish exact training state without attempting restore
  so the run can serve as an uninterrupted control or interrupted parent.

### Requirement: Exact Resume Requires Strict Deterministic Replay Admission

`exact_same_world_size` SHALL require the versioned deterministic-runtime mode
`strict_cuda_replay_v1`. Before Accelerator, model, or CUDA materialization on
every rank, admission MUST verify the launcher-supplied deterministic
environment and establish fail-closed deterministic algorithm settings. The
resolved seed, policy version, dependency/runtime baseline, world size, and
rank/device mapping MUST be included in exact-resume compatibility identity.
The compatibility runtime policy MAY remain available only when exact resume
is disabled.

#### Scenario: Exact resume selects the compatibility runtime policy

- **WHEN** `resume.mode` is `exact_same_world_size` but the deterministic mode
  is not `strict_cuda_replay_v1`
- **THEN** config or runtime admission MUST fail before model or CUDA setup
- **AND** it MUST NOT silently strengthen or weaken the authored policy.

#### Scenario: A launcher prerequisite is absent

- **WHEN** strict replay is selected but a required launcher-provided CUDA,
  CUBLAS, or FlashAttention determinism prerequisite is absent or malformed
- **THEN** admission MUST fail before CUDA initialization
- **AND** the process MUST NOT synthesize a value after launch and claim exact
  compatibility.

#### Scenario: Rank runtime identities disagree

- **WHEN** ranks disagree on seed, policy, dependency baseline, world size, or
  rank/device mapping
- **THEN** all live ranks MUST converge the incompatibility before mutable
  training state is restored.
