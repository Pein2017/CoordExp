## ADDED Requirements

### Requirement: Production Relaunch Readiness Is Contracted Infrastructure

CoordExp-Swift production relaunches SHALL be governed by explicit readiness
artifacts, not by ad hoc launch notes. The completed V1 rebuild baseline MUST
remain historical authority, and follow-on relaunch work MUST use a separate
change or clearly labeled follow-on contract.

#### Scenario: Completed rebuild baseline exists

- **WHEN** a production relaunch needs new readiness tasks
- **THEN** the completed rebuild task ledger MUST NOT be unchecked or rewritten
- **AND** the relaunch readiness work MUST live in a follow-on change.

### Requirement: Seed Control Before Stochastic Setup

The configured runtime seed SHALL be applied before Qwen model loading, fresh
DoRA adapter initialization, selected embedding setup, optimizer setup, and
runtime setup. Runtime MAY reapply the seed at backend setup time, but this MUST
be recorded as a separate phase and MUST NOT create a new public config knob.

#### Scenario: Fresh DoRA adapter is initialized

- **WHEN** a training config initializes a fresh DoRA adapter
- **THEN** seed setup MUST have already run using the configured runtime seed
- **AND** artifacts MUST include a runtime seed-control receipt linked from the
  manifest.

### Requirement: Derived Batch And Schedule Evidence

Public training configs SHALL continue to author
`training.effective_batch_size`, while backend accumulation remains derived
from world size. Production relaunch artifacts MUST record the derived runtime
batch and planned schedule before training is interpreted.

#### Scenario: EBS64 production relaunch on 8 GPUs

- **WHEN** the r16/a32 EBS64 warmup0p1 production config runs on 8 GPUs with
  unchanged pack cardinality
- **THEN** runtime MUST derive `resolved_grad_accum_steps: 8`
- **AND** the planned schedule SHOULD resolve to `resolved_max_steps: 917`,
  `tail_fill_pack_count: 48`, and warmup steps 92
- **AND** artifacts MUST include a manifest-linked scheduler receipt that
  records the authored warmup surface and the derived warmup step count.

### Requirement: Smoke-To-Production Promotion Gate

Production relaunch SHALL require artifact-backed smoke evidence before the
long 8-GPU run starts. The smoke ladder MUST prove strict config loading,
runtime seed receipts, scheduler warmup derivation, distributed runtime batch
derivation, finite metrics, eval-forward health, FA2 proof policy, rank-local
completion, packing-cache status, and final checkpoint writing.

#### Scenario: Final production-relevant smoke completes

- **WHEN** the final 8-GPU r16/a32 EBS64 smoke completes
- **THEN** rank-zero artifacts MUST include resolved config, run manifest,
  runtime seed-control receipt, runtime setup receipt, packing plan,
  optimizer/scheduler/trainable-surface receipts, eval-forward summary, finite
  metrics, FA2 proof receipt, and `checkpoints/checkpoint-final.json`
- **AND** nonzero rank directories MUST finalize cleanly.

### Requirement: Relaunch Scope Boundaries

The r16/a32 EBS64 warmup0p1 relaunch SHALL remain an Accelerate production
training relaunch. DeepSpeed production support, hidden-state caches, KV caches,
runtime feature caches, new scheduler semantics such as min-LR parity, separate
dataset-seed config surfaces, and token-embedding LR changes are outside this
change unless explicitly approved later.

#### Scenario: DeepSpeed production support is mentioned

- **WHEN** relaunch artifacts or docs describe backend support
- **THEN** they MUST NOT claim DeepSpeed production support
- **AND** the production launch target MUST remain Accelerate unless a separate
  systems-smoke change approves DeepSpeed.
