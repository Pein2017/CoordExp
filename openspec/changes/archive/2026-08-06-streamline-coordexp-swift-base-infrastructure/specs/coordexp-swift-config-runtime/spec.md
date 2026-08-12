## MODIFIED Requirements

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
