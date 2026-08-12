## ADDED Requirements

### Requirement: Pre-Accelerator Launcher Identity Is Attested

Before cache admission creates Accelerator state, a distributed training launch
MUST strictly validate its launcher-provided global rank and world size. The
validated launcher rank zero MAY select the one shared run directory and own
preflight artifacts through a bounded CPU control plane. After admission,
Accelerator MUST resolve exactly the same global rank and world size before
model, adapter, optimizer, or material GPU setup. A missing, malformed, or
mismatched distributed identity MUST fail closed and MUST NOT create an
alternate rank-suffixed run tree.

#### Scenario: Launcher and Accelerator identities agree

- **WHEN** cache admission succeeds using a validated launcher rank and world
  size
- **THEN** Accelerator setup may continue only after resolving the same rank
  and world size
- **AND** the existing run directory remains owned by the same logical rank
  zero

#### Scenario: Distributed launcher identity is only partially present

- **WHEN** exactly one of `RANK` or `WORLD_SIZE` is present, or either value is
  malformed or outside the declared world
- **THEN** launch fails before the CPU control plane, artifact owner, cache
  admission, or Accelerator is constructed
- **AND** the missing field is not filled from direct-launch defaults

#### Scenario: Accelerator identity differs after admission

- **WHEN** Accelerator resolves a rank or world size different from the
  validated launcher identity
- **THEN** every live rank converges the same bounded failure before model work
- **AND** the existing shared run artifact records the identity mismatch while
  `cache_admission` is still the terminal phase
- **AND** no second run tree is created

### Requirement: Infrastructure Optimization Policies Are Strict And Explicit

The canonical runtime configuration MUST expose explicit, typed controls for
packing policy, input-provider mode, resume mode, and upstream attention-backend
selection, plus a versioned deterministic replay policy. Source-order next-fit,
the accepted synchronous provider, disabled exact resume, the legacy runtime
policy, and the currently accepted FA2 route SHALL remain compatibility defaults
until their named promotion gates pass. Exact same-world-size resume MUST select
`strict_cuda_replay_v1`; unknown, retired, or mutually incompatible values MUST
fail before cache preparation or expensive setup.

#### Scenario: A compatibility configuration is loaded

- **WHEN** a current configuration omits all newly optional optimization
  controls
- **THEN** strict loading resolves the compatibility defaults and records them
  explicitly in the run receipt

#### Scenario: An incompatible combination is requested

- **WHEN** exact resume is combined with a non-replayable packing policy or a
  non-strict replay policy, or a backend is requested that lacks the required
  packed-boundary contract
- **THEN** configuration validation fails before cache or model work begins and
  identifies the conflicting fields

#### Scenario: Strict replay environment is not launcher-bound

- **WHEN** `strict_cuda_replay_v1` is selected but its exact FlashAttention or
  CUBLAS environment value was not already supplied by the launcher
- **THEN** runtime admission fails before CUDA initialization and does not
  silently synthesize or weaken the missing setting

#### Scenario: An environment override changes an optimization policy

- **WHEN** a supported diagnostic environment override changes a resolved
  provider, packing, or backend value
- **THEN** the resolved value and override source are recorded in the run
  receipt, and production mode rejects any unpersisted semantic override

### Requirement: Upstream And Attention Backend Baseline Is Fail Closed

This change MUST keep the accepted Transformers, FlashAttention, Torch, CUDA,
Accelerate, PEFT, tokenizers, and FA2 route fixed. Strict runtime admission SHALL
bind the resolved dependency source/binary identities and attention backend to
the recorded accepted baseline. The current fixed point is pinned
runtime-baseline schema 3 with digest
`cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784`.
An unknown or unapproved drift MUST fail before cache preparation or model setup
rather than silently selecting a compatible-looking implementation. An upgrade
requires a separate approved compatibility change and is not an execution arm
of this change.

#### Scenario: The installed dependency baseline is used

- **WHEN** the resolved imported dependencies and attention backend match the
  recorded accepted baseline
- **THEN** the accepted installed dependency route remains active and its exact
  provenance is recorded

#### Scenario: The current native runtime fixed point is attested

- **WHEN** minimal CUDA initialization maps the admitted Torch, CUDA runtime,
  driver, and cuDNN objects under schema 3
- **THEN** current receipt
  `outputs/probes/coordexp_swift/wave8_native_runtime/2026-08-10-r2/receipt.json`
  binds baseline digest
  `cc486f03edb88e4fa1c9d41dc6fa98c97f25a633400e6baf98077e8beaf2b784`
  and internal receipt SHA-256
  `829d86ec8d977f30d37e8e979acba6527b3640180825e551785078a4876b1043`
- **AND** the receipt is native-runtime identity evidence only and cannot claim
  model loading, cache publication, training quality, or throughput
- **AND** the earlier r1 receipt remains historical pre-owner-expansion evidence
  rather than a current admission alternative

#### Scenario: An imported dependency or backend drifts

- **WHEN** an imported source/binary identity or resolved attention backend
  differs from the accepted baseline without a separately approved change
- **THEN** admission fails before cache preparation or model setup and names the
  mismatched component and identities

#### Scenario: A future upgrade is requested

- **WHEN** an operator requests a Transformers, FlashAttention, Torch, CUDA,
  Accelerate, PEFT, tokenizers, or attention-backend upgrade
- **THEN** runtime/config surfaces do not promote it through this change
- **AND** the request is routed to a separate compatibility and throughput
  change while the accepted baseline remains unchanged
