## ADDED Requirements

### Requirement: CUDA-required non-dry inference runtime
CoordExp-swift inference SHALL require CUDA for every non-dry execution path.
The runtime MUST fail before Qwen model loading if no CUDA device is visible.
This applies to direct single-rank execution and controller/worker execution.
This feature MUST NOT silently fall back to CPU execution.

#### Scenario: CUDA unavailable
- **WHEN** a non-dry inference run starts with no visible CUDA device
- **THEN** the run fails before Qwen model loading
- **AND** the failure summary does not claim benchmark eligibility

#### Scenario: Dry run without CUDA
- **WHEN** `debug.dry_run: true` is configured and no CUDA device is visible
- **THEN** config and run-directory validation may complete without model
  loading
- **AND** no scored artifact set or benchmark eligibility claim is written

### Requirement: No public GPU-id config surface
The V1 data-parallel inference resource boundary SHALL be the visible CUDA environment, not stable config GPU ids.
The strict inference config MUST NOT add a public stable field such as
`gpu_ids` or `num_gpus` for this change. Tests MAY use private overrides for
device discovery, but production users restrict devices through
`CUDA_VISIBLE_DEVICES`.

#### Scenario: User restricts devices
- **WHEN** the user wants inference on four specific GPUs
- **THEN** the supported V1 mechanism is launching with
  `CUDA_VISIBLE_DEVICES=<four tokens>`
- **AND** the resolved inference config remains focused on semantic inference
  settings rather than cluster resource selection

### Requirement: Parallelism metadata in resolved runtime evidence
Every data-parallel inference run SHALL record its resolved parallelism policy.
The evidence MUST include parent visible CUDA tokens, active rank count,
per-device decode batch size, worker binding policy, shard-plan fingerprint,
and whether the run used direct single-process or controller/worker execution.

#### Scenario: Parallel run metadata
- **WHEN** a multi-rank inference run completes
- **THEN** manifest or provenance evidence records active rank count,
  per-device batch size, and rank-to-device mapping

#### Scenario: Direct single-rank path
- **WHEN** only one active rank is needed
- **THEN** manifest or summary evidence records that the direct single-process
  path was used
- **AND** non-dry direct execution still records CUDA-required runtime evidence
- **AND** the manifest records direct-runtime device evidence after model load,
  including logical device and model first-parameter device when available
