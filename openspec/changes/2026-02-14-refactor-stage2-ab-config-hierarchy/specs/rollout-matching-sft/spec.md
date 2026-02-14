## MODIFIED Requirements

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, the trainer SHALL support generating rollouts for multiple samples in a single backend decode call (HF `generate()` or vLLM `/infer/`), controlled by a normalized rollout configuration knob:

- `decode_batch_size` (int) in the trainer’s injected rollout config contract.

Config source semantics (normative):
- For Stage-2 AB canonical profile authoring, rollout backend selection MUST come from `rollout_matching.rollout_backend`.
- For Stage-2 AB canonical profile authoring, `decode_batch_size` MUST come from `rollout_matching.decode_batch_size`.
- For Stage-2 AB rollout knobs that currently exist under `custom.extra.rollout_matching.*`, canonical migration MUST be path-only relocation to `rollout_matching.*` with the same subkey names.
- Legacy Stage-2 alias keys under `custom.extra.rollout_matching.*` are unsupported and MUST fail fast with migration guidance.
- The trainer/runtime contract MUST expose a single resolved `decode_batch_size` value to rollout execution code.

Semantics (normative):
- `decode_batch_size` denotes the maximum number of sequences decoded per rollout GPU in one generation call.
- The trainer MUST enforce this bound for both HF and vLLM backends.

Defaulting (normative):
- If `decode_batch_size` is unset, the implementation MUST default it to `1` (conservative).
- Higher-level experiment templates MAY set a larger default explicitly (e.g., Stage2-AB YAML under `configs/stage2_ab/**` uses `4`).

#### Scenario: Canonical Stage-2 key controls decode microbatching
- **WHEN** a Stage-2 AB config sets `rollout_matching.decode_batch_size: M` where `M > 1`
- **THEN** rollout generation uses `M` as the resolved decode batch size in trainer rollout config.

#### Scenario: Microbatching increases decode parallelism without changing outputs format
- **WHEN** rollout-matching training runs with resolved `decode_batch_size=M` where `M > 1`
- **THEN** the trainer performs batched decode calls for up to `M` samples per rollout GPU
- **AND** it returns per-sample `response_token_ids` suitable for strict token-aligned parsing.

#### Scenario: Legacy decode key path fails fast
- **WHEN** a Stage-2 config sets `custom.extra.rollout_matching.decode_batch_size`
- **THEN** config loading fails fast with guidance to migrate to `rollout_matching.decode_batch_size`.

#### Scenario: Legacy rollout backend key path fails fast
- **WHEN** a Stage-2 config sets `custom.extra.rollout_matching.rollout_backend`
- **THEN** config loading fails fast with guidance to migrate to `rollout_matching.rollout_backend`.

## ADDED Requirements

### Requirement: Trainer rollout contract is source-agnostic after normalization
Rollout-matching trainer internals MUST consume a source-agnostic rollout config contract after normalization from canonical grouped keys.

Normative behavior:
- Runtime normalization MUST produce one `rollout_matching_cfg` mapping injected into rollout-aware trainers.
- Rollout execution code MUST read resolved rollout values from the injected contract and MUST NOT branch on original YAML source path.
- Normalized contract fields MUST include at least:
  - rollout backend selection,
  - vLLM mode/server/sync settings,
  - decoding parameters,
  - repeat-terminate settings,
  - matching/packing runtime knobs consumed by trainer validation and execution.

#### Scenario: Canonical grouped config drives trainer contract
- **WHEN** a config defines rollout settings under canonical grouped keys
- **THEN** trainer-side `rollout_matching_cfg` contains the normalized rollout contract used by rollout execution.
