## MODIFIED Requirements

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, rollout config keys and nested structures MUST be validated through schema-derived strict contracts before runtime rollout execution.

Normative behavior additions:
- Rollout config key acceptance MUST be derived from typed schema definitions and enforced at config-load time.
- Unknown rollout keys (top-level or nested) MUST fail fast with dotted-path error messages.
- Runtime rollout validators MAY enforce execution-dependent constraints (runtime mode compatibility, numeric bounds) but MUST NOT be the long-term owner of static schema key acceptance.
- `rollout_matching.decode_batch_size` remains the single source of truth for rollout decode/evaluation microbatching.
- Unknown-key dotted paths MUST include list indices when present (e.g., `rollout_matching.vllm.server.servers[0].unknown_flag`).
- Rollout server schema supports only `rollout_matching.vllm.server.servers[]`; legacy paired-list form (`vllm.server.base_url` + `vllm.server.group_port`) is removed and MUST fail fast with migration guidance.

#### Scenario: Unknown rollout key fails before rollout trainer execution
- **WHEN** config includes `rollout_matching.unknown_rollout_key`
- **THEN** loader fails fast during schema parsing
- **AND** rollout trainer is not constructed.

#### Scenario: Unknown nested decoding key fails before execution
- **WHEN** config includes `rollout_matching.decoding.unknown_decoding_key`
- **THEN** loader fails fast during schema parsing with nested dotted-path context.

#### Scenario: Legacy rollout server paired-list shape fails fast
- **WHEN** config uses `rollout_matching.vllm.server.base_url` and `rollout_matching.vllm.server.group_port`
- **THEN** schema loading fails fast with guidance to migrate to `rollout_matching.vllm.server.servers[]`.

## ADDED Requirements

### Requirement: Rollout schema validation ownership is centralized
The rollout schema contract MUST be owned centrally by config schema parsing, and reused consistently by runtime/preflight consumers.

Normative behavior:
- One schema-driven rollout contract defines accepted rollout keys and nested structures.
- Preflight/runtime consumers read normalized rollout config from the shared loader path and MUST NOT define conflicting parallel schema ownership.
- Contract changes for rollout keys MUST be implemented by updating typed schema definitions, not by adding independent manual allowlists in each consumer.
- Runtime duplicate static-key checks may remain temporarily during migration as safety gates, but MUST be removed once loader-level parity checks pass.
- `src/sft.py` continues to inject normalized rollout configuration into trainer `rollout_matching_cfg`; schema refactor MUST keep this runtime interface stable.

#### Scenario: Preflight/runtime consume a shared validated rollout contract
- **WHEN** canonical rollout config is valid and loaded
- **THEN** both launcher preflight and trainer runtime observe the same validated rollout contract.

#### Scenario: Runtime rollout injection path stays stable
- **WHEN** rollout config passes schema validation
- **THEN** runtime still injects `rollout_matching_cfg` from the normalized loader contract
- **AND** rollout trainers do not depend on alternative legacy config source paths.
