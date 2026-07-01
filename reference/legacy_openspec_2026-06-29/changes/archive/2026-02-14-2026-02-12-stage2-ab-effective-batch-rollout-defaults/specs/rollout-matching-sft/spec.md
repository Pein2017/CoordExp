## ADDED Requirements

### Requirement: Legacy rollout batch-size knobs fail fast
The system MUST fail fast if a config provides any legacy rollout batch-size knob under `custom.extra.rollout_matching`, with actionable guidance to migrate to the unified decode batching knob `custom.extra.rollout_matching.decode_batch_size`.

Legacy knobs (normative):
- `custom.extra.rollout_matching.rollout_generate_batch_size`
- `custom.extra.rollout_matching.rollout_infer_batch_size`

#### Scenario: rollout_generate_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.rollout_generate_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use `custom.extra.rollout_matching.decode_batch_size`.

#### Scenario: rollout_infer_batch_size fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.rollout_infer_batch_size`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove it and use `custom.extra.rollout_matching.decode_batch_size`.

### Requirement: Window-aware post-rollout packing scope knob is removed
The system MUST fail fast if a config provides `custom.extra.rollout_matching.post_rollout_pack_scope` (any value), with actionable guidance to remove it.

Normative behavior:
- Post-rollout packing behavior MUST be the standardized micro-scope dynamic packing semantics.
- Window-scoped packing is not supported.

#### Scenario: post_rollout_pack_scope fails fast when present
- **GIVEN** rollout-matching training is enabled
- **AND** the config provides `custom.extra.rollout_matching.post_rollout_pack_scope`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to remove `post_rollout_pack_scope`.

## MODIFIED Requirements

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, the trainer SHALL support generating rollouts for multiple samples in a single backend decode call (HF `generate()` or vLLM `/infer/`), controlled by a YAML knob:
- `custom.extra.rollout_matching.decode_batch_size` (int)

Semantics (normative):
- `decode_batch_size` denotes the maximum number of sequences decoded per rollout GPU in one generation call.
- The trainer MUST enforce this bound for both HF and vLLM backends.

Defaulting (normative):
- If `custom.extra.rollout_matching.decode_batch_size` is unset, the implementation MUST default it to `1` (conservative).
- Higher-level experiment templates (e.g., Stage2-AB YAML under `configs/stage2_ab/**`) MAY set a larger default (such as `4`) explicitly in config.

#### Scenario: Microbatching increases decode parallelism without changing outputs format
- **GIVEN** rollout-matching training is enabled
- **AND** `custom.extra.rollout_matching.decode_batch_size: M` where `M > 1`
- **WHEN** the trainer generates rollouts for a batch of `M` samples on one rank
- **THEN** the trainer performs one batched decode call for those `M` samples
- **AND** it returns per-sample `response_token_ids` suitable for strict token-aligned parsing.

### Requirement: vLLM server mode derives per-rank request chunking from server world size
When rollout-matching training uses vLLM server mode (`custom.extra.rollout_matching.rollout_backend=vllm` and `custom.extra.rollout_matching.vllm.mode=server`), the trainer MUST derive per-rank request chunk sizing from the rollout server world size and learner DDP world size to preserve the per-rollout-GPU cap defined by `decode_batch_size`.

Semantics (normative):
- The trainer MUST query each configured server’s `${base_url}/get_world_size/` endpoint and cache one `world_size` per server entry.
- Let `server_world_sizes = [s_0, s_1, ...]` be those cached values, and define:
  - `S = sum(server_world_sizes)` (total rollout inference device count across servers; DP replicas)
  - `W = learner_world_size` (training DDP world size)
- **Feasibility**: If `decode_batch_size * S < W`, the trainer MUST fail fast with actionable guidance (the cap cannot be preserved if every learner rank must issue at least one request concurrently).
- Otherwise, the trainer MUST derive a per-learner-rank chunk size:
  - `chunk = floor(decode_batch_size * S / W)`
- The trainer MUST distribute requests across servers in a capacity-aware deterministic way (proportional to `server_world_sizes`) so that rollout GPUs are not overloaded when servers are heterogeneous.

#### Scenario: vLLM server mode fails fast when cap is infeasible
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_backend=vllm` and `vllm.mode=server`
- **AND** `decode_batch_size=1`, `S=2`, and `W=4`
- **WHEN** training starts
- **THEN** config validation fails fast with guidance to increase rollout server world size, reduce learner world size, or increase `decode_batch_size`.

## REMOVED Requirements

### Requirement: Stage-2 Window-Aware Post-Rollout Packing (Training Only)
**Reason**: Window-aware packing couples training semantics to the gradient-accumulation window and introduces additional scheduling/feasibility knobs that are unnecessary under the standardized step-budgeted Channel-B pathway.

**Migration**: Remove `custom.extra.rollout_matching.post_rollout_pack_scope` from configs. The system uses standardized micro-scope dynamic post-rollout packing.
