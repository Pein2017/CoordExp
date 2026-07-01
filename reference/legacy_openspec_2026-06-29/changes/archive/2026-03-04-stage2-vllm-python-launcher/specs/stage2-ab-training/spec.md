# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 AB canonical rollout namespace is normalized before trainer injection
Stage-2 AB canonical profile authoring MUST use a grouped rollout namespace outside `custom.extra`, and the loader/runtime MUST normalize this namespace into the trainer-consumed rollout config through a schema-derived strict parsing path.

Normative behavior:
- Stage-2 AB profiles MUST author rollout settings under canonical grouped section `rollout_matching.*`.
- For rollout knobs that previously lived under `custom.extra.rollout_matching.*`, canonical migration is path-only relocation to `rollout_matching.*` with unchanged subkey names.
- Config shape validation for Stage-2 profiles MUST be driven by typed schema contracts (dataclass field definitions), not duplicated manual nested allowlists in multiple modules.
- Unknown keys MUST fail fast with full dotted-path errors during config load before trainer construction.
- Runtime trainer validators MAY enforce runtime-dependent invariants, but MUST NOT be the primary owner of static config shape/key acceptance.
- Removed Stage-2 Channel-B knobs MUST fail fast at config-load time:
  - `stage2_ab.channel_b.reordered_gt_sft`
  - `stage2_ab.channel_b.desc_ce_weight_matched`
  - `stage2_ab.channel_b.semantic_desc_gate`
- Any legacy Stage-2 rollout key placement under `custom.extra.rollout_matching.*` remains unsupported and MUST fail fast with actionable migration guidance to `rollout_matching.*`.
- `src/sft.py` rollout normalization/injection path remains authoritative for trainer wiring:
  - normalized top-level `rollout_matching` is injected as `rollout_matching_cfg`,
  - parser refactor MUST NOT alter this injection contract semantics.
- Before trainer construction, runtime MUST normalize canonical grouped rollout fields into the rollout config object injected into Stage-2 AB / rollout-matching trainers.
- For rollout-aware trainer variants, rollout decode/evaluation microbatching MUST be driven by `rollout_matching.decode_batch_size` as the single source of truth.
- `training.per_device_eval_batch_size` and similar per-device eval knobs MUST NOT independently control rollout decode/evaluation batching behavior.
- For `custom.trainer_variant=stage2_two_channel`, top-level `rollout_matching` remains required and missing it MUST fail fast.

Launcher preflight / orchestration (updated for Python launcher):
- Stage-2 server-mode launcher orchestration MUST resolve rollout settings from the same shared normalization contract used by runtime, and MUST NOT maintain a divergent raw-field contract.
- `scripts/train_stage2.sh` MUST remain a supported operator-facing entrypoint for selecting:
  - the YAML config path, and
  - the runtime GPU split (server vs learner),
  but it MUST behave as a thin wrapper that delegates orchestration to a repo-owned Python launcher module.
- The bash wrapper MUST NOT parse `rollout_matching.*` keys directly in bash.
- The Python launcher MUST call the shared Python loader/normalizer via `resolve_stage2_launcher_preflight(config_path)` (or an equivalent side-effect-free path based on `ConfigLoader.load_materialized_training_config(...)`) and consume machine-readable normalized fields rather than parsing rollout keys directly in bash.
- The Python launcher MUST construct a machine-readable preflight payload (JSON-serializable mapping) with keys:
  - `rollout_backend` (string),
  - `vllm_mode` (string or null),
  - `server_base_urls` (array of strings).
- When `rollout_matching.rollout_backend: vllm` and `rollout_matching.vllm.mode: server`, `server_base_urls` MUST be a launch-time projection derived from normalized `rollout_matching.vllm.server.servers[].base_url`.
  - The rollout server endpoint (base_url + port) MUST be YAML-only; the launcher MUST NOT accept runtime overrides for base_url/host/port.
- Otherwise, `server_base_urls` MUST be `[]`.
- Single-server constraint (this stack): when `rollout_matching.rollout_backend: vllm` and `rollout_matching.vllm.mode: server`:
  - `rollout_matching.vllm.server.servers[]` MUST have length `1`, and
  - `server_base_urls` MUST have length `1`.
  - If length != 1, launcher preflight MUST fail fast before spawning any GPU subprocesses, with actionable guidance.
  - Single-server constraint refers to rollout-server endpoints only (not GPU count).
    - A single rollout server process may still use multiple GPUs via vLLM TP/DP.
- The legacy bash-eval preflight contract is removed:
  - `ROLLOUT_CONTRACT_JSON` is removed,
  - the bash wrapper MUST NOT use `eval $(python ...)` (or equivalent),
  - Python MUST NOT emit shell assignment lines for bash evaluation.
- The preflight projection above MUST NOT replace canonical schema validation for `rollout_matching.vllm.server.servers[]` (including `group_port`) defined in rollout-matching contracts.
- The 3-key preflight payload above is the minimum normative contract; additional preflight payload keys are allowed but MUST NOT weaken or replace the minimum contract.
- Launcher preflight MUST fail fast (non-zero exit) and MUST NOT launch training when config normalization fails, required preflight fields are missing, or any required field is typed incorrectly.
- The normalization output MUST preserve existing rollout semantics (backend, server, decoding, repeat-terminate, matching, and packing-related runtime knobs).
- Cutover ordering is atomic for canonical profiles: leaf YAML migration, runtime normalization/injection, and launcher preflight consumption of normalized fields MUST land together before strict legacy-key fail-fast gates are enabled.

Normalization mapping sketch (minimum required):
- `rollout_matching.rollout_backend` -> `rollout_matching_cfg.rollout_backend`
- `rollout_matching.decode_batch_size` -> `rollout_matching_cfg.decode_batch_size`
- `rollout_matching.vllm.mode` -> `rollout_matching_cfg.vllm.mode`
- `rollout_matching.vllm.server.servers[].base_url` -> `rollout_matching_cfg.vllm.server.servers[].base_url`
- Additional rollout fields keep existing key names while relocating from `custom.extra.rollout_matching.*` to top-level `rollout_matching.*` (no compatibility aliasing).

#### Scenario: Canonical grouped rollout config is visible to trainer
- **WHEN** a Stage-2 AB profile defines rollout settings under `rollout_matching.*`
- **THEN** trainer initialization receives equivalent rollout settings through injected `rollout_matching_cfg`.

#### Scenario: Any legacy rollout key path fails fast
- **WHEN** a Stage-2 AB profile sets `custom.extra.rollout_matching.decode_batch_size` (with or without canonical keys)
- **THEN** config loading fails fast with guidance to migrate to `rollout_matching.decode_batch_size`.

#### Scenario: Launcher uses normalized rollout contract
- **WHEN** a Stage-2 AB profile defines rollout settings only under `rollout_matching.*`
- **THEN** Stage-2 launcher preflight resolves server/backend settings successfully through shared normalization
- **AND** it does not require `custom.extra.rollout_matching.*` keys.

#### Scenario: Launcher preflight fails on invalid normalization
- **WHEN** shared config normalization fails or required normalized fields cannot be materialized
- **THEN** Stage-2 launcher exits non-zero and blocks training launch with actionable diagnostics.

#### Scenario: Bash wrapper does not eval python output
- **WHEN** `scripts/train_stage2.sh` launches Stage-2 server-mode via the Python launcher
- **THEN** it does not use `eval $(python ...)` (or equivalent) at any point
- **AND** all orchestration is performed inside the Python launcher.

#### Scenario: Multi-server configuration fails fast
- **GIVEN** `rollout_matching.vllm.mode: server`
- **AND** `rollout_matching.vllm.server.servers` contains 2 entries
- **WHEN** Stage-2 server-mode launch is attempted
- **THEN** launch fails fast before spawning `swift rollout` or `torchrun`
- **AND** the error indicates only single-server is supported.

#### Scenario: Rollout batching ignores eval per-device mismatch
- **WHEN** a Stage-2 AB rollout profile sets `rollout_matching.decode_batch_size=4` and `training.per_device_eval_batch_size=1`
- **THEN** rollout decode/evaluation behavior uses microbatch size `4`
- **AND** `training.per_device_eval_batch_size` does not alter rollout decode/evaluation batching.

#### Scenario: Unknown nested rollout key fails at schema load
- **WHEN** a Stage-2 AB config defines `rollout_matching.vllm.server.servers[0].unknown_flag`
- **THEN** config loading fails fast before trainer init
- **AND** the error reports a full nested path including list index (e.g., `rollout_matching.vllm.server.servers[0].unknown_flag`).

#### Scenario: Runtime validator does not own static key allowlists
- **WHEN** schema parsing succeeds for canonical rollout keys
- **THEN** trainer initialization does not require duplicate static unknown-key allowlist ownership for the same config shape.

#### Scenario: sft injection contract remains stable
- **WHEN** canonical rollout config is valid and loaded
- **THEN** `src/sft.py` injects normalized rollout config into trainer `rollout_matching_cfg`
- **AND** parser architecture changes do not require alternate rollout source paths.

### Requirement: Stage-2 config sections are parsed by schema-derived strict contracts
Stage-2 training config loading MUST enforce unknown-key fail-fast across all top-level sections using schema-derived strict parsing.

Normative section coverage:
- `model`
- `quantization`
- `template`
- `data`
- `tuner`
- `training`
- `rlhf`
- `custom`
- `debug`
- `stage2_ab`
- `rollout_matching`
- `deepspeed`
- `global_max_length`
- `extra` (reserved/rejected at top-level)

Normative behavior:
- The loader MUST derive accepted keys from typed section contracts.
- Any unsupported key in any covered section MUST fail fast with full dotted-path key reporting.
- Semantic/value constraints (range checks, required combinations) MUST remain enforced after shape validation.
- Explicit schema extension buckets remain allowed only where declared by contract (currently `custom.extra`).
- Extension-bucket allowance MUST NOT relax strictness for canonical grouped sections.
- Top-level `extra:` is not an author-facing extension bucket; presence of top-level `extra:` MUST fail fast.
- Section-coverage entry `extra` means loader-level explicit detection/rejection ownership; it does not permit arbitrary top-level `extra` payload parsing.
- Dotted-path unknown-key reporting format MUST include list indices where applicable (`field[index].subfield`).
- Regression verification MUST include fixture coverage for:
  - strict `custom.extra` policy behavior,
  - Stage-2 smoke/profile configs using canonical top-level `rollout_matching.*`,
  - `stage2_two_channel` requirement for top-level `rollout_matching`.

#### Scenario: Unknown top-level section key fails fast
- **WHEN** a config includes an unsupported top-level key under canonical Stage-2 profile loading
- **THEN** loader fails fast and reports the unknown key.

#### Scenario: Unknown key inside covered section fails fast
- **WHEN** a config includes `custom.unknown_knob`
- **THEN** loader fails fast and reports `custom` section unknown-key details.

#### Scenario: Removed Channel-B semantic-desc gate key fails fast
- **WHEN** a config includes `stage2_ab.channel_b.semantic_desc_gate`
- **THEN** loader fails fast before trainer init
- **AND** error includes dotted path `stage2_ab.channel_b.semantic_desc_gate`
- **AND** error message indicates removed-or-unknown key with actionable removal guidance.

#### Scenario: Removed Channel-B reordered-gt key fails fast
- **WHEN** a config includes `stage2_ab.channel_b.reordered_gt_sft`
- **THEN** loader fails fast before trainer init
- **AND** error includes dotted path `stage2_ab.channel_b.reordered_gt_sft`
- **AND** error message indicates removed-or-unknown key with actionable removal guidance.

#### Scenario: Removed Channel-B desc-ce-weight key fails fast
- **WHEN** a config includes `stage2_ab.channel_b.desc_ce_weight_matched`
- **THEN** loader fails fast before trainer init
- **AND** error includes dotted path `stage2_ab.channel_b.desc_ce_weight_matched`
- **AND** error message indicates removed-or-unknown key with actionable removal guidance.

#### Scenario: Extension bucket accepts minor residual keys
- **WHEN** a config includes `custom.extra.some_minor_toggle`
- **THEN** config loading can succeed if all canonical grouped sections remain valid.

#### Scenario: Top-level extra presence fails fast
- **WHEN** a config includes top-level `extra:` (including empty `{}`)
- **THEN** config loading fails fast with guidance that only `custom.extra` is the escape-hatch bucket.

#### Scenario: Missing top-level rollout_matching still fails for stage2_two_channel
- **WHEN** `custom.trainer_variant=stage2_two_channel` and top-level `rollout_matching` is omitted
- **THEN** config loading fails fast with actionable requirement text for top-level `rollout_matching`.

### Requirement: Channel selection is deterministic and step-driven
The trainer SHALL choose between Channel-A and Channel-B **deterministically** as a function of (`global_step`, `stage2_ab.schedule.b_ratio`).

Definition (normative):
- `global_step` MUST refer to the **optimizer-step** counter (post gradient-accumulation), i.e. the value that increments exactly once per optimizer update.
- The selected channel for a given `global_step` MUST remain fixed for the entire accumulation window (all micro-batches that contribute to that optimizer update).
- On resume from checkpoint, the schedule MUST continue from the restored `global_step` (no re-randomization).

Schedule definition (normative minimum):
- `stage2_ab.schedule.b_ratio` MUST be a float in `[0.0, 1.0]`.
- `stage2_ab.schedule.b_ratio` MUST be explicitly provided (no implicit default).
- Let optimizer step be `s` (0-indexed).
- The trainer MUST select Channel-B at step `s` iff:
  - `floor((s+1) * b_ratio) > floor(s * b_ratio)`.
  - Otherwise it MUST select Channel-A.

Special cases:
- If `b_ratio == 0.0`, the trainer MUST always select Channel-A.
- If `b_ratio == 1.0`, the trainer MUST always select Channel-B.

Legacy schedule handling (normative):
- The legacy list-based schedule knob `stage2_ab.schedule.pattern` is not supported.
- If a config provides `stage2_ab.schedule.pattern`, configuration parsing MUST fail fast with guidance to migrate to `stage2_ab.schedule.b_ratio`.

Legacy rollout-buffer behavior:
- `rollout_matching.rollout_buffer` is removed.
- Any config that provides `rollout_matching.rollout_buffer` MUST fail fast with actionable guidance.

Legacy rollout namespace placement:
- Legacy placement under `custom.extra.rollout_matching.*` is removed and MUST fail fast with actionable guidance to migrate to top-level `rollout_matching.*`.

#### Scenario: b_ratio=0.5 alternates deterministically by optimizer step
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.5`
- **WHEN** the trainer selects channels for `global_step` `s = 0, 1, 2, 3`
- **THEN** it selects Channel-A at steps `0` and `2`
- **AND** it selects Channel-B at steps `1` and `3`.

#### Scenario: b_ratio edge cases are deterministic
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-A.

- **GIVEN** `stage2_ab.schedule.b_ratio: 1.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-B.

#### Scenario: Channel selection continues across checkpoint resume
- **GIVEN** a run that has completed optimizer step `global_step = s`
- **WHEN** training resumes from a checkpoint that restores `global_step = s`
- **THEN** the channel selected for step `s` is identical to the pre-resume selection for step `s`.

#### Scenario: stage2_ab.schedule.pattern fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_two_channel`
- **AND** `stage2_ab.schedule.pattern: ["A","B"]` is provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to use `stage2_ab.schedule.b_ratio`.

#### Scenario: Missing b_ratio fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_two_channel`
- **AND** `stage2_ab.schedule.b_ratio` is not provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to set `stage2_ab.schedule.b_ratio`.

#### Scenario: Legacy rollout_buffer config fails fast
- **GIVEN** a Stage-2 AB config that provides `rollout_matching.rollout_buffer`
- **WHEN** configuration is parsed/materialized
- **THEN** it fails fast with guidance to remove `rollout_buffer`.

#### Scenario: Multi-process learner uses rank0 broadcast for step kind
- **GIVEN** Stage-2 AB training is enabled under `torchrun` with `world_size=2`
- **WHEN** one optimizer step executes
- **THEN** rank0 broadcasts the step kind (`A` or `B`) for that optimizer step
- **AND** all ranks execute the same step kind for all micro-steps in the accumulation window.

### Requirement: Channel-B vLLM rollouts honor repeat-aware termination settings
When Stage-2 AB Channel-B performs rollouts through vLLM rollout server backend, repeat-aware termination MUST be applied according to rollout-matching config.

Normative behavior:
- Channel-B rollout path MUST propagate the full `rollout_matching.repeat_terminate` subtree (`enabled`, `min_new_tokens`, `max_consecutive_token_repeats`, `ngram_size`, `ngram_repeats`, optional `max_object_keys`) into the active vLLM rollout serving startup path.
- Because the rollout server is launched as a separate process (external dependency stack), the full subtree MUST be transmitted into that server startup process.
  - Recommended compliant approach: the server launcher:
    - sets `COORDEXP_VLLM_REPEAT_TERMINATE_CONFIG_JSON=<json>` and enables injection with `COORDEXP_ENABLE_VLLM_REPEAT_TERMINATE_INJECTION=1`, and
    - launches `swift rollout` with `--external_plugins <repo-owned-plugin>` so the server can attach repeat-aware processing at startup without external library source edits.
- For this stack, Channel-B MUST NOT assume request-time logits-processor fields in rollout request payloads; repeat-aware activation is validated at rollout-server startup.
- If `rollout_matching.repeat_terminate.enabled: true` and startup activation is unavailable, Stage-2 AB MUST fail before entering training steps.
- Channel-B MUST preserve FP/matching contracts and MUST NOT change geometry supervision semantics due to repeat-aware processing.
- Channel-B logs/metrics MUST emit concrete audit keys (as entries in the neutral trainer-metrics payload `metrics` map; see `src/metrics/payload_contract.py`):
  - `rollout/repeat_terminate_active` (0 or 1),
  - `rollout/repeat_terminate_triggered_sequences` (counter).
  - Metric meaning (normative):
    - `rollout/repeat_terminate_active`: 1 iff repeat-aware processing is active for the step under the current rollout backend/mode when `rollout_matching.repeat_terminate.enabled: true`; otherwise 0.
    - `rollout/repeat_terminate_triggered_sequences`: number of rollout sequences in the step for which repeat-aware processing **triggered at least once** and forced EOS due to configured repeat thresholds.
      - This key MUST be derived from an explicit trigger signal produced by the repeat-aware processor/server stack (not inferred from finish-reason heuristics alone).
- The vLLM server `/infer/` response MUST expose the explicit per-sequence trigger signal in an additive-only wrapper envelope.
  - A compliant per-output schema is:
    - `{"response": <ChatCompletionResponse-dict>, "coordexp": {"repeat_terminate_triggered": 0|1}}`
  - The wrapper MUST be additive-only:
    - it MUST NOT remove or rename fields within the inner `response` payload compared to the unwrapped server output,
    - and it MUST preserve the detail fields required for strict alignment and token-aligned parsing (at minimum `prompt_token_ids` and `choices[0].token_ids` when `request_config.return_details: true`).
  - The learner MUST compute `rollout/repeat_terminate_triggered_sequences` from this wrapper signal (sum of `repeat_terminate_triggered` across sequences in the step), not from stop-reason heuristics.
- All metrics emitted via the neutral trainer-metrics payload `metrics` map (see `src/metrics/payload_contract.py`) MUST be emitted as **global** aggregates after:
  - micro-batch/gradient-accumulation aggregation to one optimizer-step payload, and
  - distributed aggregation across ranks (e.g., DDP all-reduce) when `world_size > 1`.
  - Global aggregation semantics are defined per metric family (normative):
    - counters: global sum,
    - wall-time seconds (e.g., `time/*_s`): global max,
    - boolean-style activation flags: global max,
    - rates: ratio of globally-summed numerator/denominator (never mean of rank-local ratios).
  - For this requirement's audit keys:
    - `rollout/repeat_terminate_active` MUST remain in `{0,1}` after global aggregation (a compliant approach is a global max over rank-local 0/1 values).
    - `rollout/repeat_terminate_triggered_sequences` MUST be a non-negative global counter for the step (a compliant approach is a global sum over rank-local counts).
- For auditability, Channel-B rollout steps MUST emit tail-control metrics (as entries in the neutral trainer-metrics payload `metrics` map; see `src/metrics/payload_contract.py`):
  - `rollout/gen_new_tokens_p99`,
  - `rollout/parse_truncated_rate`,
  - `rollout/parse_dropped_invalid`.
  - Metric definitions/aggregation (normative, distributed):
    - `rollout/parse_truncated_rate` MUST be computed as `(sum(num_truncated_samples) / sum(num_rollout_samples))` over ranks for the step (0 when `sum(num_rollout_samples) == 0`).
    - `rollout/gen_new_tokens_p99` MUST be computed as a global conservative proxy using only all-reduce:
      - compute rank-local p99 over that rank's rollout samples for the step,
      - then compute the global metric as `max(rank_local_p99)` via all-reduce max.
      - Rationale: this preserves a simple, reproducible global metric without requiring all-gather of variable-length lists.

#### Scenario: Channel-B run in vLLM mode uses repeat-aware termination
- **GIVEN** Stage-2 AB with Channel-B and vLLM rollout backend
- **AND** `rollout_matching.repeat_terminate.enabled: true`
- **WHEN** a rollout sequence enters degenerate repetition
- **THEN** Channel-B rollout output is terminated early for that sequence by repeat-aware logic
- **AND** downstream parse/match training continues for the batch.

#### Scenario: Channel-B startup fails when repeat-aware contract is enabled but inactive
- **GIVEN** Stage-2 AB Channel-B with vLLM backend
- **AND** `rollout_matching.repeat_terminate.enabled: true`
- **WHEN** rollout server startup cannot activate repeat-aware processing
- **THEN** trainer startup fails with an error that reports the missing processor activation path
- **AND** no training step is executed.

#### Scenario: Tail-control audit metrics are emitted
- **GIVEN** Stage-2 AB with Channel-B and vLLM rollout backend
- **WHEN** a Channel-B rollout step executes
- **THEN** logs include `rollout/gen_new_tokens_p99`, `rollout/parse_truncated_rate`, and `rollout/parse_dropped_invalid`.
