# stage2-ab-training Specification

## Purpose
Define the Stage-2 AB training contract (Channel-A supervised, Channel-B rollout-matching) including the configuration surface and reproducibility-critical invariants.
## Requirements
### Requirement: Stage-2 AB trainer variant is selectable via YAML
When training config sets `custom.trainer_variant: stage2_two_channel`, the system SHALL use the Stage-2 AB trainer implementation.

The system MUST reject `custom.trainer_variant: stage2_ab_training` (fail fast) with actionable guidance to use
`stage2_two_channel`.

The trainer MUST be configurable via YAML and MUST NOT require new CLI flags.

Canonical config location (typed):
- Stage-2 AB knobs MUST be expressed under the top-level `stage2_ab` mapping (parallel to `training` and `custom`).
- Unknown keys under `stage2_ab` MUST fail fast with actionable guidance (to avoid silent drift from typos).

#### Scenario: Selecting the trainer variant with typed `stage2_ab`
- **GIVEN** a training config with `custom.trainer_variant: stage2_two_channel`
- **AND** a top-level `stage2_ab` mapping is provided
- **WHEN** training starts
- **THEN** the Stage-2 AB trainer is constructed and used for training
- **AND** the trainer reads Stage-2 AB knobs from `stage2_ab`.

#### Scenario: Unknown stage2_ab keys fail fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_two_channel`
- **AND** a top-level `stage2_ab` mapping contains an unknown key (e.g., a typo)
- **WHEN** training starts
- **THEN** configuration parsing fails fast with guidance to fix/remove the unknown key.

#### Scenario: Selecting the trainer variant
- **GIVEN** a training config with `custom.trainer_variant: stage2_two_channel`
- **WHEN** training starts
- **THEN** the Stage-2 AB trainer is constructed and used for training.

### Requirement: Stage-2 AB objective weights are pipeline-only (no flat objective knobs)
When `custom.trainer_variant: stage2_two_channel`, the Stage-2 AB objective MUST be fully determined by the declared module pipeline under:
- `stage2_ab.pipeline.objective[]` and `stage2_ab.pipeline.diagnostics[]`.

Normative behavior:
- `stage2_ab.pipeline` MUST be present (no implicit default objective manifest).
- Flat objective knobs are removed and MUST fail fast when present (non-exhaustive):
  - `stage2_ab.desc_ce_weight`, `stage2_ab.fmt_struct_ce_weight`
  - `stage2_ab.bbox_smoothl1_weight`, `stage2_ab.bbox_ciou_weight`
  - `stage2_ab.coord_ce_weight`, `stage2_ab.coord_gate_weight`, `stage2_ab.text_gate_weight`
- Legacy aux-loss config surfaces MUST be rejected for Stage-2 AB, including `custom.coord_soft_ce_w1.*`.

#### Scenario: Missing pipeline fails fast
- **WHEN** a Stage-2 AB config sets `custom.trainer_variant: stage2_two_channel`
- **AND** `stage2_ab.pipeline` is absent
- **THEN** config loading fails fast before trainer init
- **AND** the error indicates `stage2_ab.pipeline` is required.

### Requirement: Stage-2 AB pipeline specs are explicit and complete (no implicit defaults)
Stage-2 AB pipeline module specs MUST be authored with explicit fields and complete module configs to prevent silent drift from default injection.

Normative behavior:
- Each entry in `stage2_ab.pipeline.objective[]` and `stage2_ab.pipeline.diagnostics[]` MUST include:
  - `name`, `enabled`, `weight`, `channels`, `application`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `application` MUST be an explicitly authored mapping with exactly one key:
  - `preset`
- `application.preset` MUST be valid for the referenced module:
  - `token_ce`: `anchor_text_only`, `rollout_text_only`
  - `loss_duplicate_burst_unlikelihood`: `rollout_only`
  - `bbox_geo`, `bbox_size_aux`, `coord_reg`:
    - `anchor_only`
- Presets that imply a deprecated final Channel-A self-context pass MUST be
  rejected with actionable migration guidance:
  - `token_ce.application.preset: anchor_text_plus_final_struct`
  - `bbox_geo.application.preset: anchor_if_single_iter_else_final`
  - `bbox_geo.application.preset: final_only`
  - `bbox_geo.application.preset: anchor_and_final`
  - and the equivalent `bbox_size_aux` / `coord_reg` preset values.
- `config` MUST include exactly the allowlisted keys for the referenced module:
  - missing required keys MUST fail fast (no implicit defaults),
  - unknown keys MUST fail fast (no escape-hatch aliases).

#### Scenario: Missing module spec field fails fast
- **WHEN** a Stage-2 AB config omits `stage2_ab.pipeline.objective[i].channels`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the missing required field and its full path.

#### Scenario: Missing application preset fails fast
- **WHEN** a Stage-2 AB config omits `stage2_ab.pipeline.objective[i].application.preset`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the missing required field and its full path.

#### Scenario: Deprecated final-pass preset fails fast
- **GIVEN** a Stage-2 AB config that authors
  `token_ce.application.preset: anchor_text_plus_final_struct`
- **WHEN** the config is loaded
- **THEN** validation fails fast
- **AND** the error explains the single-pass replacement
  `anchor_text_only`.

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)
Stage-2 AB pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `bbox_size_aux.config` MUST accept only:
  - `log_wh_weight`
  - `oversize_penalty_weight`
  - `oversize_area_frac_threshold`
  - `oversize_log_w_threshold`
  - `oversize_log_h_threshold`
  - `eps`
- `coord_reg.config` MUST accept only:
  - `coord_ce_weight`
  - `soft_ce_weight`
  - `w1_weight`
  - `coord_gate_weight`
  - `text_gate_weight`
  - `temperature`
  - `target_sigma`
  - `target_truncate`
- Legacy alias keys (e.g., `bbox_smoothl1_weight`, `coord_soft_ce_weight`, `coord_w1_weight`) MUST be rejected.

#### Scenario: Alias key in module config fails fast
- **WHEN** `stage2_ab.pipeline.objective[*].name=bbox_size_aux`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the canonical `bbox_size_aux.config.*` key family
  must be used instead.

### Requirement: Stage-2 AB supports text_gate via coord_reg module config
Stage-2 AB MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `stage2_ab.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution when coord-vocab mass appears at supervised text positions (subject to registry masking).

#### Scenario: Non-zero text_gate_weight is effective
- **WHEN** `coord_reg.config.text_gate_weight > 0`
- **AND** the model places substantial coord-vocab probability mass at supervised `type=struct|desc` positions
- **THEN** the emitted `text_gate` objective atom increases relative to the same run with `text_gate_weight = 0`:
  - Channel-A: `loss/coord/text_gate`
  - Channel-B: `loss/B_coord/text_gate`
- **AND** the increase is attributable to the `text_gate` sub-term inside `coord_reg`.

### Requirement: Stage-2 AB objective application is explicit and non-redundant
Stage-2 AB SHALL route Channel-A objective provenance through `application.preset`
instead of duplicating loss strengths across separate `a1_*` config families.

Normative behavior:
- `bbox_geo`, `bbox_size_aux`, and `coord_reg` MUST express Channel-A routing
  through `stage2_ab.pipeline.objective[*].application.preset`.
- The canonical non-redundant Channel-A preset is now `anchor_only`:
  - Channel-A bbox/coord atoms MUST emit only under `loss/coord/*`.
- `token_ce.application.preset=anchor_text_only` MUST keep
  `loss/text/{struct_ce,desc_ce}` on the GT anchor path and MUST NOT emit
  `loss/A1_*` or `loss/A2_*`.
- Stage-2 AB module configs MUST NOT reintroduce duplicated Channel-A routing
  weights such as `a1_smoothl1_weight`, `a1_ciou_weight`,
  `a1_log_wh_weight`, `a1_oversize_penalty_weight`,
  `a1_soft_ce_weight`, or `a1_w1_weight`.
- Stage-2 AB module configs MUST NOT use final-pass/self_context aliases or
  weight families to recreate A2 routing.

#### Scenario: Anchor-only Channel-A routes bbox/coord atoms to the normal coord group
- **GIVEN** `bbox_geo.application.preset: anchor_only`
- **AND** `bbox_size_aux.application.preset: anchor_only`
- **AND** `coord_reg.application.preset: anchor_only`
- **WHEN** Channel-A executes
- **THEN** bbox/coord objective atoms emit under `loss/coord/*`
- **AND** the same step does not emit `loss/A1_*` or `loss/A2_*`.

### Requirement: Coord diagnostics are attributed to the normal coord group and B provenance in Stage-2 two-channel
When Stage-2 two-channel emits coord-distribution diagnostics, it MUST
attribute them only to the supported forward surfaces that still exist.

Normative behavior:
- `coord_diag/*`: computed from the Channel-A GT-anchor logits.
- `coord_diag/B/*`: computed from Channel-B rollout-context logits.
- `coord_diag/A1/*` and `coord_diag/A2/*` MUST NOT be emitted.
- The trainer MUST NOT emit ambiguous bare `coord_diag/*` keys for these
  monitors beyond the normal single-pass Channel-A coord group.

#### Scenario: Channel-A diagnostics use the normal coord group
- **WHEN** Stage-2 AB runs with `coord_diag` enabled
- **THEN** emitted coord diagnostics may include `coord_diag/*` and
  `coord_diag/B/*` when the relevant channel runs
- **AND** the same run does not emit `coord_diag/A1/*` or `coord_diag/A2/*`.

### Requirement: Stage-2 AB profile hierarchy supports the current shared-base layout
Stage-2 AB experiment profiles under `configs/stage2_two_channel/` SHALL use the current repo-owned inheritance layout so shared prod/smoke behaviors remain reusable without flattening every downstream profile into a one-hop leaf.

Normative structure:
- `configs/stage2_two_channel/base.yaml` MUST remain the canonical shared base for Stage-2 AB profile runs.
- Canonical profile discovery continues to target `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml`.
- Canonical prod and smoke profiles MAY extend other Stage-2 profiles inside `configs/stage2_two_channel/` when those parents capture a supported reusable sub-profile.
- Multi-hop inheritance within the Stage-2 profile tree is allowed.
- Config inheritance cycles MUST still fail fast.

Validation behavior:
- Config loading for canonical Stage-2 profiles MUST validate the **resolved** profile rather than requiring one-hop inheritance.
- Error messages for invalid canonical profiles MUST identify the missing resolved dotted-path fields.
- Strict profile validation continues to target `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml`.

#### Scenario: Canonical prod profile may extend base directly
- **WHEN** a canonical Stage-2 prod profile under `configs/stage2_two_channel/prod/` extends `../base.yaml`
- **THEN** config loading succeeds for hierarchy validation.

#### Scenario: Canonical smoke profile may extend a prod profile
- **WHEN** a canonical Stage-2 smoke profile under `configs/stage2_two_channel/smoke/` extends an intermediate Stage-2 prod profile
- **THEN** config loading succeeds so long as the fully resolved profile satisfies the required Stage-2 contract.

#### Scenario: Canonical profile discovery is scoped to prod/smoke
- **WHEN** a config discovery utility scans canonical Stage-2 profiles
- **THEN** it includes only `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml`

### Requirement: Offline image-budget validation is distinct from runtime template max_pixels
Stage-2 training SHALL distinguish the offline prepared-data image budget from the runtime processor/server `template.max_pixels` setting.

Normative behavior:
- `custom.offline_max_pixels` MAY be provided as the canonical offline image-budget contract for launch prechecks and dataset runtime validation.
- When `custom.offline_max_pixels` is provided:
  - launcher JSONL prechecks MUST use `custom.offline_max_pixels`,
  - dataset runtime max-pixel enforcement MUST use `custom.offline_max_pixels`,
  - `template.max_pixels` remains available for runtime processor/server behavior and MUST NOT replace the offline validation contract.
- When `custom.offline_max_pixels` is absent:
  - launch prechecks and dataset runtime MAY fall back to `template.max_pixels` for backward compatibility.

#### Scenario: Stage-2 config disables runtime resize but keeps offline validation strict
- **GIVEN** a Stage-2 config sets a large `template.max_pixels` to disable HF auto-resize
- **AND** `custom.offline_max_pixels` is set to the offline prepared-data budget
- **WHEN** launcher prechecks and dataset runtime validate image sizes
- **THEN** they use `custom.offline_max_pixels`
- **AND** the large runtime `template.max_pixels` does not weaken offline data validation.

### Requirement: Stage-2 AB canonical profiles resolve high-signal knobs after inheritance
Each canonical Stage-2 AB profile under `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml` MUST resolve the following high-signal run and ablation knobs after inheritance.

Required resolved fields:
- `model.model`
- `training.run_name`
- `training.output_dir`
- `training.logging_dir`
- `training.learning_rate`
- `training.vit_lr`
- `training.aligner_lr`
- `training.effective_batch_size`
- `training.eval_strategy`
- `training.eval_steps`
- `training.save_strategy`
- `training.save_steps`
- `stage2_ab.schedule.b_ratio`

Rationale for resolved-profile explicitness:
- The LR trio (`training.learning_rate`, `training.vit_lr`, `training.aligner_lr`) is treated as MUST for canonical profiles to avoid hidden optimizer-group drift across ablations, even when these values are supplied by an intermediate parent.

Validation behavior:
- Canonical Stage-2 AB profile loading MUST fail fast if any required field is missing from the fully resolved profile.
- Error text MUST identify missing fields by full key path.

#### Scenario: Canonical profile with resolved high-signal fields is accepted
- **WHEN** a Stage-2 AB canonical profile resolves all required high-signal keys after inheritance
- **THEN** config loading succeeds and the profile is considered self-consistent.

#### Scenario: Missing resolved run identity fails fast
- **WHEN** a Stage-2 AB canonical profile does not resolve `training.run_name`
- **THEN** config loading fails fast and reports `training.run_name` as missing.

#### Scenario: Missing resolved model path fails fast
- **WHEN** a Stage-2 AB canonical profile does not resolve `model.model`
- **THEN** config loading fails fast and reports `model.model` as missing.

### Requirement: Stage-2 AB canonical rollout namespace is normalized before trainer injection
Stage-2 AB canonical profile authoring MUST use a grouped rollout namespace outside `custom.extra`, and the loader/runtime MUST normalize this namespace into the trainer-consumed rollout config through a schema-derived strict parsing path.

Normative behavior:
- Stage-2 AB profiles MUST author rollout settings under canonical grouped section `rollout_matching.*`.
- For rollout knobs that previously lived under `custom.extra.rollout_matching.*`, canonical migration MUST target strict top-level `rollout_matching.*` keys defined by the current schema (including any required key renames).
- Config shape validation for Stage-2 profiles MUST be driven by typed schema contracts (dataclass field definitions), not duplicated manual nested allowlists in multiple modules.
- Unknown keys MUST fail fast with full dotted-path errors during config load before trainer construction.
- Runtime trainer validators MAY enforce runtime-dependent invariants, but MUST NOT be the primary owner of static config shape/key acceptance.
- Removed Stage-2 Channel-B knobs MUST fail fast at config-load time:
  - `stage2_ab.channel_b.reordered_gt_sft`
  - `stage2_ab.channel_b.desc_ce_weight_matched`
  - `stage2_ab.channel_b.semantic_desc_gate`
  - `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`
- Any legacy Stage-2 rollout key placement under `custom.extra.rollout_matching.*` remains unsupported and MUST fail fast with actionable migration guidance to `rollout_matching.*`.
- `src/sft.py` rollout normalization/injection path remains authoritative for trainer wiring:
  - normalized top-level `rollout_matching` is injected as `rollout_matching_cfg`,
  - parser refactor MUST NOT alter this injection contract semantics.
- Before trainer construction, runtime MUST normalize canonical grouped rollout fields into the rollout config object injected into Stage-2 AB / rollout-matching trainers.
- For rollout-aware trainer variants, rollout decode/evaluation microbatching MUST be driven by canonical explicit keys:
  - `rollout_matching.channel_b_decode_batch_size` for training-time Channel-B rollouts,
  - `rollout_matching.eval_decode_batch_size` for evaluation-time rollouts.
- `training.per_device_eval_batch_size` and similar per-device eval knobs MUST NOT independently control rollout decode/evaluation batching behavior.
- For `custom.trainer_variant=stage2_two_channel`, top-level `rollout_matching` remains required and missing it MUST fail fast.
- Stage-2 launcher preflight (`scripts/train_stage2.sh`) MUST resolve rollout settings from the same shared normalization contract used by runtime, and MUST NOT maintain a divergent raw-field contract.
- Launcher preflight MUST call the shared Python loader/normalizer (`ConfigLoader.load_materialized_training_config(...)` path) and consume machine-readable normalized fields rather than parsing rollout keys directly in bash.
- Launcher preflight machine-readable output MUST be newline-terminated single-line JSON with keys:
  - `rollout_backend` (string),
  - `vllm_mode` (string or null),
  - `server_base_urls` (array of strings; empty allowed when backend/mode does not require server URLs).
- `server_base_urls` MUST be a launch-time projection derived from normalized `rollout_matching.vllm.server.servers[].base_url`.
- Launcher preflight JSON contract defined for Stage-2 canonical rollout normalization remains required:
  - Python resolver emits shell assignment lines; the `ROLLOUT_CONTRACT_JSON` value MUST be newline-terminated single-line JSON,
  - required keys/types: `rollout_backend` (string), `vllm_mode` (string or null), `server_base_urls` (array of strings),
  - invalid/missing JSON contract fields MUST fail fast before launch.
- The preflight projection above MUST NOT replace canonical schema validation for `rollout_matching.vllm.server.servers[]` (including `group_port`) defined in rollout-matching contracts.
- The 3-key JSON contract above is the minimum normative contract; additional preflight payload keys are allowed but MUST NOT weaken or replace the minimum contract.
- Launcher preflight MUST fail fast (non-zero exit) and MUST NOT launch training when config normalization fails, JSON is invalid, or any required key is missing/typed incorrectly.
- The normalization output MUST preserve existing rollout semantics (backend, server, decoding, matching, and packing-related runtime knobs).
- Cutover ordering is atomic for canonical profiles: leaf YAML migration, runtime normalization/injection, and launcher preflight consumption of normalized fields MUST land together before strict legacy-key fail-fast gates are enabled.

Normalization mapping sketch (minimum required):
- `rollout_matching.rollout_backend` -> `rollout_matching_cfg.rollout_backend`
- `rollout_matching.channel_b_decode_batch_size` -> `rollout_matching_cfg.channel_b_decode_batch_size`
- `rollout_matching.eval_decode_batch_size` -> `rollout_matching_cfg.eval_decode_batch_size`
- `rollout_matching.vllm.mode` -> `rollout_matching_cfg.vllm.mode`
- `rollout_matching.vllm.server.servers[].base_url` -> `rollout_matching_cfg.vllm.server.servers[].base_url`
- Additional rollout fields MUST keep canonical top-level `rollout_matching.*` key names (no `custom.extra.rollout_matching.*` compatibility aliasing).

#### Scenario: Canonical grouped rollout config is visible to trainer
- **WHEN** a Stage-2 AB profile defines rollout settings under `rollout_matching.*`
- **THEN** trainer initialization receives equivalent rollout settings through injected `rollout_matching_cfg`.

#### Scenario: Any legacy rollout key path fails fast
- **WHEN** a Stage-2 AB profile sets `custom.extra.rollout_matching.channel_b_decode_batch_size` (with or without canonical keys)
- **THEN** config loading fails fast with guidance to migrate to `rollout_matching.channel_b_decode_batch_size`.

#### Scenario: Launcher preflight uses normalized rollout contract
- **WHEN** a Stage-2 AB profile defines rollout settings only under `rollout_matching.*`
- **THEN** `scripts/train_stage2.sh` preflight resolves server/backend settings successfully through shared normalization and does not require `custom.extra.rollout_matching.*` keys.

#### Scenario: Launcher preflight fails on invalid normalization JSON contract
- **WHEN** shared normalization output is invalid JSON or omits required keys/types (`rollout_backend`, `vllm_mode`, `server_base_urls`)
- **THEN** `scripts/train_stage2.sh` exits non-zero and blocks training launch with actionable contract error text.

#### Scenario: Rollout batching ignores eval per-device mismatch
- **WHEN** a Stage-2 AB rollout profile sets `rollout_matching.channel_b_decode_batch_size=4`, `rollout_matching.eval_decode_batch_size=2`, and `training.per_device_eval_batch_size=1`
- **THEN** training-time Channel-B rollout decode uses microbatch size `4`
- **AND** eval-time rollout decode uses microbatch size `2`
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

#### Scenario: Removed Channel-B invalid-structure multiplier key fails fast
- **WHEN** a config includes `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`
- **THEN** loader fails fast before trainer init
- **AND** error includes dotted path `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`
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

#### Scenario: Preflight contract format is strict
- **WHEN** launcher preflight emits contract JSON
- **THEN** the `ROLLOUT_CONTRACT_JSON` value contains the minimum 3-key contract as newline-terminated single-line JSON
- **AND** malformed format or missing required keys fails fast before launch.

### Requirement: Stage-2 AB remains compatible with ms-swift and Transformers (no upstream patches)
Stage-2 AB training MUST be implemented in a way that is compatible with:
- the ms-swift training harness (trainer construction, dataloader/collator semantics, and checkpoint/resume flow), and
- HuggingFace Transformers model forward contracts (no monkey-patching or editing upstream model files).

Normative constraints:
- The implementation MUST rely on existing ms-swift/Transformers mechanisms for loading Stage-1 pretrained checkpoints and resuming from checkpoints; it MUST NOT introduce a custom weight-loading pipeline.
- The trainer MUST preserve access to raw sample fields required by Channel-B (e.g., `messages`, `assistant_payload`) via an ms-swift compatible data-collation path (e.g., identity collator + `remove_unused_columns=False`), so rollout→parse→match construction can run.
- The trainer MUST filter out ms-swift / Trainer-injected helper keys before every model forward in both channels, and MUST NOT pass unknown/non-forward kwargs into the model.
  - At minimum, keys like `labels`, `compute_loss_func`, `loss_scale`, `text_position_ids`, and `channel` MUST NOT reach `model(...)` in either channel.
- The trainer MUST ensure that `outputs.logits` contains the full sequence-length logits needed for CE and coord decoding; it MUST NOT enable or rely on any logits-slicing mechanisms (e.g., `logits_to_keep`) that would drop positions required by the loss. If such a knob is configured or injected by the harness, the trainer MUST disable it (or fail fast with actionable guidance).
- When `training.packing` is enabled for Stage-2 AB, dataset-level packing MUST be disabled; both channels MUST use dynamic post-rollout packing for the teacher-forced forward pass (segment-level invariants are shared with rollout-matching: atomic segments, no splitting, deterministic selection, hard `packing_length` cap).
  - Stage-2 AB is **step-budgeted** in raw samples, so packing selection SHOULD be **pool-aware**: it SHOULD attempt to minimize the total number of packed sequences required to consume the per-rank pool for the optimizer step (fewer forward/backward calls).
  - To avoid tiny remainder packs within the per-step pool, selection MAY trade off “current pack total length vs FIFO-greedy baseline for the same buffer state” when this reduces the overall number of packs or avoids underfilled remainders, as long as all packing invariants are preserved.
  - When `training.packing` is enabled, Stage-2 AB MUST ensure **both** channels run with the padding-free packing metadata needed for Qwen3-VL correctness (i.e., `text_position_ids` + mRoPE `position_ids`) so each forward can pass the required 4-row `position_ids` format.
  - If the required packing metadata cannot be produced, the trainer MUST fail fast during initialization with actionable guidance (e.g., disable `training.packing`).

#### Scenario: Stage-2 AB runs under ms-swift with raw-sample collation
- **GIVEN** a config that selects `custom.trainer_variant: stage2_two_channel`
- **WHEN** training starts under ms-swift
- **THEN** the trainer receives raw samples with fields required for Channel-B (including `messages` and `assistant_payload`)
- **AND** no upstream Transformers files are modified to enable the run.

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
- The legacy list-based schedule knob `schedule.pattern` is not supported.
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

### Requirement: Bbox-only v1 guardrails are enforced
The Stage-2 AB trainer MUST enforce bbox-only v1 guardrails on both GT objects and predicted rollout objects.

Stage-2 AB is bbox-only v1:
- GT objects MUST contain exactly one geometry field and it MUST be `bbox_2d` with exactly 4 coordinates.
  - If any GT object contains `poly` or any other geometry key, the trainer MUST fail fast with an error.
  - If any GT object contains malformed `bbox_2d` (wrong length, non-decodable values, or out-of-range values outside `[0, 999]` in norm1000 space), the trainer MUST fail fast.
    - Coercion contract (normative): values MUST be parsed as `int(round(float(x)))`; any value that cannot be coerced or lands outside `[0, 999]` is invalid.
  - If any GT bbox is invalid in ordering (`x2 < x1` or `y2 < y1`), the trainer MUST fail fast.

Predicted rollout objects MUST be filtered deterministically (no repair, no conversion):
- Instances whose geometry is not `bbox_2d` (including `poly`) MUST be dropped.
- Instances with missing/empty `desc` MUST be dropped.
- Instances with invalid bbox arity or invalid coord values MUST be dropped.

Diagnostics (normative):
- The trainer MUST expose strict-drop counters:
  - `N_valid_pred` (valid predicted instances kept),
  - `N_drop_invalid` (instances dropped by strict validation),
  - and reason buckets at minimum including: `missing_desc`, `missing_geom`, `wrong_arity`, `non_coord_token`, `poly_unsupported`, `unknown_geom`, `bbox_invalid`, `key_invalid`.
- The trainer MUST emit these counters as stable training metrics/log keys at least once per optimizer step that runs Channel-B.

#### Scenario: Drop diagnostics include valid and invalid counts
- **GIVEN** a Channel-B rollout containing a mix of valid bbox instances and invalid instances
- **WHEN** the trainer applies strict validation
- **THEN** invalid instances are dropped deterministically (no repair)
- **AND** the trainer exposes `N_valid_pred` and `N_drop_invalid` with at least one reason bucket incremented.

#### Scenario: GT poly fails fast
- **GIVEN** a training sample whose GT `assistant_payload` contains an object with `poly`
- **WHEN** the trainer prepares the sample for either channel
- **THEN** it raises an error indicating bbox-only v1 requires filtering out polygons upstream.

### Requirement: Deprecated self-context knobs fail fast in Stage-2 two-channel configs
Stage-2 two-channel MUST reject authored config knobs that exist only for the
deprecated self-context loop.

Normative behavior:
- The following keys MUST fail fast with actionable migration guidance when
  authored under `stage2_ab`:
  - `n_softctx_iter`
  - `softctx_grad_mode`
  - `softctx_temperature`
  - `coord_ctx_embed_mode`
- The error MUST explain that Channel-A now uses a single GT-anchored forward
  and MUST point users to the supported preset replacements.

#### Scenario: Deprecated self-context knob fails fast
- **GIVEN** a Stage-2 AB config with `stage2_ab.softctx_grad_mode: unroll`
- **WHEN** schema validation runs
- **THEN** validation fails
- **AND** the error explains that self-context iteration is deprecated and
  unsupported.

### Requirement: Channel-A forward path is compatible with Qwen3-VL multimodal semantics
For Qwen3-VL (dense) models, each forward MUST provide **exactly one** of `input_ids` or `inputs_embeds`.

Normative behavior:
- The canonical Stage-2 two-channel Channel-A path is a single teacher-forced
  forward and MUST NOT depend on iterative coord-slot embedding rewrites.
- Channel-A MAY use ordinary teacher-forced `input_ids` as its supported
  default path.
- If a specialized `inputs_embeds` path is ever used for another reason, it
  MUST preserve multimodal placeholder semantics and MUST NOT perturb the
  placeholder rows needed for model-internal feature insertion.
- The forward call MUST remain compatible with multimodal feature insertion
  (placeholder token count matches provided visual feature length).
- When using padding-free packing with Qwen3-VL, the trainer MUST pass the
  4-row `position_ids` format consistently for the Channel-A forward.

#### Scenario: Single-pass multimodal Channel-A remains compatible
- **WHEN** Stage-2 two-channel executes a multimodal Channel-A update
- **THEN** the Channel-A forward remains compatible with Qwen3-VL visual
  feature insertion
- **AND** it does not rely on iterative self-context forwards.

### Requirement: Channel-B reuses rollout-matching infra (clean-prefix parse/match + mandatory FN append)
Channel-B MUST reuse rollout generation and matching infrastructure, but its positive supervision contract is now clean-prefix based rather than raw-prefix based.

Normative behavior:
- Rollout generation MUST remain configured under `rollout_matching`.
- Parsing MUST use bounded container salvage plus strict record acceptance.
- Matching MUST be deterministic and MUST operate on `accepted_objects_clean`, not on the raw parsed bbox list.
- The positive teacher-forced prefix MUST be canonical serialization of `accepted_objects_clean`.
- FN append MUST remain mandatory so all GT objects are present in the final teacher-forced target.

#### Scenario: Channel-B teacher-forced target uses the clean accepted prefix
- **GIVEN** Channel-B is selected and rollout generation succeeds
- **WHEN** the trainer builds the teacher-forced target
- **THEN** the positive prefix is canonical serialization of `accepted_objects_clean`
- **AND** later correct objects are teacher-forced on that clean prefix rather than the raw rollout prefix.

### Requirement: Stage-2 serialized object field order follows shared config
Stage-2 AB serialization paths SHALL honor `custom.object_field_order` exactly as stage-1 serialization does.

Scope:
- Channel-A teacher-forced assistant payload construction.
- Channel-B FN append serialization path used to build `Y_train`.

Normative behavior:
- `desc_first`: per-object payload order is `desc` then concrete geometry key (`bbox_2d` or `poly`).
- `geometry_first`: per-object payload order is concrete geometry key (`bbox_2d` or `poly`) then `desc`.
- Object instance ordering and object key numbering remain unchanged.
- The serializer MUST NOT emit a synthetic key literally named `geometry`.

#### Scenario: Channel-A uses geometry-first payload when configured
- **GIVEN** `custom.object_field_order: geometry_first`
- **WHEN** Channel-A constructs teacher-forced assistant payload text
- **THEN** each serialized object payload places its concrete geometry key before `desc`
- **AND** object keys remain sequential (`object_1`, `object_2`, ...).

#### Scenario: Channel-B uses geometry-first for FN append when configured
- **GIVEN** `custom.object_field_order: geometry_first`
- **AND** Channel-B appends unmatched GT objects
- **WHEN** `Y_train` is constructed
- **THEN** appended object payloads place their concrete geometry key before `desc`
- **AND** matching/masking logic remains unchanged.

#### Scenario: Explicit desc-first behavior is preserved in both channels
- **GIVEN** `custom.object_field_order: desc_first`
- **WHEN** Channel-A or Channel-B serializes object payloads
- **THEN** payloads remain `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Stage-2 object instance ordering contract is unchanged
`custom.object_field_order` SHALL NOT modify stage-2 object instance ordering behavior.

Normative behavior:
- When Channel-A is selected, teacher-forced assistant payload construction SHALL follow `custom.object_ordering`.
- When Channel-A is selected, canonical clean-prefix construction SHALL serialize objects in the same effective order as the Channel-A teacher-forced assistant payload.
- For Channel-A, `custom.object_ordering: sorted` means canonical top-left ordering by `(minY, minX)`.
- For Channel-A, `custom.object_ordering: random` means the trainer SHALL use the current epoch’s deterministic dataset order for that sample.
- Channel-A object key numbering (`object_1`, `object_2`, ...`) SHALL follow the effective configured instance order.
- Channel-B object sequence remains determined by existing pipeline semantics (parsed rollout appearance order, matching/index continuation logic, and FN append order).
- Only intra-object field order is configurable through `custom.object_field_order`.

#### Scenario: geometry-first does not change rollout appearance order handling
- **GIVEN** rollout parsed objects appear in a specific raw-text order
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** Stage-2 performs matching and FN append
- **THEN** parsed predicted order remains the same as raw-text appearance
- **AND** only field order inside serialized object payloads differs.

#### Scenario: Channel-A sorted ordering uses canonical top-left order
- **GIVEN** Channel-A is selected
- **AND** `custom.object_ordering: sorted`
- **WHEN** the trainer builds the teacher-forced assistant payload and canonical clean prefix
- **THEN** both surfaces serialize GT objects in canonical top-left order
- **AND** object key numbering follows that same order.

#### Scenario: Channel-A random ordering reuses the sample’s current epoch order
- **GIVEN** Channel-A is selected
- **AND** `custom.object_ordering: random`
- **WHEN** the trainer builds the teacher-forced assistant payload and canonical clean prefix
- **THEN** both surfaces serialize GT objects in the sample’s current epoch order
- **AND** object key numbering follows that same order.

### Requirement: Channel-B invalid rollouts fall back deterministically (no silent skips)
When Channel-B is selected and a rollout response cannot be recovered into an append-ready `{"objects": [...]}` prefix, the trainer MUST:

- mark the rollout invalid for that sample,
- fall back to the canonical empty prefix `{"objects": [`,
- treat the rollout as containing zero valid predicted objects,
- append all GT objects as FN and continue training that sample.

#### Scenario: Invalid rollout falls back to the canonical empty objects prefix
- **GIVEN** Channel-B is selected for a sample
- **AND** the rollout response does not yield an append-ready `{"objects": [...]}` prefix
- **WHEN** the trainer parses the rollout for matching
- **THEN** it marks the rollout invalid for that sample
- **AND** it uses `{"objects": [` as the prefix and FN-appends all GT objects
- **AND** the sample is still included in teacher-forced training.

#### Scenario: Closure-resolution ambiguity keeps the sample on the FN-tail fallback path
- **GIVEN** Channel-B has already built a deterministic clean-prefix teacher-forced target
- **AND** explicit closure-marker bookkeeping cannot be resolved unambiguously for that target
- **WHEN** the trainer finalizes per-sample Channel-B supervision metadata
- **THEN** it keeps the sample in teacher-forced training
- **AND** it falls back to the normal FN-tail supervision path without dropping the sample.

### Requirement: Channel-B clean boundaries and duplicate bursts are canonical
The clean-prefix Channel-B contract SHALL define boundary-indexed duplicate bursts over the deduplicated clean sequence.

Normative behavior:
- Clean boundaries are indexed by insertion position in `accepted_objects_clean`:
  - boundary `0` before the first clean object,
  - boundary `N` after the last clean object, where `N = len(accepted_objects_clean)`.
- If `accepted_objects_clean` is empty, exactly one boundary exists: `0`.
- Duplicate bursts MUST attach to these canonical clean boundaries.

#### Scenario: Empty clean sequence still exposes one valid boundary
- **WHEN** sequential dedup yields `accepted_objects_clean = []`
- **THEN** duplicate bursts are still indexed against boundary `0`
- **AND** duplicate-ul target construction remains well-defined.

### Requirement: Generic unmatched clean extras remain prefix-visible while staying outside desc and coord supervision
Accepted clean objects that are unmatched after Hungarian MAY remain in the clean prefix as context, but they MUST remain outside desc, bbox, coord, and duplicate-ul supervision.

Normative behavior:
- Unmatched clean extras MAY populate global rollout-prefix struct masks when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`.
- Unmatched clean extras MUST NOT populate coord/bbox supervision groups.
- Unmatched clean extras MUST NOT create extra positive desc targets.
- Unmatched clean extras MUST NOT create duplicate-ul positives.

#### Scenario: Unmatched clean extra stays in context with shared prefix structure CE only
- **WHEN** Channel-B retains an unmatched clean accepted object in the clean prefix
- **THEN** that object remains visible in the canonical teacher-forced prefix
- **AND** it contributes zero desc CE, zero bbox loss, zero coord loss, and zero duplicate-ul positives
- **AND** it may still participate in the global rollout-prefix structure CE surface.

### Requirement: Channel-B rollout seeding is deterministic and logged
Channel-B rollouts MUST be fully deterministic under greedy decoding given the same model weights, the same training seed, and the same `global_step`.

If stochastic decoding is enabled, the trainer MUST apply a deterministic seeding plan and SHOULD be reproducible under a fixed run configuration (same backend, same world size/sharding, same batch shapes/order). Exact per-request determinism for stochastic decoding is not required in this capability.

Definition (normative):
- `training_seed` refers to HuggingFace `TrainingArguments.seed` (ms-swift `train_args.seed`), i.e., the same seed source used by `rollout_matching_sft`.

Deterministic decoding constraints (normative):
- The trainer MUST support deterministic greedy decoding for Channel-B rollouts under both HF and vLLM backends.
- If the run enables stochastic decoding (e.g., `temperature > 0` and/or `do_sample: true`), the trainer MUST make it reproducible via deterministic seeding derived from `rollout_seed_base`.
  - For `rollout_backend: "vllm"`, seeded stochastic decoding MUST be implemented by passing per-request seeds via the backend request config (or equivalent).
  - For `rollout_backend: "hf"`, if stochastic decoding is enabled, the trainer MUST at least set a deterministic *global* RNG seed before rollout generation (e.g., `seed_everything(rollout_seed_base)`), and MUST log `rollout_seed_base` and the decoding params used.
    - Per-request RNG isolation is OPTIONAL (and may be added later), but is not required for landing this capability.

The trainer MUST derive a deterministic rollout seed base consistent with the rollout-matching seeding contract:
- `rollout_seed_base = (training_seed + global_step * 1000003) & 0x7FFFFFFF`

Per-request seeds MUST be derived deterministically from `rollout_seed_base` and the within-batch request index (or an equivalent stable index plan for multi-server chunking).

The trainer MUST log (or expose via metrics) the effective `rollout_seed_base` at least once per optimizer step when Channel-B performs fresh rollouts.

#### Scenario: Same step produces identical derived rollout seed base
- **GIVEN** `training_seed = 123` and `global_step = 7`
- **WHEN** Channel-B derives the rollout seed base twice
- **THEN** both runs produce the same `rollout_seed_base`
- **AND** the derived `rollout_seed_base` is logged for reproducibility.

### Requirement: Hybrid objective preserves Channel-A anchoring and uses clean-prefix Channel-B supervision
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- **Token CE anchor at the GT teacher-forced forward**:
  - CE on non-coord tokens MUST be computed from the GT-context logits.
  - Coord tokens MUST NOT contribute to CE, to avoid double-supervision.
- **Geometry + distribution regularizers from the same single-pass Channel-A
  logits**:
  - Geometry losses and any distribution-level losses MUST be computed from the
    supported single-pass Channel-A logits, not from a deprecated final
    self-context pass.

Channel-B:
- **Clean-prefix positive supervision**:
  - the positive teacher-forced prefix MUST be canonical serialization of `accepted_objects_clean`,
  - matched clean prefix objects MUST receive structure-only CE,
  - generic unmatched clean extras MAY remain in the clean prefix as context but MUST remain neutral,
  - FN objects MUST be appended to the clean target and receive structure+desc CE.
- **FP-neutral geometry**:
  - geometry losses MUST be computed for matched clean prefix objects and FN-injected objects,
  - generic unmatched clean extras MUST NOT receive geometric gradients.
- **Duplicate-ul supervision**:
  - duplicate-certified continuations MUST be removed from the positive clean prefix,
  - duplicate UL MUST target the first true LCP-divergence token relative to the clean continuation at the same clean boundary,
  - same-boundary duplicates that share the same divergence token MUST collapse to one UL term.
- **Closure supervision stays on**:
  - the outermost JSON closure `}` and `<|im_end|>` MUST remain CE-supervised,
  - if closure-marker bookkeeping becomes ambiguous after the clean target is built, the sample MUST stay on the deterministic FN-tail fallback path rather than being dropped.

Configurable desc supervision (both channels):
- Desc CE weights MUST be expressed via the declared pipeline module configs:
  - Channel-A: `stage2_ab.pipeline.objective[name=token_ce].config.desc_ce_weight`
  - Channel-B (FN tail): `stage2_ab.pipeline.objective[name=token_ce].config.rollout_fn_desc_weight`

Bbox geometry losses (both channels) are computed from coord distributions:
- The trainer MUST decode coordinates from coord-token distributions via CoordExp expectation decoding (not argmax):
  - Let bins `k ∈ {0..999}` correspond to the coord-token sub-vocabulary.
  - Let `p(k)` be the softmax distribution over these 1000 bins for a coord slot, taken from the standard causal shift:
    - For a coord token at input position `p`, `p(k)` MUST be computed from logits at position `p-1` (consistent with CE).
  - The decoded normalized coordinate MUST be: `c_hat = Σ_k p(k) * (k/999)` in `[0, 1]`.
- The trainer MUST compute bbox losses in normalized coordinate space `[0, 1]`:
  - GT bbox ints in `[0, 999]` MUST be converted to floats by dividing by `999`.
  - Predicted bbox coords MUST be the decoded normalized floats from `c_hat` above.
- Geometry loss MUST use logits from:
  - the single Channel-A GT-anchor forward, and
  - the Channel-B clean-prefix teacher-forced logits.

Loss form (normative):
- The trainer MUST use SmoothL1 (Huber) + CIoU as the bbox regression terms.
- The trainer MUST NOT use GIoU in Stage-2 AB.

Numerical stability (normative):
- The trainer MUST canonicalize predicted boxes before CIoU:
  - `(x1,x2) := (min(x1,x2), max(x1,x2))`, `(y1,y2) := (min(y1,y2), max(y1,y2))`.
- The trainer MUST ensure the geometry losses do not produce NaNs/Infs, including early training when predictions are degenerate.

#### Scenario: Desc can be fully masked while keeping structure CE
- **GIVEN** `stage2_ab.pipeline.objective[name=token_ce].config.desc_ce_weight: 0`
- **WHEN** CE labels are built for a batch
- **THEN** JSON structure tokens remain supervised by CE
- **AND** desc value tokens do not contribute to CE.

#### Scenario: Expectation decoding uses probability-weighted mean (not argmax)
- **GIVEN** a coord-slot distribution with `p(k=0)=0.5` and `p(k=999)=0.5`
- **WHEN** the trainer decodes the coordinate via expectation decoding
- **THEN** the decoded value is approximately `0.5` (i.e., `(0*0.5 + 999*0.5)/999`)
- **AND** it is not equal to an argmax decode of `0` or `1`.

#### Scenario: Geometry losses operate on normalized coordinates
- **GIVEN** a GT bbox coordinate bin value `k=999`
- **WHEN** the trainer converts GT bins to normalized floats for geometry loss
- **THEN** the converted value is `999/999 = 1.0`
- **AND** geometry losses are computed using normalized floats in `[0, 1]`.

#### Scenario: Channel-A geometry no longer depends on a final iteration
- **WHEN** Stage-2 AB computes Channel-A geometry or coord regularization
- **THEN** the same single-pass Channel-A logits feed those terms
- **AND** no final self-context logits are required.

#### Scenario: Channel-B geometry includes matched clean prefix objects and FN but excludes unmatched clean extras
- **GIVEN** Channel-B where clean-prefix matching yields non-empty matched clean objects, unmatched clean extras, and FN sets
- **WHEN** Channel-B losses are computed
- **THEN** geometry loss is accumulated for matched clean prefix objects and FN-injected objects
- **AND** unmatched clean extras contribute zero geometry loss.

#### Scenario: Closure-supervision brace target is the same brace used for injection
- **GIVEN** Channel-B injects FN entries before the outermost close brace resolved by brace-depth scan
- **WHEN** CE masks are produced
- **THEN** that same outermost close brace token position remains CE-supervised
- **AND** `<|im_end|>` remains CE-supervised.

#### Scenario: CE masking follows matched-clean / unmatched-clean / FN policy
- **GIVEN** Channel-B contains one matched clean object, one unmatched clean extra, and one FN-injected object
- **WHEN** CE weights are materialized
- **THEN** matched clean structure tokens are supervised while matched clean desc tokens are masked
- **AND** unmatched clean extra structure/desc/coord tokens are all masked
- **AND** FN-injected structure and desc tokens are supervised.

#### Scenario: Channel-B supervises top-level closure and `<|im_end|>`
- **GIVEN** Channel-B builds a teacher-forced target that ends with a top-level `}` followed by `<|im_end|>`
- **WHEN** the trainer builds CE labels/weights for Channel-B
- **THEN** it keeps CE supervision on that top-level `}` token position
- **AND** it keeps CE supervision on `<|im_end|>`.

#### Scenario: Closure marker resolution fallback keeps the sample in training
- **GIVEN** a Channel-B sample where the trainer cannot deterministically locate the outermost `}` / `<|im_end|>` marker positions after building the clean target
- **WHEN** the trainer finalizes CE labels/weights for Channel-B
- **THEN** it keeps the sample in Channel-B supervision
- **AND** it increments `stage2_ab/channel_b/closure_supervision/N_drop` as the legacy-named fallback-activation counter.

#### Scenario: No valid predictions fall back to the canonical empty clean prefix
- **GIVEN** strict validation yields `N_valid_pred == 0`
- **WHEN** Channel-B builds the clean teacher-forced target
- **THEN** it uses the canonical empty clean prefix `{"objects": [`
- **AND** this is equivalent to appending all GT objects as FN-supervised targets.

### Requirement: Coord quantization is globally consistent (k/999 only)
The Stage-2 AB trainer MUST use a single consistent coord quantization scheme:
- Encode: `k = clamp(round(999*c), 0, 999)`
- Decode: `c = k/999`

Semantics (normative):
- Coord tokens of the form `<|coord_k|>` denote an integer bin index `k ∈ [0, 999]` under the project’s 1000-bin convention (bins 0..999).
- In this convention, `(k_x, k_y) = (0, 0)` corresponds to the top-left pixel corner, and `(999, 999)` corresponds to the bottom-right pixel corner (i.e., the inclusive image bounds).
  - No `/1000`-based "bin -> normalized float" mapping is allowed anywhere in the repository. Any existing `/1000`-based helpers/metadata MUST be removed or updated to `/999` so that `k=999` maps to `1.0` consistently.

The trainer MUST NOT use any `1000`-based normalization (e.g., `round(1000*c)` or decode `k/1000`).

#### Scenario: Upper bound is 999 and decodes to 1.0
- **GIVEN** a normalized coordinate `c = 1.0`
- **WHEN** it is encoded to a bin index
- **THEN** the bin index is `k = 999`
- **AND** decoding returns `k/999 = 1.0`
- **AND** no bin index `1000` is ever produced.

#### Scenario: Lower bound is 0 and decodes to 0.0
- **GIVEN** a normalized coordinate `c = 0.0`
- **WHEN** it is encoded to a bin index
- **THEN** the bin index is `k = 0`
- **AND** decoding returns `k/999 = 0.0`.

### Requirement: Channel-B step mode supports an in-step bounded pipeline queue between rollout and learning
When Channel-B executes in step mode with packing enabled, the trainer SHALL support overlapping rollout generation with learner compute within the optimizer step using a bounded producer/consumer queue (size 1 is sufficient).

Normative safety guardrail:
- If the in-step pipeline queue is enabled, rollouts MUST run on dedicated GPUs via vLLM server mode.
  - Concretely: the trainer MUST require `rollout_matching.rollout_backend=vllm` and `rollout_matching.vllm.mode=server`.
  - If this condition is not met, the trainer MUST error fast with a clear message (to avoid unsafe concurrent rollout+train on the same process/device).

#### Scenario: Rollout and learner overlap within a step
- **GIVEN** rollout runs on dedicated GPUs via vLLM server mode
- **AND** learner training runs on a separate GPU
- **WHEN** Channel-B executes one optimizer step
- **THEN** the trainer overlaps rollout generation and learner forward/backward where feasible
- **AND** the trainer does not build an unbounded rollout pool.

### Requirement: Channel-B rollout decode batching is configurable and independent of learner microbatch
When Channel-B executes, the trainer SHALL allow configuring rollout decode batching independently of learner microbatch size (which remains 1 under packing).

Configuration (normative):
- The decode batching knob MUST be `rollout_matching.channel_b_decode_batch_size` (int).
- It MUST denote the maximum number of sequences decoded per rollout GPU in one backend generation call.

#### Scenario: Rollout decode batch size 2 with learner microbatch 1
- **GIVEN** `training.per_device_train_batch_size=1`
- **AND** `rollout_matching.channel_b_decode_batch_size: 2`
- **WHEN** rollouts are generated
- **THEN** the rollout backend generates responses for 2 samples in one decode call
- **AND** learner training still runs one packed sequence per forward/backward.

#### Scenario: Rollout decode batch size 4 with learner microbatch 1
- **GIVEN** `training.per_device_train_batch_size=1`
- **AND** `rollout_matching.channel_b_decode_batch_size: 4`
- **WHEN** rollouts are generated
- **THEN** the rollout backend generates responses with per-device decode batch size bounded by 4
- **AND** learner training still runs one packed sequence per forward/backward
- **AND** legacy key path `custom.extra.rollout_matching.channel_b_decode_batch_size` remains fail-fast.

### Requirement: Deprecated legacy coord-loss knobs are silently ignored
To enable config refactors without blocking training runs, the configuration system MUST silently ignore deprecated legacy coord-loss knobs under `custom.*` that are no longer supported by the project’s coord-loss contract.

Normative minimum:
- If `custom.coord_loss` is present in a YAML config, configuration parsing MUST NOT raise, and the value MUST be ignored.

#### Scenario: custom.coord_loss does not hard error
- **GIVEN** a config that includes `custom.coord_loss` (legacy)
- **WHEN** configuration is parsed
- **THEN** parsing succeeds and the legacy field is ignored.

### Requirement: Channel-B step mode is step-budgeted in raw rollouts and learns-to-completion under packing
When Stage-2 AB training is enabled, Channel-B SHALL interpret the Channel-B batch size in terms of **raw rollouts per optimizer step**, not “packed sequences per optimizer step”.

Config contract (normative):
- `training.effective_batch_size` MUST be provided.
- `training.effective_batch_size` MUST be divisible by:
  - `training.per_device_train_batch_size × learner_world_size`,
  so `training.gradient_accumulation_steps` is an exact integer (no ceil overshoot).
- `training.gradient_accumulation_steps` MAY be omitted (recommended; derived from `effective_batch_size`).
  - If `training.gradient_accumulation_steps` is provided explicitly, it MUST equal the derived exact value implied by `effective_batch_size` and topology, otherwise config validation MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.mode` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.async` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.rollouts_per_step` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.enable_pipeline` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.rollout_decode_batch_size` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.

Normative behavior:
- Define `rollouts_per_step := training.effective_batch_size`.
- The trainer MUST collect `rollouts_per_step` raw rollouts **globally across all train ranks** for the optimizer step.
  - Under DDP, each rank MUST collect a deterministic share `local_rollouts_per_step` such that the sum over ranks equals `rollouts_per_step`.
- The trainer MUST then construct per-sample teacher-forced segments (rollout prefix + mandatory FN append).
- When `training.packing=true`, the trainer MUST pack these segments into a **variable** number of packed sequences under the `packing_length` cap derived from `global_max_length`.
- The trainer MUST run forward/backward once per packed sequence and accumulate gradients, then perform **exactly one** optimizer update for the optimizer step.

#### Scenario: 32 raw rollouts pack into fewer than 32 packed sequences
- **GIVEN** `training.effective_batch_size=32`
- **AND** `training.packing=true` and `global_max_length=12000`
- **WHEN** Channel-B executes for one optimizer step
- **THEN** the trainer collects 32 raw rollouts
- **AND** packs them into `N_packs` packed sequences where `N_packs` MAY be less than 32
- **AND** performs one optimizer update for the step.

#### Scenario: Channel-B executes only on the final micro-step under grad accumulation
- **GIVEN** `training.per_device_train_batch_size=1`, `learner_world_size=4`, and `training.gradient_accumulation_steps=8`
- **WHEN** Channel-B is selected for one optimizer step
- **THEN** each rank buffers its raw rollouts across the first 7 micro-steps without running the Channel-B loop
- **AND** the full Channel-B loop (rollout→pack→learn-to-completion) runs on the 8th (final) micro-step
- **AND** the outer Trainer performs exactly one optimizer update for the step.

### Requirement: Removed Channel-B semantic-desc gate knobs fail fast
Training-time semantic-desc gating is removed from Stage-2 AB Channel-B and MUST NOT be configurable.

Normative behavior:
- `stage2_ab.channel_b.semantic_desc_gate` MUST fail fast during config loading with actionable removal guidance.
- Stage-2 AB desc supervision MUST follow the unified rollout-prefix + FN-append contract without semantic-gate toggle surfaces.

#### Scenario: Semantic-desc gate config is rejected
- **WHEN** a config includes `stage2_ab.channel_b.semantic_desc_gate`
- **THEN** config loading fails fast before trainer construction
- **AND** the error reports the removed dotted path with removal guidance.

### Requirement: Async actor-learner mode knobs are removed
Stage-2 AB MUST reject async actor-learner config surfaces for Channel-B.

Normative behavior:
- `stage2_ab.channel_b.mode` MUST fail fast when provided.
- `stage2_ab.channel_b.async` MUST fail fast when provided.
- Channel-B execution remains the single step-budgeted pathway (with optional bounded in-step overlap as defined by the step-mode pipeline requirement).

#### Scenario: Channel-B async mode knob is rejected
- **WHEN** a config sets `stage2_ab.channel_b.mode: async`
- **THEN** config loading fails fast with guidance to remove `stage2_ab.channel_b.mode`.

### Requirement: DDP-safe Channel-B execution semantics for multi-GPU learners
When `world_size > 1`, Channel-B MUST be executed in a DDP-safe way:
- Each micro-step MUST perform exactly one packed forward/backward per rank.
- The trainer MUST NOT run any inner loops that cause different ranks to perform different numbers of forwards within the same micro-step.

Legacy guardrail (v1):
- Under all world sizes (including DDP), `stage2_ab.channel_b.mode` is removed and MUST fail fast when provided.

#### Scenario: Removed mode knob fails fast under DDP
- **GIVEN** Stage-2 AB is launched with `world_size=2`
- **AND** config sets `stage2_ab.channel_b.mode: step`
- **WHEN** training starts
- **THEN** the trainer fails fast with guidance to remove `stage2_ab.channel_b.mode`.

### Requirement: Legacy Channel-B ablation knobs are removed
Legacy Channel-B ablation knobs are removed and MUST NOT be configurable:
- `stage2_ab.channel_b.reordered_gt_sft`
- `stage2_ab.channel_b.desc_ce_weight_matched`
- `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`

#### Scenario: Removed reordered-gt knob fails fast
- **WHEN** a config includes `stage2_ab.channel_b.reordered_gt_sft`
- **THEN** config loading fails fast with actionable removal guidance.

#### Scenario: Removed desc-ce-weight knob fails fast
- **WHEN** a config includes `stage2_ab.channel_b.desc_ce_weight_matched`
- **THEN** config loading fails fast with actionable removal guidance.

#### Scenario: Removed invalid-structure multiplier knob fails fast
- **WHEN** a config includes `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`
- **THEN** config loading fails fast with actionable removal guidance.

### Requirement: Repeat-terminate rollout knobs are unsupported in Stage-2 AB
Stage-2 AB rollout config MUST treat repeat-terminate keys as unsupported.

Normative behavior:
- `rollout_matching.repeat_terminate` MUST fail fast as an unknown/unsupported key.
- `custom.extra.rollout_matching.*` (including any `repeat_terminate` subtree) MUST fail fast with migration guidance to canonical top-level rollout keys.
- Stage-2 AB MUST NOT claim repeat-terminate-specific rollout metrics (`rollout/repeat_terminate_active`, `rollout/repeat_terminate_triggered_sequences`) unless this capability is reintroduced by a future approved change.

#### Scenario: Top-level repeat-terminate key is rejected
- **WHEN** a config includes `rollout_matching.repeat_terminate`
- **THEN** strict config parsing fails fast with unknown-key diagnostics.

#### Scenario: Legacy custom.extra repeat-terminate path is rejected
- **WHEN** a config includes `custom.extra.rollout_matching.repeat_terminate`
- **THEN** config loading fails fast with guidance to remove legacy `custom.extra.rollout_matching.*` usage.

### Requirement: Stage-2 AB consumes rollout helpers through public contracts only
The Stage-2 AB capability SHALL consume rollout parsing/matching/packing helpers only through a public rollout-matching contract module.
It MUST NOT import underscore-prefixed symbols from trainer implementation files.

#### Scenario: Private rollout helper removal does not break Stage-2 imports
- **WHEN** private underscore-prefixed helpers are removed from the rollout trainer implementation file
- **THEN** Stage-2 AB still imports successfully via public contract modules
- **AND** training initialization does not fail due to missing private symbols.

### Requirement: No-private-import boundary is regression-guarded
The Stage-2 AB capability SHALL include a regression guard that detects imports from underscore-prefixed rollout symbols and fails validation when such imports reappear.
This guard MUST use AST import inspection (test or static check) rather than regex text matching so formatting/comment changes do not create false signals.
The guard MUST run in routine validation for this capability.
The guard scope MUST cover the full Stage-2 AB capability surface:
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/**/*.py`

#### Scenario: Validation fails when a private rollout helper import is reintroduced
- **GIVEN** any Stage-2 AB source file in the guarded surface imports an underscore-prefixed rollout helper
- **WHEN** capability validation checks execute
- **THEN** validation fails with a boundary-violation diagnostic
- **AND** the regression is caught before merge.

### Requirement: Stage-2 AB trainer is decomposed into orchestrator plus owned components
The Stage-2 AB trainer SHALL be structured as an orchestration surface that delegates scheduling, async queue management, and channel execution to dedicated components.
The decomposition MUST preserve deterministic channel selection and existing Stage-2 contract semantics.

#### Scenario: Scheduling policy changes are isolated from trainer orchestration
- **GIVEN** a change to channel scheduling policy implementation
- **WHEN** Stage-2 AB training is run with unchanged YAML semantics
- **THEN** only scheduling component modules require modification
- **AND** the top-level trainer orchestration entrypoint remains interface-compatible.

### Requirement: Stage-2 critical invariants fail fast with contextual diagnostics
Stage-2 AB SHALL classify critical runtime invariants (queue feasibility, version-window gating, sync boundaries, required batch fields) as fail-fast conditions.
Unexpected failures on critical invariants MUST raise errors with step/channel/version context.
Best-effort diagnostics MAY continue under guarded warning paths.

#### Scenario: Async queue invariant violation raises actionable error
- **GIVEN** async mode is enabled and queue state violates required invariants for a scheduled Channel-B step
- **WHEN** Stage-2 attempts to execute that step
- **THEN** training raises with contextual diagnostics including step kind and queue/version state
- **AND** the failure is not silently suppressed.

### Requirement: Stage-2 AB objective includes coord soft-CE and W1 terms on supervised bbox slots
Stage-2 AB trainer MUST support Stage-1-style coord distribution penalties in Stage-2 training:

- `soft_ce` and `w1` MUST be computed on coord distributions for Stage-2-supervised bbox coord slots only:
  - matched-prefix groups (`bbox_groups_prefix`),
  - and FN-injected groups (`bbox_groups_fn`).
- The coord distribution for each supervised coord slot MUST follow the same causal shift contract as other Stage-2 coord losses (coord token at position `p` uses logits at `p-1`).
- These terms MUST contribute to the Stage-2 coord regularization objective (coord_reg module), and MUST be surfaced in training logs as **objective atoms** under provenance keys:
  - Channel-A: `loss/coord/{coord_soft_ce,coord_w1}`
  - Channel-B (rollout-context): `loss/B_coord/{coord_soft_ce,coord_w1}`
- The trainer MUST NOT apply these terms to unsupervised FP-only coord slots.

Weighting/config contract:
- Stage-2 uses the declared pipeline config for coord distribution penalties:
  - `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.temperature`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.target_sigma`
  - `stage2_ab.pipeline.objective[name=coord_reg].config.target_truncate`
- If `soft_ce_weight` and `w1_weight` are both `0`, Stage-2 soft-CE/W1 contributions MUST be zero.

#### Scenario: Enabled coord soft-CE/W1 increases Stage-2 coord regularization
- **GIVEN** Stage-2 config has `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight > 0`
- **AND** `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight > 0`
- **AND** a batch has supervised bbox coord slots
- **WHEN** Stage-2 computes loss
- **THEN** `loss/B_coord/coord_soft_ce` and `loss/B_coord/coord_w1` are positive
- **AND** the Channel-A coord group, when active, uses `loss/coord/*` rather
  than any `loss/A1_*` or `loss/A2_*` key.

### Requirement: Canonical Stage-2 base and prod leaves declare CIoU/coord-CE/soft-CE/W1 weights explicitly
The canonical Stage-2 AB config surfaces MUST declare CIoU and coord-distribution weights explicitly to avoid ambiguity between inherited defaults and production-tuned overrides.

Canonical base defaults (pipeline-only):
- `stage2_ab.pipeline.objective[name=bbox_geo].config.smoothl1_weight: 2.0`
- `stage2_ab.pipeline.objective[name=bbox_geo].config.ciou_weight: 0.5`
- `stage2_ab.pipeline.objective[name=coord_reg].config.coord_ce_weight: 0.0`
- `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight: 0.02`
- `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight: 0.02`

Canonical prod overrides (pipeline-only):
- `stage2_ab.pipeline.objective[name=bbox_geo].config.ciou_weight: 0.2`
- `stage2_ab.pipeline.objective[name=coord_reg].config.coord_ce_weight: 0.02`
- `stage2_ab.pipeline.objective[name=coord_reg].config.soft_ce_weight: 0.1`
- `stage2_ab.pipeline.objective[name=coord_reg].config.w1_weight: 0.1`
- `stage2_ab.pipeline.objective[name=coord_reg].config.target_truncate: 8`

#### Scenario: Canonical prod leaves pin explicit CIoU/soft-CE/W1 overrides
- **GIVEN** a canonical Stage-2 profile leaf under `configs/stage2_two_channel/prod/*.yaml`
- **WHEN** config is materialized through the supported Stage-2 profile hierarchy rooted at `configs/stage2_two_channel/base.yaml`
- **THEN** the leaf explicitly overrides effective Stage-2 loss weights with the canonical prod values above.

#### Scenario: Canonical smoke leaves inherit base CIoU/soft-CE/W1 defaults
- **GIVEN** a canonical Stage-2 profile leaf under `configs/stage2_two_channel/smoke/*.yaml`
- **WHEN** config is materialized through the supported Stage-2 profile hierarchy rooted at `configs/stage2_two_channel/base.yaml`
- **THEN** effective Stage-2 loss defaults include canonical base CIoU downweight and non-zero soft-CE/W1 terms.

### Requirement: Stage-2 two-channel training supports a config-declared objective and diagnostics pipeline
When `custom.trainer_variant: stage2_two_channel`, the system SHALL use an explicit YAML-declared objective/diagnostics pipeline for the canonical clean-prefix Channel-B contract.

Normative behavior:
- `stage2_ab.pipeline` MUST be present. There is no implicit default pipeline manifest for this contract.
- Canonical Stage-2 AB objective ordering for this contract is:
  1. `token_ce`
  2. `loss_duplicate_burst_unlikelihood`
  3. `bbox_geo`
  4. `coord_reg`
- Canonical Stage-2 AB diagnostics MAY include `coord_diag`.
- `loss_duplicate_burst_unlikelihood` MUST be present in canonical Stage-2 AB pipelines and MUST declare `channels: [B]`.
- `loss_duplicate_burst_unlikelihood` module `weight` is the only v1 scaling surface for duplicate UL.
- The old raw-prefix Channel-B contract is removed; there is no contract toggle or compatibility mode.

#### Scenario: Missing stage2_ab.pipeline fails fast
- **WHEN** a Stage-2 AB config sets `custom.trainer_variant: stage2_two_channel`
- **AND** `stage2_ab.pipeline` is absent
- **THEN** config loading fails fast before trainer init
- **AND** the error indicates `stage2_ab.pipeline` is required.

#### Scenario: Missing loss_duplicate_burst_unlikelihood in the canonical Channel-B pipeline fails fast
- **WHEN** a Stage-2 AB config declares `stage2_ab.pipeline.objective`
- **AND** the objective list omits `loss_duplicate_burst_unlikelihood`
- **THEN** config validation fails fast
- **AND** the error indicates the canonical clean-prefix Channel-B contract requires `loss_duplicate_burst_unlikelihood`.

#### Scenario: Stage-2 Two-Channel rejects rollout-matching pipeline keys
- **WHEN** `custom.trainer_variant=stage2_two_channel`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast with guidance to use `stage2_ab.pipeline`.

#### Scenario: Unknown module name fails fast
- **WHEN** a Stage-2 Two-Channel config defines `stage2_ab.pipeline` referencing an unknown module `name`
- **THEN** training initialization fails fast with actionable diagnostics.

### Requirement: Trainer variant naming is clear and stable
To reduce public-facing confusion, the system SHALL support clear trainer variant naming for Stage-2 two-channel
training.

Normative behavior:
- The system MUST accept `custom.trainer_variant: stage2_two_channel` as the canonical trainer variant string.
- The system MUST reject `custom.trainer_variant: stage2_ab_training` (fail fast) with actionable guidance to use
  `stage2_two_channel`.

#### Scenario: Legacy stage2_ab_training trainer alias is rejected
- **WHEN** configuration sets `custom.trainer_variant: stage2_ab_training`
- **THEN** config validation fails fast
- **AND** the error recommends `custom.trainer_variant: stage2_two_channel`.

### Requirement: Stage-2 Two-Channel module names are stable and discoverable
Stage-2 Two-Channel SHALL provide a strict module registry for its pipeline modules, and the module names SHALL be stable so
YAML-declared experiments remain auditable.

Normative minimum objective module names for this contract:
- `token_ce`
- `loss_duplicate_burst_unlikelihood`
- `bbox_geo`
- `coord_reg`

Normative minimum diagnostics module names:
- `coord_diag`

Normative behavior:
- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available Stage-2 Two-Channel module names.

#### Scenario: Unknown Stage-2 two-channel module names fail fast
- **WHEN** `stage2_ab.pipeline` references an objective module name not present in the Stage-2 registry
- **THEN** trainer initialization fails fast
- **AND** the error includes the unknown name and allowed module names.

### Requirement: Stage-2 Two-Channel module configs are strict and typed
Stage-2 Two-Channel SHALL validate module `config` payloads and `stage2_ab.channel_b` payloads strictly so experiments are reproducible and fail fast on schema drift.

Normative behavior:
- `loss_duplicate_burst_unlikelihood.config` MUST be an empty mapping in v1.
- `token_ce.config` no longer accepts any legacy invalid-structure amplification knob for Channel-B.
- `stage2_ab.channel_b` MUST accept only:
  - `duplicate_iou_threshold`
  - `triage_posterior`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`
  - `invalid_rollout_policy`
  - `pseudo_positive`
- `stage2_ab.channel_b.pseudo_positive` MUST be a typed mapping and MUST accept only:
  - `enabled`
  - `coord_weight`
- `stage2_ab.channel_b.triage_posterior` MUST accept only:
  - `num_rollouts`
  - `explorer_temperature`
  - `explorer_top_p`
  - `explorer_top_k`
  - `unlabeled_consistent_iou_threshold`
  - `recovered_ground_truth_weight_multiplier`
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, `stage2_ab.channel_b.triage_posterior.num_rollouts` MUST be `2`
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, `stage2_ab.channel_b.triage_posterior.num_rollouts` MUST be `>= 2`
- Unknown keys in a module `config` or in `stage2_ab.channel_b` MUST fail fast with actionable diagnostics.

#### Scenario: Non-empty loss_duplicate_burst_unlikelihood.config fails fast
- **WHEN** `stage2_ab.pipeline.objective[*].name=loss_duplicate_burst_unlikelihood`
- **AND** its `config` mapping contains any key
- **THEN** configuration parsing fails fast
- **AND** the error indicates `loss_duplicate_burst_unlikelihood.config` must be empty for v1.

#### Scenario: Legacy invalid-structure multiplier placement fails fast
- **WHEN** a Stage-2 AB config sets `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`
- **THEN** configuration parsing fails fast
- **AND** the error indicates that legacy raw-prefix invalid-structure amplification is not part of the canonical clean-prefix contract.

### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract
Stage-2 Two-Channel training SHALL implement loss naming and masking semantics per the `teacher-forcing-unified-loss-registry`
capability.

Normative behavior:
- Stage-2 Two-Channel MUST build token-type masks and object-subset masks according to the registry contexts:
  - Channel-A uses `context=gt` for CE and bbox/coord supervision.
  - Channel-B uses `context=rollout` with explicit triage-aware + EOS-enforced semantics.
- When the module pipeline is enabled, objective/diagnostics modules MUST emit metric keys consistent with the
  registry’s canonical component names.

#### Scenario: Channel-A uses GT registry context only
- **WHEN** Stage-2 Two-Channel executes a Channel-A update step
- **THEN** Channel-A loss computation uses `context=gt`
- **AND** the same step does not construct `context=self_context`.

### Requirement: Stage-2 Two-Channel removes self-context-era decode toggles from typed YAML config
Stage-2 Two-Channel SHALL reject geometry-decode toggles that were carried with
the deprecated self-context surface.

Normative behavior:
- Config MUST be expressed under the typed Stage-2 Two-Channel namespace (`stage2_ab.*`) and MUST be strict (unknown keys fail).
- `stage2_ab.coord_decode_mode` is deprecated and MUST fail fast if authored.
- `stage2_ab.coord_ctx_embed_mode` is deprecated and MUST fail fast if authored.
- Geometry decode follows the supported expectation-decode baseline without an
  authored Stage-2 override.

#### Scenario: Deprecated Stage-2 decode toggle fails fast
- **WHEN** `stage2_ab.coord_decode_mode` is authored in an active Stage-2 config
- **THEN** configuration parsing fails fast
- **AND** the error explains that Stage-2 geometry decode now uses the fixed
  expectation-decode baseline.

### Requirement: Eval-step supports COCO mAP when enabled
Stage-2 two-channel training SHALL support computing COCO-style bbox mAP during `eval_step` (in addition to the
rollout matching evaluator metrics), so evaluation is consistent across Stage-2 trainers.

Normative behavior:
- COCO/mAP evaluation MUST be controlled by the typed config surface:
  - `rollout_matching.eval_detection.enabled: bool`
- For Stage-2 trainers, `rollout_matching.eval_detection.enabled` MUST default to `true` so COCO `mAP` is a standard
  eval metric like `rollout/f1`. Disabling it MUST be explicit via YAML (`enabled: false`).
- The eval-step COCO evaluation implementation MUST reuse the same detection evaluator modules used by offline
  evaluation (see `openspec/specs/detection-evaluator`), rather than re-implementing COCO preparation/scoring logic in
  trainer code.
- When `rollout_matching.eval_detection.enabled=true`, the trainer MUST attempt to compute COCO bbox AP/mAP for the
  eval-step rollouts and MUST emit, at minimum:
  - `eval/detection/mAP` (float; COCO `bbox_AP` = AP@[.50:.95])
- The trainer MUST NOT emit additional COCO summary metric keys during eval-step (e.g., `rollout/bbox_AP50`,
  `rollout/bbox_AR100`, `rollout/segm_*`) beyond `eval/detection/mAP`. Full metric reports remain available via the offline
  evaluation pipeline.
- If COCO eval fails unexpectedly (missing dependencies, invalid records, etc.), the trainer MUST:
  - emit `eval/detection/mAP=0.0` (so the key is always present when enabled), and
  - surface the failure as a warning (not silent).

#### Scenario: Stage-2 two-channel eval reports mAP
- **GIVEN** `rollout_matching.eval_detection.enabled=true`
- **WHEN** `eval_step` runs
- **THEN** `eval/detection/mAP` is present in the eval metrics payload

### Requirement: Stage-2 AB Channel-B uses anchor/explorer triage with an optional pseudo-positive multi-view extension
When `custom.trainer_variant: stage2_two_channel`, the canonical Channel-B contract SHALL build its clean teacher-forced target from one greedy anchor rollout plus one or more explorer rollouts:

- one anchor rollout using greedy / deterministic decoding,
- one or more explorer rollouts using stochastic decoding configured under `stage2_ab.channel_b.triage_posterior`.

Normative behavior:

- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, total rollout views MUST remain `2` (`1` anchor + `1` explorer),
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, total rollout views MUST equal `stage2_ab.channel_b.triage_posterior.num_rollouts`,
- each rollout MUST independently reuse the existing bounded salvage + strict record acceptance + bbox-valid filtering + sequential dedup + Hungarian matching path,
- GT-backed semantics MUST inherit the existing Channel-B accepted-clean Hungarian + gating contract,
- the final positive target MUST be built by editing the **anchor** clean sequence rather than rebuilding a union order,
- pseudo-positive candidate discovery MUST start from unmatched anchor clean objects and use explorer agreement only as support evidence,
- explorer-only non-GT-backed objects MUST NOT be promoted into clean-prefix positives,
- a GT hit found only on the explorer side MUST project to `recovered_fn`, not to anchor retention.

#### Scenario: Channel-B builds the final target from the anchor clean sequence
- **GIVEN** anchor and explorer rollouts were both produced for a Channel-B sample
- **WHEN** the trainer constructs the teacher-forced target
- **THEN** it starts from the anchor clean accepted sequence
- **AND** it preserves anchor order for retained objects
- **AND** it does not rebuild a union ordering over anchor and explorer objects.

#### Scenario: Explorer-only GT hit does not keep a bad anchor object positive
- **GIVEN** an anchor/explorer pair-or-singleton record where the anchor side misses GT and the explorer side matches GT
- **WHEN** the trainer projects triage evidence into training actions
- **THEN** the outcome is `recovered_fn`
- **AND** the bad anchor object is not kept as an anchor GT-backed positive.

#### Scenario: Explorer-only non-GT-backed object does not become a pseudo-positive prefix object
- **GIVEN** an explorer object that does not correspond to any unmatched anchor clean object
- **WHEN** Channel-B projects pseudo-positive evidence into the final clean prefix
- **THEN** that explorer-only non-GT-backed object is not promoted into a new prefix positive
- **AND** pseudo-positive selection remains anchored on unmatched anchor clean objects.

### Requirement: Stage-2 AB Channel-B rollout and pseudo-positive knobs are typed and grouped
The Stage-2 AB config SHALL expose rollout-view and pseudo-positive knobs under `stage2_ab.channel_b`.

Normative behavior:

- `stage2_ab.channel_b.pseudo_positive` MUST be a typed mapping,
- the mapping MUST accept only:
  - `enabled`
  - `coord_weight`
- `stage2_ab.channel_b.triage_posterior` MUST be a typed mapping,
- the mapping MUST accept only:
  - `num_rollouts`
  - `explorer_temperature`
  - `explorer_top_p`
  - `explorer_top_k`
  - `unlabeled_consistent_iou_threshold`
  - `recovered_ground_truth_weight_multiplier`
- unknown keys under `stage2_ab.channel_b.triage_posterior` MUST fail fast.

#### Scenario: Unknown triage_posterior key fails fast
- **WHEN** a Stage-2 AB config includes an unknown key under `stage2_ab.channel_b.triage_posterior`
- **THEN** config loading fails fast with the full dotted path.

### Requirement: Recovered GT objects stay on the FN injection path with higher weight
The canonical v1 v3 contract SHALL treat recovered GT objects as weighted FN injections, not as a second teacher trajectory.

Normative behavior:

- `recovered GT` means “missed in anchor accepted-clean matching and hit in explorer accepted-clean matching,”
- recovered GT objects MUST remain on the same FN injection path used by ordinary FN objects,
- the configured `recovered_ground_truth_weight_multiplier` MUST increase their desc+geo+coord supervision weight relative to ordinary FN objects,
- recovered-prefix distillation MUST NOT be part of the canonical v1 contract.

#### Scenario: Recovered GT object uses weighted FN injection
- **WHEN** a GT object is missed in anchor and hit in explorer
- **THEN** it is appended through the normal FN-injection path
- **AND** it receives the configured recovered-FN positive weight
- **AND** no separate explore-prefix teacher-forced pass is created.

### Requirement: Channel-B v3 uses deterministic one-to-one anchor/explorer association
The canonical v1 v3 contract SHALL associate anchor and explorer accepted objects deterministically before projecting triage actions.

Normative behavior:

- candidate cross-rollout pairs MUST be scored by IoU,
- only pairs with `IoU >= unlabeled_consistent_iou_threshold` are eligible,
- the chosen association MUST be one-to-one and maximize IoU,
- if multiple assignments achieve the same maximum total IoU, the chosen assignment MUST be the one whose sorted pair list `[(anchor_index, explorer_index), ...]` is lexicographically smallest.

#### Scenario: Crowded-scene association is stable under tie conditions
- **WHEN** two eligible anchor/explorer candidate pairs have identical IoU
- **THEN** the selected association is resolved by the canonical lexicographic assignment tie-break rule rather than container ordering or hash iteration.

### Requirement: Channel-B v3 uses one merged teacher-forced forward
The canonical v1 v3 contract SHALL realize `L(clean_anchor) + L(explore-derived corrections)` through one merged teacher-forced forward on the edited anchor target.

Normative behavior:

- the trainer MUST run one teacher-forced forward on the final edited target,
- positive, weighted-FN, and dead-anchor UL terms MUST be derived from that same forward,
- the trainer MUST NOT require a second explore teacher-forced payload in the canonical v1 contract.

#### Scenario: Single-forward v3 target realization
- **WHEN** a Channel-B v3 sample is prepared
- **THEN** all loss terms are derived from a single teacher-forced forward over the edited anchor target
- **AND** no second teacher-forced explore payload is required.

### Requirement: Retained unmatched anchor objects remain prefix-visible with explicit pseudo-positive supervision subsets
Retained unmatched anchor objects MAY remain in the clean prefix, but the Channel-B contract SHALL distinguish between selected pseudo-positive anchors, support-positive shielded anchors, and cluster-demoted shielded anchors.

Normative behavior:

- retained unmatched anchor objects MAY participate in global rollout-prefix struct masks when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`,
- selected pseudo-positive anchors MUST receive positive bbox/coord supervision using their retained anchor coordinates and the configured pseudo-positive weight,
- support-positive retained shielded anchors that are not cluster-demoted MAY receive support-rate-weighted bbox/coord supervision,
- cluster-demoted pseudo-positive candidates MUST stay outside bbox/coord supervision groups,
- retained unmatched anchor objects MUST NOT create extra positive desc targets,
- retained unmatched anchor objects MAY remain visible in the final clean prefix as context.

#### Scenario: Support-positive shielded anchor keeps context visibility and partial coord supervision
- **WHEN** an unmatched anchor object is retained as shielded with non-zero explorer support and is not cluster-demoted
- **THEN** it may remain in the edited clean prefix
- **AND** it may still participate in the global rollout-prefix structure CE surface
- **AND** it may receive support-rate-weighted bbox/coord supervision
- **AND** it contributes no extra positive desc CE.

#### Scenario: Cluster-demoted pseudo-positive candidate stays struct-only
- **WHEN** a pseudo-positive candidate loses overlap clustering and is demoted back to shielded
- **THEN** it may remain in the edited clean prefix
- **AND** it may still participate in the global rollout-prefix structure CE surface
- **AND** it contributes no positive bbox/coord or desc supervision.

### Requirement: Stage-2 AB can add matched decoded-box size auxiliaries through `bbox_size_aux`
Stage-2 AB SHALL support optional decoded-box size auxiliaries on the existing
matched geometry path without changing bbox parameterization or decode format.

Normative behavior:

- when `bbox_size_aux.config.log_wh_weight > 0`, the trainer MUST add matched
  log-width/log-height supervision on canonicalized decoded boxes,
- when `bbox_size_aux.config.oversize_penalty_weight > 0`, the trainer MAY add the
  thresholded oversize penalty on decoded boxes for the same context,
- Channel-A and Channel-B applicability MUST remain controlled by the authored
  `channels` field on the `bbox_size_aux` module entry,
- Channel-A provenance MUST remain controlled by
  `bbox_size_aux.application.preset`, with `anchor_only` as the supported
  Channel-A preset,
- `bbox_size_aux` MUST remain separate from `bbox_geo` in the authored pipeline
  so the new size loss is an independently removable plugin module,
- `bbox_size_aux` MUST consume the current four bbox coord slots in the existing
  `xyxy` order rather than introducing a new bbox expression,
- the default canonical Stage-2 profile behavior SHOULD enable only the matched
  `log_wh` term at a small weight and keep `log_area` / `oversize` off.

#### Scenario: Channel-A matched geometry uses the normal coord-group log-size aux
- **GIVEN** a Stage-2 AB config with `bbox_size_aux.channels: [A]`
- **AND** `bbox_size_aux.application.preset: anchor_only`
- **AND** `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** Channel-A computes matched geometry loss from decoded boxes
- **THEN** `bbox_size_aux` contributes `bbox_log_wh` under `loss/coord/*`
- **AND** the same step does not emit legacy `A*` size-aux objective
  atoms.

#### Scenario: Channel-B matched rollout geometry can include log-size aux
- **GIVEN** a Stage-2 AB config with `bbox_size_aux.channels: [B]`
- **AND** `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** Channel-B computes matched rollout geometry loss from decoded boxes
- **THEN** the matched log-width/log-height auxiliary contributes on the same
  matched-clean + FN supervision set
- **AND** unmatched clean extras remain outside positive geometry supervision.

### Requirement: Stage-2 AB distributed step coordination MUST be explicit and shared across channels
When Stage-2 AB runs under DDP, the trainer SHALL use one explicit
distributed-step coordination contract for per-step barriers, rank-symmetric
failure propagation, and optimizer-step metric reduction.

Normative behavior:
- Channel-A and Channel-B MUST use the same coordination ownership model for
  bounded phase barriers and rank-symmetric failure handling.
- The active Stage-2 path MUST NOT maintain multiple independent
  optimizer-step reduction systems for the same training step.
- Any per-step distributed coordination MUST remain bounded and fail clearly
  rather than waiting indefinitely.

#### Scenario: Channel-A and Channel-B share the same bounded coordination contract
- **GIVEN** a Stage-2 AB training run under DDP
- **WHEN** Channel-A and Channel-B each require bounded synchronization within
  the training step
- **THEN** both channels use the same shared coordination contract
- **AND** the job does not depend on separate ad hoc barrier policies for
  otherwise equivalent step-boundary coordination.

### Requirement: Stage-2 AB runs MUST emit executed-runtime provenance in addition to authored config
Stage-2 AB runs SHALL persist artifacts that describe the executed runtime
after launcher and bootstrap mutation, not only the authored typed config.

Normative behavior:
- Stage-2 AB runs MUST emit:
  - `effective_runtime.json`
  - `pipeline_manifest.json`
- Stage-2 AB runs MUST also emit stable train-data provenance and, when eval
  data is configured, eval-data provenance sidecars that describe the resolved
  source identity for the executed run.
- These artifacts MUST describe the executed runtime surfaces that affect
  training behavior, including the resolved pipeline structure and relevant
  post-mutation runtime settings.
- These artifacts complement rather than replace the authored-config record.

#### Scenario: Operator can inspect executed runtime after launch mutation
- **GIVEN** a Stage-2 AB run where launcher or bootstrap logic mutates runtime
  settings derived from the authored config
- **WHEN** the run starts successfully
- **THEN** the output artifacts include an executed-runtime record and
  pipeline manifest
- **AND** an operator can distinguish authored config from executed runtime
  without re-deriving it from code.

### Requirement: Stage-2 AB checkpoint intent MUST distinguish artifact-only from restartable resume
When Stage-2 AB persists checkpoints, the run MUST make checkpoint intent
explicit rather than leaving model-selection artifacts and restartable
checkpoints ambiguous.

Normative behavior:
- `artifact_only` is the compatibility-preserving default mode for
  model-selection artifacts.
- `artifact_only` checkpoints MAY omit optimizer, scheduler, RNG, and
  repo-owned future-affecting runtime state, and MUST NOT claim exact resume
  fidelity.
- `restartable` MUST be an explicit opt-in mode.
- `restartable` checkpoints MUST include, at minimum:
  - model weights,
  - optimizer state,
  - scheduler state,
  - RNG state,
  - trainer state sufficient to restore `global_step`,
  - repo-owned future-affecting Stage-2 runtime state,
  - and restart-sensitive callback state or a recompute-safe equivalent.
- Resume preflight for `restartable` mode MUST fail fast when any required
  artifact is missing or incompatible.
- This requirement extends resume correctness while preserving the existing
  rule that schedule continuity depends on the restored `global_step`.

#### Scenario: Incomplete restartable checkpoint fails before resume
- **GIVEN** a Stage-2 AB run authored for `restartable` checkpoint mode
- **AND** a candidate checkpoint is missing required optimizer, RNG,
  runtime-state, or callback-state artifacts
- **WHEN** resume preflight runs
- **THEN** the run fails fast with actionable guidance
- **AND** it does not proceed as if the checkpoint were restartable.

### Requirement: Stage-2 AB Channel-B uses anchor-rooted rollout triage with legacy K=2 compatibility and default K=4 pseudo-positive evidence
When `custom.trainer_variant: stage2_two_channel`, the canonical Channel-B
contract SHALL build its clean teacher-forced target from rollout evidence
rooted in the anchor clean sequence.

Normative behavior:
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, the canonical
  Channel-B contract uses exactly two rollout views:
  - one anchor rollout using greedy / deterministic decoding,
  - one explorer rollout using stochastic decoding configured under
    `stage2_ab.channel_b.triage_posterior`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, the opt-in
  pseudo-positive contract uses exactly
  `stage2_ab.channel_b.triage_posterior.num_rollouts` total rollout views:
  - one anchor rollout using greedy / deterministic decoding,
  - `num_rollouts - 1` explorer rollouts using the shared stochastic decode
    profile configured under `stage2_ab.channel_b.triage_posterior`,
  - repo-authored default pseudo-positive profiles SHOULD set `num_rollouts`
    to `4`,
- each rollout MUST independently reuse the existing bounded salvage + strict
  record acceptance + bbox-valid filtering + sequential dedup + Hungarian
  matching path,
- GT-backed semantics MUST inherit the existing Channel-B accepted-clean
  Hungarian + gating contract,
- the final positive target MUST be built by editing the anchor clean sequence
  rather than rebuilding a union order,
- explorer-only non-GT-backed objects MUST be treated as dead by default,
- a GT hit found only on one or more explorer views MUST project to
  `recovered_fn`, not to anchor retention,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true` and the anchor view
  fails to complete accepted-clean preparation, the sample MUST be dropped from
  Channel-B training for that step,
- the enabled pseudo-positive contract MUST NOT use the canonical empty-prefix
  fallback for malformed anchor preparation.

#### Scenario: Enabled pseudo-positive drops malformed anchor samples
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** the anchor rollout does not complete accepted-clean preparation for a
  sample
- **THEN** that sample is dropped from Channel-B training for the step
- **AND** the trainer does not fall back to the empty-prefix FN-only path for
  that sample.

### Requirement: Stage-2 AB Channel-B v3-specific knobs are typed and grouped
The Stage-2 AB config SHALL expose v3-specific rollout knobs under
`stage2_ab.channel_b.triage_posterior`.

Normative behavior:
- `stage2_ab.channel_b.triage_posterior` MUST be a typed mapping,
- the mapping MUST accept only:
  - `num_rollouts`
  - `explorer_temperature`
  - `explorer_top_p`
  - `explorer_top_k`
  - `unlabeled_consistent_iou_threshold`
  - `recovered_ground_truth_weight_multiplier`
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, `num_rollouts`
  MUST equal `2`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, `num_rollouts` MUST
  be an integer greater than or equal to `2`,
- unknown keys under `stage2_ab.channel_b.triage_posterior` MUST fail fast.

#### Scenario: Unknown pseudo-positive key fails fast
- **WHEN** a Stage-2 AB config includes an unknown key under
  `stage2_ab.channel_b.pseudo_positive`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the full dotted path of the unknown key.

#### Scenario: Enabled pseudo-positive accepts arbitrary K at or above two total rollouts
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** config loading fails fast unless
  `stage2_ab.channel_b.triage_posterior.num_rollouts >= 2`.

### Requirement: Stage-2 AB arbitrary-K explorer observability remains deterministic and aggregate
When `stage2_ab.channel_b.pseudo_positive.enabled=true`, the Stage-2 AB trainer
SHALL keep legacy single-explorer observability surfaces meaningful by
defining them as deterministic aggregate explorer summaries.

Normative behavior:
- aggregate explorer metric families such as `rollout/explorer/*` MUST
  represent deterministic summaries across valid explorer views rather than one
  arbitrarily chosen explorer view,
- `rollout/explorer/pred_objects`, `rollout/explorer/valid_pred_objects`,
  `rollout/explorer/parse_truncated_rate`,
  `rollout/explorer/gen_new_tokens_mean`,
  `rollout/explorer/gen_new_tokens_p90`,
  `rollout/explorer/near_iou90_any`, and
  `rollout/explorer/near_iou90_same` MUST use mean-over-valid-explorer-view
  aggregation so their semantics remain comparable across different
  `num_rollouts` values,
- explorer decode-profile observability such as `rollout/explorer/temperature`,
  `rollout/explorer/do_sample`, `rollout/explorer/top_p`,
  `rollout/explorer/top_k`, and `explorer_decode_mode` MUST report the shared
  explorer decode profile used for all explorer views,
- `stage2/raw_rollouts` MUST continue to count the total raw rollout
  trajectories produced across the anchor rollout plus all explorer rollouts,
- `rollout/parse_truncated_rate` MUST continue to represent the parse-truncated
  ratio over those total raw rollouts,
- per-sample explorer-local metadata that cannot be losslessly merged, such as
  dead explorer index lists, MUST move to explorer-indexed carriers such as
  `dead_explorer_indices_by_view` rather than overloading the legacy singular
  field.

#### Scenario: Enabled arbitrary-K keeps explorer metrics aggregate rather than arbitrary
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** `stage2_ab.channel_b.triage_posterior.num_rollouts > 2`
- **THEN** legacy `rollout/explorer/*` metrics still have a defined
  deterministic meaning
- **AND** explorer-local metadata is emitted through explorer-indexed carriers
  rather than singular merged fields.

#### Scenario: Enabled pseudo-positive uses independent deterministic association and ratio-based voting per explorer
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** each explorer view is associated to the anchor view independently
  under the canonical one-to-one IoU rule
- **AND** support ratios are computed from those independent deterministic
  associations using the sample's `valid_explorer_count`.

#### Scenario: Single-forward target realization with pseudo-positive enabled
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** all loss terms are derived from a single teacher-forced forward over
  the edited anchor target
- **AND** no second teacher-forced explore payload is required.

#### Scenario: Enabled pseudo-positive does not add desc-side rollout supervision
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** only the selected pseudo-positive anchor subset is removed from
  blanket FP-neutral coord masking
- **AND** the same step does not add pseudo-positive desc CE
- **AND** retained prefix objects may still share the global rollout-prefix
  structure CE surface.

#### Scenario: Enabled pseudo-positive does not use anchor empty-prefix fallback
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** the anchor rollout cannot be recovered into accepted-clean
  preparation
- **THEN** the sample is dropped from Channel-B training for the step
- **AND** the empty-prefix fallback is not used for that sample.

### Requirement: Stage-2 AB can add decoded-box size auxiliaries through `bbox_size_aux`
Stage-2 AB SHALL support optional decoded-box size auxiliaries on the existing
coord-supervised decoded-box path without changing bbox parameterization or
decode format.

Normative behavior:
- when `bbox_size_aux.config.log_wh_weight > 0`, the trainer MUST add
  log-width/log-height supervision on canonicalized decoded boxes for every
  Channel-B coord-supervised object group,
- when `bbox_size_aux.config.oversize_penalty_weight > 0`, the trainer MAY add
  the thresholded oversize penalty on decoded boxes for the same
  coord-supervised groups,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, selected
  pseudo-positive objects MAY participate in `bbox_size_aux` through the same
  coord-supervised group path as matched-clean and FN-injection objects,
- pseudo-positive `bbox_size_aux` targets MUST use the selected anchor object's
  own canonical coordinate bins rather than explorer-consensus geometry,
- Channel-A and Channel-B applicability MUST remain controlled by the authored
  `channels` field on the `bbox_size_aux` module entry,
- `bbox_size_aux` MUST remain separate from `bbox_geo` in the authored pipeline
  so the new size loss is an independently removable plugin module,
- `bbox_size_aux` MUST consume the current four bbox coord slots in the
  existing `xyxy` order rather than introducing a new bbox expression.

#### Scenario: Pseudo-positive size aux uses anchor geometry
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** a selected pseudo-positive anchor object participates in
  `bbox_size_aux`
- **THEN** its target decoded box is derived from that anchor object's own
  canonical coordinates
- **AND** no explorer-consensus geometry target is constructed for
  `bbox_size_aux`.
