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
  - `stage2_ab.coord_ce_weight`, `stage2_ab.coord_el1_weight`, `stage2_ab.coord_ehuber_weight`
  - `stage2_ab.coord_entropy_weight`, `stage2_ab.coord_gate_weight`, `stage2_ab.text_gate_weight`
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
  - `name`, `enabled`, `weight`, `channels`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `config` MUST include exactly the allowlisted keys for the referenced module:
  - missing required keys MUST fail fast (no implicit defaults),
  - unknown keys MUST fail fast (no escape-hatch aliases).

#### Scenario: Missing module spec field fails fast
- **WHEN** a Stage-2 AB config omits `stage2_ab.pipeline.objective[i].channels`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the missing required field and its full path.

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)
Stage-2 AB pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `coord_reg.config` MUST accept only canonical keys, including:
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
- **WHEN** `stage2_ab.pipeline.objective[*].name=bbox_geo`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates `smoothl1_weight` is the only accepted key.

### Requirement: Stage-2 AB supports text_gate via coord_reg module config
Stage-2 AB MUST support `text_gate` as part of `coord_reg` with a typed weight:
- `stage2_ab.pipeline.objective[*].config.text_gate_weight`

Normative behavior:
- `text_gate_weight > 0` MUST introduce a non-zero `text_gate` contribution when coord-vocab mass appears at supervised text positions (subject to registry masking).

#### Scenario: Non-zero text_gate_weight is effective
- **WHEN** `coord_reg.config.text_gate_weight > 0`
- **AND** the model places substantial coord-vocab probability mass at supervised `type=struct|desc` positions
- **THEN** the emitted `text_gate` objective atom increases relative to the same run with `text_gate_weight = 0`:
  - Channel-A: `loss/A2_coord/text_gate`
  - Channel-B: `loss/B_coord/text_gate`
- **AND** the increase is attributable to the `text_gate` sub-term inside `coord_reg`.

### Requirement: Coord diagnostics are attributed to A1 vs A2 logits in Channel-A
Stage-2 AB SHALL provide coord-distribution monitors that let operators compare the GT-anchor logits (`A1`) versus the final softctx logits (`A2`) on the same GT coord-token positions.

Normative behavior:
- When `coord_diag` diagnostics module is enabled (non-zero effective weight for the current channel), the trainer MUST emit coord distribution monitor keys with explicit forward provenance:
  - `coord_diag/A1/*`: computed from the Channel-A A1 logits (`logits_a1`, `it==0`).
  - `coord_diag/A2/*`: computed from the Channel-A final softctx logits (`it==n_softctx_iter-1`), emitted only when `n_softctx_iter > 1`.
- The monitor set SHOULD include at least:
  - `coord_diag/<prov>/acc_top5`
  - `coord_diag/<prov>/p_gt_mean`
  - `coord_diag/<prov>/expected_bin_mae`
- These diagnostics MUST NOT affect the training objective (they are monitors only).
- The trainer MUST NOT emit ambiguous bare `coord_diag/*` keys for these monitors in Stage-2 AB logs.

#### Scenario: Channel-A diagnostics expose A1 and A2 provenance
- **WHEN** Stage-2 AB runs with `n_softctx_iter > 1` and `coord_diag` enabled
- **THEN** emitted coord diagnostics include `coord_diag/A1/*` and `coord_diag/A2/*`
- **AND** no ambiguous bare `coord_diag/*` provenance-free keys are emitted.

### Requirement: Stage-2 AB profile hierarchy is canonical and one-hop
Stage-2 AB experiment profiles under `configs/stage2_two_channel/` MUST follow a canonical one-hop hierarchy so ablation intent remains auditable from each downstream file.

Normative structure:
- `configs/stage2_two_channel/base.yaml` MUST be the canonical shared base for Stage-2 AB profile runs.
- Canonical profile leaves under `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml` MUST extend exactly one file, and that file MUST be `../base.yaml`.
- Canonical smoke leaves MUST inline smoke runtime overrides and MUST NOT use dual-parent `extends` lists.
- Canonical profile leaves MUST NOT use multi-hop inheritance chains (e.g., leaf -> intermediate -> base).
- Additional optional Stage-2 profile leaves (outside the canonical trio) are allowed only if they satisfy the same one-hop + explicit-leaf contract.

Validation behavior:
- Config loading for Stage-2 AB profile leaves MUST fail fast when one-hop structure is violated.
- Error messages MUST include actionable migration guidance (expected parent path and offending `extends` chain).
- Strict hierarchy/explicitness validation targets the canonical profile directories (`configs/stage2_two_channel/prod/*.yaml`, `configs/stage2_two_channel/smoke/*.yaml`) and is expected to pass for all files in those paths.
- Any automation that enumerates canonical Stage-2 profiles MUST target only `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml`.

#### Scenario: One-hop profile inheritance passes validation
- **WHEN** a Stage-2 AB profile leaf in `configs/stage2_two_channel/prod/` extends only `../base.yaml`
- **THEN** config loading succeeds for hierarchy validation.

#### Scenario: Multi-hop profile inheritance fails fast
- **WHEN** a Stage-2 AB profile leaf in `configs/stage2_two_channel/smoke/` extends an intermediate profile file
- **THEN** config loading fails fast with guidance to extend `../base.yaml` directly.

#### Scenario: Dual-parent smoke inheritance fails fast
- **WHEN** a Stage-2 AB smoke profile leaf uses `extends` with two parents (e.g., prod leaf + smoke base)
- **THEN** config loading fails fast with guidance to inline smoke runtime overrides in a one-hop leaf.

#### Scenario: Canonical profile discovery is scoped to prod/smoke
- **WHEN** a config discovery utility scans canonical Stage-2 profiles
- **THEN** it includes only `configs/stage2_two_channel/prod/*.yaml` and `configs/stage2_two_channel/smoke/*.yaml`

### Requirement: Stage-2 AB downstream profiles explicitly pin high-signal knobs
Each canonical Stage-2 AB profile leaf MUST explicitly declare high-signal run and ablation knobs so the file is self-consistent without traversing parent configs.

Required explicit leaf fields:
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
- `stage2_ab.n_softctx_iter`

Rationale for strict explicitness:
- The LR trio (`training.learning_rate`, `training.vit_lr`, `training.aligner_lr`) is treated as MUST for canonical leaves to avoid hidden optimizer-group drift across ablations.

Validation behavior:
- Canonical Stage-2 AB profile loading MUST fail fast if any required explicit field is missing from the leaf profile.
- Error text MUST identify missing fields by full key path.

#### Scenario: Downstream profile with explicit high-signal fields is accepted
- **WHEN** a Stage-2 AB profile leaf includes all required explicit high-signal keys
- **THEN** config loading succeeds and the profile is considered self-consistent.

#### Scenario: Missing explicit run identity fails fast
- **WHEN** a Stage-2 AB profile leaf omits `training.run_name`
- **THEN** config loading fails fast and reports `training.run_name` as missing.

#### Scenario: Missing explicit model path fails fast
- **WHEN** a Stage-2 AB profile leaf omits `model.model`
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

### Requirement: Channel-A performs iterative ST/soft self-context via N× full-forwards (no rollout)
Channel-A MUST implement iterative ST/soft self-context using `stage2_ab.n_softctx_iter` full forward passes:
- `stage2_ab.n_softctx_iter` MUST be an integer `>= 1`.
- The iteration index `m` ranges over `m = 0..n_softctx_iter-1`.
- For `n_softctx_iter = 1`, Channel-A MUST reduce to a single teacher-forced forward (pure TF baseline).
- For `n_softctx_iter > 1`, Channel-A MUST:
  - Run a teacher-forced forward to obtain logits for coord slots.
  - Construct coord-slot context embeddings from coord-token distributions, with ST embedding as the default
    (hard forward / soft backward) and soft expectation embedding as an explicit alternative.
  - Update coord-slot embeddings and re-run a full forward, repeating until `m = n_softctx_iter-1`.

The trainer MUST use the **final-iteration** logits `z^(n_softctx_iter-1)` for geometry decoding and loss computation.

Gradient semantics (normative):
- The default gradient mode MUST be fully unrolled:
  - `stage2_ab.softctx_grad_mode: "unroll"` MUST record gradients through all softctx iterations
  - and MUST NOT detach the expected coord embeddings used to update coord-slot inputs.
- An explicit EM-style fallback MAY be provided for ablations:
  - `stage2_ab.softctx_grad_mode: "em_detach"` MAY detach the expected coord embeddings (or equivalently disable grad recording for embedding updates) to reduce memory.

Causal shift convention (normative):
- For a causal LM, logits at sequence position `t` predict the token at position `t+1` (the standard shift used for CE).
- Therefore, when updating a coord-slot embedding at token position `p`, the trainer MUST use the coord distribution read from logits at position `p-1` (for `p > 0`) from the previous iteration.

#### Scenario: Unroll mode does not detach expected coord embeddings
- **GIVEN** `stage2_ab.n_softctx_iter: 2`
- **AND** `stage2_ab.softctx_grad_mode: unroll`
- **WHEN** Channel-A runs the softctx loop
- **THEN** it executes two full forward passes
- **AND** it does not detach the expected coord embeddings used to update coord-slot inputs.

#### Scenario: n_softctx_iter=2 runs exactly two forwards and uses final logits
- **GIVEN** `n_softctx_iter: 2`
- **WHEN** Channel-A runs on a batch
- **THEN** it executes two full forward passes
- **AND** it computes geometry loss using logits from the second forward only.

### Requirement: Channel-A forward path is compatible with Qwen3-VL multimodal semantics
For Qwen3-VL (dense) models, each forward MUST provide **exactly one** of `input_ids` or `inputs_embeds`.

When Channel-A uses `inputs_embeds` to implement iterative soft self-context:
- The initial embeddings MUST be computed by **calling** the model's input embedding module on the teacher-forced token ids (so embedding forward hooks apply); they MUST NOT be assembled by indexing `.weight[...]`.
- Expected coord embeddings used in soft self-context MUST also be computed via the embedding module applied to coord token ids (not via `.weight[...]`).
- The trainer MUST modify only the coord-slot embeddings and MUST NOT perturb multimodal placeholder token embeddings (e.g., image/video placeholders) **bitwise**.
- For multimodal batches, each softctx iteration MUST provide `inputs_embeds` that still contains placeholder token embeddings at placeholder positions (so Qwen3-VL can insert visual features).
  - Normative minimum: each iteration MUST rebuild a fresh base `inputs_embeds` from teacher-forced `input_ids` (embedding-module call) and then scatter-update only coord-slot rows.
  - It MUST NOT reuse embeddings after model-internal feature insertion for subsequent iterations.
- Channel-A MAY retain `input_ids` in the batch for label/slot bookkeeping, but MUST NOT pass `input_ids` into `model(...)` when `inputs_embeds` is provided.
- The forward call MUST remain compatible with multimodal feature insertion (i.e., placeholder token count must match provided visual feature length).
- When using padding-free packing with Qwen3-VL, the trainer MUST pass the 4-row `position_ids` format (`[text_position_ids; mRoPE]`) consistently for every iteration forward.
- Cache safety (normative): Channel-A training forwards MUST set `use_cache=False` and MUST NOT pass or reuse `past_key_values` across iterations. Carrying KV cache across softctx iterations is forbidden.

#### Scenario: Multimodal forward does not fail under iterative inputs_embeds
- **GIVEN** a multimodal batch with `pixel_values` and valid placeholder tokens
- **AND** Channel-A runs with `n_softctx_iter > 1`
- **WHEN** the model is forwarded with `inputs_embeds` (and `input_ids=None`)
- **THEN** the forward pass succeeds without placeholder-count mismatch errors.

### Requirement: Channel-B reuses rollout-matching infra (strict parse/match + mandatory FN append)
Channel-B MUST reuse the rollout-matching pipeline:
- Rollout generation MUST be configured under `rollout_matching` (backend `hf` or `vllm`).
- Parsing MUST be strict and token-aligned (no re-tokenization of the rollout prefix), except for a possible token-internal cut on the final token where the trainer MAY retokenize only the final token as a shorter tokenization that decodes exactly to the original substring.
- Matching MUST be deterministic.
- FN append MUST be performed (mandatory) to ensure all GT objects are present in `Y_train`.

#### Scenario: Rollout prefix + FN append produces a valid teacher-forced target
- **GIVEN** Channel-B is selected and rollout generation succeeds
- **WHEN** the trainer builds `Y_train` for teacher forcing
- **THEN** `Y_train` contains the rollout prefix (suffix-trimmed only) followed by a JSON-only FN append fragment.

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

#### Scenario: Default desc-first behavior is preserved in both channels
- **GIVEN** `custom.object_field_order` is omitted
- **WHEN** Channel-A or Channel-B serializes object payloads
- **THEN** payloads remain `desc` before the concrete geometry key (`bbox_2d` or `poly`).

### Requirement: Stage-2 object instance ordering contract is unchanged
`custom.object_field_order` SHALL NOT modify stage-2 object instance ordering behavior.

Normative behavior:
- Object sequence remains determined by existing pipeline semantics (GT order, parsed rollout appearance order, and current matching/index continuation logic).
- Only intra-object field order is configurable.

#### Scenario: geometry-first does not change rollout appearance order handling
- **GIVEN** rollout parsed objects appear in a specific raw-text order
- **AND** `custom.object_field_order: geometry_first`
- **WHEN** Stage-2 performs matching and FN append
- **THEN** parsed predicted order remains the same as raw-text appearance
- **AND** only field order inside serialized object payloads differs.

### Requirement: Channel-B invalid rollouts fall back deterministically (no silent skips)
When Channel-B is selected and a rollout response cannot be parsed into an append-ready JSON prefix (e.g., there is no top-level `{` in the response), the trainer MUST:
- Mark the rollout as invalid for that sample and expose an `invalid_rollout` counter/metric.
- Fall back to a canonical empty JSON prefix of exactly `{` (as token ids) so FN append can proceed.
- Treat the rollout as containing zero valid predicted objects for matching/supervision purposes.
- Continue training that sample by FN-appending all GT objects and running the normal teacher-forced loss (i.e., the sample is not skipped and the trainer does not raise).

This fallback MUST be deterministic given the same `response_token_ids` and tokenizer.

Normative minimum: this requirement MUST at least cover the case where the rollout response contains no top-level `{` (equivalent to the current strict parser’s “completely malformed rollout” condition).

#### Scenario: Missing opening brace falls back to `{` and trains
- **GIVEN** Channel-B is selected for a sample
- **AND** the rollout response text contains no top-level `{`
- **WHEN** the trainer parses the rollout for matching
- **THEN** it marks the rollout invalid for that sample
- **AND** it uses `{` as the prefix and FN-appends all GT objects
- **AND** the sample is still included in teacher-forced training.

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

### Requirement: Hybrid objective preserves JSON structure CE and adds bbox geometry losses
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- **Token CE anchor at A1**:
  - CE on non-coord tokens MUST be computed from the teacher-forced logits of the first forward (`z^(0)`; GT context).
  - Coord tokens MUST NOT contribute to CE (they are masked out), to avoid double-supervision.
- **Geometry + distribution regularizers from final softctx logits**:
  - Geometry losses and any distribution-level losses MUST be computed from the final-iteration logits `z^(n_softctx_iter-1)`.

Channel-B:
- **Matched-only geometry**:
  - Geometry losses MUST be computed only for matched `(pred_i -> gt_j)` pairs.
  - Unmatched predicted objects (FP under Hungarian) MUST NOT receive geometric gradients.
- **Stop-neutral CE**:
  - Channel-B CE MUST NOT supervise the stop/continue decision.
  - The trainer MUST mask CE on:
    - the top-level JSON closing brace `}` (the brace that closes the outermost assistant JSON object), and
    - `<|im_end|>` (the only turn-end token).
  - Top-level brace identification MUST be robust and deterministic:
    - the trainer MUST identify the token position of the `}` that closes the **outermost** JSON object in the rendered Channel-B teacher-forced assistant span,
    - and MUST NOT rely on “the last `}` token id in the whole sequence” without verifying it corresponds to the outermost close brace of the assistant JSON.
    - A compliant approach is to decode the assistant-span token pieces and locate the outermost close brace via a brace-depth scan, then map the character span back to token positions.
- **FN append always**:
  - FN objects MUST be appended to the B3 target so they are supervised even when they were missing from rollout.
  - If `N_valid_pred == 0` after strict validation, the trainer MUST treat all GT objects as FN (canonical GT order) and append them, which is equivalent to “FN append all GT objects”.
  - Optional weak correction: when `N_drop_invalid > 0`, the trainer MAY upweight Channel-B’s B3 structure-token CE weights to discourage “escaping supervision via invalid instances”.
    - This upweight MUST be controlled by `stage2_ab.pipeline.objective[name=token_ce].config.rollout_drop_invalid_struct_ce_multiplier` (float).
    - The multiplier MUST default to `1.0` (no effect) and MUST be constrained to a safe range `[1.0, 4.0]` (clamp or fail fast).
    - “Structure-token CE weights” refers to Channel-B CE-supervised tokens excluding:
      - coord tokens,
      - desc value tokens, and
      - stop-neutral masked token positions (`}` and `<|im_end|>`).

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
  - the final iteration `z^(n_softctx_iter-1)` in Channel-A, and
  - the Channel-B teacher-forced logits under the rollout scaffold (or B2 refined logits when enabled).

Loss form (normative):
- The trainer MUST use SmoothL1 (Huber) + CIoU as the bbox regression terms.
- The trainer MUST NOT use GIoU in Stage-2 AB.

Numerical stability (normative):
- The trainer MUST canonicalize predicted boxes before CIoU:
  - `(x1,x2) := (min(x1,x2), max(x1,x2))`, `(y1,y2) := (min(y1,y2), max(y1,y2))`.
- The trainer MUST ensure the geometry losses do not produce NaNs/Infs, including early training when predictions are degenerate.

Efficiency rule (normative):
- If Channel-B has no valid matched pairs for a sample/batch, the trainer MUST skip the B2 forward (geo-only) and run B3 only.

#### Scenario: Channel-B is stop-neutral for `}` and `<|im_end|>`
- **GIVEN** Channel-B builds a teacher-forced target that ends with a top-level `}` followed by `<|im_end|>`
- **WHEN** the trainer builds CE labels/weights for Channel-B
- **THEN** it masks CE on that top-level `}` token position
- **AND** it masks CE on `<|im_end|>`.

#### Scenario: Dropped invalid instances may upweight B3 structure CE
- **GIVEN** a Channel-B rollout with `N_drop_invalid > 0`
- **AND** `stage2_ab.pipeline.objective[name=token_ce].config.rollout_drop_invalid_struct_ce_multiplier: 1.5`
- **WHEN** Channel-B builds CE weights for structure tokens in B3
- **THEN** it MAY multiply the structure-token CE weights by `1.5` (bounded) for that sample/window.

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

#### Scenario: Channel-B geometry includes matched and FN but excludes FP
- **GIVEN** Channel-B where matching yields non-empty matched, FP, and FN sets
- **WHEN** Channel-B losses are computed
- **THEN** geometry loss is accumulated for matched and FN-injected objects
- **AND** FP objects contribute zero geometry loss.

#### Scenario: Start key avoids collision even when highest key object is invalid
- **GIVEN** retained rollout prefix contains keys `object_2` (valid) and `object_7` (invalid and dropped by strict validation)
- **WHEN** FN entries are injected
- **THEN** `max_object_index_in_prefix` is `7`
- **AND** FN key assignment starts at `object_8`.

#### Scenario: Closure-supervision brace target is the same brace used for injection
- **GIVEN** Channel-B injects FN entries before the outermost close brace resolved by brace-depth scan
- **WHEN** CE masks are produced
- **THEN** that same outermost close brace token position remains CE-supervised
- **AND** `<|im_end|>` remains CE-supervised.

#### Scenario: CE masking follows matched/FP/FN policy
- **GIVEN** Channel-B contains one matched object, one FP object, and one FN-injected object
- **WHEN** CE weights are materialized
- **THEN** matched structure tokens are supervised while matched desc tokens are masked
- **AND** FP structure/desc/coord tokens are all masked
- **AND** FN-injected structure and desc tokens are supervised.

#### Scenario: Channel-B supervises top-level closure and `<|im_end|>`
- **GIVEN** Channel-B builds a teacher-forced target that ends with a top-level `}` followed by `<|im_end|>`
- **WHEN** the trainer builds CE labels/weights for Channel-B
- **THEN** it keeps CE supervision on that top-level `}` token position
- **AND** it keeps CE supervision on `<|im_end|>`.

#### Scenario: Stop-neutral masking is not applied
- **GIVEN** Stage-2 AB Channel-B configuration
- **WHEN** CE masks are constructed for Channel-B
- **THEN** top-level `}` and `<|im_end|>` are not masked out by any stop-neutral branch
- **AND** FP-neutral masking remains limited to unmatched predicted object spans.

#### Scenario: Legacy stop-neutral config keys fail fast
- **GIVEN** Stage-2 AB config includes legacy stop-neutral keys under Channel-B
- **WHEN** trainer configuration is validated
- **THEN** startup fails fast before training
- **AND** the error indicates stop-neutral knobs are unsupported under the typed contract.

#### Scenario: Closure marker resolution failure is dropped and counted
- **GIVEN** a Channel-B sample where the trainer cannot deterministically locate the outermost `}` / `<|im_end|>` marker positions (e.g., truncation)
- **WHEN** the trainer constructs CE labels/weights for Channel-B
- **THEN** it drops the sample from Channel-B supervision for that step
- **AND** it increments `stage2_ab/channel_b/closure_supervision/N_drop`.

#### Scenario: No valid predictions fall back to canonical GT order
- **GIVEN** strict validation yields `N_valid_pred == 0`
- **WHEN** Channel-B builds `y_GT_reordered` for B3
- **THEN** it sets `y_GT_reordered := y_GT_canonical`
- **AND** this is equivalent to appending all GT objects as FN-supervised targets.

#### Scenario: Out-of-range struct CE multiplier is handled safely
- **GIVEN** `stage2_ab.pipeline.objective[name=token_ce].config.rollout_drop_invalid_struct_ce_multiplier` is outside `[1.0, 4.0]`
- **WHEN** trainer parses Channel-B config
- **THEN** it clamps the value into `[1.0, 4.0]` or fails fast (implementation choice)
- **AND** it MUST NOT run with an effective multiplier outside the safe range `[1.0, 4.0]`.

#### Scenario: B2 forward is skipped when there are no valid matched pairs
- **GIVEN** Channel-B sample/batch has zero valid matched pairs
- **WHEN** trainer executes Channel-B steps
- **THEN** it skips B2 geo-only forward
- **AND** runs B3 only.

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

#### Scenario: Removed reordered-gt knob fails fast
- **WHEN** a config includes `stage2_ab.channel_b.reordered_gt_sft`
- **THEN** config loading fails fast with actionable removal guidance.

#### Scenario: Removed desc-ce-weight knob fails fast
- **WHEN** a config includes `stage2_ab.channel_b.desc_ce_weight_matched`
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
  - Channel-A (self-context): `loss/A2_coord/{coord_soft_ce,coord_w1}`
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
- **AND** both contribute to the coord_reg objective.

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
- **WHEN** config is materialized through one-hop inheritance from `../base.yaml`
- **THEN** the leaf explicitly overrides effective Stage-2 loss weights with the canonical prod values above.

#### Scenario: Canonical smoke leaves inherit base CIoU/soft-CE/W1 defaults
- **GIVEN** a canonical Stage-2 profile leaf under `configs/stage2_two_channel/smoke/*.yaml`
- **WHEN** config is materialized through one-hop inheritance from `../base.yaml`
- **THEN** effective Stage-2 loss defaults include canonical base CIoU downweight and non-zero soft-CE/W1 terms.

### Requirement: Stage-2 two-channel training supports a config-declared objective and diagnostics pipeline
When `custom.trainer_variant: stage2_two_channel`, the system SHALL support a
YAML-declared module pipeline for the Stage-2 two-channel training objective and diagnostics.

Normative behavior:
- If `stage2_ab.pipeline` is provided, the trainer MUST interpret it according to the
  `teacher-forcing-objective-pipeline` capability.
- If `stage2_ab.pipeline` is absent, the trainer MUST construct and use a default pipeline that preserves the current
  Stage-2 two-channel objective semantics, by resolving to the **Default Pipeline Manifest** defined below.
- Pipeline parsing and module resolution MUST be strict and MUST fail fast on unknown module names.
- The pipeline identity (module list + checksum) MUST be recorded in logs for reproducibility.

#### Default Pipeline Manifest (when `stage2_ab.pipeline` is omitted)

To eliminate ambiguity, the “pipeline omitted” behavior is defined as resolving to the following manifest.

Normative default objective modules (ordered):
1) `token_ce`
2) `bbox_geo`
3) `coord_reg`

Normative default diagnostics modules (ordered):
1) `coord_diag`

Normative default module weights (all objective modules):
- `weight = 1.0`

Normative default module configs (effective values):
- `token_ce`:
  - Channel-A (Expectation):
    - `desc_ce_weight = stage2_ab.desc_ce_weight` (default `1.0`)
    - `self_context_struct_ce_weight = stage2_ab.fmt_struct_ce_weight` (default `0.1`)
  - Channel-B (Rollout):
    - `rollout_fn_desc_weight = stage2_ab.desc_ce_weight` (default `1.0`)
    - `rollout_matched_prefix_struct_weight = 1.0`
    - `rollout_drop_invalid_struct_ce_multiplier = stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` (default `1.0`)
- `bbox_geo`:
  - `smoothl1_weight = stage2_ab.bbox_smoothl1_weight` (default `1.0`)
  - `ciou_weight = stage2_ab.bbox_ciou_weight` (default `1.0`)
  - decode mode is controlled by `stage2_ab.coord_decode_mode` (default `exp`)
- `coord_reg`:
  - Coord-shape terms (rollout-context; enabled only when `custom.coord_soft_ce_w1.enabled: true`):
    - `soft_ce_weight = custom.coord_soft_ce_w1.soft_ce_weight` (default `1.0`)
    - `w1_weight = custom.coord_soft_ce_w1.w1_weight` (default `1.0`)
    - `temperature = custom.coord_soft_ce_w1.temperature` (default `1.0`)
    - `target_sigma = custom.coord_soft_ce_w1.target_sigma` (default `2.0`)
    - `target_truncate = custom.coord_soft_ce_w1.target_truncate` (default `null`)
  - Coord-shape terms (Channel-A self_context regularizer; default-on small weight):
    - `self_context_soft_ce_weight = custom.coord_soft_ce_w1.soft_ce_weight` when enabled, else `0.05`
  - Additional coord terms (Stage-2 Two-Channel typed knobs):
    - `coord_ce_weight = stage2_ab.coord_ce_weight` (default `0.0`)
    - `coord_el1_weight = stage2_ab.coord_el1_weight` (default `0.0`)
    - `coord_ehuber_weight = stage2_ab.coord_ehuber_weight` (default `0.0`)
    - `coord_huber_delta = stage2_ab.coord_huber_delta` (default `0.001`)
    - `coord_entropy_weight = stage2_ab.coord_entropy_weight` (default `0.0`)
    - `coord_gate_weight = stage2_ab.coord_gate_weight` (default `0.0`)
    - `text_gate_weight = stage2_ab.text_gate_weight` (default `0.0`)

Informative YAML expression (conceptual; values shown are symbolic):
```yaml
stage2_ab:
  # pipeline omitted => resolves to the following manifest:
  # pipeline:
  #   objective:
  #     - {name: token_ce, weight: 1.0}
  #     - {name: bbox_geo, weight: 1.0}
  #     - {name: coord_reg, weight: 1.0}
  #   diagnostics:
  #     - {name: coord_diag}
```

#### Scenario: Pipeline omitted uses current (full_idea-aligned) defaults
- **WHEN** a Stage-2 Two-Channel config does not define `stage2_ab.pipeline`
- **THEN** training starts successfully
- **AND** the Stage-2 Two-Channel objective matches the Default Pipeline Manifest (and is aligned to `full_idea`)
- **AND** canonical `loss/<component>` keys are emitted (per `teacher-forcing-unified-loss-registry`)

#### Scenario: Pipeline is applied when provided
- **WHEN** a Stage-2 Two-Channel config defines `stage2_ab.pipeline` with at least one objective module
- **THEN** the trainer uses that declared module pipeline
- **AND** the resolved module list and checksum are logged.

#### Scenario: Pipeline mode rejects duplicated flat objective knobs
- **WHEN** a Stage-2 Two-Channel config defines `stage2_ab.pipeline`
- **AND** the config also defines any objective-affecting flat knobs used by the default manifest (e.g.,
  `stage2_ab.desc_ce_weight`, `stage2_ab.fmt_struct_ce_weight`, `stage2_ab.bbox_smoothl1_weight`, `stage2_ab.bbox_ciou_weight`,
  `stage2_ab.coord_ce_weight`, `stage2_ab.coord_entropy_weight`, `stage2_ab.coord_gate_weight`,
  `stage2_ab.text_gate_weight`,
  `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier`, or `custom.coord_soft_ce_w1.*`)
- **THEN** config validation fails fast with guidance to move those values into the declared module configs.

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

Normative minimum module names (initial set; may be extended):
- Objective modules:
  - `token_ce` (masked/weighted CE per unified loss registry; includes EOS-enforced closure supervision)
  - `bbox_geo` (bbox SmoothL1 + CIoU on decoded boxes; FP-neutral in rollout context)
  - `coord_reg` (coord distribution regularizers: coord CE / softCE / W1 / entropy / gate / EL1/EHuber)
- Diagnostics modules:
  - `coord_diag` (coord distribution diagnostics; best-effort)

Normative behavior:
- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available Stage-2 Two-Channel module names.

#### Scenario: Unknown Stage-2 two-channel module names fail fast
- **WHEN** `stage2_ab.pipeline` references an objective module name not present in the Stage-2 registry
- **THEN** trainer initialization fails fast
- **AND** the error includes the unknown name and allowed module names.

### Requirement: Stage-2 Two-Channel module configs are strict and typed
Stage-2 Two-Channel SHALL validate module `config` payloads strictly so experiments are reproducible and fail fast on schema
drift.

Normative behavior:
- Module `config` mappings MUST be validated at trainer initialization (before training starts).
- Unknown keys in a module `config` MUST fail fast with actionable diagnostics listing allowed keys.

Normative config schemas (minimum set; may be extended):
- `token_ce.config`:
  - `desc_ce_weight: float` (default: `1.0`)
  - `self_context_struct_ce_weight: float` (default: `0.1`)
  - `rollout_fn_desc_weight: float` (default: `1.0`)
  - `rollout_matched_prefix_struct_weight: float` (default: `1.0`)
  - `rollout_drop_invalid_struct_ce_multiplier: float` (default: `1.0`)
- `bbox_geo.config`:
  - `smoothl1_weight: float` (default: `1.0`)
  - `ciou_weight: float` (default: `1.0`)
- `coord_reg.config`:
  - `coord_ce_weight: float` (default: `0.0`)
  - `coord_el1_weight: float` (default: `0.0`)
  - `coord_ehuber_weight: float` (default: `0.0`)
  - `coord_huber_delta: float` (default: `0.001`)
  - `coord_entropy_weight: float` (default: `0.0`)
  - `coord_gate_weight: float` (default: `0.0`)
  - `text_gate_weight: float` (default: `0.0`)
  - `soft_ce_weight: float` (default: `0.0`)
  - `self_context_soft_ce_weight: float` (default: `0.05`)
  - `w1_weight: float` (default: `0.0`)
  - `temperature: float` (default: `1.0`)
  - `target_sigma: float` (default: `2.0`)
  - `target_truncate: int|null` (default: `null`)
- `coord_diag.config`:
  - (no required keys; implementations MAY accept a strict subset of the above for convenience, but MUST document them)

Note:
- When `stage2_ab.pipeline` is omitted, the effective defaults for the Stage-2 Two-Channel objective are defined by the Default
  Pipeline Manifest above (which sources values from the typed Stage-2 Two-Channel flat schema keys and `custom.coord_soft_ce_w1`
  as applicable).

#### Scenario: Unknown Stage-2 module config keys fail fast
- **WHEN** a Stage-2 module config includes a key outside its strict allowlist
- **THEN** initialization fails fast before the first training step
- **AND** diagnostics include the invalid key and allowed keys.

### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract
Stage-2 Two-Channel training SHALL implement loss naming and masking semantics per the `teacher-forcing-unified-loss-registry`
capability.

Normative behavior:
- Stage-2 Two-Channel MUST build token-type masks and object-subset masks according to the registry contexts:
  - Channel-A uses `context=gt` for CE anchoring and `context=self_context` for geometry and optional format/closure stabilization.
  - Channel-B uses `context=rollout` with FP-neutral + EOS-enforced semantics.
- When the module pipeline is enabled, objective/diagnostics modules MUST emit metric keys consistent with the
  registry’s canonical component names.

#### Scenario: Channel-B remains FP-neutral and EOS-enforced
- **WHEN** Stage-2 Two-Channel runs a Channel-B update step
- **THEN** FP spans do not contribute to CE or geometry losses
- **AND** top-level closure `}` and `<|im_end|>` remain supervised.

### Requirement: Stage-2 Two-Channel exposes ST bridge knobs as typed YAML config
Stage-2 Two-Channel SHALL expose config knobs to enable Straight-Through (ST) behavior for:
- Channel-A coord-slot self-context embeddings, and
- geometry coord decode.

Normative behavior:
- Config MUST be expressed under the typed Stage-2 Two-Channel namespace (`stage2_ab.*`) and MUST be strict (unknown keys fail).
- When ST knobs are omitted, defaults MUST preserve current behavior (ST embeddings + expectation decode).
- When ST knobs are enabled, the forward/backward behavior MUST follow the ST semantics defined by
  `teacher-forcing-unified-loss-registry`.

Normative key names:
- `stage2_ab.coord_ctx_embed_mode: soft|st|hard` (default: `st`)
- `stage2_ab.coord_decode_mode: exp|st` (default: `exp`)

Normative mapping / identity:
- The resolved values MUST feed a single internal enum for embedding and decode behavior.
- The resolved values MUST be included in the pipeline identity checksum (so ST-vs-soft differences are auditable even
  when the pipeline module list is unchanged).

#### Scenario: ST can be enabled for Channel-A without changing CE anchoring
- **WHEN** `stage2_ab.coord_ctx_embed_mode=st` is enabled for Stage-2 Channel-A
- **THEN** Channel-A CE remains anchored to `context=gt` logits
- **AND** only coord-slot context embeddings follow ST semantics.

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
  - `rollout/mAP` (float; COCO `bbox_AP` = AP@[.50:.95])
- The trainer MUST NOT emit additional COCO summary metric keys during eval-step (e.g., `rollout/bbox_AP50`,
  `rollout/bbox_AR100`, `rollout/segm_*`) beyond `rollout/mAP`. Full metric reports remain available via the offline
  evaluation pipeline.
- If COCO eval fails unexpectedly (missing dependencies, invalid records, etc.), the trainer MUST:
  - emit `rollout/mAP=0.0` (so the key is always present when enabled), and
  - surface the failure as a warning (not silent).

#### Scenario: Stage-2 two-channel eval reports mAP
- **GIVEN** `rollout_matching.eval_detection.enabled=true`
- **WHEN** `eval_step` runs
- **THEN** `rollout/mAP` is present in the eval metrics payload
