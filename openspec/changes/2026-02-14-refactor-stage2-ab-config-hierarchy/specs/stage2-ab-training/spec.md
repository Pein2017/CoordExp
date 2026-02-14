## ADDED Requirements

### Requirement: Stage-2 AB profile hierarchy is canonical and one-hop
Stage-2 AB experiment profiles under `configs/stage2_ab/` MUST follow a canonical one-hop hierarchy so ablation intent remains auditable from each downstream file.

Normative structure:
- `configs/stage2_ab/base.yaml` MUST be the canonical shared base for Stage-2 AB profile runs.
- Canonical profile leaves under `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml` MUST extend exactly one file, and that file MUST be `../base.yaml`.
- Canonical smoke leaves MUST inline smoke runtime overrides and MUST NOT use dual-parent `extends` lists.
- Canonical profile leaves MUST NOT use multi-hop inheritance chains (e.g., leaf -> intermediate -> base).
- Additional optional Stage-2 profile leaves (outside the canonical trio) are allowed only if they satisfy the same one-hop + explicit-leaf contract.

Validation behavior:
- Config loading for Stage-2 AB profile leaves MUST fail fast when one-hop structure is violated.
- Error messages MUST include actionable migration guidance (expected parent path and offending `extends` chain).
- Strict hierarchy/explicitness validation targets canonical profile directories only (`configs/stage2_ab/prod/*.yaml`, `configs/stage2_ab/smoke/*.yaml`); profiles that cannot satisfy this contract MUST relocate to `configs/stage2_ab/legacy/` before those gates are enabled.
- Any automation that enumerates canonical Stage-2 profiles MUST target only `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml`, and MUST NOT treat `configs/stage2_ab/legacy/*.yaml` as canonical inputs.

#### Scenario: One-hop profile inheritance passes validation
- **WHEN** a Stage-2 AB profile leaf in `configs/stage2_ab/prod/` extends only `../base.yaml`
- **THEN** config loading succeeds for hierarchy validation.

#### Scenario: Multi-hop profile inheritance fails fast
- **WHEN** a Stage-2 AB profile leaf in `configs/stage2_ab/smoke/` extends an intermediate profile file
- **THEN** config loading fails fast with guidance to extend `../base.yaml` directly.

#### Scenario: Dual-parent smoke inheritance fails fast
- **WHEN** a Stage-2 AB smoke profile leaf uses `extends` with two parents (e.g., prod leaf + smoke base)
- **THEN** config loading fails fast with guidance to inline smoke runtime overrides in a one-hop leaf.

#### Scenario: Non-canonical profile is excluded from canonical hierarchy gate
- **WHEN** a Stage-2 AB profile is moved to `configs/stage2_ab/legacy/` (or another non-canonical path outside `configs/stage2_ab/{prod,smoke}/`)
- **THEN** canonical one-hop/explicitness validators are not applied to that file path.

#### Scenario: Legacy directory is excluded from canonical profile discovery
- **WHEN** a config discovery utility scans canonical Stage-2 profiles
- **THEN** it includes only `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml`
- **AND** excludes `configs/stage2_ab/legacy/*.yaml` from canonical run surfaces.

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
Stage-2 AB canonical profile authoring MUST use a grouped rollout namespace outside `custom.extra`, and the loader/runtime MUST normalize this namespace into the trainer-consumed rollout config.

Normative behavior:
- Stage-2 AB profiles MUST author rollout settings under canonical grouped section `rollout_matching.*`.
- For rollout knobs that previously lived under `custom.extra.rollout_matching.*`, canonical migration is path-only relocation to `rollout_matching.*` with unchanged subkey names.
- Before trainer construction, runtime MUST normalize canonical grouped rollout fields into the rollout config object injected into Stage-2 AB / rollout-matching trainers.
- Stage-2 launcher preflight (`scripts/train_stage2.sh`) MUST resolve rollout settings from the same shared normalization contract used by runtime, and MUST NOT maintain a divergent raw-field contract.
- Launcher preflight MUST call the shared Python loader/normalizer (`ConfigLoader.load_training_config(...)` path) and consume machine-readable normalized fields rather than parsing rollout keys directly in bash.
- Launcher preflight machine-readable output MUST be newline-terminated single-line JSON with keys:
  - `rollout_backend` (string),
  - `vllm_mode` (string or null),
  - `server_base_urls` (array of strings; empty allowed when backend/mode does not require server URLs).
- Launcher preflight MUST fail fast (non-zero exit) and MUST NOT launch training when config normalization fails, JSON is invalid, or any required key is missing/typed incorrectly.
- The normalization output MUST preserve existing rollout semantics (backend, server, decoding, repeat-terminate, matching, and packing-related runtime knobs).
- Any legacy Stage-2 rollout key placement under `custom.extra.rollout_matching.*` MUST fail fast with actionable migration guidance to `rollout_matching.*`.
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

#### Scenario: Launcher preflight uses normalized rollout contract
- **WHEN** a Stage-2 AB profile defines rollout settings only under `rollout_matching.*`
- **THEN** `scripts/train_stage2.sh` preflight resolves server/backend settings successfully through shared normalization and does not require `custom.extra.rollout_matching.*` keys.

#### Scenario: Launcher preflight fails on invalid normalization JSON contract
- **WHEN** shared normalization output is invalid JSON or omits required keys/types (`rollout_backend`, `vllm_mode`, `server_base_urls`)
- **THEN** `scripts/train_stage2.sh` exits non-zero and blocks training launch with actionable contract error text.
