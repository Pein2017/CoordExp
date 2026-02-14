## Why

Stage-2 AB configs currently mix deep inheritance, repeated overrides, and broad `custom.extra.*` nesting, which makes it hard to audit run intent from a single downstream file and increases misconfiguration risk. We need a clearer, one-hop config contract so ablation files are self-consistent, reproducible, and professional to maintain as the experiment matrix grows.

## What Changes

- Standardize Stage-2 AB to an Option-A hierarchy under `configs/stage2_ab/`:
  - one shared `base.yaml` with stable defaults,
  - canonical downstream leaves (`prod/{a_only,b_only,ab_mixed}.yaml`, `smoke/{a_only,b_only,ab_mixed}.yaml`),
  - optional additional leaves are allowed only if they also follow the same one-hop + explicit-leaf contract,
  - no multi-hop inheritance chains for leaves.
- Define an explicit base migration:
  - create `configs/stage2_ab/base.yaml` as the canonical Stage-2 AB base by flattening shared defaults currently split across `base_rollout_matching_sft.yaml` and `prod/base.yaml`,
  - make smoke profiles inherit directly from `base.yaml` (no dual-parent `extends` list),
  - retire `base_rollout_matching_sft.yaml`, `prod/base.yaml`, and `base_smoke_runtime.yaml` as canonical profile parents.
- Make each downstream ablation config self-consistent by explicitly declaring high-signal knobs even when inherited defaults exist (model path, run identity, output/log dirs, group learning rates, effective batch, eval/save policy, channel schedule).
- Replace ambiguous catch-all nesting for core rollout knobs by moving Stage-2 rollout keys from `custom.extra.rollout_matching.*` to top-level `rollout_matching.*` using path-only relocation (same subkeys), and reserve `custom.extra` only for truly minor/trivial residual toggles that do not fit established groups.
- Add strict config validation and migration guidance so hierarchy violations, ambiguous key placement, and unknown keys fail fast with actionable errors.
- Update launcher/config wiring to use one shared Python normalization path so both `src/sft.py` and `scripts/train_stage2.sh` consume the same resolved rollout config contract.
- Require launcher preflight to call the same Python resolver used by runtime (via `ConfigLoader.load_training_config(...)` + shared normalization) and consume machine-readable resolved fields, rather than parsing raw YAML rollout keys in bash.
- Scope strict validators to the canonical Stage-2 profile surface (`configs/stage2_ab/prod/*.yaml`, `configs/stage2_ab/smoke/*.yaml`); any legacy/experimental profile that cannot satisfy the new contract must relocate to `configs/stage2_ab/legacy/` before those gates are enabled.
- **BREAKING**: Canonical paths for Stage-2 rollout-related knobs change from `custom.extra.rollout_matching.*` to `rollout_matching.*` with the same subkey names (no alias support, no dual-read).
- **BREAKING**: Stage-2 profiles remove legacy alias support immediately; any legacy Stage-2 rollout key path fails fast with explicit migration guidance.
- **BREAKING**: Stage-2 AB downstream configs must follow one-hop inheritance from `configs/stage2_ab/base.yaml`; deeper inheritance chains are no longer supported for canonical experiment configs.

## Capabilities

### New Capabilities

<!-- None. This change strengthens and clarifies existing Stage-2 AB and rollout config contracts. -->

### Modified Capabilities

- `stage2-ab-training`: Tighten the YAML contract for Stage-2 AB experiment configuration so downstream ablation files are explicit, one-hop, and fail fast on ambiguous hierarchy/key placement.
- `rollout-matching-sft`: Redefine canonical rollout config namespaces to avoid ambiguous `custom.extra.*` ownership, while preserving config-first behavior and deterministic rollout/training semantics.

## Impact

- Affected configs:
  - `configs/stage2_ab/base.yaml` and downstream Stage-2 AB profile leaves under `configs/stage2_ab/prod/` and `configs/stage2_ab/smoke/`.
  - legacy parent files (`configs/stage2_ab/base_rollout_matching_sft.yaml`, `configs/stage2_ab/prod/base.yaml`, `configs/stage2_ab/base_smoke_runtime.yaml`) are retired from canonical profile inheritance.
- Affected loading/validation/wiring:
  - `src/config/schema.py`, `src/config/loader.py`, and Stage-2 launcher wiring in `scripts/train_stage2.sh` / `src/sft.py` where rollout-related config paths are resolved.
- Spec/docs updates:
  - Delta specs for `stage2-ab-training` and `rollout-matching-sft`, plus migration notes for removed legacy key paths.
- Reproducibility/eval validity:
  - The refactor is behavior-preserving for intended defaults, but key-path and hierarchy changes are contract-level and therefore versioned as breaking to avoid silent drift.
