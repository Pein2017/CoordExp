## Context

Stage-2 AB currently uses a mixed config contract across `training`, `stage2_ab`, and `custom.extra.rollout_matching`, with leaf configs inheriting through multiple layers. This makes it difficult to verify ablation intent from a single downstream file, and increases drift risk when launcher preflight, config loader, and trainer runtime evolve at different speeds.

Current data/control flow is:
- **data**: JSONL paths and sample caps from `custom.*`
- **transforms/packing**: runtime packing knobs from `training.*` and rollout decode knobs from `custom.extra.rollout_matching.*`
- **training/inference**: Stage-2 AB trainer + rollout backend wiring in `src/sft.py` and `src/trainers/*`
- **artifacts**: run naming, checkpoints, and logs from `training.output_dir` / `training.logging_dir` / `training.run_name`

Constraints:
- Config-first, YAML-driven; no new CLI flags.
- Preserve Qwen3-VL chat-template compatibility and existing geometry invariants.
- Keep upstream HF internals untouched.
- Keep runs reproducible and paper-auditable.

Stakeholders:
- Researchers authoring ablation configs under `configs/stage2_ab/**`.
- Infra/launcher maintainers (`scripts/train_stage2.sh`, `src/sft.py`, `src/config/*`).

## Goals / Non-Goals

**Goals:**
- Enforce canonical hierarchy for Stage-2 AB:
  - one canonical `configs/stage2_ab/base.yaml`,
  - canonical downstream leaves (`prod/{a_only,b_only,ab_mixed}.yaml`, `smoke/{a_only,b_only,ab_mixed}.yaml`),
  - optional additional leaves are allowed only if they follow the same one-hop + explicit-leaf contract,
  - no multi-hop leaf inheritance.
- Ensure downstream files are self-consistent and readable in isolation by explicitly declaring high-signal knobs (run identity, optimizer LRs, effective batch, eval/save policy, schedule/ablation toggles).
- Replace ambiguous rollout nesting (`custom.extra.rollout_matching`) with an explicit grouped namespace for canonical Stage-2 authoring.
- Keep runtime behavior reproducible by introducing a single normalization path from canonical YAML to trainer-consumed config.
- Fail fast on hierarchy violations and legacy key usage with actionable migration guidance.

**Non-Goals:**
- Redesigning Stage-2 AB losses, channel schedule semantics, or rollout algorithm behavior.
- Introducing new training CLI flags or changing launcher topology semantics.
- Refactoring non-Stage2 config families (`configs/dlora/**`) in this change.

## Decisions

### 1) Canonical Stage-2 AB profile topology is one-hop only

Decision:
- Standardize Stage-2 AB profiles to:
  - `configs/stage2_ab/base.yaml` (stable defaults),
  - `configs/stage2_ab/prod/{a_only,b_only,ab_mixed}.yaml`,
  - `configs/stage2_ab/smoke/{a_only,b_only,ab_mixed}.yaml`.
- Canonical leaves extend only `../base.yaml` (single parent).
- Smoke profiles pin smoke runtime knobs directly in leaf files (no dual-parent `extends` list).
- Additional leaves (e.g., iterative ablations) remain allowed, but MUST also be one-hop from `../base.yaml` and satisfy explicit-leaf rules.

Base migration decision:
- Build `configs/stage2_ab/base.yaml` by flattening shared defaults currently split across:
  - `configs/stage2_ab/base_rollout_matching_sft.yaml`
  - `configs/stage2_ab/prod/base.yaml`
- Retire `base_rollout_matching_sft.yaml`, `prod/base.yaml`, and `base_smoke_runtime.yaml` as canonical profile parents after migration.

Alternatives considered:
- Keep multi-hop inheritance and document conventions.
  - Rejected: docs-only discipline cannot prevent drift and hidden overrides.
- Split by module/layer files (data/model/runtime/schedule).
  - Rejected per user requirement: downstream file must remain self-explanatory.

Rationale:
- One-hop profiles keep override provenance obvious and reduce merge-order ambiguity.

### 2) Downstream leafs must explicitly pin high-signal knobs

Decision:
- Require each downstream leaf to set, at minimum:
  - model identity/path (`model.model`),
  - run identity (`training.run_name`, `training.output_dir`, `training.logging_dir`),
  - optimization-critical knobs (`training.learning_rate`, `training.vit_lr`, `training.aligner_lr`, `training.effective_batch_size`),
  - eval/save policy (`training.eval_strategy`, `training.eval_steps`, `training.save_strategy`, `training.save_steps`),
  - ablation-defining knobs (`stage2_ab.schedule.b_ratio`, `stage2_ab.n_softctx_iter`, any channel-B ablation toggles used by that profile).

Alternatives considered:
- Keep these only in base and override when different.
  - Rejected: downstream intent becomes implicit and harder to review.

Rationale:
- Explicit high-signal fields in leaf files improve auditability without requiring traversal.

### 3) Introduce canonical grouped rollout namespace and normalize once

Decision:
- Canonical Stage-2 rollout namespace is top-level `rollout_matching.*`.
- Migration rule: rollout knobs already present under `custom.extra.rollout_matching.*` migrate via path-only relocation to `rollout_matching.*` with the same subkey names.
- `rollout_matching.decode_batch_size` is the single source of truth for rollout decode/evaluation microbatching across rollout-aware trainer variants.
- `training.per_device_eval_batch_size` (and similar per-device eval knobs) does not independently control rollout decode/evaluation batching; rollout paths consume the resolved decode batch knob.
- Implement one shared Python normalization resolver in `src/config/*` that maps canonical grouped fields into trainer-consumed runtime config.
- Both runtime and launcher preflight MUST consume this shared resolver:
  - `src/sft.py` uses normalized config for trainer injection.
  - `scripts/train_stage2.sh` preflight calls the same resolver (instead of maintaining duplicated YAML parsing/field assumptions).
  - Launcher mechanism: preflight runs a single Python call (under `conda run -n ms`) that loads config via `ConfigLoader.load_training_config(...)`, applies shared normalization, and returns resolved rollout fields in machine-readable form for bash checks.
  - Command shape example: `conda run -n ms python -c "...load_training_config(...); print(json.dumps({'rollout_backend': ..., 'vllm_mode': ..., 'server_base_urls': [...]}))"`; bash consumes the JSON fields instead of parsing YAML.
  - Output contract: emit single-line JSON (newline-terminated, no pretty-print) so bash parsing remains deterministic.
  - Required JSON schema for preflight parsing:
    - `rollout_backend`: string
    - `vllm_mode`: string or null
    - `server_base_urls`: list of strings (empty allowed outside server-mode)
  - Failure semantics: preflight exits non-zero and blocks launch if normalization fails, JSON is invalid, or schema keys/types are missing.
- Keep `custom.extra` as a narrow escape hatch for genuinely small/trivial toggles only.

Normalization contract:
- The authoritative minimal mapping table is defined in `specs/stage2-ab-training/spec.md` under “Normalization mapping sketch (minimum required)”.
- Additional rollout fields keep their existing key names while moving from `custom.extra.rollout_matching.*` to top-level `rollout_matching.*` (no aliasing).

Alternatives considered:
- Keep `custom.extra.rollout_matching` as canonical and only tighten docs.
  - Rejected: namespace remains ambiguous and encourages config sprawl.
- Keep dual-read compatibility (`rollout_matching.*` and `custom.extra.rollout_matching.*`) during migration.
  - Rejected: allows shadowed semantics and violates fail-fast reproducibility requirements.

Rationale:
- Canonical grouped authoring with controlled normalization gives clean user-facing contracts while preserving implementation stability.

### 4) Enforce fail-fast migration boundaries

Decision:
- Validation rejects:
  - canonical leaf configs with multi-hop inheritance in `configs/stage2_ab/{prod,smoke}/*`,
  - any legacy Stage-2 rollout key placement under `custom.extra.rollout_matching.*`,
  - any ambiguous duplicated key definitions for canonical rollout settings.
- Errors include exact replacement paths and migration guidance.

Alternatives considered:
- Silent precedence rules when both legacy and canonical keys are present.
  - Rejected: unsafe and non-reproducible.

Rationale:
- Hard boundaries prevent shadowed behavior and make paper runs easier to reproduce.

## Risks / Trade-offs

- [Legacy configs break on stricter validation] → Mitigation: provide explicit migration table and deterministic replacement errors.
- [Migration blast radius across existing Stage-2 YAMLs] → Mitigation: batch-update canonical leaves and remove non-conforming/obsolete leaves from canonical `prod/` + `smoke/` surfaces before enabling strict gates.
- [Big-bang cutover risk across YAML/runtime/launcher] → Mitigation: land config relocation, `src/sft.py` normalized injection, and `scripts/train_stage2.sh` normalized preflight in one coordinated change before turning on strict fail-fast for unsupported key paths.
- [Leaf verbosity increases file length] → Mitigation: explicitly pin only high-signal knobs; keep truly stable low-signal defaults in base.
- [Potential drift between launcher preflight and runtime parser] → Mitigation: reuse shared config normalization/validation path for script preflight where possible.

## Migration Plan

1. Create `configs/stage2_ab/base.yaml` by flattening shared defaults from current Stage-2 parent files.
2. Rebuild canonical prod/smoke leaves to inherit one-hop from `../base.yaml`, and inline smoke runtime overrides in smoke leaves.
3. Migrate optional additional leaves (e.g., iterative variants) to one-hop from `../base.yaml` or relocate them outside canonical profile set.
4. Introduce shared rollout normalization resolver in `src/config/*` and consume it from both `src/sft.py` and `scripts/train_stage2.sh` preflight.
5. Add validation for one-hop canonical leaf structure, explicit leaf high-signal keys, and fail-fast on unsupported key paths in canonical validated paths.
6. Verify canonical leaf configs load via `ConfigLoader.load_training_config` and fail fast on any legacy Stage-2 rollout key paths.
7. Run Stage-2 smoke profile to ensure end-to-end wiring remains intact.

Rollback:
- Revert to previous config tree and disable new validation gates if migration blocks urgent runs.
- Because this is config/loader scoped, rollback is localized to `configs/stage2_ab/**`, `src/config/*`, and `scripts/train_stage2.sh`.

## Resolved Scope Decisions

- Canonical rollout namespace is top-level `rollout_matching.*` for Stage-2 profiles.
- Strict one-hop validation applies to canonical Stage-2 profile leaves in `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml`; every file under those paths must satisfy the contract.
- Any profile discovery tooling must treat only `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml` as canonical inputs.
- Stage-2 profiles do not keep legacy rollout alias compatibility; unsupported legacy keys fail fast immediately.
