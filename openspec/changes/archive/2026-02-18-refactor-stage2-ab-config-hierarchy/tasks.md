## 1. Stage-2 AB profile hierarchy refactor

Note:
- Parser architecture migration (schema-derived strict parsing and de-duplication of manual nested allowlists) is tracked separately in:
  - `openspec/changes/refactor-config-schema-derived-validation/`
- Tasks below remain the hierarchy/canonical-surface migration tasks for this change.

- [x] 1.1 Create `configs/stage2_ab/base.yaml` by flattening shared defaults currently split across `configs/stage2_ab/base_rollout_matching_sft.yaml` and `configs/stage2_ab/prod/base.yaml`.
- [x] 1.2 Rebuild canonical one-hop leaves under `configs/stage2_ab/prod/{a_only,b_only,ab_mixed}.yaml` and `configs/stage2_ab/smoke/{a_only,b_only,ab_mixed}.yaml` (smoke overrides inlined; no dual-parent extends).
- [x] 1.3 Retire canonical-parent usage of `configs/stage2_ab/base_rollout_matching_sft.yaml`, `configs/stage2_ab/prod/base.yaml`, and `configs/stage2_ab/base_smoke_runtime.yaml`; migrate or remove any non-conforming iterative variants so canonical `prod/` and `smoke/` surfaces remain clean.
- [x] 1.4 Verify canonical leaves load via `conda run -n ms python -c "from src.config.loader import ConfigLoader; paths=['configs/stage2_ab/prod/a_only.yaml','configs/stage2_ab/prod/b_only.yaml','configs/stage2_ab/prod/ab_mixed.yaml','configs/stage2_ab/smoke/a_only.yaml','configs/stage2_ab/smoke/b_only.yaml','configs/stage2_ab/smoke/ab_mixed.yaml']; [ConfigLoader.load_training_config(p) for p in paths]"`.

## 2. Canonical grouped rollout config namespace

- [x] 2.1 Add canonical Stage-2 rollout namespace support in config schema/loader normalization path (`src/config/schema.py`, `src/config/loader.py`) using path-only relocation from `custom.extra.rollout_matching.*` to `rollout_matching.*` (same subkeys, no alias fallback).
- [x] 2.2 Normalize canonical grouped rollout keys into the trainer-injected rollout contract (`rollout_matching_cfg`) before trainer construction in `src/sft.py`.
- [x] 2.3 Remove Stage-2 legacy rollout alias support entirely (`custom.extra.rollout_matching.*`) and fail fast with actionable migration text to canonical grouped keys.
- [x] 2.4 Ensure trainer/runtime/launcher consume only canonical rollout namespace for Stage-2 profiles (no dual-read compatibility path).
- [x] 2.5 Document and enforce the canonical normalization mapping (`rollout_matching.*` -> `rollout_matching_cfg.*`) used by both runtime and launcher preflight.
- [x] 2.6 Land YAML relocation + `src/sft.py` normalized injection + `scripts/train_stage2.sh` normalized preflight together before enabling strict fail-fast on unsupported key paths (single coordinated cutover).
- [x] 2.7 Enforce rollout-aware trainer behavior so `rollout_matching.decode_batch_size` is the single source of truth for rollout decode/eval microbatching (no independent control from `training.per_device_eval_batch_size` or similar per-device eval knobs).

## 3. Validation and launcher contract alignment

- [x] 3.1 Add Stage-2 profile validation that enforces one-hop inheritance for `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml`.
- [x] 3.2 Add Stage-2 leaf validation that requires explicit high-signal knobs (`model.model`, `run_name`, output/log dirs, group LRs, effective batch, eval/save policy, schedule fields).
- [x] 3.3 Align `scripts/train_stage2.sh` preflight with shared Python config normalization resolver used by runtime (remove drift-prone duplicated assumptions), then validate with `bash -n scripts/train_stage2.sh`.
- [x] 3.4 Enable strict validators only after canonical leaves are migrated and canonical `prod/` + `smoke/` surfaces contain only compliant profiles.
- [x] 3.5 Define and enforce launcher preflight JSON schema (`rollout_backend`, `vllm_mode`, `server_base_urls`) and fail-fast semantics for invalid/missing fields.

## 4. Downstream profile explicitness and migration

- [x] 4.1 Update each canonical Stage-2 AB downstream leaf to explicitly pin high-signal customization knobs, including `model.model`, run identity, group learning rates, rollout decode/runtime, and ablation-defining schedule fields.
- [x] 4.2 Keep `custom.extra` usage minimal by moving non-trivial rollout/runtime knobs into canonical grouped sections and leaving only tiny/trivial residuals if strictly needed.
- [x] 4.3 Add migration notes in config comments/docs for removed/relocated Stage-2 key paths and expected replacements, including iterative-variant placement policy.

## 5. Reproducibility and verification checkpoints

- [x] 5.1 Run focused config-load verification for canonical leaves and confirm fail-fast behavior on representative legacy Stage-2 key paths; record pass/fail matrix in change notes.
- [x] 5.2 Run one Stage-2 launcher preflight dry-check using canonical profile path, e.g. `config=configs/stage2_ab/smoke/ab_mixed.yaml gpus=0 bash scripts/train_stage2.sh` (or equivalent no-train preflight), and confirm resolved rollout fields are sourced from shared Python normalization output.
- [x] 5.3 Capture reproducibility checklist in artifact notes for at least one canonical profile: config path, `training.run_name`, seed source (`training.seed`), and expected output artifact roots (`training.output_dir`, `training.logging_dir`).
- [x] 5.4 Audit profile discovery tooling/globs so canonical runs include only `configs/stage2_ab/prod/*.yaml` + `configs/stage2_ab/smoke/*.yaml`.
- [x] 5.5 Add a focused mismatch check (unit/integration/log assertion) that `rollout_matching.decode_batch_size=4` with a different `training.per_device_eval_batch_size` still yields rollout decode/eval microbatching of `4`.

### Verification Notes (2026-02-18)

- **5.1 Canonical leaves load**: PASS (see task 1.4 command).
- **5.1 Fail-fast (representative legacy key paths)**:
  - `custom.extra.rollout_matching.decode_batch_size` -> `ValueError: custom.extra.rollout_matching is unsupported. Move rollout settings to top-level rollout_matching.*.`
  - `stage2_ab.schedule.pattern` -> `ValueError: stage2_ab.schedule.pattern is not supported. Use stage2_ab.schedule.b_ratio (float in [0,1]) instead.`
- **5.2 Preflight contract** (equivalent no-train preflight via `resolve_stage2_launcher_preflight("configs/stage2_ab/smoke/ab_mixed.yaml")`):
  - `{"rollout_backend":"vllm","vllm_mode":"server","server_base_urls":["http://127.0.0.1:8000"]}`
- **5.3 Repro checklist (example: `configs/stage2_ab/smoke/ab_mixed.yaml`)**:
  - `training.run_name`: `smoke_ab_mixed`
  - Seed source: `training.seed: 17` from `configs/base.yaml`
  - `training.output_dir`: `output/stage2_ab/smoke/ab_mixed`
  - `training.logging_dir`: `tb/stage2_ab/smoke/ab_mixed`
- **5.4 Profile discovery audit**:
  - `rg -n "configs/stage2_ab/(prod|smoke)"` finds only docs/specs and `scripts/train_stage2.sh` default config string; no code enumerates configs via globs beyond explicit operator-provided `config=...`.
- **5.5 Mismatch check**:
  - Added `tests/test_stage2_ab_config_contract.py::test_rollout_decode_batch_size_overrides_eval_batch_size_when_mismatched` to assert `rollout_matching.decode_batch_size` overrides `training.per_device_eval_batch_size` for rollout-aware trainer variants.
