## 1. Stage-2 AB profile hierarchy refactor

- [ ] 1.1 Create `configs/stage2_ab/base.yaml` by flattening shared defaults currently split across `configs/stage2_ab/base_rollout_matching_sft.yaml` and `configs/stage2_ab/prod/base.yaml`.
- [ ] 1.2 Rebuild canonical one-hop leaves under `configs/stage2_ab/prod/{a_only,b_only,ab_mixed}.yaml` and `configs/stage2_ab/smoke/{a_only,b_only,ab_mixed}.yaml` (smoke overrides inlined; no dual-parent extends).
- [ ] 1.3 Retire canonical-parent usage of `configs/stage2_ab/base_rollout_matching_sft.yaml`, `configs/stage2_ab/prod/base.yaml`, and `configs/stage2_ab/base_smoke_runtime.yaml`; migrate or remove any non-conforming iterative variants so canonical `prod/` and `smoke/` surfaces remain clean.
- [ ] 1.4 Verify canonical leaves load via `conda run -n ms python -c "from src.config.loader import ConfigLoader; paths=['configs/stage2_ab/prod/a_only.yaml','configs/stage2_ab/prod/b_only.yaml','configs/stage2_ab/prod/ab_mixed.yaml','configs/stage2_ab/smoke/a_only.yaml','configs/stage2_ab/smoke/b_only.yaml','configs/stage2_ab/smoke/ab_mixed.yaml']; [ConfigLoader.load_training_config(p) for p in paths]"`.

## 2. Canonical grouped rollout config namespace

- [ ] 2.1 Add canonical Stage-2 rollout namespace support in config schema/loader normalization path (`src/config/schema.py`, `src/config/loader.py`) using path-only relocation from `custom.extra.rollout_matching.*` to `rollout_matching.*` (same subkeys, no alias fallback).
- [ ] 2.2 Normalize canonical grouped rollout keys into the trainer-injected rollout contract (`rollout_matching_cfg`) before trainer construction in `src/sft.py`.
- [ ] 2.3 Remove Stage-2 legacy rollout alias support entirely (`custom.extra.rollout_matching.*`) and fail fast with actionable migration text to canonical grouped keys.
- [ ] 2.4 Ensure trainer/runtime/launcher consume only canonical rollout namespace for Stage-2 profiles (no dual-read compatibility path).
- [ ] 2.5 Document and enforce the canonical normalization mapping (`rollout_matching.*` -> `rollout_matching_cfg.*`) used by both runtime and launcher preflight.
- [ ] 2.6 Land YAML relocation + `src/sft.py` normalized injection + `scripts/train_stage2.sh` normalized preflight together before enabling strict fail-fast on unsupported key paths (single coordinated cutover).
- [ ] 2.7 Enforce rollout-aware trainer behavior so `rollout_matching.decode_batch_size` is the single source of truth for rollout decode/eval microbatching (no independent control from `training.per_device_eval_batch_size` or similar per-device eval knobs).

## 3. Validation and launcher contract alignment

- [ ] 3.1 Add Stage-2 profile validation that enforces one-hop inheritance for `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml`.
- [ ] 3.2 Add Stage-2 leaf validation that requires explicit high-signal knobs (`model.model`, `run_name`, output/log dirs, group LRs, effective batch, eval/save policy, schedule fields).
- [ ] 3.3 Align `scripts/train_stage2.sh` preflight with shared Python config normalization resolver used by runtime (remove drift-prone duplicated assumptions), then validate with `bash -n scripts/train_stage2.sh`.
- [ ] 3.4 Enable strict validators only after canonical leaves are migrated and canonical `prod/` + `smoke/` surfaces contain only compliant profiles.
- [ ] 3.5 Define and enforce launcher preflight JSON schema (`rollout_backend`, `vllm_mode`, `server_base_urls`) and fail-fast semantics for invalid/missing fields.

## 4. Downstream profile explicitness and migration

- [ ] 4.1 Update each canonical Stage-2 AB downstream leaf to explicitly pin high-signal customization knobs, including `model.model`, run identity, group learning rates, rollout decode/runtime, and ablation-defining schedule fields.
- [ ] 4.2 Keep `custom.extra` usage minimal by moving non-trivial rollout/runtime knobs into canonical grouped sections and leaving only tiny/trivial residuals if strictly needed.
- [ ] 4.3 Add migration notes in config comments/docs for removed/relocated Stage-2 key paths and expected replacements, including iterative-variant placement policy.

## 5. Reproducibility and verification checkpoints

- [ ] 5.1 Run focused config-load verification for canonical leaves and confirm fail-fast behavior on representative legacy Stage-2 key paths; record pass/fail matrix in change notes.
- [ ] 5.2 Run one Stage-2 launcher preflight dry-check using canonical profile path, e.g. `config=configs/stage2_ab/smoke/ab_mixed.yaml gpus=0 bash scripts/train_stage2.sh` (or equivalent no-train preflight), and confirm resolved rollout fields are sourced from shared Python normalization output.
- [ ] 5.3 Capture reproducibility checklist in artifact notes for at least one canonical profile: config path, `training.run_name`, seed source (`training.seed`), and expected output artifact roots (`training.output_dir`, `training.logging_dir`).
- [ ] 5.4 Audit profile discovery tooling/globs so canonical runs include only `configs/stage2_ab/prod/*.yaml` + `configs/stage2_ab/smoke/*.yaml`.
- [ ] 5.5 Add a focused mismatch check (unit/integration/log assertion) that `rollout_matching.decode_batch_size=4` with a different `training.per_device_eval_batch_size` still yields rollout decode/eval microbatching of `4`.
