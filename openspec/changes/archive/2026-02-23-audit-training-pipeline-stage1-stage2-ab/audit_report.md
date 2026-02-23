# Audit Report: Training Pipeline (Stage-1 + Stage-2 AB, Qwen3-VL)

Date: 2026-02-19

Anchored entrypoints and configs under audit:
- Stage-1 launcher: `scripts/train.sh`
- Stage-1 config: `configs/stage1/ablation/geometry_first_coco80.yaml`
- Stage-2 AB combined launcher (rollout server + learner): `scripts/train_stage2.sh`
- Stage-2 AB config: `configs/stage2_ab/prod/ab_mixed.yaml`

This report is evidence-backed (file:line handles, config keys, and CPU-only contract tests). GPU-dependent operational smokes are listed as deferred tasks (Section "Deferred GPU Smokes").

---

## Environment / Upstream Provenance (Pinned Integration Surfaces)

The audit results below assume the following environment and upstream checkouts (recorded via explicit commands at audit time):

- Python: 3.12.11
- `transformers`: 4.57.1
- `torch`: 2.8.0+cu128
- `vllm`: 0.11.0
- `swift`: 3.10.0.dev0 (local checkout at `/data/ms-swift`)
- `/data/ms-swift` git SHA: `57e9a1cb4100a52ba63983457eca2b3e695f71b6` (clean working tree at audit time)

CoordExp repo git SHA at audit time:
- `033fbe43c9ce6a67f220d68dc2e184a163d8a7a2`

Verification commands (run from repo root):
```bash
conda run -n ms python -c "import transformers,torch,swift,vllm; print(transformers.__version__); print(torch.__version__); print(getattr(swift,'__version__','n/a'), swift.__file__); print(getattr(vllm,'__version__','n/a'))"
git -C /data/ms-swift rev-parse HEAD
git -C /data/ms-swift status --porcelain | wc -l
git rev-parse HEAD
```

---

## Summary (Highest-Leverage Findings)

P0 (must-be-correct / objective-changing):
- Runtime image resizing is forbidden because it breaks grounding geometry. `template.max_pixels` is treated as a hard cap and oversize inputs hard-error on the learner before ms-swift could rescale.
- Stage-2 AB depends on strict cross-process chat-template/tokenizer boundary alignment; prompt-token-id alignment and CoordJSON strict parsing boundaries are explicitly tested and enforced.

P1 (stability / operator-safety / integration):
- vLLM server-mode rollouts now have explicit request/response schema contracts (return_details + token ids), endpoint surface contracts, and HF-rollout length coherence gating.
- Diagnostics (monitor/debug dumps) are allowed to be unbounded (max_chars=0), but are now protected against training stalls and disk exhaustion via async write + bounded queue + low-disk guard (best-effort sinks only).

P2 (paper readiness / maintainability):
- Run-manifest “source YAML copies” are best-effort but are no longer silent if they fail (warnings emitted).
- Fusion-config ROOT_IMAGE_DIR auto-default is now explicitly labeled as heuristic (warning) to avoid silent path drift.

---

## P0 Findings (Correctness / Objective Integrity)

### P0.1 No Runtime Image Rescaling (Geometry Is Sacred)

Risk:
- Upstream ms-swift templates can rescale images when `template.max_pixels` is exceeded, which would invalidate grounding coordinates unless geometry is transformed identically.

Policy:
- CoordExp forbids runtime resizing. Images must be pre-rescaled offline, and JSONL `width`/`height` must match the rescaled images.
- Oversize inputs must hard-error on the learner.

Evidence:
- Hard cap configured: `configs/base.yaml:16` sets `template.max_pixels: 786432` (`768*32*32`).
- Learner hard-error check based on JSONL dimensions: `src/datasets/dense_caption.py:263` (`BaseCaptionDataset._enforce_max_pixels`).

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_max_pixels_enforcement.py
```

Residual risk / operator note:
- This enforcement relies on JSONL `width`/`height`. If the on-disk image file does not match JSONL metadata, geometry will be wrong regardless. The offline pipeline must keep image pixels and JSONL metadata consistent.

---

### P0.2 Strict CoordJSON Parse Boundary (No Direct json.loads on CoordJSON)

Risk:
- Assistant payloads contain CoordJSON with bare coord-token literals; direct `json.loads` is invalid and/or silently changes semantics.

Policy:
- CoordJSON must be transpiled to strict JSON before `json.loads` and before matching/eval.

Evidence:
- CoordJSON transpilation boundary: `src/utils/coordjson_transpiler.py:599` (`parse_coordjson` strict/salvage behavior).
- Inventory gates exist to prevent reintroducing direct `json.loads` on CoordJSON (tests named below).

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_coordjson_parsing_inventory_gate.py tests/test_coordjson_parse_boundary_inventory.py
```

---

## P1 Findings (Stability / Integration / Operator Safety)

### P1.1 Stage-2 Launcher Preflight: Deterministic Path Resolution + Single-Server Contract

Risk:
- Relative JSONL/model paths resolving against caller CWD (instead of config/repo root) can silently point training at the wrong data.
- Multi-server configs could be silently ignored by the combined launcher if it only uses one URL.

Evidence:
- Preflight is designed to be CWD-independent: `src/trainers/rollout_matching/preflight.py:42` (`resolve_stage2_launcher_preflight` docstring).
- Explicit single-server fail-fast: `src/trainers/rollout_matching/preflight.py:125` (`build_stage2_launcher_preflight` checks `len(server_base_urls) != 1` and raises actionable error).
- ROOT_IMAGE_DIR resolution for server mode comes from `custom.train_jsonl` directory: `src/trainers/rollout_matching/preflight.py:125` (`root_image_dir_resolved` derived from `train_jsonl_path.parent`).

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_stage2_preflight_path_resolution.py tests/test_stage2_launcher_preflight_contract.py
```

---

### P1.2 Tokenizer / Prompt Boundary Alignment (Learner vs vLLM Server Mode)

Risk:
- If the learner and server disagree on chat-template rendering, special tokens, or tokenizer vocab/IDs, packed training targets can be misaligned (wrong prompt_len, wrong label masking, wrong gradients).

Evidence:
- Prompt-token-id prefix alignment checks: `src/trainers/stage2_ab_training.py` (prompt alignment checks inside batch prep; see tests below).
- vLLM server-mode request construction and response parsing are pinned via dedicated helpers and tests:
  - `src/trainers/rollout_matching_sft.py:1136` (`_parse_vllm_server_output`)
  - `src/trainers/rollout_matching_sft.py:1185` (`_build_vllm_server_infer_requests`)

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_stage2_ab_prompt_alignment_contract.py
conda run -n ms python -m pytest -q tests/test_vllm_server_rollout_contract.py tests/test_vllm_server_multimodal_payload_contract.py
```

---

### P1.3 HF Rollout Length Coherence Gate

Risk:
- For `rollout_backend=hf`, `prompt_len + max_new_tokens` can exceed `model.config.max_position_embeddings`, causing truncation or undefined behavior.

Evidence:
- Fail-fast gate: `src/trainers/rollout_matching_sft.py:3671` (`_enforce_hf_rollout_max_position_embeddings`).

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_rollout_hf_length_coherence_gate.py
```

---

### P1.4 Stage-2 AB A/B Timing Metrics (No Misleading Zero Rollout Timings)

Risk:
- Stage-2 AB Channel A steps (teacher-forced) should not emit `time/rollout_*` scalars as zeros; this breaks throughput diagnosis and hides Channel-B bottlenecks.

Evidence:
- Rollout pipeline timings are gated on actual rollout execution: `src/trainers/rollout_matching_sft.py:5796` (`_build_train_rollout_log_payload` only emits `time/rollout_*` when rollouts ran).

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_rollout_time_metrics_gating.py tests/test_stage2_ab_time_metrics_gating.py
```

---

### P1.5 Dump Safety (Unbounded Dumps Allowed, But Cannot Stall Training)

Risk:
- `monitor_dump` and `vllm.server.debug_dump` can be large/unbounded (`max_chars=0`), which can stall training (slow FS), blow up memory (unbounded pending writes), or brick runs (disk full).

Policy:
- Dumps are diagnostic sinks only. They must be best-effort and never crash the learner.
- Allow unbounded text dumps, but protect runtime stability.

Evidence (config surface is strict-parsed):
- `rollout_matching.monitor_dump` keys: `src/config/rollout_matching_schema.py:32` (`RolloutMonitorDumpConfig` includes async write + disk guard fields).
- `rollout_matching.vllm.server.debug_dump` keys: `src/config/rollout_matching_schema.py:66` (`VllmServerDebugDumpConfig` includes async write + disk guard fields).

Evidence (implementation):
- Monitor dump writer uses async + bounded queue + low-disk guard: `src/trainers/rollout_matching_sft.py:1443` (`_write_monitor_dump`).
- vLLM server debug dump is also best-effort + bounded: `src/trainers/rollout_matching_sft.py:2978` (`_maybe_debug_dump_vllm_server_rollouts`).

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_monitor_dump_io_best_effort.py
conda run -n ms python -m pytest -q tests/test_dump_clip_text_safety.py
conda run -n ms python -m pytest -q tests/test_vllm_server_debug_dump_io_best_effort.py
```

---

### P1.6 Swift Rollout Endpoint Surface Contract (Launcher Dependency)

Risk:
- The combined Stage-2 launcher depends on specific HTTP endpoints existing and staying stable across ms-swift upgrades.

Evidence:
- Endpoint contract test: `tests/test_swift_rollout_endpoints_contract.py` constructs the FastAPI app without starting a worker and asserts required endpoints exist.

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_swift_rollout_endpoints_contract.py
```

---

## P2 Findings (Paper Readiness / Maintainability / Diagnostics Hygiene)

### P2.1 Run Manifest Source YAML Copies: Best-Effort But Not Silent

Risk:
- Losing the exact YAML inputs used for a run (even if resolved_config.json exists) reduces auditability. Silent failures are particularly bad.

Evidence:
- Run manifest YAML source copies warn on failure (still best-effort): `src/utils/run_manifest.py:163` and `src/utils/run_manifest.py:173`.

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_run_manifest_files.py
```

---

### P2.2 Fusion ROOT_IMAGE_DIR Auto-Default Is Heuristic (Now Explicit)

Context:
- Fusion-config training is legacy/experimental in CoordExp.

Risk:
- Auto-defaulting `ROOT_IMAGE_DIR` from `custom.fusion_config` file directory can silently point at the wrong root if images are elsewhere.

Evidence:
- Warning is explicit when the heuristic is used: `src/sft.py:553` (logs a warning indicating the heuristic and preferred remediation).

Verification:
- Manual: run a fusion config with `ROOT_IMAGE_DIR` unset and confirm the warning is emitted.

---

## Config Contract (No Ad-hoc Hyperparameter Flags)

Policy:
- Training behavior changes must be YAML-specified and strict-parsed (unknown keys fail fast).
- Shell launchers may expose runtime-only knobs (GPU topology, server DP/TP, dtype) but must not create hidden YAML drift.

Evidence:
- Strict unknown-key fail-fast: `tests/test_training_config_strict_unknown_keys.py`.
- YAML-visible rollout-matching surfaces are dataclass-specified (strict parser): `src/config/rollout_matching_schema.py:1`.
- New dump safety rails are YAML keys (not hidden flags) and strict-parsed:
  - `src/config/rollout_matching_schema.py:32` (`RolloutMonitorDumpConfig`)
  - `src/config/rollout_matching_schema.py:66` (`VllmServerDebugDumpConfig`)

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py
conda run -n ms python -m pytest -q tests/test_strict_dataclass_parser.py
```

---

## Silent-Failure Policy (Audit Scan Result)

Policy:
- Avoid `except Exception: pass` in core code paths.
- Best-effort exception suppression is limited to diagnostics/I/O sinks and should emit warnings where appropriate.

Evidence:
- Dedicated regression test ensures no `except Exception: pass` exists under `src/`: `tests/test_no_silent_except_exception_pass.py`.

Verification:
```bash
conda run -n ms python -m pytest -q tests/test_no_silent_except_exception_pass.py
```

---

## Deferred GPU Smokes (Operational, Run Later)

These are intentionally deferred due to current GPU unavailability.

### Stage-1 smoke (GPU required)
```bash
bash scripts/train.sh gpus=0 config=configs/stage1/smoke/geometry_first_coco80.yaml debug=true
```
Acceptance checks:
- Trainer starts, dataset loads, packing active, and checkpoints saved as weight-only (no optimizer state).
- No runtime resize occurs (no max_pixels violations; no upstream resize path triggered).
- Key metrics present: loss scalars, packing telemetry, eval keys (if do_eval enabled).

### Stage-2 AB server-mode smoke (GPU required)
```bash
bash scripts/train_stage2.sh server_gpus=0,1,2,3 train_gpus=4,5,6,7 config=configs/stage2_ab/smoke/ab_mixed.yaml
```
Acceptance checks:
- Preflight passes (single-server, max_model_len coherence, resolved ROOT_IMAGE_DIR).
- Server readiness gates pass (`/health/`, `/get_world_size/`).
- A/B schedule telemetry: `stage2_ab/b_ratio_realized` tracks configured `b_ratio`.
- Channel-B steps emit `time/rollout_*` and `rollout/gen_tokens_per_s` (Channel-A steps do not emit `time/rollout_*`).
- Weight sync is stable under DDP; failure propagation is synchronized.
