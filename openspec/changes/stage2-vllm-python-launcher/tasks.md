## 1. Python Launcher Entry Point

- [x] 1.1 Add a new Python launcher module (e.g., `src/launchers/stage2_vllm_server.py`) with a `main()` that accepts **environment variables only** (no positional args).
- [x] 1.2 Reuse `resolve_stage2_launcher_preflight(config_path)` to build a single authoritative preflight payload for both server and learner.

## 2. Validation (Fail-Fast, Python-Owned)

- [x] 2.1 Validate config path resolution (absolute, repo-relative, `configs/*`) deterministically and fail with actionable errors.
- [x] 2.2 Validate runtime GPU split in Python:
  - `server_gpus` and `train_gpus` are non-empty lists of device ids,
  - sets are disjoint,
  - `len(server_gpus) % rollout_matching.vllm.tensor_parallel_size == 0`,
  - derive `server_dp = len(server_gpus) / tp`.
- [x] 2.3 Validate the rollout server base_url:
  - scheme `http|https`,
  - host not `0.0.0.0`,
  - port present,
  - port is free before launching.

## 3. Server Launch + Readiness

- [x] 3.1 Spawn `swift rollout` with:
  - `CUDA_VISIBLE_DEVICES=server_gpus`
  - `ROOT_IMAGE_DIR` from preflight
  - vLLM flags from YAML preflight (tp/eager/gpu_mem/max_model_len/etc)
  - template flags from YAML preflight (`--template`, `--max_pixels`, `--max_length`, `--truncation_strategy`)
- [x] 3.2 Poll ms-swift rollout endpoints for readiness:
  - `GET /health/` must return HTTP 200 within timeout
  - `GET /get_world_size/` must return `world_size == len(server_gpus)`
- [x] 3.3 On readiness timeout or mismatch:
  - print a concise diagnostic (endpoint, last payload),
  - terminate server subprocess and exit non-zero.

## 4. Learner Launch (Parity with `scripts/train.sh`)

- [x] 4.1 Spawn learner via `torchrun -m src.sft --config <yaml>` with:
  - `CUDA_VISIBLE_DEVICES=train_gpus`
  - explicit `MASTER_ADDR/MASTER_PORT` semantics (match current scripts)
  - proxy hygiene for localhost server endpoints (NO_PROXY/no_proxy + unset proxies)
- [x] 4.2 Preserve stage2 run metadata (currently exported env vars) in a deterministic, Python-owned way:
  - either as env vars (compatible), or
  - write a small JSON metadata file next to `output_dir` (preferred for reproducibility).

## 5. Lifecycle / Cleanup

- [x] 5.1 Implement signal handling:
  - Ctrl+C terminates both server and learner cleanly,
  - launcher exits with learner return code when learner finishes.
- [x] 5.2 Ensure no orphaned server processes on any failure path (server launch failure, readiness timeout, learner crash).

## 6. Tests (No-GPU)

- [x] 6.1 Add unit tests for command construction:
  - server cmd includes all required flags and uses YAML-derived values,
  - learner cmd uses `torchrun -m src.sft --config`.
- [x] 6.2 Add unit tests for validation logic:
  - GPU overlap detection,
  - TP divisibility check,
  - base_url parsing and 0.0.0.0 rejection.
- [x] 6.3 Add a lifecycle test that simulates a failing server readiness check and asserts subprocess cleanup is executed.

## 7. Docs / Runbook

- [x] 7.1 Update or add a short runbook snippet documenting the new launcher invocation (env vars + example).
- [x] 7.2 Document the “single rollout server only” limitation and recommended external orchestration for multi-server setups.

## 8. Manual Validation Checklist

- [ ] 8.1 Single-node split run (e.g., 6 server GPUs / 2 learner GPUs) with prod config; verify:
  - server readiness gate passes,
  - learner connects and trains,
  - no proxy-related failures on localhost.
- [ ] 8.2 Intentionally misconfigure:
  - overlapping GPUs → fail fast,
  - TP not dividing server GPUs → fail fast,
  - base_url port occupied → fail fast.
