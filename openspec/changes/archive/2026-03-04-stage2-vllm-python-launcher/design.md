## Context

Current launcher shape:
- `scripts/train_stage2.sh` orchestrates:
  - config resolution + preflight via `src.trainers.rollout_matching.preflight.resolve_stage2_launcher_preflight`,
  - vLLM rollout server startup via `swift rollout ... --infer_backend vllm`,
  - readiness checks via HTTP:
    - `GET /health/` → `{"status":"ok"}`
    - `GET /get_world_size/` → `{"world_size": ...}`
  - learner startup via `scripts/train.sh` (`torchrun -m src.sft --config <yaml>`),
  - cleanup on exit (kill server + learner).

Upstream (ms-swift) rollout server contract:
- Endpoint registration (paths):
  - `swift/llm/infer/rollout.py::SwiftRolloutDeploy._register_rl_rollout_app`
- Readiness endpoints:
  - `swift/llm/infer/rollout.py::SwiftRolloutDeploy.health`
  - `swift/llm/infer/rollout.py::SwiftRolloutDeploy.get_world_size`
- `get_world_size` returns `vllm_tensor_parallel_size * vllm_data_parallel_size`, which we can use to validate that the server process matches the intended GPU allocation.

Constraints:
- YAML-first; avoid new CLI flags (prefer env vars).
- Preserve Qwen3-VL template compatibility (server prompt encoding must match learner template).
- Preserve geometry invariants (coords never dropped/reordered).
- Do not modify upstream ms-swift or upstream model code.

## Goals / Non-Goals

Goals:
- Replace bash Stage-2 server-mode orchestration with a Python launcher.
- Centralize all validation in Python (config invariants, runtime GPU split checks, server readiness).
- Keep vLLM server hyperparameters YAML-driven; keep runtime GPU split out-of-YAML.
- Ensure deterministic, unit-testable command construction (server cmd + learner cmd).
- Own lifecycle (signals, termination) to prevent orphan `swift rollout` processes.

Non-Goals:
- Changing training objective, dataset formats, or coordinate conventions.
- Replacing ms-swift rollout implementation.
- Multi-node orchestration (this launcher targets single-node GPU splits first).
- Supporting multiple rollout servers in one process (keep current “exactly one server base_url” contract).

## Decisions

### 1) Interface: env-only parameters (no new CLI flags)

Decision:
- The Python launcher accepts environment variables only (no positional args), mirroring `scripts/train*.sh` patterns:
  - `config`/`CONFIG`: path to YAML
  - `server_gpus`/`SERVER_GPUS`: e.g. `0,1,2,3,4,5`
  - `train_gpus`/`TRAIN_GPUS`: e.g. `6,7`
  - optional operational runtime knobs: `wait_timeout`, `wait_interval`
    - These MUST NOT override YAML-driven rollout server connectivity (`base_url`, `group_port`) or vLLM hyperparameters.

Why:
- Keeps compatibility with current workflow and follows “avoid new CLI flags”.

### 2) Single authoritative preflight payload

Decision:
- Reuse `resolve_stage2_launcher_preflight(config_path)` as the single resolver for YAML-driven values:
  - server model path
  - template flags (`template`, `max_pixels`, truncation strategy, max_length)
  - vLLM knobs (`tensor_parallel_size`, `enforce_eager`, `gpu_memory_utilization`, `max_model_len`, `enable_lora`)
  - server endpoint (base_url + group_port)
  - NOTE: rollout server endpoint (including port) is YAML-only; the launcher MUST NOT accept runtime overrides for base_url/host/port/group_port.
  - NOTE: launcher MAY apply operational proxy hygiene (e.g., `NO_PROXY` for the rollout server host), but this MUST NOT be a behavior-changing runtime knob.

Why:
- Keeps configuration semantics centralized and avoids duplicating schema logic in a new launcher.

### 2.1) Legacy bash-eval contract removed

Decision:
- The launcher MUST NOT rely on `eval $(python ...)` or `ROLLOUT_CONTRACT_JSON` for preflight/contract propagation.
- Orchestration MUST remain JSON-native inside Python (testable, deterministic).

### 3) Runtime GPU split validation in Python

Decision:
- Validate in Python before starting any GPU process:
  - `server_gpus` and `train_gpus` non-empty and disjoint,
  - `len(server_gpus) % tensor_parallel_size == 0`,
  - derived server data-parallel size `dp = len(server_gpus) / tp`,
  - reject any “port already in use” condition for the requested base_url.

Why:
- Prevents silent desynchronization (e.g., server binds a different port than config, or mismatch between intended TP and allocated GPUs).

### 4) Server readiness checks use ms-swift endpoints

Decision:
- After spawning `swift rollout`, poll:
  - `GET /health/` expecting HTTP 200
  - `GET /get_world_size/` expecting `world_size == len(server_gpus)`

Why:
- Aligns with upstream contract and catches stale servers / port mismatch quickly.

### 5) Learner launch uses the same entrypoint as `scripts/train.sh`

Decision:
- Spawn learner with `torchrun -m src.sft --config <yaml>` (matching the existing training script), using:
  - `CUDA_VISIBLE_DEVICES=train_gpus`
  - explicit `MASTER_ADDR/MASTER_PORT` (same semantics as current scripts).

Why:
- Minimizes behavioral differences; launcher refactor should not change the training runtime.

### 6) Process lifecycle + signal handling in Python

Decision:
- Parent launcher process owns:
  - stdout/stderr redirection options (default: inherit, user can redirect outer command),
  - signal forwarding (SIGINT/SIGTERM) to children,
  - cleanup/termination on failure (server start failure, readiness timeout, learner crash).

Why:
- Prevents orphan server processes and makes failure modes explicit.

## Risks / Trade-offs

- [Risk] Subprocess lifecycle bugs can leak server processes.
  - Mitigation: unit tests for termination paths; explicit `try/finally` cleanup; robust signal handling.
- [Risk] Differences vs bash environment defaults (NCCL/proxy/alloc conf).
  - Mitigation: mirror the current scripts’ runtime env defaults inside the Python launcher, but keep them as “runtime-only” (not hyperparameters).
- [Risk] Users may rely on `TRAIN_ENV` shell injection.
  - Mitigation: allow an opt-in “extra env vars” passthrough (explicit allowlist or `TRAIN_ENV` parsing) but keep it minimal and auditable.

## Migration Plan

1. Introduce Python launcher (new entrypoint) with parity behavior.
2. Update `scripts/train_stage2.sh` to delegate to Python launcher (thin wrapper), keeping legacy entrypoint stable.
3. Add tests around preflight contract + command construction + lifecycle (no GPU required).
4. Optional: once stable, deprecate/remove bash-only orchestration paths.

## Open Questions

- Should we keep `scripts/train_stage2.sh` as a wrapper indefinitely, or migrate docs/workflows to the Python entrypoint?
- Do we want a structured “launcher metadata” artifact (e.g., JSON) written into `output_dir` for exact command/env reproduction?
- Should the launcher enforce `disable_proxy=true` unconditionally in server-mode runs, or keep it configurable?
