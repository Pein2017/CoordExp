## Why

Stage-2 server-mode training (`swift rollout` vLLM server + learner DDP) is currently orchestrated by `scripts/train_stage2.sh`. While the underlying training/eval logic is config-driven, the launcher itself is still split across:

- bash orchestration (process lifecycle, health checks, port checks, GPU split), and
- inline Python snippets (`eval $(python -c ...)`) for config preflight and value extraction.

This makes it too easy to diverge from YAML (via environment overrides), and it makes correctness/reproducibility checks hard to test and evolve.

We want a single Python launcher (similar in spirit to `src/sft.py`) that:

- treats YAML as the single source of truth for server hyperparameters,
- keeps only the runtime GPU split outside YAML, and
- consolidates all validation + command construction in Python (testable, deterministic, and easier to extend).

## What Changes

- Add a Python Stage-2 launcher that replaces the logic in `scripts/train_stage2.sh`:
  - resolves config (extends, repo-relative paths) and builds a single preflight payload,
  - validates GPU split and vLLM parallelism deterministically,
  - launches the vLLM rollout server via `swift rollout`,
  - waits for readiness via ms-swift rollout endpoints (`/health/`, `/get_world_size/`),
  - launches the learner via `torchrun -m src.sft --config <yaml>`,
  - owns signal handling + child process cleanup (no orphan vLLM servers).
- Keep **runtime GPU split** as the only non-YAML knob that affects model/server hyperparameters or endpoints:
  - `server_gpus` (vLLM actors)
  - `train_gpus` (learner ranks)
- Optional operational runtime knobs (timeouts/debug/proxy hygiene) MAY exist, but they MUST NOT override YAML-driven rollout server
  connectivity (`rollout_matching.vllm.server.servers[0].base_url` / `group_port`) or vLLM hyperparameters.
- Optionally keep `scripts/train_stage2.sh` as a thin wrapper for backwards compatibility, but eliminate bash-side parsing/validation (no `eval` of Python output, no duplicated checks).

## Capabilities

### New Capabilities

- A single Python entrypoint to launch Stage-2 server-mode runs reproducibly (YAML-driven, fail-fast, testable).

### Modified Capabilities

- `stage2-ab-training`: Stage-2 server-mode orchestration and validation move from bash to Python; training/eval behavior remains unchanged. The launcher contract no longer relies on `eval`-style bash preflight or `ROLLOUT_CONTRACT_JSON`.
- `rollout-matching-sft`: Clarify and enforce single-server-only rollout server configuration and canonicalize rollout sampling + repeat-termination knobs under top-level `rollout_matching.*` (no `custom.extra` usage).

## Impact

- Reproducibility:
  - YAML remains the single source of truth for vLLM server knobs (e.g., tensor parallel size, eager mode, memory utilization, max_model_len) and rollout server connectivity (base_url + group_port).
  - Only the GPU split remains runtime-defined; other runtime knobs are operational (do not change training semantics).
- Maintainability:
  - Launcher logic becomes unit-testable without spawning GPUs.
  - Reduced duplication between `scripts/train_stage2.sh` and `scripts/train.sh` prechecks.
- Risk:
  - Process/signal-handling must be correct to avoid leaked server processes; this is addressed explicitly in design + tests.

Code impacted (expected):
- New module under `src/` (launcher entrypoint + helpers).
- `scripts/train_stage2.sh` (convert to thin wrapper, or mark deprecated).
- Tests under `tests/` (launcher contract + command construction).

Dependencies:
- Uses ms-swift rollout server endpoints and CLI semantics (no modifications to ms-swift; treat as upstream contract).
