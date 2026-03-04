## Why

Stage-2 evaluation currently runs full rollout-based decoding via the HF backend, which is slower than vLLM and makes it harder to run frequent, production-like eval on long multimodal sequences. We want to switch **evaluation rollouts only** to vLLM without running a separate rollout server, while keeping runs reproducible and keeping GPU memory usage minimal on 8x A100-80GB.

## What Changes

- Add a YAML-driven way to use **vLLM for `eval_step` rollouts only** (Stage-2 trainers) without requiring a separate vLLM rollout server.
- Enforce **full weight sync** for vLLM rollouts (no adapter-only sync), since adapter sync is unsupported in our setup.
- Use standard colocate lifecycle for eval-only vLLM (sleep mode disabled by default for stability in long-lived DDP jobs), while keeping memory relief via offload/server-mode options.
- Make vLLM token-logprob tracing for eval-step confidence scoring **non-experimental** by adding targeted regressions/guards and removing "experimental" caveats once validated.
- Make eval rollouts robust: token-trace violations fall back to constant-score (tracked via `eval/trace_fallback_count`), and per-sample vLLM decode failures are skipped (tracked via `eval/vllm_decode_error_count`). Engine-level vLLM failures (init / missing lifecycle APIs / eval OOM) fail fast.
- Provide production-style example configs for `configs/stage2_two_channel/prod/` that enable vLLM eval rollouts for Channel-A-only runs (and leave room to extend to low-`b_ratio` Channel-B later).

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `rollout-matching-sft`: evaluation-time backend selection, full-sync enforcement, and standard colocate guidance (sleep mode disabled by default; optional advanced sleep mode).

## Impact

- Training/eval runtime:
  - Stage-2 trainers will be able to run eval rollouts using vLLM colocate without a separate server process.
  - Multimodal eval runs will always use full merged-weight sync (DoRA/LoRA compatible) rather than adapter-only sync.
  - Default colocate behavior prioritizes teardown stability (sleep mode disabled); for additional memory relief use `rollout_matching.offload` or vLLM `server` mode.
  - Eval-step confidence-based scoring (e.g., `confidence_postop`-style token-trace scoring used by detection eval) becomes a first-class supported path under vLLM (no "experimental" behavior).
- Code impacted (expected):
  - `src/trainers/stage2_rollout_aligned.py` (evaluation loop and vLLM colocate lifecycle).
  - `src/config/rollout_matching_schema.py` and `src/config/schema.py` (new YAML knobs / validation).
  - `configs/stage2_two_channel/prod/` (new example profile(s)).
- Dependencies:
  - ms-swift `swift.llm.VllmEngine` APIs used by the colocate backend; no in-process vLLM shutdown during training.
