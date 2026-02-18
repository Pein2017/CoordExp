# Stage-2 Training Runbook (Rollout-Matching & AB)

This document is the consolidated runbook for Stage-2 training workflows in CoordExp:

- **Rollout-matching SFT**: `custom.trainer_variant: rollout_matching_sft`
- **Stage-2 AB** (scheduler over channels A/B): `custom.trainer_variant: stage2_ab_training`

Stage-2 aims to align the model with its own decoded outputs while recovering missing GT objects.

---

## Quickstart Commands

From repo root:

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>]
```

Recommended wrapper (config-driven; handles `PYTHONPATH`, `torchrun`, proxy hygiene):

```bash
# Single GPU
bash scripts/train.sh config=<yaml> gpus=0

# Multi-GPU
bash scripts/train.sh config=<yaml> gpus=0,1,2,3
```

Multi-GPU:

```bash
PYTHONPATH=. conda run -n ms torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]
```

Multi-GPU + vLLM server mode (recommended topology for long rollouts): this is supported, but requires
the default Stage2-AB Channel-B step-budgeted pathway.

Requirements:
- `rollout_matching.rollout_backend: vllm`
- `rollout_matching.vllm.mode: server`
- Under multi-process learners (`torchrun`, `world_size > 1`), `rollout_matching.vllm.sync.mode` must resolve to `full`
  (DDP-safe rank0-only full-weight sync with strict barriers).

Where this lives in code:
- Rollout-matching SFT trainer: `src/trainers/rollout_matching_sft.py`
- Stage-2 AB trainer: `src/trainers/stage2_ab_training.py`
- Training entrypoint (YAML loader + wiring): `src/sft.py`

---

## Core Objective (What Stage-2 Trains)

Stage-2 performs:

rollout (no grad) -> strict parse -> match -> build one teacher-forced target -> masked losses

The canonical assistant training target is:

`Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

Key policies:
- Rollout parsing is STRICT (no JSON repair). Invalid predicted objects are dropped.
- Missing GT objects (FN) are always appended in the tail (recall recovery stays mandatory).
- Bbox geometry loss uses **SmoothL1 + CIoU** on expectation-decoded coords (no GIoU; boxes are canonicalized for CIoU stability).
- Text/structure CE is supervised with explicit masking/weights (Channel-A CE@A1; Channel-B supervises top-level `}` + `<|im_end|>` and supports semantic-tolerant matched desc masking).

---

## Stage-2 AB (Scheduler + Channels)

Stage-2 AB composes two channels:

- **Channel-A** (Expectation Loop): builds teacher-forced targets from GT (no rollouts), then runs packed SFT forward/backward.
- **Channel-B** (Rollout Matching): generates rollouts (no grad), parses + matches, builds `Y_train`, then runs packed SFT forward/backward.

Scheduler:
- Config: `stage2_ab.schedule.b_ratio: float` in `[0,1]` (0.0=A-only, 1.0=B-only, 0.05≈5% Channel-B).
- Runtime: deterministic Bresenham-style schedule from `TrainerState.global_step`:
  - Channel-B iff `floor((s+1)*b_ratio) > floor(s*b_ratio)`, else Channel-A.

### Channel-B Execution (Single Step-Budgeted Path)

Channel-B is standardized to a single step-budgeted pathway (legacy `micro/async` modes removed).

Key semantics:
- `training.effective_batch_size` is REQUIRED for `custom.trainer_variant: stage2_ab_training`.
- One optimizer step consumes exactly `effective_batch_size` raw rollout samples globally (across all learner ranks).
- Stage2-AB requires `effective_batch_size` to be divisible by `learner_world_size`, so each learner rank receives exactly `effective_batch_size / learner_world_size` raw samples.
- Raw rollouts are dynamically post-packed into a variable number of packed sequences (<= `global_max_length`), and the trainer runs
  multiple forward/backward passes inside the optimizer step (using `no_sync` for intermediate packs under DDP).

Rollout decode batching:
- Single knob: `rollout_matching.decode_batch_size`.
  - Definition: per rollout GPU cap for one `generation()` / `/infer/` call.
- HF + vLLM colocate:
  - Each learner rank chunks its local requests to `decode_batch_size`.
- vLLM server mode:
  - The learner queries each rollout server DP `world_size` via `/get_world_size/` and derives a per-rank request chunk size so the
    per-GPU cap holds when all learner ranks generate concurrently.
  - Pipeline overlap (produce segments while consuming packs) is enabled automatically in server mode.

Worked example (default launcher):
- 6 rollout GPUs (server DP world size = 6), 2 learner GPUs (DDP world size = 2), `decode_batch_size=4`:
  - per-rank chunk size = `floor(4 * 6 / 2) = 12` requests per call.
  - Across 2 ranks: 24 requests per synchronized round; average 4 per rollout GPU.

### Channel-A Contract (Expectation Loop)

- Default grad semantics: `stage2_ab.softctx_grad_mode: unroll` (no detach anywhere in the soft self-context loop). Use `em_detach` only for explicit ablations.
- CE anchor split: Channel-A computes CE on the **A1** teacher-forced logits and computes geometry (bbox loss + coord regularizers) from the **final** softctx iteration logits.

### Channel-B Contract (FP-neutral + Closure Supervision)

- Unified Channel-B is the only supported contract (rollout prefix + FN injection; no reorder path).
- CE masking policy (unified Channel-B):
  - matched prefix objects: structure CE ON, desc CE OFF, coord CE OFF;
  - FP prefix objects: structure/desc/coord CE all OFF;
  - FN-injected objects: structure CE ON, desc CE ON, coord CE OFF.
- FP-neutral geometry: Channel-B geometry loss includes matched prefix objects and FN-injected objects; FP objects contribute no geometry loss.
- Deterministic FN injection:
  - retain an append-ready rollout prefix inside `{"objects": [ ...`,
  - append unmatched GT records as extra `objects[]` elements,
  - insert a leading comma iff the retained prefix body already has object entries.
- Closure supervision: keep CE ON for the same outermost `}` used as FN injection anchor, and keep CE ON for `<|im_end|>` (no stop-neutral masking).
- Strict-drop diagnostics: invalid predicted objects are dropped deterministically (no repair) but counted in metrics:
  - `stage2_ab/channel_b/strict_drop/N_valid_pred`
  - `stage2_ab/channel_b/strict_drop/N_drop_invalid`
  - `stage2_ab/channel_b/strict_drop/reason/<bucket>`
  - Optional structure-token CE upweight: `stage2_ab.channel_b.drop_invalid_struct_ce_multiplier` (clamped to `[1.0, 4.0]`).

---

## Post-Rollout Packing (Important Gotchas)

Packing is supported post-rollout only:
- Enable with `training.packing: true`.
- Rollout generation remains un-packed (padded batch). The trainer temporarily disables padding-free / packing during rollouts.
- Stage-2 uses dynamic post-rollout packing inside the trainer (dataset-level packing wrappers are not used).
- Selection uses a deterministic constant-volume binpacking heuristic; the `binpacking` dependency is required when packing is enabled.
- Carry-only mode requires `training.packing_drop_last: true` (the trainer does not run flush steps at the end).
- Stage-2 uses micro-scope dynamic post-rollout packing only (window lookahead removed).

The rollout prefix is treated as immutable in token space:
- Only suffix-only trimming is allowed (no decode+re-encode of earlier tokens).

---

## Rollout Parsing Policy (Current)

The current rollout behavior commonly includes a trailing `<|im_end|>` token and can sometimes truncate mid-object.

Parsing policy:
- Treat `<|im_end|>` as a hard stop (strip it, even when fused into the final token).
- If the rollout is truncated mid-object, suffix-trim to the last complete object boundary.
- Keep the prefix append-ready inside the top-level `objects` array (`{"objects": [` or `{"objects": [{...}`).
- Failure fallback:
  - If no valid `{"objects": [...]}` container exists, or no append-ready prefix can be produced, use `Y_rollout_prefix = "{\"objects\": ["`
    (empty predicted set; all GT are appended as FN).

---

## Matching Knobs (Starting Point)

Reasonable smoke defaults:
- `candidate_top_k: 5`
- `maskiou_gate: 0.3`

Interpretation:
- `candidate_top_k` prunes GT candidates per predicted object before expensive geometry.
- `maskiou_gate` rejects low-quality matches early; rejected GT remain FN and are appended.

---

## Config Checklist

Start from a template config and fill in dataset + rollout knobs:

- Rollout-matching Stage-2 base: `configs/stage2_ab/base.yaml`
- Stage-2 AB examples: `configs/stage2_ab/`

Minimum required edits:
- Set `custom.train_jsonl` / `custom.val_jsonl`.
- Set top-level `rollout_matching.*` (including `decoding.*` + matching knobs).
- If using Stage-2 AB (`custom.trainer_variant: stage2_ab_training`), provide a top-level `stage2_ab` section (typed) including:
  - `stage2_ab.schedule.b_ratio`
- Set `training.packing: true` if you want post-rollout packing for the teacher-forced forward pass.

Breaking config migrations (no backward compatibility):
- Rollout sampling knobs are configured under `rollout_matching.decoding.*`:
  - `decoding.temperature`, `decoding.top_p`, `decoding.top_k`
- Rollout-matching settings must be authored under top-level `rollout_matching.*`:
  - `custom.extra.rollout_matching.*` is removed and MUST fail fast if present.
- Legacy keys are removed and MUST fail fast if present:
  - `rollout_matching.temperature`, `rollout_matching.top_p`, `rollout_matching.top_k`
  - `rollout_matching.rollout_buffer` (buffered reuse is removed; use vLLM server mode + derived chunking + `decode_batch_size` to scale throughput)
  - `stage2_ab.channel_b.reordered_gt_sft` (removed; unified Channel-B only)
  - `stage2_ab.channel_b.desc_ce_weight_matched` (removed; no matched-desc CE knob)
  - `stage2_ab.channel_b.semantic_desc_gate` (removed; no training-time semantic gating)

Logging tip:
- Stage-2 metrics are logged once per optimizer step (aggregated across gradient accumulation).
- If you reuse the same `training.run_name` and `training.logging_dir`, multiple `events.out.tfevents.*` files can accumulate.
  Prefer unique run names, or leave `training.logging_dir` unset (default unique per run).

---

## Colocate Offload (Peak Memory Relief During Rollouts)

If you use vLLM colocate mode and hit peak memory issues during rollouts, Stage-2 can optionally offload training state to CPU during rollout generation:

- `rollout_matching.offload.enabled: true`
- `rollout_matching.offload.offload_model: true` moves training model params to CPU during rollouts.
- `rollout_matching.offload.offload_optimizer: true` moves optimizer state to CPU during rollouts.

Notes:
- Offload is currently not supported with DeepSpeed/ZeRO in this trainer; if you need ZeRO, disable offload or switch `rollout_backend: hf`.

---

## Rollout Backend Options

### vLLM (Default)

Set `rollout_matching.rollout_backend: vllm`.

Note: vLLM rollouts currently support **non-beam decoding only** (`decode_mode=greedy` is enforced in code).
Sampling can still be enabled via `decoding.temperature/top_p/top_k` (see Decoding Tips below). Use `rollout_backend: hf` if you need beam search.

vLLM has a mode switch under `rollout_matching.vllm.mode`:

- `colocate` (default): learner instantiates a local vLLM engine on the same GPU(s) as training.
  - Requires `rollout_matching.vllm.max_model_len`.
  - Weight sync modes:
    - Recommended (default): `rollout_matching.vllm.enable_lora: false`
      - The trainer merges adapters into full weights and loads merged full weights into vLLM on rollout steps.
    - Optional: `rollout_matching.vllm.enable_lora: true`
      - The trainer pushes adapter tensors into vLLM via `add_lora` (faster, but can be unstable on multimodal stacks).

- `server` (recommended for long rollouts): learner connects to a pre-launched `swift rollout` server and generates rollouts on dedicated GPUs.
  - Supports multi-process learner (`torchrun`, `world_size > 1`).
  - Under `world_size > 1`, the trainer performs rank0-only weight sync with strict barriers and requires `rollout_matching.vllm.sync.mode: full`.
  - Connectivity is configured in YAML under `rollout_matching.vllm.server`.
  - Weight sync is configured under `rollout_matching.vllm.sync`:
    - `sync.mode: full` (default): full merged-weight sync (robust for multimodal + DoRA).
    - `sync.mode: adapter`: adapter-only sync (requires server launched with `--vllm_enable_lora true`).
    - `sync.fallback_to_full: true` permanently falls back to full sync if adapter sync fails at runtime.
  - Deploy-readiness gates (enforced by `scripts/stage2_ab_server_train.sh`):
    - `rollout_matching.rollout_backend=vllm` and `rollout_matching.vllm.mode=server`
    - `rollout_matching.vllm.server.servers[0].base_url` must be `http(s)://<host>:<port>`
    - `model.model` must point to a local model directory (avoid accidental Hub-ID resolution)
    - `server_gpus` and `train_gpus` must be disjoint device sets
    - no external repeat-terminate plugin is required

### HF (Fallback)

Set `rollout_matching.rollout_backend: hf`.

Notes:
- Rollout batching is controlled by a single knob: `rollout_matching.decode_batch_size`.
- For rollout-aware trainer variants, `training.per_device_eval_batch_size` and similar per-device eval knobs do not independently control rollout decode/eval chunking.
- HF `generate()` (and vLLM colocate) chunk per rank using the resolved `rollout_matching.decode_batch_size`.

---

## Decoding Tips

- Start with deterministic non-beam decoding for stability: `decode_mode: greedy`, `decoding.temperature: 0.0`.
- `decode_mode` is a **beam vs non-beam selector** in Stage-2 configs; sampling is controlled by `decoding.temperature/top_p/top_k`.
  - `decode_mode: greedy` can still produce **sampling** rollouts when `decoding.temperature > 0.0`.
  - Metrics tip: use `rollout/do_sample` + `rollout/temperature` to disambiguate sampling vs deterministic, not `rollout/decode_non_beam_count`.
- vLLM rollout backends currently enforce `decode_mode=greedy` (non-beam only); use `rollout_backend: hf` if you need beam search.
- For long dense JSON generations, set a mild `repetition_penalty` (e.g. `1.05`) to reduce loop-y rollouts.
- Ensure `max_new_tokens` is large enough to avoid systematic truncation (dense detection outputs can be very long).

---

## Optional: Description Monitor (Metrics Only)

Stage-2 can optionally monitor `desc` quality on matched pairs. This does not affect training loss.

Enable under `rollout_matching.desc_monitor`:

```yaml
rollout_matching:
  desc_monitor:
    enabled: true
    # 'exact'|'semantic'|'both'
    mode: semantic
    every_steps: 20
    semantic_model: sentence-transformers/all-MiniLM-L6-v2
    semantic_threshold: 0.6
    semantic_device: cpu
    semantic_batch_size: 32
    max_pairs: 64
```

---

## GPU Topology (Server Mode)

Server mode (recommended for long rollouts) runs rollouts on dedicated GPUs and supports a multi-GPU learner:

- Constraint: `server_gpus` and `train_gpus` must be disjoint.
- vLLM parallelism constraint: `len(server_gpus) == server_dp * server_tp`.

Recommended starting points (when each GPU can fit the full model):

- 8 GPUs (balanced; **server data-parallel**, learner DDP):
  - `server_gpus=0,1,2,3 train_gpus=4,5,6,7`
  - Default launcher behavior: `server_tp=1` (so `server_dp=4`).
  - Why: maximizes rollout throughput while keeping learner throughput high.

- 4 GPUs (minimal server, multi-GPU learner):
  - `server_gpus=0 train_gpus=1,2,3`
  - Why: keeps rollouts on a dedicated GPU while preserving a multi-GPU learner.

If the model / long-context KV cache does **not** fit as a single replica, use tensor-parallel server mode:
- Example: `server_gpus=0,1,2,3 server_tp=4` -> `server_dp=1` (one sharded server engine).

Notes:
- Server and learner GPU sets must be disjoint.
- Server GPUs will be idle on steps that do not call the rollout backend (e.g., Stage-2 AB Channel-A steps). This is expected.
  - In this idle state, vLLM may still reserve a large amount of VRAM for weights/KV cache, so `nvidia-smi` can show high memory usage with near-zero utilization.

---

## Launch (vLLM Server Mode)

Recommended launcher (starts server + learner, disables proxies, waits for `/health/`, cleans up on exit):

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost

bash scripts/stage2_ab_server_train.sh \
  server_gpus=0,1,2,3 train_gpus=4,5,6,7 \
  vllm_gpu_memory_utilization=0.75 \
  config=configs/stage2_ab/prod/ab_mixed.yaml
```

Launcher knobs (runtime-only; no YAML drift):
- `server_tp=<int>`: tensor-parallel degree for vLLM rollout server (default: 1).
- `server_dp=<int>`: data-parallel degree for vLLM rollout server (default: derived from `len(server_gpus) / server_tp`).
- `server_torch_dtype=bfloat16|float16|float32|None`: server model dtype passed to `swift rollout` (default: `bfloat16`).
- `server_vllm_enforce_eager=true|false`: eager mode for vLLM server (default: `true`).

Operational tip:
- Run the launcher inside `tmux` so a single `Ctrl-C` cleanly terminates both learner and server and frees GPU memory quickly.

---

## Evaluation (Production-Style)

Stage-2 evaluation runs a production-style pipeline on `custom.val_jsonl`:

rollout (no grad) -> strict parse -> Hungarian match -> report metrics

Important:
- Eval intentionally skips teacher-forced encoding and loss computation to keep eval fast and reflective of real rollout performance.
- As a result, Stage-2 eval does not report `eval_loss`.

Eval metrics include:
- `eval_rollout/precision`, `eval_rollout/recall`, `eval_rollout/f1`
- Counters: `eval_rollout/pred_objects`, `eval_rollout/gt_objects`, `eval_rollout/matched`, `eval_rollout/fp`, `eval_rollout/fn`
- Parse health: `eval_rollout/parse_truncated_rate`, `eval_rollout/parse_dropped_invalid`, `eval_rollout/parse_dropped_ambiguous`
- Sample health: `eval_rollout/sample_valid_pred_rate`, `eval_rollout/sample_any_match_rate`
- Geometry quality: `eval_rollout/matched_maskiou_mean`

Best-checkpoint selection:
- Prefer `training.metric_for_best_model: rollout/f1` and `training.greater_is_better: true`.

---

## Monitoring & Metrics

### Key Health Metrics (Most Load-Bearing)

Channel scheduling (AB):
- `stage2/channel_a` (1 on A steps)
- `stage2/channel_b` (1 on B steps)

Rollout health:
- `rollout/parse_truncated_rate`
- `rollout/sample_valid_pred_rate`
- `rollout/f1`

Throughput:
- `time/rollout_generate_s`
- `rollout/gen_tokens_per_s`

Stage-2 AB extras:
- `stage2_ab/channel_b/strict_drop/*` (strict-drop diagnostics; see Channel-B contract above)
- `stage2_ab/channel_b/closure_supervision/N_drop` (closure-marker resolution drops; should stay near 0)

### Qualitative Monitoring Dumps

Enable periodic dumps of (Prompt, Rollout, Target) triplets (rank0 only):

```yaml
rollout_matching:
  monitor_dump:
    enabled: true
    # If omitted, follows training.logging_steps.
    every_steps: 4
    max_events: 50
    max_samples: 1
    max_text_chars: 4000
```

Outputs land in `<training.output_dir>/monitor_dumps/` (both `.json` and `.md` per dump).

---

## Troubleshooting

### Stuck at First Rollout Step (Server Mode)

Symptom:
- `/health/` returns 200, but training hangs at 0% GPU util.

Interpretation:
- `/health/` validates the HTTP server, but `group_port` is a separate TCP port used to initialize a communicator for weight sync.
- If `group_port` is not listening/reachable, the learner can block before the first rollout step.

Checks:
1) Confirm HTTP health:
   - `curl --noproxy '*' -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health/`
2) Confirm `group_port` is open:
   - `conda run -n ms python - <<'PY'\nimport socket\nhost='127.0.0.1'; port=51216\ns=socket.socket(); s.settimeout(2)\ntry:\n    s.connect((host, port))\n    print('group_port connect: ok')\nexcept Exception as e:\n    print('group_port connect: failed', e)\nfinally:\n    s.close()\nPY`

Mitigations:
- Change `vllm.server.servers[].group_port` to an unused port and restart server and learner.
- Ensure localhost connections are not routed through proxies (prefer the helper launcher, or unset proxies + set `NO_PROXY`).

### Channel-B never executes

Symptom:
- `stage2/channel_b` stays 0.0 across training (Channel-A runs instead).

Interpretation:
- In the standardized pathway, Channel-B execution is deterministic and schedule-driven. This typically indicates:
  - `stage2_ab.schedule.b_ratio` is 0.0 (A-only), or
  - you are not actually running `custom.trainer_variant: stage2_ab_training`.

Checks:
1) Confirm schedule:
   - `stage2_ab.schedule.b_ratio` (1.0 for B-only; 0.0 for A-only).
2) Confirm trainer variant:
   - `custom.trainer_variant: stage2_ab_training`

Mitigations (smoke/debug runs):
- Reduce rollout length (`rollout_matching.max_new_tokens`) for faster steps.
- Reduce rollout decode batching by setting `rollout_matching.decode_batch_size: 1`.
- Lower `vllm_gpu_memory_utilization` or increase server GPUs if the server is capacity-bound.

### Length Constraints ("Long rollout" failures)

Two separate limits interact:
- `rollout_matching.max_new_tokens`: rollout generation budget.
- `global_max_length`: model/server max length (also passed as vLLM `--vllm_max_model_len` in server mode).

Rule of thumb:
- Ensure `global_max_length >= prompt_len + max_new_tokens`.
- If `max_new_tokens` is ~3k, `global_max_length: 2048` is too small; use `global_max_length: 4096` (or higher).

---

## Preflight Validation (Suggested)

- Unit tests (Stage-2):
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_rollout_matching_sft.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_rollout_offload_context.py`

---

## Rollout Backend Benchmarking (HF vs vLLM)

If you want to benchmark rollout generation throughput (analysis only; not an official launch script):

- Runner: `scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py`
- Example configs: `configs/bench/rollout_backend_bench*.yaml`

General expectations:
- vLLM is typically substantially faster for long rollouts but can reserve a large KV cache.
- `gpu_memory_utilization` primarily controls how much VRAM vLLM can reserve (mostly KV cache); lowering it can reduce peak VRAM.

---

## See Also

- **Metrics Guide**: [`METRICS_LOSSES.md`](METRICS_LOSSES.md)
- **Packing Guide**: [`../data/PACKING.md`](../data/PACKING.md)
- **Stage-1 SFT**: [`../data/README.md`](../data/README.md)
