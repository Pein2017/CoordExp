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

Multi-GPU (NOT for vLLM server mode):

```bash
PYTHONPATH=. conda run -n ms torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]
```

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
- Coord supervision remains token-distributional (softCE + W1 + gate) at coord slots.
- `desc` value tokens are NOT supervised by CE in Stage-2 (JSON structure remains supervised).

---

## Stage-2 AB (Scheduler + Channels)

Stage-2 AB composes two channels:

- **Channel-A** (Expectation Loop): builds teacher-forced targets from GT (no rollouts), then runs packed SFT forward/backward.
- **Channel-B** (Rollout Matching): generates rollouts (no grad), parses + matches, builds `Y_train`, then runs packed SFT forward/backward.

Scheduler:
- Config: `custom.extra.stage2_ab.schedule.pattern: ["A", "A", "B"]` (example).
- Runtime: channel is chosen by `TrainerState.global_step % len(pattern)` (A/B only).

### Channel-B Modes (Step vs Micro)

Channel-B supports two execution modes under `custom.extra.stage2_ab.channel_b.mode`:

- `micro` (legacy): each micro-step independently runs rollout -> pack -> learn.
- `step` (recommended / current default): "step-budgeted" rollouts + packing.
  - Collect raw dataset samples across the full gradient-accumulation window.
  - Run rollout generation + parse/match.
  - Pack into a variable number of packed sequences (each capped by `global_max_length`) and run forward/backward once per pack.
  - The outer Trainer still performs exactly one `optimizer.step()` for the optimizer step.

Constraints for `mode: step`:
- Requires `training.packing: true`.
- Requires `training.dataloader_drop_last: true` to avoid partial accumulation windows at epoch end.
- Keep `custom.extra.rollout_matching.post_rollout_pack_scope: micro` (step mode does packing inside the optimizer step).

### Channel-B Pipelining (Server Mode Only)

`custom.extra.stage2_ab.channel_b.enable_pipeline: true` overlaps rollout generation (server GPUs) with learner packing/learning.

Requirements:
- `custom.extra.rollout_matching.rollout_backend: vllm`
- `custom.extra.rollout_matching.vllm.mode: server`

Tuning:
- `custom.extra.stage2_ab.channel_b.rollout_decode_batch_size` controls the decode chunk size used by the pipeline.

---

## Post-Rollout Packing (Important Gotchas)

Packing is supported post-rollout only:
- Enable with `training.packing: true`.
- Rollout generation remains un-packed (padded batch). The trainer temporarily disables padding-free / packing during rollouts.
- Stage-2 uses dynamic post-rollout packing inside the trainer (dataset-level packing wrappers are not used).
- Selection uses a deterministic constant-volume binpacking heuristic; the `binpacking` dependency is required when packing is enabled.
- Carry-only mode requires `training.packing_drop_last: true` (the trainer does not run flush steps at the end).
- Optional train-only scheduling knob:
  - `custom.extra.rollout_matching.post_rollout_pack_scope: micro` (default): pack immediately per micro-step / per step budget.
  - `custom.extra.rollout_matching.post_rollout_pack_scope: window`: accumulate segments across a full gradient-accumulation window and schedule packing within-window.

The rollout prefix is treated as immutable in token space:
- Only suffix-only trimming is allowed (no decode+re-encode of earlier tokens).

---

## Rollout Parsing Policy (Current)

The current rollout behavior commonly includes a trailing `<|im_end|>` token and can sometimes truncate mid-object.

Parsing policy:
- Treat `<|im_end|>` as a hard stop (strip it, even when fused into the final token).
- If the rollout is truncated mid-object, suffix-trim to the last complete object boundary.
- Make the prefix append-ready by dropping the final top-level `}` (open JSON object).
- Failure fallback:
  - If no opening `{` exists, or no append-ready prefix can be produced, use `Y_rollout_prefix = "{"`
    (no prefix supervision; all GT become FN and are appended).

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

- Rollout-matching SFT template: `configs/stage2_ab/rollout_matching_sft_template.yaml`
- Stage-2 AB examples: `configs/stage2_ab/`

Minimum required edits:
- Set `custom.train_jsonl` / `custom.val_jsonl`.
- Set `custom.extra.rollout_matching.*` (decode + matching knobs).
- Set `training.packing: true` if you want post-rollout packing for the teacher-forced forward pass.

Logging tip:
- Stage-2 metrics are logged once per optimizer step (aggregated across gradient accumulation).
- If you reuse the same `training.run_name` and `training.logging_dir`, multiple `events.out.tfevents.*` files can accumulate.
  Prefer unique run names, or leave `training.logging_dir` unset (default unique per run).

---

## Rollout Buffering (E-step / M-step Reuse)

Buffered rollouts enable caching + reuse of one completed accumulation window across multiple optimizer steps:

- Enable with `custom.extra.rollout_matching.rollout_buffer.enabled: true`.
- `custom.extra.rollout_matching.rollout_buffer.m_steps: <int>` controls how many optimizer steps reuse one window.
  - `m_steps=1` disables reuse.
- When buffering is enabled with `m_steps > 1`, the Stage-2 trainer repeats each gradient-accumulation window `m_steps` times per rank to avoid silently skipping dataset samples.
- Final partial accumulation windows (< `gradient_accumulation_steps`) are processed once and must not be repeated.
  - If you want strict reuse only on full windows, set `training.dataloader_drop_last: true`.
- Evaluation/prediction forces `m_steps=1` (buffering disabled) to keep metrics interpretable.
- Checkpoint/resume: the buffer is runtime-only and starts empty after resume (first step regenerates).

---

## Colocate Offload (Peak Memory Relief During Rollouts)

If you use vLLM colocate mode and hit peak memory issues during rollouts, Stage-2 can optionally offload training state to CPU during rollout generation:

- `custom.extra.rollout_matching.offload.enabled: true`
- `custom.extra.rollout_matching.offload.offload_model: true` moves training model params to CPU during rollouts.
- `custom.extra.rollout_matching.offload.offload_optimizer: true` moves optimizer state to CPU during rollouts.

Notes:
- Offload is currently not supported with DeepSpeed/ZeRO in this trainer; if you need ZeRO, disable offload or switch `rollout_backend: hf`.

---

## Rollout Backend Options

### vLLM (Default)

Set `custom.extra.rollout_matching.rollout_backend: vllm`.

Note: vLLM rollouts currently support `decode_mode=greedy` only (enforced in code). Use `rollout_backend: hf` if you need beam search.

vLLM has a mode switch under `custom.extra.rollout_matching.vllm.mode`:

- `colocate` (default): learner instantiates a local vLLM engine on the same GPU(s) as training.
  - Requires `custom.extra.rollout_matching.vllm.max_model_len`.
  - Weight sync modes:
    - Recommended (default): `custom.extra.rollout_matching.vllm.enable_lora: false`
      - The trainer merges adapters into full weights and loads merged full weights into vLLM on rollout steps.
    - Optional: `custom.extra.rollout_matching.vllm.enable_lora: true`
      - The trainer pushes adapter tensors into vLLM via `add_lora` (faster, but can be unstable on multimodal stacks).

- `server` (recommended for long rollouts): learner connects to a pre-launched `swift rollout` server and generates rollouts on dedicated GPUs.
  - v1 constraint: learner must run as a single process (`world_size == 1`); do not launch learner with `torchrun`.
  - Connectivity is configured in YAML under `custom.extra.rollout_matching.vllm.server`.
  - Weight sync is configured under `custom.extra.rollout_matching.vllm.sync`:
    - `sync.mode: full` (default): full merged-weight sync (robust for multimodal + DoRA).
    - `sync.mode: adapter`: adapter-only sync (requires server launched with `--vllm_enable_lora true`).
    - `sync.fallback_to_full: true` permanently falls back to full sync if adapter sync fails at runtime.

### HF (Fallback)

Set `custom.extra.rollout_matching.rollout_backend: hf`.

Notes:
- HF rollouts are batched only if you increase `training.per_device_train_batch_size` (otherwise each rank rolls out 1 sample).
- `custom.extra.rollout_matching.rollout_generate_batch_size` controls the per-rank microbatch size for HF `generate()`.

---

## Decoding Tips

- Start with greedy for stability: `decode_mode: greedy`, `temperature: 0.0`.
- vLLM rollout backends currently support `decode_mode=greedy` only; use `rollout_backend: hf` if you need beam search.
- For long dense JSON generations, set a mild `repetition_penalty` (e.g. `1.05`) to reduce loop-y rollouts.
- If HF rollouts occasionally generate repetitive garbage until `max_new_tokens`, enable `repeat_terminate` to force EOS for the offending sequences.
- Ensure `max_new_tokens` is large enough to avoid systematic truncation (dense detection outputs can be very long).

---

## Optional: Description Monitor (Metrics Only)

Stage-2 can optionally monitor `desc` quality on matched pairs. This does not affect training loss.

Enable under `custom.extra.rollout_matching.desc_monitor`:

```yaml
custom:
  extra:
    rollout_matching:
      desc_monitor:
        enabled: true
        # 'exact'|'semantic'|'both'
        mode: semantic
        every_steps: 20
        semantic_model: sentence-transformers/all-MiniLM-L6-v2
        semantic_threshold: 0.6
        semantic_device: cpu
        semantic_batch_size: 64
        max_pairs: 64
```

---

## GPU Topology (Server Mode)

Server mode (recommended for long rollouts) runs rollouts on dedicated GPUs and keeps the learner single-process:

- 4 GPUs: 3 server (rollouts) + 1 learner (training) -> `server_gpus=0,1,2 train_gpus=3`.
- 8 GPUs: 7 server + 1 learner -> `server_gpus=0,1,2,3,4,5,6 train_gpus=7`.

Notes:
- The learner must be `world_size == 1` in server mode.
- Server GPUs will be idle on steps that do not call the rollout backend (e.g., Stage-2 AB Channel-A steps). This is expected.

---

## Launch (vLLM Server Mode)

Recommended launcher (starts server + learner, disables proxies, waits for `/health/`, cleans up on exit):

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost

bash scripts/stage2_ab_server_train.sh \
  server_gpus=0,1,2 train_gpus=3 \
  vllm_gpu_memory_utilization=0.75 \
  config=configs/stage2_ab/prod/ab_mixed.yaml
```

---

## Evaluation (Production-Style)

Stage-2 evaluation runs a production-style pipeline on `custom.val_jsonl`:

rollout (no grad) -> strict parse -> Hungarian match -> report metrics

Important:
- Eval intentionally skips teacher-forced encoding and loss computation to keep eval fast and reflective of real rollout performance.
- As a result, Stage-2 eval does not report `eval_loss`.

Eval metrics include:
- `eval_rollout_precision`, `eval_rollout_recall`, `eval_rollout_f1`
- Counters: `eval_rollout_pred_objects`, `eval_rollout_gt_objects`, `eval_rollout_matched`, `eval_rollout_fp`, `eval_rollout_fn`
- Parse health: `eval_rollout_parse_truncated_rate`, `eval_rollout_parse_dropped_invalid`, `eval_rollout_parse_dropped_ambiguous`
- Sample health: `eval_rollout_sample_valid_pred_rate`, `eval_rollout_sample_any_match_rate`
- Geometry quality: `eval_rollout_matched_maskiou_mean`

Best-checkpoint selection:
- Prefer `training.metric_for_best_model: eval_rollout_f1` and `training.greater_is_better: true`.

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
- `time/rollout_generate_s` (forced to 0 on buffer reuse steps)
- `rollout/gen_tokens_per_s`

### Buffering Note (E-step Only for Quality)

Under rollout buffering, interpret rollout-quality metrics on fresh-rollout steps only:
filter to `rollout/buffer_reuse == 0`.

### Qualitative Monitoring Dumps

Enable periodic dumps of (Prompt, Rollout, Target) triplets (rank0 only):

```yaml
custom:
  extra:
    rollout_matching:
      monitor_dump:
        enabled: true
        # If omitted, follows training.logging_steps. In buffered mode, using
        # every_steps == rollout_buffer.m_steps aligns dumps to fresh-rollout steps.
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
   - `python - <<'PY'\nimport socket\nhost='127.0.0.1'; port=51216\ns=socket.socket(); s.settimeout(2)\ntry:\n    s.connect((host, port))\n    print('group_port connect: ok')\nexcept Exception as e:\n    print('group_port connect: failed', e)\nfinally:\n    s.close()\nPY`

Mitigations:
- Change `vllm.server.servers[].group_port` to an unused port and restart server and learner.
- Ensure localhost connections are not routed through proxies (prefer the helper launcher, or unset proxies + set `NO_PROXY`).

### Length Constraints ("Long rollout" failures)

Two separate limits interact:
- `custom.extra.rollout_matching.max_new_tokens`: rollout generation budget.
- `global_max_length`: model/server max length (also passed as vLLM `--vllm_max_model_len` in server mode).

Rule of thumb:
- Ensure `global_max_length >= prompt_len + max_new_tokens`.
- If `max_new_tokens` is ~3k, `global_max_length: 2048` is too small; use `global_max_length: 4096` (or higher).

---

## Preflight Validation (Suggested)

- Unit tests (Stage-2):
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_rollout_matching_sft.py -q`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_rollout_buffer.py -q`

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
