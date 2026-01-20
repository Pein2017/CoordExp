# Stage-2 (Rollout-Matching) SFT Runbook

This doc is a minimal “paper-ready” checklist for running the rollout-matching SFT
trainer (stage_2), enabled via:

`custom.trainer_variant: rollout_matching_sft`

Authoritative requirements live under:
- `openspec/changes/2026-01-15-add-rollout-matching-trainer/specs/rollout-matching-sft/spec.md`
- `openspec/changes/2026-01-16-add-stage2-post-rollout-packing/specs/rollout-matching-sft/spec.md`
- `openspec/changes/2026-01-19-add-stage2-rollout-buffer-offload/specs/rollout-matching-sft/spec.md`
- `openspec/changes/2026-01-19-update-stage2-post-rollout-packing-binpacking/specs/rollout-matching-sft/spec.md`

## What Stage-2 Does (One Forward Pass)

Stage_2 performs:

rollout (no grad) -> strict parse -> match -> build one teacher-forced target -> masked losses

The canonical assistant training target is:

`Y_train = Y_rollout_prefix + SerializeAppend(FN_gt_objects) + EOS`

Key policies:
- Rollout parsing is STRICT (no JSON repair). Invalid predicted objects are DROPPED.
- Missing GT objects (FN) are ALWAYS appended in the tail (recall recovery stays mandatory).
- Coord supervision stays token-distributional (softCE + W1 + gate) at coord slots.
- `desc` string VALUE tokens are not supervised by CE in stage_2 (JSON structure remains supervised).

## Hard Constraints / Gotchas

- Packing is supported **post-rollout only**:
  - Enable with `training.packing: true`.
  - Rollout generation remains un-packed (padded batch) and the trainer temporarily disables
    `template.padding_free/template.packing` during rollouts.
  - Stage_2 uses **dynamic post-rollout packing inside the trainer** (dataset-level packing wrappers are not used).
  - Selection uses a deterministic, ms-swift-like constant-volume binpacking heuristic; the `binpacking` dependency is
    required when packing is enabled.
  - Carry-only mode requires `training.packing_drop_last: true` (the trainer does not run flush steps at the end).
- The rollout prefix is treated as immutable in token space:
  - Only suffix-only trimming is allowed (no decode+re-encode of earlier tokens).

## Rollout Parsing Policy (Current Rollouts)

The current rollout behavior (20-sample smoke at
`output/infer/rollout_ckpt3106_smoke/pred.jsonl`) commonly includes a trailing
`<|im_end|>` token and occasionally true truncation mid-object.

Parsing policy:
- Treat `<|im_end|>` as a hard stop (strip it, even when fused into the final token).
- If the rollout is truncated mid-object, suffix-trim to the last complete object boundary.
- Make the prefix append-ready by dropping the final top-level `}` (open JSON object).
- Failure fallback:
  - If no opening `{` exists, or no append-ready prefix can be produced, use
    `Y_rollout_prefix = "{"` (no prefix supervision; all GT become FN and are appended).

## Recommended Matching Knobs (Starting Point)

These are reasonable smoke defaults (tune later with logs):
- `candidate_top_k: 5`
- `maskiou_gate: 0.3`

Interpretation:
- `candidate_top_k` prunes GT candidates per predicted object before expensive geometry.
- `maskiou_gate` rejects low-quality matches early; rejected GT remain FN and are appended.

## Config Checklist

Start from: `configs/rollout_matching_sft_template.yaml`

Set:
- `custom.train_jsonl`
- `custom.val_jsonl`
- `custom.extra.rollout_matching.*` (decode + matching knobs)
- `training.packing: true` to enable post-rollout packing for the teacher-forced forward pass

Optional desc monitoring (metrics only; does not affect loss):
- Enable with `custom.extra.rollout_matching.desc_monitor.enabled: true`.
- Suggested starting point:

```yaml
custom:
  extra:
    rollout_matching:
      desc_monitor:
        enabled: true
        # 'exact'|'semantic'|'both'
        mode: semantic
        # Run on every N optimizer steps (only runs on E-steps when buffering is enabled).
        every_steps: 20
        # Sentence embedding model used for semantic matching.
        semantic_model: sentence-transformers/all-MiniLM-L6-v2
        semantic_threshold: 0.6
        semantic_device: cpu
        semantic_batch_size: 64
        # Cap matched pairs per batch for monitoring cost control.
        max_pairs: 64
```

Buffered rollouts (E-step / M-step reuse):
- `custom.extra.rollout_matching.rollout_buffer.enabled: true` enables caching + reuse of one completed
  accumulation window across multiple optimizer steps.
- `custom.extra.rollout_matching.rollout_buffer.m_steps: <int>` controls how many optimizer steps reuse one
  window (counts in `TrainerState.global_step` units; `m_steps=1` disables reuse).
- When buffering is enabled with `m_steps > 1`, the stage_2 trainer repeats each *gradient accumulation window*
  `m_steps` times per rank (e.g., `A,B,C, A,B,C, ...` for `gas=3, m_steps=2`) to avoid silently skipping dataset
  samples.
- Final partial accumulation windows (< `gradient_accumulation_steps`) are processed once and MUST NOT be repeated.
  If you want strict reuse only on full windows, set `training.dataloader_drop_last: true`.
- Evaluation/prediction forces `m_steps=1` (buffering disabled) to keep metrics interpretable,
  and uses a production-style evaluator (rollout -> parse -> match only; no teacher-forced loss).
- Checkpoint/resume: the buffer is runtime-only and starts empty after resume (first step regenerates).

Colocate vLLM offload (peak memory relief during rollouts):
- `custom.extra.rollout_matching.offload.enabled: true` enables offload during vLLM colocate rollouts only.
- `custom.extra.rollout_matching.offload.offload_model: true` moves training model params to CPU during rollouts.
- `custom.extra.rollout_matching.offload.offload_optimizer: true` moves optimizer state to CPU during rollouts.
- Offload is currently **not supported** with DeepSpeed/ZeRO in this trainer; if you need it, disable offload or
  switch `rollout_backend: hf`.

Rollout backend:
- Default: vLLM colocate (`custom.extra.rollout_matching.rollout_backend: vllm`)
  - Requires `custom.extra.rollout_matching.vllm.max_model_len`
  - Weight sync modes:
    - **Recommended (default):** `custom.extra.rollout_matching.vllm.enable_lora: false`
      - The trainer merges adapters into the training model weights and loads the merged full weights into vLLM
        on E-steps ("GRPO-style"). This is significantly more robust for Qwen3-VL multimodal stacks.
    - Optional: `custom.extra.rollout_matching.vllm.enable_lora: true`
      - The trainer pushes adapter tensors into vLLM via `add_lora` (faster, but can be unstable on multimodal).
- Fallback (explicit): HF (`custom.extra.rollout_matching.rollout_backend: hf`)
  - To get *batched* HF rollouts, you must also increase `training.per_device_train_batch_size` (otherwise each rank rolls out 1 sample).
  - `custom.extra.rollout_matching.rollout_generate_batch_size` controls the per-rank microbatch size for HF generate().

Decoding notes:
- Start with greedy (`decode_mode: greedy`, `temperature: 0.0`) for stability.
- For long dense JSON generations, set a mild `repetition_penalty` (e.g. `1.05`) to reduce loop-y rollouts.
- If HF rollouts occasionally get stuck generating repetitive garbage until `max_new_tokens`, enable
  `repeat_terminate` to force EOS for the offending sequences (batch-friendly; does not stop the whole batch).
- Ensure `max_new_tokens` is large enough to avoid systematic truncation
  (LVIS dense outputs can be ~11k text tokens in the tail).

## Command

From repo root:

`PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>]`

4 GPUs:

`PYTHONPATH=. conda run -n ms torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]`

## Evaluation (Production-Style)

Stage_2 evaluation (`training.eval_strategy`/`training.eval_steps`) runs a production-style
pipeline on `custom.val_jsonl`:

rollout (no grad) -> strict parse -> Hungarian match -> report metrics

Important:
- Eval intentionally **skips** teacher-forced encoding and loss computation to keep eval fast and
  reflective of real rollout performance on unseen data.
- As a result, Stage_2 eval does **not** report `eval_loss`.

Eval metrics:
- `eval_rollout_precision`, `eval_rollout_recall`, `eval_rollout_f1`
- `eval_rollout_pred_objects`, `eval_rollout_gt_objects`, `eval_rollout_matched`, `eval_rollout_fp`, `eval_rollout_fn`
- Parse health: `eval_rollout_parse_truncated_rate`, `eval_rollout_parse_dropped_invalid`, `eval_rollout_parse_dropped_ambiguous`
- Sample health: `eval_rollout_sample_valid_pred_rate`, `eval_rollout_sample_any_match_rate`
- Geometry quality: `eval_rollout_matched_maskiou_mean`
- (Optional) Desc monitor:
  - `eval_rollout_desc_pairs_total`, `eval_rollout_desc_exact_acc_on_matched`
  - `eval_rollout_desc_sem_enabled`, `eval_rollout_desc_sem_acc_on_matched`
  - `eval_rollout_desc_sem_sim_mean`

Best-checkpoint selection:
- For Stage_2 runs, prefer `training.metric_for_best_model: eval_rollout_f1` and
  `training.greater_is_better: true`.

## Health Counters to Watch

Stage_2 logs both rollout health counters and lightweight quality metrics. Under
rollout buffering, interpret rollout-quality metrics on **E-steps only**:
filter to `rollout/buffer_reuse == 0` (fresh rollouts). M-steps reuse cached
targets and should not be used to judge on-policy rollout quality.

Parsing/matching counters:
- `rollout/parse_dropped_invalid`
- `rollout/parse_dropped_ambiguous`
- `rollout/parse_truncated`
- `rollout/parse_truncated_rate`
- `rollout/parse_obj_valid_frac`
- `rollout/valid_pred_objects`
- `rollout/matched_for_supervision`
- `rollout/excluded_from_supervision`
- `rollout/fn_appended`
- `rollout/gating_rejections`
- `rollout/gating_rejection_rate`

Quality metrics (E-step only recommended):
- `rollout/precision`
- `rollout/recall` (same as `rollout/match_rate`)
- `rollout/f1`
- `rollout/matched_maskiou_mean` (mean maskIoU over matched pairs; norm1000 space)

Generation/throughput (helps diagnose long-rollout slowdowns and OOM risk):
- `rollout/gen_new_tokens_mean`
- `rollout/gen_new_tokens_p90`
- `rollout/gen_new_tokens_p99`
- `rollout/gen_tokens_per_s`
- `time/rollout_generate_s` (forced to 0 on buffer reuse steps)

Loss breakdown:
- `loss/ce`
- `loss/coord`
- `loss/coord_prefix`
- `loss/coord_tail`

Buffered-mode diagnostics:
- `rollout/buffer_reuse` (1 on M-steps, 0 on E-steps)
- `rollout/buffer_window_step0`
- `rollout/buffer_completed_steps`
- `rollout/buffer_micro_idx`

## Qualitative Monitoring Dumps (Rollout vs GT vs Training Target)

For paper/debug runs, you can periodically dump a small number of qualitative
examples (rank0 only), aligned to the same optimizer-step notion as buffering.

Enable under `custom.extra.rollout_matching.monitor_dump`:

```yaml
custom:
  extra:
    rollout_matching:
      monitor_dump:
        enabled: true
        # If omitted, follows training.logging_steps. In buffered mode, using
        # every_steps == rollout_buffer.m_steps aligns dumps to E-steps.
        every_steps: 4
        max_events: 50
        max_samples: 1
        max_text_chars: 4000
```

Outputs land in:
- `<training.output_dir>/monitor_dumps/step_000012.json`
- `<training.output_dir>/monitor_dumps/step_000012.md`

Each dump includes:
- prompt `messages`
- rollout text (raw)
- append-ready prefix used for matching
- training target text (prefix + FN append)
- GT/pred objects + match details (with per-pair maskIoU)

## Minimal Preflight Validation

- Spec validity:
  - `openspec validate 2026-01-15-add-rollout-matching-trainer --strict`
  - `openspec validate 2026-01-19-add-stage2-rollout-buffer-offload --strict`
- Unit tests:
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_rollout_matching_sft.py -q`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_rollout_buffer.py -q`

## Tiny Ablation: Rollout Backend Efficiency (HF vs vLLM)

Stage-2 rollouts are **inference-only** (no grad), so the rollout stage can be run
with either:
- HF backend: `swift.llm.PtEngine`
- vLLM backend: `swift.llm.VllmEngine`

Benchmark runner (analysis-only, not an official launch script):
- `analysis/rollout_backend_bench/benchmark_rollout_backends.py`

Configs used:
- `configs/bench/rollout_backend_bench_ckpt3106.yaml` (vLLM `gpu_memory_utilization=0.85`)
- `configs/bench/rollout_backend_bench_ckpt3106_vllm05.yaml` (vLLM `gpu_memory_utilization=0.5`)

Hardware + settings (kept identical across backends):
- 2× A100 80GB (CUDA 0,1)
- checkpoint: `output/12-24/coord_loss-merged/ckpt-3106`
- dataset: `public_data/lvis/rescale_32_768_poly_20/*coord.jsonl`
- decoding: greedy (`temperature=0.0`), `max_new_tokens=256`
- context: `max_length=4096`, `max_model_len=4096`
- protocol: 32 samples / GPU, 2 repeats (4 runs total)

Results (rollouts/s is higher-is-better; peak mem is per-GPU):
- vLLM (0.85): ~**1.81×** faster than HF, but **~67.8 GB** peak GPU mem
  - run: `output/bench/rollout_backend_bench/20260117_084925/aggregate_report_corrected.json`
- vLLM (0.5): ~**1.84×** faster than HF, with **~39.4 GB** peak GPU mem
  - run: `output/bench/rollout_backend_bench/20260117_094726/aggregate_report.json`

Interpretation / gotchas:
- vLLM speedup is substantial for rollout generation, but vLLM reserves a large **KV cache** by default.
- `gpu_memory_utilization` primarily controls how much VRAM vLLM can reserve (mostly KV cache). Lowering it
  can reduce peak VRAM dramatically without harming single-request throughput (as seen at 0.5 here).
- vLLM is still inference-only: you cannot run SFT backward in vLLM, and you generally should not colocate
  a vLLM engine on the same GPU as stage-2 training forward/backward unless you have ample headroom.

Reproduce (CUDA 0,1):
- `PYTHONPATH=. /root/miniconda3/envs/ms/bin/python analysis/rollout_backend_bench/benchmark_rollout_backends.py --config configs/bench/rollout_backend_bench_ckpt3106.yaml --multi_gpu --gpus 0,1`
- `PYTHONPATH=. /root/miniconda3/envs/ms/bin/python analysis/rollout_backend_bench/benchmark_rollout_backends.py --config configs/bench/rollout_backend_bench_ckpt3106_vllm05.yaml --multi_gpu --gpus 0,1`

Optional visual sanity check (renders GT vs HF vs vLLM):
- `export ROOT_IMAGE_DIR=public_data/lvis/rescale_32_768_poly_20`
- `PYTHONPATH=. /root/miniconda3/envs/ms/bin/python analysis/rollout_backend_bench/vis_rollout_backend_compare.py --compare_jsonl output/bench/rollout_backend_bench/<run>/compare_gpu0_seed17.jsonl --save_dir vis_out/rollout_backend_compare --limit 50`
