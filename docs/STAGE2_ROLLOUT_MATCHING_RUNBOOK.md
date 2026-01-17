# Stage-2 (Rollout-Matching) SFT Runbook

This doc is a minimal “paper-ready” checklist for running the rollout-matching SFT
trainer (stage_2), enabled via:

`custom.trainer_variant: rollout_matching_sft`

Authoritative requirements live under:
- `openspec/changes/2026-01-15-add-rollout-matching-trainer/specs/rollout-matching-sft/spec.md`

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

- Packing is NOT supported:
  - Set `training.packing: false`.
  - The trainer fails fast if packing is enabled.
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
- `training.packing: false`

Decoding notes:
- Start with greedy (`decode_mode: greedy`, `temperature: 0.0`) for stability.
- Ensure `max_new_tokens` is large enough to avoid systematic truncation
  (LVIS dense outputs can be ~11k text tokens in the tail).

## Command

From repo root:

`PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m src.sft --config <yaml> [--base_config <yaml>]`

4 GPUs:

`PYTHONPATH=. /root/miniconda3/envs/ms/bin/torchrun --nproc_per_node 4 -m src.sft --config <yaml> [--base_config <yaml>]`

## Health Counters to Watch

The trainer logs rollout health without logging IoU/maskIoU numeric metrics
(maskIoU is internal to matching only).

Parsing/matching counters:
- `rollout/parse_dropped_invalid`
- `rollout/parse_truncated`
- `rollout/valid_pred_objects`
- `rollout/matched_for_supervision`
- `rollout/excluded_from_supervision`
- `rollout/fn_appended`
- `rollout/gating_rejections`

Loss breakdown:
- `loss/ce`
- `loss/coord`
- `loss/coord_prefix`
- `loss/coord_tail`

## Minimal Preflight Validation

- Spec validity:
  - `openspec validate 2026-01-15-add-rollout-matching-trainer --strict`
- Unit tests:
  - `PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m pytest -q tests/test_rollout_matching_sft.py -q`

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
