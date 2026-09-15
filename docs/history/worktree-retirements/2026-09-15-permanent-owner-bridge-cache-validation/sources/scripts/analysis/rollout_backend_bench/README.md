# Rollout Backend Benchmark (HF vs vLLM)

This folder contains **analysis-only** utilities used to benchmark rollout-stage
inference for stage-2 rollout-matching SFT:

rollout generation (no grad) -> strict parse -> (optional) match + metrics

It is intentionally *not* wired as an "official" launch script.

## Runner

- `scripts/analysis/rollout_backend_bench/benchmark_rollout_backends.py`
  - HF backend: `swift.llm.PtEngine`
  - vLLM backend: `swift.llm.VllmEngine`

Outputs are written to a timestamped run directory under:
- `output/bench/rollout_backend_bench/<YYYYMMDD_HHMMSS>/`

## Configs

Example config lives under `configs/bench/`:
- `configs/bench/rollout_backend_bench.yaml`

Bench-specific knobs are sourced from:
- `custom.extra.rollout_backend_bench.*`

## What `gpu_memory_utilization` Controls

`gpu_memory_utilization` is a vLLM knob that limits how much of the GPU's VRAM
vLLM is allowed to reserve. In practice, it mostly controls the size of vLLM's
**KV cache** (which sets the maximum supported context length + concurrency).

Lowering it:
- reduces peak VRAM usage (KV cache shrinks)
- may reduce throughput only when you rely on high concurrency
- can fail if too low to support the chosen `max_model_len`
