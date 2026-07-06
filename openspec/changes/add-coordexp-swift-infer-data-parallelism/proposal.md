## Why

CoordExp-swift inference currently supports HF batched decoding inside one
process, but full validation and benchmark-style runs remain unnecessarily slow
because one visible GPU carries all rows. Data-parallel inference extends the
accepted baseline in `openspec/changes/build-coordexp-swift-inference-infra/`:
it lets the existing single-GPU HF path scale across visible CUDA devices while
preserving the same canonical scored artifacts expected by evaluation.

The change protects accuracy and precision first by making merge/provenance
strict: no missing rows, duplicate rows, identity drift, or partial worker
failure can produce benchmark-looking top-level artifacts. It improves
efficiency by decoding independent rows concurrently across GPUs, keeps
simplicity by using self-orchestrated subprocess workers rather than DDP, and
preserves future extension by shaping the shard contract so a later vLLM backend
can plug into the same merge layer.

## What Changes

- Add an inference data-parallel controller for `python -m src.infer --config`.
- Treat `openspec/changes/build-coordexp-swift-inference-infra/` as the current
  baseline contract for config, prompt/parsing, scoring, provenance, evaluator
  compatibility, and artifact row semantics until those specs are archived or
  synced into stable specs.
- Default production inference to all useful GPUs visible through
  `CUDA_VISIBLE_DEVICES`.
- Require CUDA for every non-dry inference execution path; no CPU fallback is
  accepted for direct single-rank or controller/worker execution.
- Preserve `generation.batch_size` as immutable per-device decode batch size.
- Split input rows into deterministic decode-batch blocks and assign blocks
  round-robin across active ranks.
- Launch one subprocess worker per active rank, binding each worker to exactly
  one CUDA device by narrowing the worker `CUDA_VISIBLE_DEVICES`.
- Keep worker-local runtime device semantics simple: inside a worker, the only
  supported CUDA device is logical `cuda:0`.
- Write rank-local artifact sets under shard directories, then strictly merge
  them into the existing top-level artifact contract.
- Record parallelism, rank-to-device mapping, shard plan, shard artifact hashes,
  row coverage, rank/device trace metadata, and merge status in run manifests
  and in the merged scored provenance sidecar.
- Keep V1 implementation and acceptance HF-only while preserving a
  backend-neutral shard/merge interface for future vLLM.

No breaking change is intended for single-GPU inference artifact consumers. The
canonical top-level artifacts remain the same after a successful merge.

## Capabilities

### New Capabilities

- `coordexp-swift-infer-data-parallel-runtime`: controller/worker launch,
  CUDA binding, rank planning, deterministic row sharding, and strict merge
  orchestration for data-parallel inference.

### Modified Capabilities

- `coordexp-swift-infer-config-runtime`: production inference now has explicit
  CUDA availability and visible-device semantics for data-parallel launch.
- `coordexp-swift-infer-pipeline`: pipeline orchestration gains multi-rank
  shard planning, worker execution, and top-level merge behavior while keeping
  `generation.batch_size` per device.
- `coordexp-swift-infer-scoring-artifacts`: scored artifacts and provenance
  gain strict shard-to-merged validation and parallelism metadata.
- `coordexp-swift-infer-backend-trace`: trace/provenance records may include
  rank/device identity while preserving backend-neutral decode records.
- `coordexp-swift-infer-benchmark-smoke`: acceptance now requires a real
  multi-GPU HF smoke before production data-parallel benchmark claims.

## Impact

Affected surfaces:

- `src/infer.py` and `src/inference/pipeline.py` for controller orchestration.
- `src/inference/runtime.py` and Qwen loading boundaries for worker-local CUDA
  validation before model loading.
- `src/inference/artifacts.py` for shard artifact validation and strict merge.
- `src/config/inference.py` for runtime contract validation and metadata
  surfaces, without adding stable GPU-id CLI flags.
- `configs/coordexp_swift/infer/**` for production and smoke leaves that rely
  on per-device `generation.batch_size`.
- `tests/inference/**` for controller planning, worker binding, merge
  contracts, CUDA failure behavior, and real multi-GPU smoke gates.

The implementation must not change model architecture, training behavior, loss
semantics, or evaluator metric reduction. It only changes how offline inference
rows are scheduled across GPUs and how rank-local artifacts are merged back into
the existing evaluator-readable artifact family.

Implementation is blocked unless the baseline inference change is treated as
current authority or its deltas have been archived/synced into stable specs.
