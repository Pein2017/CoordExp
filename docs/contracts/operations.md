# Operations and acceptance

The stable package name is `src`, and public commands use `python -m`.
The repository retains configuration support for single-node launches with
one through four GPUs and provides an explicit eight-GPU smoke. The host may
use `conda run -n ms <command>`; that environment name is not part of the
product contract.

## Static acceptance

Before a runtime claim, verify the retained package metadata, import/help
surface, default test collection, direct evaluator paths, COCO/LVIS
provenance checks, and the absence of removed compatibility surfaces.

## Eight-GPU smoke

From the repository root:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export GLOO_SOCKET_IFNAME=lo
python -m src.prepare_train_cache --config configs/smoke/eight_gpu_qwen3_vl_2b_coco.yaml
python -m src.prepare_train_cache --config configs/smoke/eight_gpu_qwen3_vl_2b_coco.yaml --require-all-hit
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m src.train --config configs/smoke/eight_gpu_qwen3_vl_2b_coco.yaml
```

This preserves production BF16/FA2, DoRA, loss/optimizer and effective batch
48. It applies four updates and saves/evaluates at steps 2 and 4. The parent
enables exact training-state publication; a fresh resume overlay selects its
step-2 checkpoint and keeps the total four-update schedule.

This single-host command binds Gloo control communication to loopback. The
recorded host-interface attempt failed before model loading; the same source
and workload passed parent training with loopback. This does not define a
multi-node interface policy.

The [current acceptance record](../../openspec/changes/optimize-production-training-and-eight-gpu-smoke/acceptance.md)
separates completed eight-rank training, exact resume and HF consumer evidence
from the pending vLLM contract/qualification gate. A topology interface is not proof of an executed topology;
the eight-rank witness does not imply a two-GPU, multi-node, full-training or
model-quality result.

Composed vLLM requires an explicit absolute
`COORDEXP_EXECUTION_MODEL_CACHE_ROOT`, current `src.qualify_vllm run` and
`admit` receipts, then the real inference entry. Direct detection evaluation
uses `python -m scripts.evaluate_detection --artifact-dir <infer-run> --out-dir <evaluation-dir>`.

## Operational boundaries

- Do not weaken exact-resume, pack-cache, payload, qualification, or evaluator
  failures to keep a launch moving.
- Keep inference payloads separate from exact-resume state.
- Preserve COCO/LVIS raw and processed data; do not turn a local smoke into a
  dataset or model-quality claim.
- Optional research may live under `research/`; it is not a default runtime,
  configuration, or test surface.
