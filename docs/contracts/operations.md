# Operations and acceptance

The stable package name is `src`, and public commands use `python -m`.
The repository supports code-shaped single-node launches with one through four
GPUs. The host may use `conda run -n ms <command>`; that environment name is
not part of the product contract.

## Static acceptance

Before a runtime claim, verify the retained package metadata, import/help
surface, default test collection, direct evaluator paths, COCO/LVIS
provenance checks, and the absence of removed compatibility surfaces.

## Runtime acceptance still pending

The required runtime witness is a short, production-shaped two-GPU Qwen3-VL
2B plus COCO vertical run. It must cross data read, template and packing,
forward/backward/optimizer, checkpoint save, a fresh exact resume, HF
inference, vLLM inference, scoring, and direct evaluation.

Until that witness is recorded, the one-to-four GPU statement is
code-supported only. It is not evidence of a completed two-GPU run, an
eight-GPU run, or multi-node support.

## Operational boundaries

- Do not weaken exact-resume, pack-cache, payload, qualification, or evaluator
  failures to keep a launch moving.
- Keep inference payloads separate from exact-resume state.
- Preserve COCO/LVIS raw and processed data; do not turn a local smoke into a
  dataset or model-quality claim.
- Optional research may live under `research/`; it is not a default runtime,
  configuration, or test surface.
