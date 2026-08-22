# CoordExp

CoordExp is a neutral infrastructure base for coordinate-token vision-language
training and direct detection evaluation. The supported product surface is:

- Qwen, Transformers, and PEFT training;
- deterministic packing and pack-cache preparation;
- checkpointing with opt-in exact resume;
- COCO and LVIS preparation with provenance;
- HF and vLLM inference; and
- direct scoring, evaluation, and visualization artifacts.

The importable package is `src`; stable entrypoints use `python -m`.

## Start here

- [Training and packing](docs/contracts/train.md)
- [COCO/LVIS data and provenance](docs/contracts/data.md)
- [Checkpoints and artifacts](docs/contracts/artifacts.md)
- [Inference, scoring, evaluation, and visualization](docs/contracts/infer-eval.md)
- [Operations and acceptance](docs/contracts/operations.md)
- [Current compatibility contract](openspec/README.md)

## Entry points

```bash
python -m src.train --config <train-config.yaml>
python -m src.prepare_train_cache --config <train-config.yaml>
python -m src.infer --config <infer-config.yaml>
CUDA_VISIBLE_DEVICES=<one-physical-index-or-GPU-UUID> python -m src.qualify_vllm run --config configs/infer/vllm.yaml --output-root <absent-external-root>
python -m src.qualify_vllm admit --config configs/infer/vllm.yaml --receipts-root <external-root>
python -m scripts.evaluate_detection --artifact-dir <inference-artifact-dir> --out-dir <eval-dir>
python -m scripts.visualize_detection gt-vs-pred --run-dir <run-dir> --out-dir <image-dir>
```

Use a configuration and process launcher appropriate to the local run. The
source contract supports single-node launches with one through four GPUs; the
two-GPU Qwen3-VL 2B plus COCO vertical witness is an acceptance task, not a
completed claim.

The local `conda run -n ms ...` wrapper is a host convention, not product
identity.
