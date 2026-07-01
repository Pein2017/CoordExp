---
doc_id: docs.standards.upstream
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Upstream dependency boundaries, routing, and maintenance notes.
updated: 2026-06-07
---

# Upstream Dependencies

Use this page as the upstream dependency router. Detailed notes live under
[`upstream/`](upstream/). The current source of truth is the active `ms`
environment plus the current ms-swift checkout, not stale paths from older
CoordExp runs.

Local handle variables used below:

- `TRANSFORMERS_ROOT`: `python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)"`
- `FLASH_ATTN_ROOT`: `python -c "import pathlib, flash_attn; print(pathlib.Path(flash_attn.__file__).parent)"`
- `MS_SWIFT_ROOT`: local ms-swift checkout.

## Version Stamp

Last reviewed locally on 2026-06-07:

| Package | Version / state | Primary local handle |
|---|---|---|
| `torch` | `2.9.1+cu128` | active `ms` Python |
| `transformers` | `4.57.1` | `${TRANSFORMERS_ROOT}` |
| `flash_attn` | `2.8.3` | `${FLASH_ATTN_ROOT}` |
| `ms-swift` | `4.2.2`, local checkout commit `f2797138dba0e224cfff735cd89a528a08d8732a` | `${MS_SWIFT_ROOT}` |
| `accelerate` | `1.10.1` | active `ms` Python |
| `peft` | `0.17.1` | active `ms` Python |
| `trl` | `0.23.1` | active `ms` Python |
| `vllm` | `0.14.1` | active `ms` Python |
| `deepspeed` | `0.17.5` | active `ms` Python |
| `liger-kernel` | not installed | verify before enabling |

Every training, eval, or synchronization run that depends on upstream behavior
should record at least `transformers`, `ms-swift`, `flash_attn`,
`attn_implementation`, processor classes, `processor_do_resize`, `peft`, `trl`,
`vllm`, and `deepspeed` when relevant.

## Pages

- [`upstream/QWEN_VL.md`](upstream/QWEN_VL.md)
  - Qwen2-VL, Qwen2.5-VL, and Qwen3-VL model, processor, grid, RoPE,
    `logits_to_keep`, and LoRA/freezing boundaries.
- [`upstream/MS_SWIFT.md`](upstream/MS_SWIFT.md)
  - live ms-swift SFT route, YAML conventions, dataset/template/collator stack,
    packing, padding-free, sequence parallel, extension maps, and launch
    failures.
- [`upstream/FLASH_ATTENTION.md`](upstream/FLASH_ATTENTION.md)
  - FlashAttention v2 / Transformers attention interface, varlen tensors,
    dtype/device constraints, determinism, and packed attention debugging.
- [`upstream/TRAINING_ECOSYSTEM.md`](upstream/TRAINING_ECOSYSTEM.md)
  - Accelerate, PEFT, TRL, Liger, adapter targeting, trainer wrappers, and
    custom-loss boundaries.

## Boundary Rules

- Do not edit installed Hugging Face model files such as
  `modeling_qwen3_vl.py`. Transformers Qwen3-VL files are generated from
  modular upstream sources; local edits are not a maintainable integration
  strategy.
- Treat ms-swift as the training integration boundary. CoordExp should extend
  ms-swift config, template, dataset, trainer, callback, loss, or plugin maps
  before importing raw TRL/HF Trainer classes directly.
- Preserve CoordExp geometry. Runtime training uses offline-prepared images and
  `do_resize=false`; never rely on upstream processor resizing unless the run is
  explicitly designed around that semantic.
- For packed or padding-free runs, treat attention isolation and supervision
  ownership as separate contracts:

```text
attention isolation -> upstream-compatible forward tensors
supervision ownership -> CoordExp sidecars / loss metadata
```

- Prefer `attn_implementation: flash_attention_2` through Transformers/ms-swift
  configuration instead of monkeypatching attention modules.
- Keep full-vocabulary logits for CoordExp token/loss diagnostics. Compact
  logits over selected physical rows are allowed only when a CoordExp-owned
  physical-position map is recorded and validated by `LossContext`;
  `logits_to_keep` slices hidden states before `lm_head` in Qwen3-VL and changes
  downstream coordinate systems.
- For LoRA/DoRA targeting, inspect real loaded module names and prefer ms-swift
  knobs such as `target_regex`, `target_parameters`, `modules_to_save`,
  `freeze_llm`, `freeze_vit`, `freeze_aligner`, and `use_dora`.
- For sequence parallel or Liger, verify task support before launch. These paths
  alter loss ownership and may reject padding-free, GRPO, custom loss, or
  `device_map` combinations.

## High-Signal Handles

Local upstream:

- Qwen3-VL model: `${TRANSFORMERS_ROOT}/models/qwen3_vl/modeling_qwen3_vl.py`
- Transformers FlashAttention utility:
  `${TRANSFORMERS_ROOT}/modeling_flash_attention_utils.py`
- FlashAttention interface:
  `${FLASH_ATTN_ROOT}/flash_attn_interface.py`
- ms-swift SFT pipeline: `${MS_SWIFT_ROOT}/swift/pipelines/train/sft.py`
- ms-swift SFT args: `${MS_SWIFT_ROOT}/swift/arguments/sft_args.py`
- ms-swift Qwen templates: `${MS_SWIFT_ROOT}/swift/template/templates/qwen.py`
- ms-swift trainer loss path: `${MS_SWIFT_ROOT}/swift/trainers/seq2seq_trainer.py`
- ms-swift sequence parallel: `${MS_SWIFT_ROOT}/swift/sequence_parallel/ulysses.py`

Upstream links:

- [Transformers Qwen3-VL docs v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl)
- [Transformers Qwen3-VL source tag v4.57.1](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/qwen3_vl)
- [Transformers attention interface v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)
- [Dao-AILab flash-attention v2.8.3](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3)
- [modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- [ms-swift command-line parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [Accelerate docs](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora)
- [TRL docs](https://huggingface.co/docs/trl/index)

## Refresh Checklist

When bumping any upstream package:

1. Record package versions and local source roots.
2. Re-check Qwen model signatures, `logits_to_keep`, processor placeholder
   expansion, image/video grid handling, and `position_ids`.
3. Re-check ms-swift CLI route, SFT pipeline path, dataclass keys, packing
   checks, template support, trainer loss hooks, and extension maps.
4. Re-check FlashAttention varlen dispatch, dtype/device checks,
   `cu_seqlens` shape/dtype expectations, and `output_attentions` support.
5. Re-check PEFT target matching, TRL import compatibility, Liger install
   status, and sequence-parallel loss behavior.
6. Update this router and the affected companion page before interpreting
   changed training/eval behavior.

Detailed handles and troubleshooting live in the companion pages under
[`upstream/`](upstream/).
