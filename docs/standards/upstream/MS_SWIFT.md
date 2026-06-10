---
doc_id: docs.standards.upstream.ms-swift
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: ms-swift local training framework notes for CoordExp.
updated: 2026-06-07
---

# ms-swift

Source scope: local checkout `/data/ms-swift`, version `4.2.2`, commit
`f2797138dba0e224cfff735cd89a528a08d8732a`.

## Live Route

- Package entrypoint: `setup.py` exposes `swift=swift.cli.main:cli_main` and
  `megatron=swift.cli._megatron.main:cli_main`.
- `swift sft` routes through `swift/cli/main.py` to `swift/cli/sft.py`, then
  `swift.pipelines.sft_main`.
- The live SFT implementation is `swift/pipelines/train/sft.py`, not old
  `swift/llm/train/sft.py` paths. In this checkout `swift/llm/...` is not the
  SFT source to cite.
- `SftArguments` lives in `swift/arguments/sft_args.py` and combines tuner,
  base, and HF `Seq2SeqTrainingArguments` surfaces.

## YAML And Launch Conventions

- If the first post-route argument is `.yaml`, `.yml`, or `.json`, ms-swift
  expands top-level keys into CLI args.
- A top-level `ENV` block becomes process environment defaults.
- YAML lists become repeated CLI values; nested dicts are JSON-serialized.
- Unknown keys surface as `remaining_argv` unless `ignore_args_error` is set.
- `NPROC_PER_NODE` or `NNODES` triggers `torch.distributed.run` for supported
  routes.
- `training_args.remove_unused_columns` is forced false for multimodal tensors,
  but template encoding preserves extra kwargs only when the data-side
  `remove_unused_columns` behavior also allows it.

## Pipeline Stack

`SwiftSft` does the following:

1. load model and processor,
2. prepare template,
3. load train/val datasets,
4. encode through lazy, packed, or streaming preprocessors,
5. prepare LoRA/full tuning,
6. select trainer class through `TrainerFactory`,
7. call `trainer.train()`.

Important files:

- `/data/ms-swift/swift/pipelines/train/sft.py`
- `/data/ms-swift/swift/arguments/sft_args.py`
- `/data/ms-swift/swift/trainers/trainer_factory.py`
- `/data/ms-swift/swift/trainers/seq2seq_trainer.py`
- `/data/ms-swift/swift/trainers/mixin.py`

## Template And Geometry

- `Template` owns multimodal placeholders, bbox normalization, loss-scale
  application, encoding, packing-row merge, forward hooks, and data collation.
- Qwen-VL templates call processors with `do_resize=False` during encoding.
  CoordExp must keep image/coordinate alignment upstream and avoid implicit
  processor resizing.
- Base `Template.normalize_bbox()` uses `norm_bbox` and image dimensions.
  Qwen2.5-VL sets `norm_bbox='none'`; the base default is `norm1000`.
- Custom multimodal templates should set `support_padding_free = True` only
  after validating MRoPE/position IDs and `packing_row()` behavior.

## Packing, Padding-Free, And Logits

- `PackingDataset` uses `binpacking.to_constant_volume`, requires precomputed
  `lengths`, and sets both `template.packing = True` and
  `template.padding_free = True`.
- `SftArguments._check_padding_free()` turns `packing` into `padding_free` and
  rejects non-flash `attn_impl`.
- `use_logits_to_keep` is an optimization with semantics. It may be disabled
  for multimodal models depending on Transformers/model support and is not
  implemented under sequence parallel in `Seq2SeqTrainer.prepare_logits_to_keep`.
- CoordExp should pin `use_logits_to_keep: false` for token/loss debugging and
  any sidecar remap that expects full logits.

## Sequence Parallel

- `sequence_parallel_size > 1` globally patches attention and registers model
  hooks.
- Ring attention requires `padding_free=true`.
- Attention heads must divide across sequence-parallel ranks.
- Sequence parallel changes loss ownership; do not treat logits/labels as
  ordinary full local tensors in custom losses.

## Extension Points

Prefer extension maps and plugins over editing upstream files:

- `external_plugins` imports local Python registration files.
- Dataset: `register_dataset()` / `register_dataset_info()`.
- Model/template: model and template registration maps.
- Loss: `swift/loss/mapping.py`.
- Metrics: `swift/metrics/mapping.py`.
- Callbacks: `swift/callbacks/mapping.py`.
- Optimizers: `swift/optimizers/mapping.py`.

Distinguish:

- `loss_scale`: template-time token weighting and message masking.
- `loss_type`: trainer-time custom loss selected from `loss_map` and passed as
  `compute_loss_func`.

## Common Failures

- Unknown YAML key: check dataclass fields and `remaining_argv`.
- Missing dataset: `dataset` or `cached_dataset` is required.
- `cached_dataset` plus streaming: unsupported.
- `lazy_tokenize` plus packing or streaming: unsupported.
- `truncation_strategy: split`: only a narrow pretraining-like causal LM path.
- Packing without flash attention: rejected.
- Unsupported template padding-free: rejected before training.
- DeepSpeed plus `device_map`/model parallel: rejected unless Ray path applies.
- FSDP2 plus DeepSpeed or some save/gradient-checkpointing modes: rejected or
  warned.
- `predict_with_generate`: temporarily disables packing/padding-free and writes
  `predict.jsonl`; do not compare its memory behavior to teacher-forced eval.
- Offline environments: set `check_model: false` when ModelScope freshness
  checks are not available.

## Handles

- Local checkout identity: `/data/ms-swift/swift/version.py`
- CLI/config: `/data/ms-swift/swift/cli/main.py`
- SFT pipeline: `/data/ms-swift/swift/pipelines/train/sft.py`
- SFT args: `/data/ms-swift/swift/arguments/sft_args.py`
- Template args: `/data/ms-swift/swift/arguments/base_args/template_args.py`
- Data args: `/data/ms-swift/swift/arguments/base_args/data_args.py`
- Packing: `/data/ms-swift/swift/dataset/packing.py`
- Qwen template: `/data/ms-swift/swift/template/templates/qwen.py`
- Trainer/loss: `/data/ms-swift/swift/trainers/seq2seq_trainer.py`
- Sequence parallel: `/data/ms-swift/swift/sequence_parallel/ulysses.py`
- Upstream repo: [modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- Upstream command-line docs:
  [Command-line parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
