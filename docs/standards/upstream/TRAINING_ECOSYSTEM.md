---
doc_id: docs.standards.upstream.training-ecosystem
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Accelerate, PEFT, TRL, Liger, and training wrapper notes.
updated: 2026-06-07
---

# Training Ecosystem

Source scope: active `ms` environment and local `${MS_SWIFT_ROOT}`.

## Boundary

CoordExp training should integrate through ms-swift config/trainer surfaces, not
raw HF Trainer or TRL imports. ms-swift wraps, patches, and subclasses several
upstream libraries; direct imports can bypass required compatibility handling.

CoordExp-specific supervision metadata is not upstream model state. Packing
maps, teacher-forcing target IR, row-coverage state, token-type metrics, and
debug ownership records should remain sidecars that are stripped or consumed
before `model(**inputs)`. The model-forward payload must stay compatible with
Transformers/ms-swift keys such as `input_ids`, labels when allowed,
`attention_mask`, `position_ids`, `cu_seq_lens_q`, `cu_seq_lens_k`,
`max_length_q`, `max_length_k`, and visual tensors.

For packed training, keep two contracts separate:

```text
attention isolation -> upstream-compatible forward tensors
supervision ownership -> CoordExp sidecars and loss metadata
```

The physical packed sequence remains the batch-budget unit: gradient
accumulation and distributed batch accounting count packed rows capped by the
run max length. Logical samples, row states, target atoms, and base images are
throughput/exposure counters, not hidden replacements for the physical budget.

## Accelerate

- `Accelerator` owns device placement, mixed precision, DeepSpeed/FSDP plugins,
  wrapping through `prepare`, gradient scaling/backward, `unwrap_model`, gather,
  and metrics helpers.
- CoordExp code usually sees Accelerate through HF Trainer/ms-swift
  `self.accelerator`, not through a raw user-instantiated `Accelerator`.
- Do not assume `model` is the base model. It may be PEFT, DDP, FSDP,
  DeepSpeed, or ms-swift wrapped. Use `self.accelerator.unwrap_model(model)` or
  ms-swift helpers for inspection, saving, generation, and adapter checks.
- Avoid model-identity checks on the wrapped object. Resolve the base model
  through ms-swift/Accelerate before testing Qwen family, PEFT status, router
  auxiliary logits, generation support, or save semantics.

## PEFT

- `get_peft_model` wraps/mutates the base model and returns `PeftModel` or a
  task-specific PEFT wrapper.
- `LoraConfig` supports `target_modules`, `modules_to_save`, `use_dora`, and
  `target_parameters`.
- ms-swift adds PEFT compatibility shims for `lora_dtype`, LoRA+, checkpoint
  metadata, and multimodal target expansion.
- Use ms-swift config fields over ad hoc module mutation:

```text
target_modules
target_regex
target_parameters
modules_to_save
freeze_llm
freeze_vit
freeze_aligner
use_dora
lora_dtype
lorap_lr_ratio
```

- `target_parameters` is for raw parameters, especially MoE parameters that are
  not `nn.Linear`; local PEFT `0.17.1` supports this.
- `all-linear` with `freeze_vit=true` and `freeze_aligner=true` can correctly
  avoid adding LoRA to vision/aligner modules. This is not necessarily a bug.
- Reapplying PEFT to an already adapted model can produce confusing adapter
  state; inspect wrapper status before re-wrapping.
- DoRA has extra overhead and should be merged for inference when applicable.

## TRL And RLHF

- ms-swift uses TRL as an algorithm base layer but heavily subclasses it.
- Trainer selection maps causal LM to ms-swift `Seq2SeqTrainer`; preference and
  RLHF tasks go through ms-swift wrappers.
- ms-swift DPO deletes/reimplements parts of upstream `DPOTrainer` init/loss.
- ms-swift GRPO patches vLLM/TRL compatibility before importing and uses a
  generation-batch lifecycle, not ordinary supervised minibatching.
- Raw `from trl import GRPOTrainer` can fail in the local environment because
  installed TRL and vLLM symbols have moved. Import through ms-swift unless a
  debug script explicitly applies the same compatibility patch.
- Do not infer supervised SFT behavior from TRL/RLHF trainer shapes. GRPO uses
  generation buffers, reward functions, vLLM or Transformers rollout engines,
  and task-specific loss plumbing that are not equivalent to Stage-1
  teacher-forced minibatches.

## Loss And Sequence Parallel

- ms-swift `Seq2SeqTrainer.compute_loss` is not vanilla HF Trainer CE. It
  handles `compute_loss_func`, `loss_scale`, channel loss, router auxiliary loss,
  sequence-parallel per-token loss, label shifting, and denominator correction.
- CoordExp custom losses should either plug into `compute_loss_func` with the
  expected signature or explicitly reproduce ms-swift loss semantics.
- The `compute_loss_func` boundary receives model outputs, labels,
  `num_items_in_batch`, optional `loss_scale`, and the trainer. CoordExp losses
  should do coordinate/sidecar remapping before this point or in a thin trainer
  bridge, not inside Objective modules that should already see physical
  positions.
- Sequence parallel owns loss gathering/reduction. Do not treat labels/logits as
  ordinary full local tensors under SP.
- Sequence parallel and custom loss can conflict; verify before launch.
- `loss_scale` is template-time token weighting, while custom objectives should
  keep their own explicit normalization rule. Packed sidecar support must not
  silently change the denominator merely because more logical segments fit into
  one physical row.
- `logits_to_keep` changes the logits time axis. CoordExp token diagnostics and
  segment-aware sidecar losses should require full logits until there is an
  explicit projection map from physical causal rows to sliced logits rows.

## Packing Interface

- ms-swift static packing and padding-free paths are the upstream integration
  surface for long packed rows. CoordExp should extend the collator/trainer
  bridge around that path rather than replacing the training loop.
- Segment-aware packing should validate both boundary maps before forward:
  packed segment ranges for supervision, and FlashAttention/position boundary
  tensors for causal isolation.
- Same-base-image packing is a useful validation rung, not the semantic
  constraint. The semantic constraint is segment correctness: every target,
  label, visual occurrence, row state, and loss owner must still map to the
  right physical positions after packing.
- Unsupported packed sidecar combinations should fail closed with issue codes
  and compact debug handles. Silent fallback to partial packing is harder to
  interpret than an explicit rejection.
- Throughput claims should report logical exposure counters alongside physical
  budget counters: segments, base images, row states, supervised atoms,
  supervised label tokens, fill ratio, padding slack, and wall-clock rate.

## Liger

- `liger-kernel` is not installed in the current `ms` environment.
- ms-swift applies Liger through `liger_kernel.transformers.apply_liger_kernel_*`
  functions by model type.
- Liger has task and mode restrictions. In ms-swift docs/guards, check at least:
  causal-LM-only notes, no `device_map`, GRPO restrictions, padding-free
  restrictions, entropy/log-entropy restrictions, and sequence-parallel
  restrictions.

## Handles

- Accelerate docs:
  [Accelerator](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- PEFT docs:
  [LoRA reference](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora)
- TRL docs:
  [TRL index](https://huggingface.co/docs/trl/index)
- Liger docs:
  [TRL Liger integration](https://huggingface.co/docs/trl/main/liger_kernel_integration)
- ms-swift adapter/config:
  `${MS_SWIFT_ROOT}/swift/arguments/tuner_args.py`
- ms-swift tuner:
  `${MS_SWIFT_ROOT}/swift/pipelines/train/tuner.py`
- ms-swift PEFT shim:
  `${MS_SWIFT_ROOT}/swift/tuners/peft.py`
- ms-swift trainer factory:
  `${MS_SWIFT_ROOT}/swift/trainers/trainer_factory.py`
- ms-swift supervised loss:
  `${MS_SWIFT_ROOT}/swift/trainers/seq2seq_trainer.py`
- ms-swift RLHF trainers:
  `${MS_SWIFT_ROOT}/swift/rlhf_trainers/`
- ms-swift sequence parallel:
  `${MS_SWIFT_ROOT}/swift/sequence_parallel/`
