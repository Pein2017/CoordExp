thread_id: 019d8fed-7c0a-7401-b276-1c6b45fbc929
updated_at: 2026-04-15T07:41:49+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/15/rollout-2026-04-15T06-56-38-019d8fed-7c0a-7401-b276-1c6b45fbc929.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-1 Qwen3-VL training path: tied-head coord-token tuning plus multimodal dLoRA, with explicit gradient-flow explanation

Rollout context: The user first asked for a detailed audit of the Stage-1 SFT pipeline for `configs/stage1/profiles/2b/cxcywh_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml` and `scripts/train.sh`, focusing on (1) tied `lm_head`/embeddings, (2) coord-token vocab expansion via `<|coord_*|>`, and (3) LoRA vs embedding updates. They later asked follow-ups about making the behavior “full-FT embedding/lm_head” versus the current delta-style coord tuning, and finally asked how the computational graph, backprop, and gradient accumulation work for the `tie_head` case.

## Task 1: Analyze Stage-1 parameterization / current-vs-target behavior

Outcome: success

Preference signals:
- The user repeatedly asked for explanations in simpler terms after the initial deep dive, indicating that for follow-up conceptual questions they want concrete, plain-language distinctions rather than a dense architecture sketch.
- The user’s wording “I stil don't get the difference” and later “So the current optimization learning process is already what I wanted?” shows they want the assistant to explicitly map *current behavior* vs *proposed behavior* in direct terms.
- The user asked “Would learning the final row instead of learn the change be more optimization feasible or smoother or easier?” showing a preference for optimization intuition framed around equivalence, not just implementation detail.

Key steps:
- Traced the Stage-1 profile YAML and `scripts/train.sh` into `src/sft.py`, then into ms-swift/PEFT and the local coord-offset adapter.
- Verified the runtime-resolved config for the actual run under `output/stage1_2b/.../resolved_config.json` rather than trusting comments in the YAML.
- Compared the current setup against the user’s target parameterization and the upstream PEFT behavior.

Failures and how to do differently:
- The current setup is **not** “LoRA on embeddings/lm_head + full finetune of the entire embedding matrix.” Instead, it uses a separate dense coord-offset adapter for selected coord-token IDs and LoRA/DoRA for multimodal linear layers.
- The user’s intended “full-FT embedding/lm_head target” is closer to **row-selective dense tuning** than to naive full unfreezing. A future explanation should lead with that distinction.
- The wildcard token `<|coord_*|>` is excluded from coord-offset training, so “all newly added coord tokens” is not literally true in the current config unless that exclusion is changed.

Reusable knowledge:
- `scripts/train.sh` is just the launcher; the actual trainable surface comes from `src/sft.py` plus config inheritance.
- The resolved Stage-1 2B run had `train_type: lora`, `use_dora: true`, `target_modules: ["all-linear"]`, `freeze_llm: false`, `freeze_vit: false`, `freeze_aligner: false`, and `optimizer: "multimodal_coord_offset"`.
- The local coord-offset adapter freezes the base `embed_tokens.weight` and `lm_head.weight`, then applies a learned offset table only for coord-token IDs.
- With `tie_head: true`, the adapter uses one shared dense table (`embed_offset`) for both the embedding-side hook and the head-side hook.
- This makes the current behavior functionally very close to “train the selected coord rows densely” while still protecting the original vocab rows.
- PEFT’s `trainable_token_indices` exists upstream and is the closest direct row-selective dense-tuning primitive, but this repo currently routes coord-token training through its own coord-offset adapter instead.

References:
- [1] `configs/stage1/profiles/2b/cxcywh_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml` resolved to a run with `coord_offset.enabled: true`, `tie_head: true`, and `ids: 151670..152669`.
- [2] `src/coord_tokens/offset_adapter.py:3-6, 17-18, 77-85, 97-145, 188-190` shows the base embedding/head are frozen and coord tokens are updated via a dense offset table plus forward hooks.
- [3] `src/sft.py:1451-1479` installs the coord-offset adapter, appends `coord_offset_adapter` to `modules_to_save`, and logs the coord-offset LRs.
- [4] `src/optim/coord_offset_optimizer.py:36-112` creates separate optimizer buckets for coord-offset parameters and multimodal dLoRA buckets.
- [5] `output/.../resolved_config.json` showed the effective runtime config: `train_type: lora`, `use_dora: true`, `optimizer: multimodal_coord_offset`, `coord_offset.enabled: true`, `coord_offset.tie_head: true`.
- [6] `output/.../checkpoint-716/adapter_config.json` from the sibling Stage-1 checkpoint showed `modules_to_save: ["coord_offset_adapter"]`, `target_modules` only on multimodal linears, and no `embed_tokens`/`lm_head` targets.
- [7] `output/.../checkpoint-716/adapter_model.safetensors` contained `coord_offset_adapter.coord_ids` and `coord_offset_adapter.embed_offset`, but no full embedding or `lm_head` matrices; the checkpoint had `total_params 27509736`, with `coord_offset_params 2048000` and `lora_ab_params 24641536`.

## Task 2: Explain `tie_head` forward/backward and gradient accumulation

Outcome: success

Preference signals:
- The user explicitly asked: “How to train,backprob, grad accum on the `tie_head` case, how the `computational graph` is created?” and then “WIll the gradient computed/accumulated repeated?” This indicates they want the computational graph and gradient flow described step-by-step, not just summarized.
- The user then asked a yes/no follow-up about whether the process is already what they wanted, so future explanations should keep the gradient-flow answer tightly aligned to the current parameterization.

Key steps:
- Read `src/coord_tokens/offset_adapter.py` with line numbers to inspect the hooks.
- Confirmed the embedding hook adds the offset to selected coord-token rows and the head hook adds extra logits computed from the same tensor when `tie_head=True`.
- Checked the local Transformers trainer loop to confirm gradient accumulation is the standard repeated backward + delayed optimizer-step flow.

Failures and how to do differently:
- The main pitfall is to confuse “same parameter used in two branches” with “double-counting bug.” In this case, both branches are intentional and autograd sums both gradient contributions into the same `.grad` buffer.
- Another pitfall is to conflate tied-parameter gradient accumulation with optimizer gradient accumulation across microbatches; these are separate layers of accumulation.

Reusable knowledge:
- In `tie_head=True`, the same trainable tensor `embed_offset` is used twice in one forward pass:
  - embedding-side: selected input token embeddings get `+ embed_offset`
  - head-side: selected output logits get an additive term from `hidden @ embed_offset.T`
- Autograd builds a graph with both branches pointing to the same parameter, so backprop produces `dLoss/d(embed_offset)` as the sum of both branches’ contributions.
- Gradient accumulation across microbatches is standard trainer behavior: each microbatch runs forward/backward, gradients accumulate in `.grad`, then the optimizer steps once per accumulation window and zeroes gradients afterward.
- Therefore, in the `tie_head` case, the coord-row parameter receives the sum of all valid gradient paths: embedding-side signal, head-side signal, and all microbatch contributions inside the accumulation window.

References:
- [1] `src/coord_tokens/offset_adapter.py:77-85` defines `embed_offset` and conditionally omits `head_offset` when `tie_head=True`.
- [2] `src/coord_tokens/offset_adapter.py:97-119` embedding hook: coord-token positions get `+ embed_offset`.
- [3] `src/coord_tokens/offset_adapter.py:121-145` head hook: `extra_logits = hidden_states @ embed_offset.T`, then `scatter_add_` into coord-token columns.
- [4] `src/coord_tokens/offset_adapter.py:188-190` freezes the base `embed_tokens.weight` and `lm_head.weight` so only the adapter learns.
- [5] `src/sft.py:1451-1479` installs the adapter before `prepare_model` and logs the coord-offset config.
- [6] Transformers trainer loop references observed locally include `trainer.py:4057`, `trainer.py:4064`, `trainer.py:4071`, `trainer.py:2740`, `trainer.py:2752`, confirming the usual backward-then-step-then-zero pattern for gradient accumulation.
