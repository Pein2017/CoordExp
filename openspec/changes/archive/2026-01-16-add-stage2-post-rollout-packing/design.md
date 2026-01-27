## Context
Rollout-matching SFT (stage_2) is implemented in `src/trainers/rollout_matching_sft.py` as:

1) rollout generation (no grad) via `RolloutMatchingSFTTrainer._rollout_one` (currently batch_size=1)
2) strict token-aligned parse -> matching -> build `Y_train`
3) teacher-forced forward on `Y_train` and masked losses in `compute_loss`

Key constraints:
- Rollout generation is autoregressive. It cannot use *sequence packing* (concatenating multiple sequences into one) without changing semantics.
- The post-rollout forward pass is standard teacher-forcing. It *can* use packing (concatenate multiple samples) as long as:
  - per-sample supervision masks remain correct,
  - required metadata survives packing (prompt_len, prefix_len, train_len, etc),
  - rollout generation continues to run un-packed (padded batches).

Existing project packing infrastructure:
- Dataset packing wrapper: `src/datasets/wrappers/packed_caption.py` groups encoded samples and relies on ms-swift template packing/padding-free collator.
- ms-swift "packing row" implementation: `/data/home/xiaoyan/AIteam/data/ms-swift/swift/llm/template/base.py`:
  - `Template.packing_row(...)` concatenates `input_ids/labels/loss_scale` and resets `position_ids` per segment.
  - Packing / padding-free collation is driven by template flags:
    - `template.padding_free` / `template.packing` control whether `Template.data_collator(batch, padding_to=...)` collapses a multi-sample batch into a single packed row.
    - `Template.data_collator(...)` does NOT accept a `padding_free=` argument.

## Runtime Assumptions (current target environment)
- Distributed training is DeepSpeed ZeRO-2/3 (data-parallel). Each rank runs rollout + forward on its rank-local mini-batch, then synchronizes on backward/allreduce. Stragglers dominate step time.
- Finetuning is LoRA-only (dLoRA). This matters for:
  - rollout speed under ZeRO-3 (generation can cause repeated parameter gathering),
  - vLLM feasibility (LoRA-only sync is the only realistic per-step sync strategy).
- Hardware budget is 4 GPUs total. Any vLLM rollout engine must either co-locate with training (memory contention) or steal GPUs from training (throughput trade-off).

## Goals / Non-Goals
Goals:
- Keep rollout generation un-packed (per-sample semantics), but allow post-rollout packing for the forward pass.
- Maintain correctness: matching results and loss computation should be unchanged relative to the un-packed baseline, modulo the usual "packing concatenation context" behavior already used in stage_1.
- Improve throughput by:
  - batching rollout decoding within each rank (microbatching), and
  - reducing padding waste in the post-rollout forward pass.
- Preserve YAML-first configuration (no new CLI flags).

Non-goals:
- No changes to the rollout parsing/matching logic itself (that is correctness-critical).
- No model-architecture changes (do not touch HF model files).
- vLLM server mode is out-of-scope (we standardize on colocate mode for this change).
- No automatic fallback behavior (fail fast on vLLM errors; fallback is explicit via YAML).

## Decisions (locked for this change)
- Packing is enabled via `training.packing: true`, but stage_2 uses **dynamic post-rollout packing inside the trainer** (NOT dataset-level packing).
- Dynamic packing uses **carry** only: each teacher-forced `Y_train` is an atomic segment, never split across packs; leftover segments remain in a rank-local buffer.
- Rollout generation batching is supported for HF rollout via `rollout_generate_batch_size` (default 1; tunable; KV cache permitting).
- Rollout backend defaults to **vLLM colocate** for rollout generation only; teacher-forced forward/backprop stays on the training model.
  - Default vLLM settings for safety on 4xGPU: `tensor_parallel_size=4`, `gpu_memory_utilization=0.45`.
  - Server mode is not supported in this change (no extra-GPU budget; avoid networking complexity).
  - Fail-fast on incompatibilities (vLLM missing, TP mismatch, OOM, etc); explicit YAML can switch backend to HF.
- Multimodal + LoRA policy: start with **ViT frozen** when using vLLM LoRA (LLM + aligner LoRA is supported; ViT LoRA is optional and explicitly risky).

## Technical Corrections (from ecosystem exploration)
- `Template.data_collator(batch, *, padding_to=None)` has no `padding_free=` argument; use `template.padding_free` / `template.packing` flags instead.
- For vLLM multimodal prompts, encoding MUST run with `template.mode="vllm"` so image/mm kwargs are attached (do not rely on `template.generate_context()` which forces `pt` mode).
- Qwen3-VL's standalone `vllm_backend.py` is inference-oriented and not a drop-in rollout backend for rollout-matching (template/token-id invariants differ); prefer ms-swift's vLLM engine pattern.

## Proposed Architecture: Two-Phase Batching
The training step becomes a two-phase pipeline:

raw samples
  -> (A) rollout generation (no grad, padded batches, microbatched)
  -> parse/match (CPU)
  -> build per-sample `Y_train` token ids
  -> (B) teacher-forced encode (per sample, exact ids)
  -> post-rollout packing (concatenate multiple samples)
  -> forward + loss (packed)

### Phase A: Un-packed rollout generation (microbatched)
Current code calls `template.generate(...)` once per sample (`_rollout_one`).
This is correct but decode-bound and underutilizes GPU.

Proposed change:
- Add a rollout backend abstraction that supports:
  - HF backend: microbatched `template.generate(...)` over a standard padded batch.
  - vLLM backend (default): ms-swift-style colocate engine that returns token ids.

HF microbatching path:
- Add `_rollout_many(samples: list[dict]) -> list[RolloutResult]` that:
  - encodes each sample under `template.generate_context()` (pt mode),
  - collates as a standard padded batch (NOT padding-free),
  - calls `template.generate(...)` once per microbatch,
  - returns per-sample `response_token_ids`, decoded text, decode_mode, and prompt prefix ids.

vLLM path (colocate):
- Encode prompts with `template.mode="vllm"` to get:
  - `prompt_token_ids` (exact ids),
  - multimodal payload / mm kwargs (images, `mm_processor_kwargs`) suitable for vLLM.
- Call vLLM engine inference to get:
  - `response_token_ids` (stop-trimmed),
  - `prompt_token_ids` (for prefix alignment sanity check).

YAML knob (suggested):
- `custom.extra.rollout_matching.rollout_generate_batch_size: int` (default 1)
- `custom.extra.rollout_matching.rollout_backend: "vllm" | "hf"` (default `"vllm"`)
- `custom.extra.rollout_matching.vllm.*` engine settings (see below)

Important interaction with packing:
- If stage_2 enables packing for the post-rollout forward pass, ms-swift may set `template.padding_free=True`.
- Rollout generation must explicitly force `padding_free=False` (and `packing=False`) for its internal collation, otherwise a multi-sample padded batch would be collapsed into a single packed row.

Implementation detail:
- Add a small context manager in the trainer, e.g.:
  - `_with_template_flags(padding_free: bool, packing: bool)` that restores original flags on exit.
- Wrap rollout collation/generation in `padding_free=False, packing=False`.

### Phase B: Post-rollout packing for teacher-forced forward
After rollouts are complete and each sample is converted into a teacher-forced encoding, we want to pack (concatenate) multiple samples into one forward pass to reduce padding overhead.

#### Why dynamic post-rollout packing is necessary
Packing membership must be decided using the **true** teacher-forced sequence length, but stage_2 only knows the final `Y_train` length after:
rollout -> parse/match -> FN append -> encode.

Therefore stage_2 packing is implemented inside the trainer as:
- build `Y_train` per sample,
- compute actual `encoded["length"]`,
- pack segments to respect `global_max_length` by construction.

This avoids the classic failure mode where "static length estimates" undercount stage_2 `Y_train` and cause overflow.

#### Dynamic packing algorithm (carry-only, one packed forward per step)
State:
- maintain a rank-local buffer `buffer = List[(encoded, meta)]`.

Per training step:
1) Generate rollouts for the raw batch (Phase A) and build per-sample `(encoded, meta)` segments (Phase B-pre).
2) Append segments to `buffer`.
3) Select a subset of segments from `buffer` such that:
   - `sum(seg.length) <= packing_length` (where `packing_length` comes from `global_max_length/template.max_length`),
   - selection tries to maximize fill (best-effort; log achieved fill ratio),
   - no segment is ever split.
4) Remove selected segments from `buffer` (carry the rest to future steps).
5) Collate selected segments with `template.padding_free=True` to produce a single packed row (`bsz=1`) and run forward+loss.

Config reuse (YAML-first):
- `training.packing: true` enables dynamic post-rollout packing (stage_2 only).
- `training.packing_buffer` caps how many segments we are allowed to keep buffered (fail fast if exceeded).
- `training.packing_min_fill_ratio` is treated as a performance target; log warnings if achieved fill is consistently below target.
- `training.packing_drop_last: true` is required for carry-only mode (trainer does not run "extra flush steps" after max_steps/epoch end).

### Packing Metadata Contract (for loss correctness)
Current `compute_loss` assumes:
- `meta` is a list aligned to batch dimension `bsz`.
- each `meta[b]` describes one sequence in `input_ids[b, :]`.

Packed forward pass breaks this assumption because `input_ids` becomes `[1, total_len]` but `meta` needs to describe multiple segments.

Proposed meta contract:
- Always attach `batch["_rollout_matching_meta"]`.
- In un-packed mode: `meta` is `List[Dict]` where `len(meta) == bsz` (existing behavior).
- In packed mode: `meta` is `List[Dict]` where `bsz == 1` and `meta` is the per-segment list for that packed row (order matches concatenation order).
  - Each meta dict MUST include:
    - `encoded_len`: total encoded sequence length for that segment (prompt + assistant + EOS).
    - existing keys: `prompt_len`, `prefix_len`, `train_len`, `prompt_ids`, and coord supervision metadata.

Packed loss algorithm:
- Compute `offset` cumulatively using `encoded_len`.
- For each segment:
  - slice `seg_input_ids = input_ids[0, offset : offset + encoded_len]`,
  - run the existing `_build_labels_and_coord_targets_for_sample(...)` on the slice using the segment's `prompt_len/prefix_len/train_len`,
  - write the resulting labels into `labels_masked[0, offset:...]`,
  - shift any coord-supervised positions by `offset` before aggregating.

This keeps all mask logic unchanged, only re-indexed into the packed row.

## Batch Decoding Distribution Analysis (DDP/FSDP)
### What happens today
- Data parallelism (DDP/accelerate) distributes *data* across ranks. Each GPU process executes `_prepare_batch_inputs` on its local mini-batch.
- Inside each rank, rollout generation is currently serial over samples (`for sample in inputs: _rollout_one(sample)`), which means:
  - GPU utilization is typically low during rollout phase (decode-bound, small batch).
  - step time is gated by the slowest rank ("straggler") because backward/allreduce synchronizes after forward.

### Can rollouts be batched across samples (even if not packed)?
Yes. "Un-packed" here should mean "not sequence-packed", but you can still:
- run `generate` on a padded batch of size > 1,
- let attention_mask handle differing prompt lengths.

This is the primary low-risk win before vLLM.

### Expected GPU utilization patterns
- Rollout phase: decode is step-by-step autoregressive; per-token compute may be memory/KV-cache bound, and with batch_size=1 utilization is often poor.
- Forward phase (teacher-forced): much higher compute density; packing increases effective tokens/forward and reduces padding waste.

### Recommendations to maximize parallel rollout throughput
1) Add rollout microbatching per rank:
   - `rollout_generate_batch_size > 1` to increase decode batch size.
   - Keep it small enough to avoid OOM from KV cache (tune empirically).
2) Reduce inter-rank variance:
   - Use length-aware batching (group_by_length) if available, using prompt length as proxy.
   - Avoid mixing very-long and very-short samples in the same global step.
3) Prefer greedy decoding unless beam search is required:
   - beam multiplies decode compute and KV memory.
4) Instrument phase timings explicitly:
   - log `rollout/sec`, `forward/sec`, `time_generate`, `time_match`, `time_forward`.

Note on ZeRO-3 / FSDP:
- If generation requires parameter gathering, it can introduce cross-rank communication overhead.
- In practice, stage_2 may run faster and more stably on DDP/ZeRO-2 than ZeRO-3 when rollouts are frequent.

## vLLM Integration Feasibility (Rollout Backend)
This section focuses on using vLLM *only for rollout generation* (no-grad autoregressive decode), while keeping the
post-rollout teacher-forced forward/backprop on the normal training model.

### What stage_2 needs from a rollout backend
Rollout-matching depends on strict token-level alignment, so a rollout backend must provide:
- `response_token_ids`: the sampled/decoded assistant tokens, *exactly as token ids*, before any post-hoc text parsing.
- `prompt_token_ids` (or at least `prompt_len` + a sanity-checkable prefix): stage_2 verifies that the teacher-forced
  prompt prefix matches the rollout prompt prefix.
- Identical tokenizer/template semantics as training (special tokens, stop tokens, response prefix behavior), otherwise
  `parse_rollout_for_matching(...)` will silently misbehave.

### Relevant code references (existing ecosystem)
- ms-swift rollout + vLLM infrastructure (used by GRPO/GKD):
  - `/data/home/xiaoyan/AIteam/data/ms-swift/swift/trainers/rlhf_trainer/rollout_mixin.py`:
    - colocate rollout mode (`_colocate_rollout`) with TP-group gather/slice
    - weight sync (`_move_model_to_vllm`) gated by `global_step`
    - optional sleep/wake for KV cache and weights
  - `/data/home/xiaoyan/AIteam/data/ms-swift/swift/llm/infer/infer_engine/grpo_vllm_engine.py`:
    - vLLM engine wrapper returning `RolloutOutput` with `token_ids` and `prompt_token_ids`
- Qwen3-VL vLLM backend (inference-oriented, helpful as a multimodal request reference):
  - `/data/home/xiaoyan/AIteam/data/Qwen3-VL/src/generation/backends/vllm_backend.py`
  - demonstrates vLLM VLM request payloads (`multi_modal_data: {"image": ...}`) and stop handling.

### vLLM colocate mode (the only supported mode for this change)
Concept:
- Create a vLLM engine inside the training job, using the same GPUs as training.
- Optionally use vLLM tensor parallelism by grouping ranks (world size divisible by `vllm_tensor_parallel_size`).

How ms-swift does it (reference pattern):
- For `vllm_tensor_parallel_size > 1`, it gathers inputs inside each TP subgroup, runs one engine infer on the gathered
  list, then slices outputs back per-rank.
- See `_colocate_rollout` in `/data/home/xiaoyan/AIteam/data/ms-swift/swift/trainers/rlhf_trainer/rollout_mixin.py`.

Pros:
- No additional GPUs needed.
- Avoids network hop to server.

Cons:
- KV cache competes with training activations/optimizer state (OOM risk).
- Must tune vLLM memory utilization down to leave headroom; throughput may become unstable.
- Still requires weight sync (below).

Operational defaults:
- `tensor_parallel_size=4` (TP=4) on a 4-GPU run; all ranks are one TP group.
- `gpu_memory_utilization=0.45` as a conservative starting point to preserve training headroom.
- No automatic fallback; if this is too conservative (wasted decode throughput), tune up; if training OOMs, tune down.

### Weight synchronization strategies (on-policy constraint)
Stage_2 rollouts are on-policy: rollouts should reflect current training weights closely enough to be meaningful.

vLLM uses a separate inference model instance, so "one model does rollout + backprop" in practice means:
**two runtimes, one set of weights**, kept in sync.

#### Full finetune (generally not feasible for vLLM rollouts)
- To be truly on-policy, full weights would need to be transferred into vLLM frequently.
- This transfer is typically too expensive per step; it can erase vLLM's decode advantage.
- Recommendation: treat "vLLM rollout + full finetune" as unsupported (or accept deliberately stale/off-policy rollouts).

#### LoRA-only finetune (feasible path)
ms-swift's pattern (important reference):
- Do a base sync once (or rarely), then sync adapter weights frequently.
- In colocate mode, ms-swift gates sync by `global_step`:
  - see `_fast_infer` (`if self.state.global_step != self._last_loaded_step: self._move_model_to_vllm()`).
- This implicitly allows some staleness within gradient accumulation steps, which is usually acceptable.

Implication for this change:
- vLLM LoRA MUST be enabled for rollout-matching training when the rollout backend is vLLM (adapter-only per-step sync).
- If multimodal LoRA touches the vision tower and vLLM LoRA becomes unstable, the expected fallback is:
  - freeze ViT in config (preferred), or
  - switch rollout backend to HF explicitly.

### Multimodal + LoRA compatibility analysis (high risk area)
ms-swift warns about vLLM LoRA for multimodal models:
- In `/data/home/xiaoyan/AIteam/data/ms-swift/swift/trainers/rlhf_trainer/rollout_mixin.py`, when `vllm_enable_lora`
  is enabled it warns that multimodal LoRA may misbehave if LoRA touches the ViT component.

Implications for CoordExp stage_2:
- If we want a robust vLLM rollout backend, we likely need **LLM-only LoRA**:
  - freeze the vision tower / avoid including ViT modules in `target_modules`,
  - ensure any adapters that touch vision are disabled for rollout (or fall back to HF rollout).
- Validate multimodal vLLM support early with a single-image prompt on the target checkpoint, before building more code.

### Integration points for `RolloutMatchingSFTTrainer` (minimal-change seam)
No part of parsing/matching/teacher-forcing needs to change.
The seam is restricted to: "given a raw sample, produce `(response_token_ids, decoded_text, prompt_token_ids)`".

Proposed (conceptual) insertion point:
- Replace `_rollout_one` / `_rollout_many` internal generation call with a backend object:
  - HF backend: uses current `template.generate(...)` (microbatched).
  - vLLM backend: uses ms-swift-style engine/client that returns token ids.

Non-negotiable invariants for stage_2 correctness:
- The backend must return token ids compatible with the *same tokenizer* as `template.tokenizer`.
- The backend must preserve stop-token stripping semantics (stage_2 uses `template.skip_stop_tokens(...)` today).

### Performance expectations and trade-offs
When vLLM is likely to help:
- Rollout phase dominates step time (decode-bound).
- Rollout batch sizes are small/variable-length (vLLM continuous batching shines here).
- You can amortize any sync/IPC overhead with enough rollout tokens per step.

Main overheads that can erase gains:
- Full-weight sync or too-frequent sync.
- Colocate memory contention (OOM -> smaller max lengths / batch sizes -> worse throughput).
- Extra collectives / gather/slice patterns under TP=4 (must be implemented correctly, but overhead is typically small vs decode).

Recommendation for practical progression:
1) Default to vLLM colocate rollout with conservative memory reservation (`gpu_memory_utilization=0.45`) and TP=4.
2) If training becomes the bottleneck (e.g., long packed sequences OOM), tune:
   - vLLM `gpu_memory_utilization` down (prioritize training stability), and/or
   - reduce training max length / utilization ratio (fail-fast; user tunes).
3) Keep HF rollout backend available as a deterministic fallback for debugging or emergency recovery.

## Risks / Trade-offs
- Packing changes attention context across samples (standard packing behavior).
  - Mitigation: this is already accepted for stage_1 packing in this repo; stage_2 should be consistent.
- Carry buffers can introduce slight rollout "staleness":
  - A carried segment's rollout is generated at an earlier step than when it is used for backprop.
  - Mitigation: keep the carry buffer small in steady-state (tune raw batch size so produced tokens per step do not outpace one packed forward), and prefer FIFO selection to bound staleness.
- Packed-mode loss indexing bugs can silently corrupt training.
  - Mitigation: add explicit unit tests for packed vs un-packed loss-mask equivalence on synthetic token sequences.
- Batched generation increases KV cache footprint.
  - Mitigation: keep rollout microbatch size small and configurable; provide OOM-safe fallback to microbatch_size=1.

## Validation Plan (what to measure)
Correctness:
- Unit tests: packed-mode `compute_loss` produces identical label masking and coord supervision targets to un-packed mode for synthetic inputs.
- Smoke run: compare counters (`rollout/*`) before/after; ensure match rates and losses are consistent.

Efficiency:
- Throughput: steps/sec and (ideally) tokens/sec for:
  - rollout phase, forward phase, and end-to-end step time.
- GPU utilization: `nvidia-smi dmon` or Nsight Systems during rollout vs forward.
- Straggler analysis: per-rank rollout time variance (log rank-local timers).
