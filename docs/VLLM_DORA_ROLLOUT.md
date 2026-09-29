# Local DoRA rollout for research probes

`src.qwen.vllm_rollout.VllmDoraRollout` keeps a vLLM engine resident beside
the caller's HF training model. The model extension lives entirely in this
checkout; it does not edit the Conda installation or merge/export the base
model after an update. HF continues to own forward, backward and optimizer state.

## Supported boundary

- vLLM 0.29.0, Qwen3-VL, BF16 nonquantized frozen base, TP=1 and PP=1.
- One language-tower DoRA adapter with a fixed target set/rank, plus independent
  FP32 selected-token input and output embedding deltas.
- Full-image, empty-history greedy rollout with explicit prompt IDs, EOS and
  token budgets. Other sampling/history policies are not implemented here.
- One isolated spawned engine per trainer GPU, avoiding the trainer's DDP
  process group. Default KV allocation is 2 GiB; max context is 16,000 tokens
  and max concurrent requests is 3. The full model and activations are additional.

The local linear wrapper retains PEFT's unmerged magnitude/base/low-rank
calculation and packed QKV/gate/up slices. Norms are computed at load/refresh,
then reused for generation. Selected-token deltas remain separate additions.
This does not promise bitwise-identical HF and vLLM full-model logits or greedy
trajectories: their attention, base GEMM and batching kernels can differ.

## Training integration

`probes.online_row_credit run` and `readback` accept `--rollout-backend vllm`.
HF remains the default. A **new** qualification must explicitly contain
`"rollout_backend": "vllm"` and bind the current source; old qualification hashes
and stopped-run receipts must never be rewritten to admit this backend.
This option changes only acquisition: the existing plan, replay loss, image
weights, optimizer, save schedule and update barriers retain their owners.

Each refresh transfers the complete current adapter and both deltas in memory,
checks tensor keys/shapes/dtypes, recomputes magnitude scales, updates buffers in
place, clears prefix state and acknowledges the new producer identity. A failed
refresh closes the engine; it cannot silently generate from stale weights.
Generation verifies returned prompt IDs and stop-token evidence. Batched timing
is explicitly recorded as batch wall time divided by request count.

The generic API also supports a caller-owned loop:

```python
with VllmDoraRollout(base_model=base, checkpoint=start, identity=version,
                     log_path=output / "vllm.log") as rollout:
    # requests are NativeRequest objects with expected_token_ids and image identity.
    results = rollout.generate(requests, budgets=budgets, eos_token_id=eos,
                               pad_token_id=pad, identity=version)
    # The caller runs its existing HF forward/backward/optimizer step.
    rollout.refresh(model, embedding_deltas, identity=next_version)
```

## Technical qualification

The bounded executable is
`scripts/probes/coordexp_infras/vllm_dora_rollout.py --checkpoint <checkpoint>
--output <fresh-output> --max-new-tokens 128 --learning-step`.
It records startup separately from warm generation, realized token counts,
HF/vLLM shared prefixes, HF replay of vLLM chosen log probabilities, and one
technical update followed by in-place refresh. The update is a plumbing check,
not a research objective or efficacy experiment. Full training and any change
to the stopped run's scientific recipe still belong to its research owner.

Results and accepted launch limits are recorded in
`/data/CoordExp/outputs/runtime-optimization/2026-09-29-vllm-dora/`.
