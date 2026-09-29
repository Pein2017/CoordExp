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
- Native full-decode CUDA graphs are enabled for batches 1..3, without
  TorchInductor compilation. `enforce_eager=True` disables them. The startup
  free-memory gate is 20% of device capacity (`gpu_memory_utilization=0.2`);
  explicit KV bytes still determine cache allocation. This is not a hard limit
  on total GPU memory and leaves room for the separate HF/DDP trainer.

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
The smoke also restores the original parameters in the same engine and requires
exact reproduction of the original token IDs and raw log probabilities.

On one A100 80GB PCIe, the step-256 instance-margin checkpoint and images
1584/2299 produced 255 tokens with a 128-token per-image cap:

| Warm generation | Wall seconds | Realized tokens/second |
|---|---:|---:|
| HF, sequential | 25.27 | 10.09 |
| vLLM eager, two-image batch | 7.60 | 33.56 |
| vLLM decode graphs, two-image batch | 1.77 | 144.22 |

These are single bounded measurements, not an eight-rank/full-training speedup
claim. Graph generation was 14.29x faster than its co-resident HF baseline.
The graph engine's cold startup took 45.68 seconds; transferring an updated
snapshot, recalculating scales and acknowledging it took 0.63 seconds end to end.
Its reported peak PyTorch allocation was 6.96 GiB, excluding the HF process and
non-PyTorch allocations. Eager and graph vLLM produced identical initial token
IDs; updating weights changed both outputs and restoring them reproduced both
token IDs and raw log probabilities exactly.

HF/vLLM shared greedy prefixes were only 14/127 and 33/128 tokens. Chosen-token
log probabilities over the first 32 tokens differed by up to 0.30. This backend
is an explicit change of rollout implementation: it must not be inserted into
a frozen HF experiment or represented as numerically identical. A fresh research
qualification owns whether this difference is acceptable for its intended use.

Receipts: `eager-smoke-03/receipt.json`, `graph-smoke-01/receipt.json`. The
distributed smoke runs via `python -m torch.distributed.run --standalone
--nproc_per_node=2 scripts/probes/coordexp_infras/vllm_dora_rollout.py` with the
same checkpoint/output arguments and `--learning-step --max-new-tokens 32`.
It uses distinct images, actual DDP backward, equal updated snapshot hashes,
one independent vLLM engine per rank and the same restore check.
`ddp-smoke-02/rank-{0,1}/receipt.json` passed: each rank had 590 finite,
nonzero gradients, identical updated snapshot hashes, changed post-update
outputs/scores, and exact restoration. End-to-end refresh took 1.03–1.04 seconds.
The initial DDP attempt hit vLLM's default 92% free-memory startup gate; the
explicit 20% gate fixes that co-residency failure without changing the KV budget.
Eight-rank, full 3,084-token acquisition and the research loss itself remain
the owning experiment's fresh qualification, not claims of this bounded smoke.

Results and accepted launch limits are recorded in
`/data/CoordExp/outputs/runtime-optimization/2026-09-29-vllm-dora/`.
The lead's bounded infrastructure acceptance is `final-acceptance.json` there.
