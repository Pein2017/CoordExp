## Context

The online residual-trie vLLM gate failed before the first useful rollout
because native vLLM Qwen3VL rejected `coord_offset_adapter.*` during full
weight sync. The failure is not an OOM signal; it is a mismatch between the
learner state dict, which contains CoordExp PEFT `modules_to_save` state, and
the vLLM server model, which only accepts ordinary model weights.

The research requirement is subtle but firm:

- the base model should remain frozen for optimizer and checkpoint semantics;
- training should update and save adapters only;
- rollout generation should still see the same effective policy as the learner;
- native vLLM server should not need to know about CoordExp private modules in
  the first implementation slice.

## Key Decisions

### Use a Runtime Materialized Sync Snapshot

Full-sync should construct a transient vLLM-bound state dict that represents the
learner's effective inference weights. This state dict may contain ordinary
weights with LoRA/DoRA and token-row deltas applied, but those changes must not
survive into later training steps or be saved as a merged checkpoint. The
existing PEFT full-sync path may temporarily call merge/unmerge operations; this
change requires no lasting live-parameter mutation and must keep unmerge
cleanup exception-safe.

This keeps the experimental semantics:

```text
training state: frozen base + trainable adapters
checkpoint state: adapter-only
vLLM sync state: transient ordinary-weight snapshot
```

Existing PEFT full-sync canonicalization remains responsible for realizing
standard LoRA/DoRA deltas into ordinary tensors. This change owns CoordExp
token-row and `modules_to_save` materialization after that canonicalization, and
owns final hygiene validation before native vLLM receives the snapshot.

### Materialize Token-Row Adapters Before Filtering

`coord_offset_adapter` affects both input embedding rows and output logit rows.
Skipping it makes vLLM rollout semantically different from HF learner rollout.

The sync helper should:

1. find the active coord/token-row adapter after PEFT wrapping;
2. read `coord_ids`, `embed_offset`, and optional `head_offset`;
3. find canonicalized `embed_tokens.weight` and optional `lm_head.weight` keys
   in the state dict being sent to vLLM;
4. clone only tensors that need row updates;
5. add row deltas with dtype/device conversion;
6. delete `coord_offset_adapter.*` and related `modules_to_save` keys after
   successful materialization.

### Tie-Head Semantics

For `tie_head=true`, `embed_offset` is the shared row delta. If the sync state
dict contains both `embed_tokens.weight` and `lm_head.weight`, both should be
patched so the ordinary-weight snapshot is backend-robust. If only
`embed_tokens.weight` exists, patching only the embedding table is acceptable
only when the model confirms tied output embeddings through config and/or shared
input/output embedding storage. Unknown tying must fail fast instead of assuming
vLLM will infer the tie correctly.

For `tie_head=false`, `lm_head.weight` must be present or safely materialized in
the sync snapshot. If that cannot be done with native vLLM loader semantics, the
run must fail fast.

### Keep Server vLLM Native in Phase 1

Phase 1 should not modify vLLM site-packages or upstream HF model files. It
should adapt the learner-side snapshot before calling native vLLM
`load_weights()`.

This is intentionally more conservative than true adapter-only sync. It costs
more bandwidth but avoids a large runtime extension before the 6-server /
2-learner memory and throughput gate is understood.

### Do Not Trust Fire-And-Forget Sync Failures

ms-swift server-mode weight update currently acknowledges request receipt before
worker-side `load_weights()` completes. Phase 1 is therefore scoped to
learner-side sync snapshot preparation, validation, and DDP-rank-symmetric
abort before broadcasting known-invalid weights. Longer term, server-side
load-error acknowledgement or health checks should be tightened, but that is a
separate contract.

## Proposed Architecture

Add one small sync adapter module, for example:

```text
src/trainers/rollout_runtime/vllm_sync_materialization.py
```

Suggested public helper:

```python
def materialize_state_dict_for_vllm_full_sync(
    *,
    model: Any,
    state_dict: Mapping[str, torch.Tensor],
    logger: Any | None = None,
) -> dict[str, torch.Tensor]:
    ...
```

Responsibilities:

- operate on an already PEFT-canonicalized state dict;
- discover active coord/token-row adapter state from `model`;
- fail fast if coord/token-row adapter keys are present in the state dict but
  the active adapter cannot be discovered;
- patch ordinary row weights into cloned tensors;
- remove learner-only adapter keys;
- validate no forbidden key families remain;
- return a new dictionary suitable for vLLM `load_weights()`.

Call sites:

- `src/trainers/rollout_runtime/vllm_server.py::sync_vllm_server_full_weights`
  after PEFT canonicalization and before `_vllm_server_update_state_dict`.
- `src/trainers/rollout_runtime/vllm_engine.py::sync_vllm_full_weights_if_needed`
  after PEFT canonicalization and before `engine.inner_model.load_weights`.

## Failure Policy

Fail fast when:

- adapter state has `coord_ids` without a compatible `embed_offset`;
- `coord_ids` exceed vocab rows in the selected embedding tensor;
- no `embed_tokens.weight` key is present while active token-row adapter state
  exists;
- `tie_head=true` lacks `lm_head.weight` and the model does not confirm tied
  output embeddings;
- `tie_head=false` has `head_offset` but no safe `lm_head.weight`
  materialization route;
- forbidden keys remain after materialization.

Do not fail merely because no active coord/token-row adapter exists; ordinary
full-sync without row materialization remains valid.

## Alternatives Considered

### Filter Only

Filtering `coord_offset_adapter.*` without materializing rows avoids the crash
but silently changes rollout behavior. Rejected.

### vLLM Adapter-Only Sync

ms-swift has a LoRA tensor adapter path, but vLLM LoRA does not support generic
PEFT `modules_to_save`, and multimodal LoRA support is limited. This is a
future performance project, not the Phase 1 compatibility fix.

### Permanent Merge To Base

Permanently merging adapter state into a derived base model would make vLLM
loading easy, but it pollutes checkpoint lineage and weakens the frozen-base /
adapter-only training contract. Keep it only for export/inference baselines.

### Custom vLLM Model/Worker Extension

This can eventually support low-bandwidth row-sync, but it introduces vLLM
version coupling, TP/DP shard semantics, prefix-cache invalidation, and runtime
state management. Defer until full-sync cost is measured as the bottleneck.

## Verification Strategy

- Unit-test row materialization parity against hook behavior on a toy model.
- Unit-test `tie_head=true` and `tie_head=false` behavior.
- Unit-test state-dict hygiene after materialization.
- Unit-test no-op behavior when no coord/token-row adapter is active.
- Unit-test fail-fast behavior for missing embedding/head rows and invalid
  shapes.
- Run targeted Stage-2 rollout runtime tests.
- Run `--cfg-only` for
  `configs/stage2_two_channel/smoke/compact_full_et_rmp_ce_ckpt3664_hf_coco80_view_train128_val64_4steps_online_residual_trie_tail_append_zero_fp_lr1e5_decode4_vllm_6srv2lr_gate.yaml`.
- After approval and implementation, rerun the vLLM gate and inspect GPU memory,
  `temp/vllm6srv2lr_tail_gate.log`, first rollout sync logs, and
  decode/invalid counters.
