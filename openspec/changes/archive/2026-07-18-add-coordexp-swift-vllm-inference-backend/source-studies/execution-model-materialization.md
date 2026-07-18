# Execution-Model Materialization Source Study

## Current Model Composition

Canonical HF inference loads a base Qwen3-VL model, applies a PEFT adapter with
`peft_type: LORA` and `use_dora: true`, then installs the selected-token
embedding delta. The delta payload covers the four schema wrappers and 1,000
coordinate tokens. It is additive, validated against tokenizer/base metadata,
and applied to tied input embedding and lm-head semantics.

## Selected Materialization Contract

1. Validate base, tokenizer, adapter, and delta identities through their owner
   modules before mutation.
2. Load the base model on CPU directly in the configured target dtype.
3. Inspect adapter content through `src.adapters.dora`, load it with
   `PeftModel.from_pretrained(..., is_trainable=False,
   autocast_adapter_dtype=False)`, and call
   `merge_and_unload(safe_merge=True, adapter_names=["default"])`.
   Disabling adapter autocast is required to match the existing Transformers
   inference path: PEFT otherwise promotes BF16 adapter tensors to FP32 before
   the merge and changes the resulting executable weights.
4. Validate that no PEFT wrappers, DoRA magnitude vectors, LoRA tensors, or
   parametrization wrappers remain in the standard model.
5. Inspect delta content through `src.qwen.special_token_embeddings`; add the
   FP32 selected-token delta exactly once to target-dtype indexed rows of the
   tied input embedding/output-head weight. Do not cast the whole model after
   this fold.
6. Verify tied storage and selected-token values after folding.
7. Save model, config, tokenizer, and processor through standard HF
   `save_pretrained` surfaces.
8. Hash the completed `snapshot/`, write `coordexp_materialization.json` beside
   it, and atomically publish the composition directory.

## Fingerprint Determinants

- Base model config, tokenizer/processor files, and all weight shards.
- Adapter config and every required adapter tensor file.
- Delta metadata and tensor files, selected token ids/strings, and source dtype.
- Target dtype, tied-weight expectation, algorithm version, Transformers and
  PEFT versions.

The source determinants produce a `composition_key`; the actual published
snapshot bytes produce a separate `snapshot_fingerprint`. Derived entries use
`vllm_materialized/<composition_key>/snapshot/` and locks use
`vllm_materialized/.locks/<composition_key>.lock`. The receipt remains outside
the snapshot to avoid self-referential hashing. Paths, timestamps, lock owner,
and staging names are provenance only.

Adapter identity hashes `adapter_config.json` and every adapter tensor payload.
Delta identity hashes metadata and tensor files plus tensor key/shape/dtype,
selected token ids/strings, base-config identity, and tokenizer identity. A
checkpoint-handoff manifest is an optional path source, not a prerequisite.

## Composition Fidelity

Reload the materialized HF composition and require exact owner-recorded merged
target weights, selected-token rows after target-dtype casting, and tied
storage. Run dynamic and materialized HF independently on the same real
multimodal fixture and retain fixed-prefix raw logits, selected-token logits,
and greedy ids as BF16 behavioral diagnostics. FP32 full-vocabulary diagnostics
use reference `rtol=1e-4`, `atol=5e-3`; selected-token diagnostics use
`rtol=1e-4`, `atol=2e-3`. Exceeding those references does not invalidate exact
state composition. Materialized HF is the vLLM execution oracle; canonical
dynamic HF remains supported and is the val200 behavioral baseline.

## Executed BF16 Finding

The real step-4887 base-plus-DoRA-plus-delta probe established two separate
facts:

- After disabling PEFT adapter autocast, all 196 owner-recorded safely merged
  DoRA target weights remain bitwise equal after standard snapshot save/reload.
  The effective dynamic-HF selected-token rows, materializer-folded rows, and
  reloaded materialized-HF rows also share one exact content hash.
- The existing unmerged DoRA forward and the equivalent merged BF16 linear
  layer are not execution-identical because they evaluate the same algebra
  through different BF16 operation orderings. On the accepted row-0 fixture,
  the dynamic/materialized comparison observed full-vocabulary maximum
  absolute logit difference `0.921875`, selected-vocabulary maximum absolute
  difference `0.75390625`, and the first greedy-token mismatch at generated
  index `4`. A layer-level FP32 merge did not remove the discrepancy.

Therefore the owner-defined state composition is exact while dynamic-forward
behavior is not execution-identical. The user approved preserving canonical
dynamic HF, treating this dynamic/materialized comparison as diagnostic, using
materialized HF as the vLLM oracle, and retaining matched val200 metrics as the
behavioral compatibility gate.

## Failure Policy

Incomplete staging is never published. A corrupt completed cache entry fails
closed and is not auto-deleted or auto-repaired. Multi-rank execution builds in
the controller before GPU workers start.
