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
   `PeftModel.from_pretrained(..., is_trainable=False)`, and call
   `merge_and_unload(safe_merge=True, adapter_names=["default"])`.
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

## Qualification

Reload dynamic and materialized HF compositions independently. Compare tied
weights, selected-token weight rows, fixed-prefix raw logits, selected-token
logits, and exact greedy token ids on a real multimodal fixture. Selected rows
must be bitwise equal after target-dtype casting. FP32 full-vocabulary logits
use `rtol=1e-4`, `atol=5e-3`; FP32 selected-token logits use `rtol=1e-4`,
`atol=2e-3`. Tied storage and greedy token ids are exact checks. Only then may
vLLM consume the derived snapshot.

## Failure Policy

Incomplete staging is never published. A corrupt completed cache entry fails
closed and is not auto-deleted or auto-repaired. Multi-rank execution builds in
the controller before GPU workers start.
