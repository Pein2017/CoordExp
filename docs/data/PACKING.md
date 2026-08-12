---
doc_id: docs.data.packing
layer: docs
doc_type: reference
status: canonical
domain: data
summary: Current CoordExp-Swift packing and pack-cache routing guide.
updated: 2026-08-12
---

# Packing Policy

For current `main` behavior, packing is owned by `src/packing/` and the
training pack-cache seam in `src/training/pack_cache.py`. Exact compatibility
semantics are defined by
[`coordexp-swift-packing-forward`](../../openspec/specs/coordexp-swift-packing-forward/spec.md)
and
[`coordexp-swift-pack-cache-semantic-identity`](../../openspec/specs/coordexp-swift-pack-cache-semantic-identity/spec.md).

## Current policy

- Packing concatenates examples without padding into physical segments.
- `global_max_length` is the hard sequence limit for the resolved training
  config.
- Logical supervision spans are remapped to physical packed positions by
  `src/packing/supervision.py`.
- A sample whose atomic sequence cannot satisfy the configured hard cap fails
  through the current contract; it is not silently truncated or reordered.
- Pack-cache v3 manifests record the resolved determinant registry, source-code
  identities, materialization metadata, chunks, and hashes. Worker count is
  provenance, not semantic cache identity.
- Cache fingerprints bind raw/image content, tokenizer/processor front-end
  assets, realized vocabulary groups, serialized micro-step runtime fields, and
  the declared renderer/parser/geometry/Qwen/packing/supervision/schema/
  serializer owners. Inventory sizes are bounded and fail closed.
- Each cache is published immutably at
  `<cache-root>/coordexp-swift-pack-cache-v3/<fingerprint>`. `Rebuild` resolves a
  new identity and publishes only to an absent target; preparation does not
  repair, replace, delete, or garbage-collect existing cache directories.
- Required chunks are opened without following symlinks, authenticated as one
  bounded stable byte snapshot, and restricted-decoded from those exact bytes.
- Multi-GPU launches use a model-free admission lifecycle. Run the preparation
  subprocess with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `FLASH_ATTENTION_DETERMINISTIC=1` already present in its child environment;
  the preparation entrypoint verifies but never synthesizes those values. Run
  it once before
  `accelerate launch`; startup validates the current rank's required train
  payloads and every eval payload before constructing Accelerate or loading
  model weights. Distributed ranks are cache consumers only and fail fast on a
  cache miss.

## Ownership map

| Concern | Owner |
| --- | --- |
| Physical segment planning | `src/packing/planner.py` |
| Packed supervision positions | `src/packing/supervision.py` |
| Cache manifest/fingerprint | `src/training/pack_cache.py` |
| Assembly and cache receipt | `src/training/pipeline.py` |
| Token-level supervision records | `src/supervision/tokens.py` |

## Verification

Use the targeted packing and cache tests before a broader training smoke:

- `tests/packing/`
- `tests/training/test_pack_cache.py`
- `tests/training/test_pack_cache_determinant_registry.py`
- `tests/training/test_pack_cache_runtime_constructor.py`
- `tests/training/test_pipeline_cache_preflight.py`
- `tests/training/test_pipeline_assembly.py`
- `tests/qwen/` for position/forward assumptions that affect packed inputs

For a production config, attest the startup boundary directly before launch:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1 python -m src.prepare_train_cache --config configs/coordexp_swift/prod/<config>.yaml
```

When a change affects packing determinants, update the stable OpenSpec contract
and the cache-identity tests together. Do not make a documentation-only
cleanup by adding a new cache knob or compatibility layer.

## Historical note

Older Stage-1 static-packing matrices under `configs/stage1/` are preserved for
old-run interpretation. They are not the current Swift packing owner or a
current config route.
