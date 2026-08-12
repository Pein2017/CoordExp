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
- Cache preparation renders and tokenizes the dataset once before training. It
  uses the bounded 16-process fork pool by default, restores results to exact
  source order, and records the materialization strategy/worker count. Training
  does not expose a second runtime-tokenization mode.
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

## Cache retention

Pack caches are disposable acceleration state, not the provenance owner for a
dataset, run, checkpoint, or research conclusion. Retain a payload only while
it uses the current supported format and is referenced by active work, or while
an in-progress exact-resume/checkpoint receipt explicitly requires it.

Older-format payloads are not migration inputs and may be retired after all of
the following checks pass:

1. no live training/cache-preparation process or open file references the
   candidate directory;
2. the target is an exact fingerprint child of the documented cache root, not
   the root itself;
3. current configs, run manifests, checkpoints, receipts, and active research
   controllers do not reference it; and
4. a compact retirement receipt records the cache root, format, fingerprints,
   aggregate bytes, exclusions, and evidence trade-off.

Current-format cache payloads referenced by active exact-resume evidence remain
out of scope even when an older experiment created them. Model weights,
processed/raw data, run outputs, checkpoints, and TensorBoard or evaluation
artifacts are never governed by this cache-retirement policy. Training startup
and cache preparation stay non-destructive; retirement is a separate operator
action so a launch cannot silently garbage-collect another run's state.

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

The 2026-08-12 retirement of obsolete pack-cache and vLLM-materialization
payloads is recorded in
[`docs/history/cache-retirement/2026-08-12.md`](../history/cache-retirement/2026-08-12.md).
