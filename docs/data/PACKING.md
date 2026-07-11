---
doc_id: docs.data.packing
layer: docs
doc_type: reference
status: canonical
domain: data
summary: Current CoordExp-Swift packing and pack-cache routing guide.
updated: 2026-07-11
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
- Pack-cache manifests record the resolved determinants, code identity,
  materialization metadata, chunks, and hashes. Worker count is provenance, not
  semantic cache identity.
- Cache fingerprints include the renderer, Qwen encoding/position/forward,
  packing planner, and supervision-token construction code identities where the
  stable spec requires them.

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
- `tests/training/test_pipeline_assembly.py`
- `tests/qwen/` for position/forward assumptions that affect packed inputs

When a change affects packing determinants, update the stable OpenSpec contract
and the cache-identity tests together. Do not make a documentation-only
cleanup by adding a new cache knob or compatibility layer.

## Historical note

Older Stage-1 static-packing matrices under `configs/stage1/` are preserved for
old-run interpretation. They are not the current Swift packing owner or a
current config route.
