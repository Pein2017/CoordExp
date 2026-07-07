## Context

CoordExp-Swift now owns the local supervised training path, Qwen packed
forward boundary, checkpoint writing, and HF inference/eval infrastructure.
The remaining risk addressed here is not that those paths are missing; it is
that two artifact boundaries are still too easy to misuse:

- Packing caches are intentionally reusable and efficient, but their semantic
  identity must cover every source producer that can change the packed forward
  contract.
- Checkpoints can be useful on disk while still being ambiguous for future
  inference if base model, adapter, selected-token embedding delta, tokenizer,
  processor, template, and intended inference family are recomposed manually.

The active baseline remains the rebuilt CoordExp-Swift design and OpenSpec
artifacts. Archived OpenSpec material and MS-Swift are reference material only.

## Goals / Non-Goals

**Goals:**

- make stale cache reuse less likely by adding Qwen forward-side source
  identities to packing-cache determinants;
- preserve cache-hit efficiency and the existing cache payload shape;
- make checkpoint-to-inference identity canonical for the `handoff` gate
  through `checkpoint_handoff.json`;
- keep inference provenance honest by labeling handoff-backed composition
  separately from research/dev manual composition;
- add small tests and receipts that prove the contracts instead of relying on
  operator memory.

**Non-Goals:**

- no broad `run_training_pipeline` refactor in this change;
- no loss-runner or zero-weight loss implementation change in this change;
- no redesign of cache payload structure;
- no new public config knob for packing-cache worker count or source identity;
- no change to prompt rendering, object order, tokenization, Qwen position
  computation, FA2 execution, loss math, optimizer behavior, or evaluator
  metric reduction;
- no DeepSpeed, vLLM, or broad production-readiness claim.

## Decisions

### Pack Cache Determinants Use Conservative Source Identity

`src/training/pack_cache.py` keeps the existing semantic fingerprint mechanism
and adds forward-side producer files to the source identity set:

- `src/qwen/positions.py`
- `src/qwen/fa2.py`
- `src/qwen/forward.py`

Alternative considered: cache only lower-level packed/supervision data and
rebuild Qwen position/FA2/forward decorations after every cache load. That is
cleaner in theory, but it is a larger migration with more artifact churn. The
approved V1 path is conservative invalidation first because it protects
correctness without broad refactor.

Worker count remains provenance. It MUST NOT enter the cache fingerprint
because it is an execution strategy for materialization, not a semantic
description of the packed examples.

### Handoff Is The Identity Boundary

`checkpoint_handoff.json` is the canonical bridge from training to inference
for runtime handoff identity. The `handoff` gate checks base model, adapter
payload, selected-token embedding delta, tokenizer, processor, prompt/template
identity, and intended inference family. The `eval` gate layers accepted eval
artifact roots on top. A requested `production` gate returns hold in V1 because
production readiness remains out of scope for this narrowed implementation.

Manual base/adapter/delta paths remain useful for research and debugging, but
they must be marked as noncanonical evidence. This avoids treating a manually
assembled run as equivalent to an audited handoff identity.

### Readiness Validator Is Read-Only And Small

The readiness validator should inspect an existing run/checkpoint artifact
tree and return pass/fail with concrete missing or mismatched handles. It is
not a reporting framework and should not own training or inference behavior.

The implemented V1 surface keeps the validator small and read-only. Canonical
inference discovers a neighboring `checkpoint_handoff.json` from configured
checkpoint adapter or special-token embedding payload paths, validates it, and
records the validated identities. Direct inference config fields that point to a
handoff manifest or `checkpoint-final.json` alias are intentionally left for a
future change.

## Risks / Trade-offs

- **More cache misses after source edits** -> accepted; rebuilding is cheaper
  than silently training through stale packed-forward semantics.
- **Source identity is conservative rather than semantic AST diffing** -> keep
  file hashing simple and auditable. False-positive invalidation is acceptable.
- **Handoff validation can become a large checklist** -> keep V1 read-only and
  limited to artifact identities needed to prevent wrong checkpoint inference.
- **Manual inference paths remain possible** -> require explicit research/dev
  provenance so they are not cited as canonical handoff evidence.

## Migration Plan

1. Commit the roadmap and this OpenSpec baseline.
2. Add failing tests proving Qwen forward-side source files affect packing
   cache fingerprints while worker count does not.
3. Add the minimal source identity implementation.
4. Run targeted packing-cache tests and OpenSpec validation.
5. Implement the handoff/readiness validator and neighboring-handoff inference
   provenance after the cache identity patch is green.

Rollback is simple: revert the source identity file-list change and its tests.
Existing cache manifests remain readable because the payload shape is unchanged.

## Open Questions

No user-blocking questions remain for this narrowed change. A future CLI,
direct `checkpoint-final.json` inference resolver, and broader production gate
may be proposed later, provided they remain separate from the V1 read-only
handoff validator and do not turn this change into a report framework.
