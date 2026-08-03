## Why

Prepared packing caches currently avoid JSONL rendering and packing, but every
distributed rank still performs two complete checksum-plus-unpickle passes over
the train payload before the first forward.  The first eight-rank production
launch measured about 10.5 minutes of GPU-idle admission.

A production-cache audit rejected the first lazy-stream draft: one 431 MB chunk
took 7.39 seconds to verify and decode, implying about 32 minutes of aligned
in-loop stalls across eight epochs without proven overlap.  The user-owned
objective is shortest final training wall clock, so this change selects the
smaller deterministic optimization instead of adding a prefetch subsystem.

## What Changes

- Make cache verification depth explicit: `manifest` validates semantic and
  structural identity without reading payload bytes; `payloads` additionally
  verifies every digest and restricted-unpickles every chunk.
- Keep single-process preparation, cache publication, cache completeness checks,
  and eager eval on `payloads` verification.
- During distributed train assembly, admit the prepared train cache with
  `manifest` verification, then let the existing eager rank loader perform the
  one and only full validated payload pass before forward.
- Preserve the existing eager rank-local tuple, canonical schedule, eval path,
  cache v2, chunk size, 16-worker preparation, and all training semantics.
- Reject lazy decoding and background prefetch in this wave because neither is
  needed to remove the duplicate pass and neither has decision-grade evidence
  of a shorter final wall clock.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-swift-pack-cache-semantic-identity`: distinguish structural manifest
  admission from full payload verification and require explicit lifecycle
  intent.
- `coordexp-swift-packing-forward`: require distributed train startup to perform
  no more than one full validated payload pass before forward.

## Impact

The implementation is limited to `src/training/pack_cache.py`, verification
intent in `src/training/pipeline.py`, and focused tests.  It changes no cache
schema, fingerprint, payload, data, prompt, geometry, evaluator, config, resume
behavior, or production process.  The coordinate-independent hunks can be
backported to the clean `coordexp-swift` owner without merging the mixed source.
