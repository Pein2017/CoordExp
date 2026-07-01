## Why

The public-data pipeline has grown multiple partially-overlapping implementations of the same concepts:

- Artifact layout + naming:
  - Canonical pipeline artifacts use `<split>.raw.jsonl`, but we also materialize `<split>.jsonl` as a full file copy for compatibility (`public_data/pipeline/stages.py` + `public_data/pipeline/writer.py` + `public_data/run.sh`).
  - Max-object preset naming currently supports both `_max_60` and `_max60`, with “reuse legacy if exists” behavior (`public_data/pipeline/naming.py`).
- Validation:
  - Training preflight used a dedicated max-pixels validator while the pipeline used `public_data/scripts/validate_jsonl.py::JSONLValidator` (two validators with duplicated policy knobs).
- Image materialization + safety:
  - “No symlinked images/ for rescale presets” is enforced in multiple places (rescale, validate, pipeline validate, materialize tool), and the pipeline planner copies image trees for derived presets (`public_data/pipeline/planner.py`).
- Parallelism:
  - Different steps use different multiprocessing primitives and worker normalization logic across scripts/stages, increasing maintenance cost and making performance tuning ad-hoc.

This shows up as:

- Disk/time redundancy (duplicated JSONLs, unnecessary image tree copies).
- Policy drift (same contract enforced in multiple tools with slightly different edge cases).
- Higher cognitive load: it is hard to answer “what is the canonical artifact for this preset?” and “which validator is authoritative?”.

This change proposes a “refresh” that removes legacy/back-compat behaviors and converges on a single canonical pipeline layout + shared implementation building blocks.

## What Changes

### Breaking Changes (Intentional)

- **Canonical preset JSONL naming becomes single-source**:
  - Pixel-space preset JSONL uses `<split>.jsonl` (no `<split>.raw.jsonl` + alias copy).
- **Single canonical max-object suffix**:
  - Canonical suffix is `_max{N}` (for example `rescale_32_768_bbox_max60`).
  - Remove `_max_{N}` support and remove “reuse legacy if exists” behavior (single strict resolver).
- **Single-source preset resolution**:
  - `public_data/run.sh` stops pre-resolving “effective preset” names; it passes `--preset <base>` plus `--max-objects` through to the pipeline factory.
  - The internal pipeline planner becomes the only source of truth for `effective_preset` naming and directory selection.
- **Derived preset image duplication is removed**:
  - Max-object derived presets no longer byte-copy resized `images/` trees.
  - Derived presets reuse base-preset resized images via hardlinks, keeping JSONL `images[0]` paths stable as `images/...` (see design).
- **Image immutability is enforced**:
  - Preset `images/` are treated as immutable once written by rescale.
  - Pipeline tools only create missing images; they MUST NOT overwrite existing image files in-place.
- **Validation tool consolidation**:
  - Training preflight and pipeline validation use the same underlying validator implementation.
  - Remove the “two validators” split (and the drift risk it introduces).

### Non-Breaking (Preserved)

- Geometry invariants are preserved (no dropping/reordering of coords).
- Training continues to run with `do_resize=false` (offline preparation remains mandatory).
- Qwen3-VL chat-template compatibility is preserved (no upstream HF model edits).

## Impact

- `public_data/pipeline/*` (planner, stages, naming, writer).
- `public_data/run.sh` and the pipeline factory entrypoint script.
- `scripts/train.sh` and `scripts/train_stage2.sh` dataset preflight validation.
- OpenSpec specs:
  - `openspec/specs/public-data-pipeline/spec.md`
  - `openspec/specs/public-data-adapter-factory/spec.md`
