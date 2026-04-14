## Context

The current `cxcy_logw_logh` experiment keeps canonical `xyxy` data at the
external boundary, but reconstructs model-facing bbox slots online inside the
training data path. That work is currently split across dataset preparation,
runtime dataset mutation, builder serialization, prompt selection, and cache
identity. The recent double-conversion bug showed that multiple runtime owners
of the same transform create silent train/infer asymmetry and weaken the
experiment’s evidentiary value.

CoordExp already has a strong offline public-data pipeline:
raw annotations/images -> dataset converter -> rescale -> norm/coord-token
artifacts -> training. The proposed refactor extends that offline pipeline with
an explicit bbox-format derivation branch inside `public_data/` so
non-canonical model-facing parameterizations are prepared once, stored
separately, and consumed directly by training. Inference and evaluation
continue to emit canonical `xyxy` artifacts, and Qwen3-VL chat-template
compatibility plus `do_resize=false` remain non-negotiable constraints.

Stakeholders:
- training/runtime maintainers who need simpler invariants and fewer hidden
  branches,
- experiment owners comparing `xyxy` vs `cxcy_logw_logh`,
- reproducibility/debugging workflows that depend on stable artifacts and cache
  provenance.

## Goals / Non-Goals

**Goals:**
- Move model-facing bbox-format derivation out of the runtime dataset/builder
  path and into the offline public-data pipeline.
- Create a fully separate prepared-data branch for non-canonical bbox formats,
  beginning with `cxcy_logw_logh`.
- Export branch-local split artifacts for both train and val when present,
  using the same split naming pattern inside the derived branch root
  (`train.jsonl`, `val.jsonl`, `train.coord.jsonl`, `val.coord.jsonl`).
- Keep canonical raw and evaluation-facing artifacts in `xyxy`.
- Add strict provenance and fail-fast guards so training cannot silently mix a
  canonical dataset with a non-canonical runtime conversion request.
- Reduce redundant conversion logic and make cache/run metadata auditable.

**Non-Goals:**
- Changing the external inference/eval contract from canonical pixel `xyxy`.
- Redesigning Stage-2 or rollout training around non-canonical bbox formats.
- Editing upstream HF/Qwen3-VL internals.
- Supporting arbitrary new bbox charts in this change beyond establishing the
  branch architecture and first implementation surface for `cxcy_logw_logh`.

## Decisions

### 1. Keep canonical presets untouched; add explicit derived branch roots

Decision:
- Canonical prepared presets remain the source of truth under
  `public_data/<dataset>/<preset>/`.
- Non-canonical model-facing artifacts live under a dedicated derived branch
  root, for example:
  `public_data/<dataset>/<preset>/bbox_formats/<format>/`.

Rationale:
- This preserves existing `xyxy` consumers, keeps evaluation/visualization
  simple, and makes branch identity visible in paths.
- It avoids mutating canonical artifacts in place or overloading one preset
  directory with mixed semantics.

Alternatives considered:
- Separate top-level dataset trees per bbox format:
  clearer isolation, but duplicates too much path logic and weakens the link to
  the canonical preset lineage.
- In-place overwrite of preset artifacts:
  rejected because it destroys the canonical baseline and makes provenance
  ambiguous.

### 2. Derive non-canonical artifacts inside `public_data/` and export mirrored split JSONLs

Decision:
- The offline derivation stage runs inside the `public_data/` workflow and
  reads canonical preset split artifacts after shared rescale preparation.
- For each available split, it emits a derived branch-local JSONL set using the
  same split naming pattern:
  - `train.jsonl` / `val.jsonl` contain offline-prepared numeric bbox tuples in
    the requested model-facing chart,
  - `train.coord.jsonl` / `val.coord.jsonl` contain the tokenized version of
    those same prepared tuples.
- It does not derive from runtime dataset objects or from ad hoc prompt text.

Rationale:
- Keeping the branch inside `public_data/` makes derivation part of the same
  offline preparation authority as dataset conversion and resize.
- Mirroring the split naming pattern makes train/val handling explicit and keeps
  the non-canonical workflow branch legible to operators.
- Emitting both numeric and coord-token JSONLs provides an inspectable offline
  branch while still preserving the tokenized training surface.

Alternatives considered:
- Derive directly from canonical `*.coord.jsonl`:
  rejected because it re-entangles token parsing with geometry conversion.
- Emit only derived `*.coord.jsonl` with no numeric JSONL companion:
  rejected because it weakens inspectability and makes validation/debugging
  depend on token parsing alone.

### 3. Preserve the established `cxcy_logw_logh` slot order

Decision:
- The offline-prepared `cxcy_logw_logh` chart continues to mean
  `[cx, cy, logw, logh]`.
- This slot order is recorded in branch manifests and record-level metadata.

Rationale:
- The existing inference/eval and geometry helpers already assume that chart.
- Changing slot order at the same time as the offline refactor would introduce
  a second confound and broaden the contract change unnecessarily.

Alternatives considered:
- Switch to `[cx, cy, logh, logw]` in the same change:
  rejected because it would create an incompatible new chart rather than
  isolating the current chart into an offline branch.

### 4. Runtime training stops owning bbox-format conversion

Decision:
- Stage-1 training with non-canonical bbox formats SHALL require an
  offline-prepared dataset branch whose provenance explicitly matches the
  requested `custom.bbox_format`.
- The supported online conversion path is removed for that workflow.

Rationale:
- A single conversion owner is the strongest safety guard against double apply,
  stale cache reuse, and hidden prompt/data mismatch.
- This turns `custom.bbox_format` from “please reinterpret the dataset at
  runtime” into “please validate that the dataset branch and prompt semantics
  match.”

Alternatives considered:
- Keep runtime conversion as a fallback with extra guards:
  rejected because it retains exactly the kind of mixed ownership that caused
  the current bug.
- Keep both paths and let configs choose:
  rejected because it preserves too much subtle risk for paper-critical
  geometry experiments.

### 5. Prepared branches are self-describing and provenance-gated

Decision:
- Each derived bbox-format branch emits a manifest recording:
  - canonical source artifact(s),
  - source bbox format,
  - derived bbox format,
  - derived bbox slot order,
  - coord-token/norm contract,
  - conversion version,
  - producer command/config.
- Derived records also carry per-record metadata sufficient for downstream
  validation if files are moved outside the original branch directory.

Rationale:
- Training, cache identity, and later audits need self-describing artifacts, not
  path-only conventions.
- Branch manifests make migration and debugging explicit.

Alternatives considered:
- Path-only identification:
  simpler but too fragile if artifacts are copied or merged.
- Manifest-only with no per-record stamp:
  weaker for downstream validation when individual JSONLs move.

### 6. `all` remains canonical; derived bbox branches are explicit opt-in steps

Decision:
- The canonical `public_data/run.sh <dataset> all` flow continues to produce raw
  -> rescale -> coord -> validate outputs only.
- Derived bbox-format generation is a distinct offline branch command/stage, not
  an implicit side effect of `all`.

Rationale:
- This keeps canonical data preparation stable and avoids silently creating new
  semantics for existing pipelines.
- Experiment owners opt in when they actually want a non-canonical branch.

Alternatives considered:
- Automatically generate derived branches inside `all`:
  convenient for one experiment, but expands the blast radius and makes default
  outputs less predictable.

### 7. Cache identity and config validation key off prepared branch provenance

Decision:
- Encoded-sample cache fingerprints and run metadata incorporate the prepared
  bbox-format branch identity and conversion manifest version.
- Training validation rejects mismatched combinations such as:
  canonical `xyxy` dataset + `custom.bbox_format=cxcy_logw_logh`, or
  derived `cxcy_logw_logh` dataset + `custom.bbox_format=xyxy`.

Rationale:
- Once bbox-format conversion moves offline, provenance becomes part of dataset
  identity rather than a runtime transform knob.
- This prevents stale cache reuse across semantically different dataset
  branches.

Alternatives considered:
- Rely on file paths alone for cache invalidation:
  insufficient when roots are shared or files are copied.

## Risks / Trade-offs

- [More prepared artifacts on disk] -> Keep canonical and derived branches
  clearly separated and derive from validated preset split surfaces to minimize
  recomputation/debug time.
- [Migration churn for existing configs] -> Make the failure mode explicit and
  actionable: configs must point to the derived branch for non-canonical
  experiments.
- [Over-scoping the first refactor] -> Limit the initial implementation to the
  branch architecture plus `cxcy_logw_logh`; preserve canonical inference/eval
  behavior unchanged.
- [Confusion between canonical raw data and model-facing derived data] ->
  Require explicit manifest metadata and record-level provenance stamps.
- [Potential incompatibility with existing cache contents] -> Treat this as a
  cache-fingerprint boundary and force cache separation rather than attempting
  mixed reuse.

## Migration Plan

1. Add the offline bbox-format derivation stage and branch manifest contract in
   `public_data/`.
2. Generate the first `cxcy_logw_logh` derived branch under `public_data/`,
   including branch-local `train.jsonl` / `val.jsonl` and matching
   `train.coord.jsonl` / `val.coord.jsonl` outputs when those splits exist.
3. Update training config validation so non-canonical bbox formats require a
   matching prepared branch and reject runtime conversion.
4. Update Stage-1 configs/examples to point at the derived branch.
5. Invalidate or segregate encoded-sample caches via the new provenance-aware
   fingerprint.
6. Retrain the affected `cxcy_logw_logh` experiments from the offline-prepared
  branch and re-run the existing infer/eval workflow.

Rollback:
- Canonical `xyxy` training/eval remains available because canonical preset
  artifacts are unchanged.
- If the derived-branch implementation proves insufficient, the change can be
  rolled back by removing the branch generator and keeping configs on canonical
  `xyxy` artifacts; no canonical dataset rewrite is required.

## Open Questions

- Should the derived branch path use `bbox_formats/<format>/` or another
  canonical subdirectory name such as `model_facing/<format>/`?
- Should the derived branch keep only `train.jsonl` / `val.jsonl` plus
  `train.coord.jsonl` / `val.coord.jsonl`, or should it also emit a separate
  decoded parity helper artifact for audits against canonical `xyxy`?
