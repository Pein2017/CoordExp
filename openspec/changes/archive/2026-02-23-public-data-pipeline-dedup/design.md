## Context

CoordExp’s “public data” preparation currently has:

- A unified runner surface (`public_data/run.sh`) and a python pipeline/factory (`public_data/scripts/run_pipeline_factory.py`).
- An internal stage pipeline (`public_data/pipeline/{planner,stages,writer}.py`) that writes multiple intermediate artifacts (`*.raw.jsonl`, `*.norm.jsonl`, `*.coord.jsonl`) and then emits a “legacy” alias copy (`*.jsonl`).
- Multiple validators and multiple sites enforcing the “no symlinked images/ for rescale presets” policy.

This change focuses on reducing redundancy and forcing a single refreshed, canonical behavior.

Constraints:

- Config-first: behavior is driven via config/schema and the pipeline plan (avoid adding new ad-hoc flags).
- Preserve geometry invariants and keep training `do_resize=false`.
- Keep Qwen3-VL chat-template compatibility; do not edit upstream HF model internals.

Note on CLI flags:
- This change may add CLI flags to `public_data/scripts/validate_jsonl.py` only to replace legacy dedicated preflight max-pixels flags.
- No new runner flags are introduced for `public_data/run.sh`; the external runner grammar remains stable.

Note on max-object configuration:
- `PUBLIC_DATA_MAX_OBJECTS` remains the supported user-facing knob for `public_data/run.sh` (runner interface).
- Internally, this is carried as pipeline config field `max_objects` and passed as `--max-objects` to the factory entrypoint.
- Precedence / conflict behavior:
  - The canonical source of truth is the pipeline config field `max_objects`.
  - If `max_objects` is provided via more than one input surface (for example both config and `--max-objects`), then:
    - if the values are identical, proceed,
    - otherwise, fail fast with an actionable error (do not silently override).

## Goals / Non-Goals

Goals:

- Single canonical artifact layout in preset directories (remove duplicate alias outputs).
- Single canonical naming scheme for max-object derived presets (remove “reuse legacy if exists”).
- Single shared validator implementation for both pipeline validation and training preflight.
- Remove unnecessary image tree copying for derived presets while staying safe (no symlinked images/ write hazards).
- Keep multiprocessing only for heavy work (resizing/materializing images, per-record transforms), not metadata scans.

Non-Goals:

- No new datasets or adapter rewrites.
- No architectural changes to model training/eval beyond the preflight validator call sites.
- No attempt to make runtime resizing acceptable; offline preparation remains required.

## Decisions

1) **Canonical preset JSONL layout**

- Decision: for pixel-space preset JSONL, emit `<split>.jsonl` as the only canonical output.
- Rationale: aligns with existing docs and user-facing paths, removes the redundant full-file copy (`<split>.raw.jsonl -> <split>.jsonl`) and the `LegacyAliasStage`.

2) **Canonical max-object suffix**

- Decision: canonical suffix is `_max{N}` (for example `..._max60`).
- Treat `_max_{N}` as invalid after migration (fail-fast with an actionable rename hint).
- Rationale: dual-resolution (`_max60` vs `_max_60`) makes reproducibility and path resolution ambiguous; the pipeline should be strict.

3) **Single-source preset resolution**

- Decision:
  - `public_data/run.sh` passes `--preset <base_preset>` and `--max-objects <N>` through to the pipeline factory (no pre-resolve step).
  - The pipeline planner is the only authority that computes `effective_preset`.
- Rationale: removes duplicated resolution logic and prevents runner/pipeline drift on naming + layout.

4) **Derived preset image sharing (no byte-copy)**

- Decision: when `max_objects` causes `effective_preset != base_preset`, the derived preset MUST NOT byte-copy resized images.
- Mechanism:
  - Keep JSONL `images[0]` stable as `images/...` (no `..` path components).
  - Materialize the derived preset’s `images/` as a real directory containing hardlinks to the base preset’s resized images.
  - If hardlink creation fails for any reason, the pipeline MUST fail fast (no byte-copy fallback).
  - Never use a directory symlink for `images/` (same safety policy as rescale presets).
- Rationale:
  - Filtering drops records; it does not modify images.
  - Copying full image trees is disk- and time-expensive and duplicates content.
  - Hardlinks eliminate data duplication while preserving a “normal” on-disk layout expected by training + tools.

Clarification (what "dedup" means operationally):
- Rescaling happens in the base preset only.
  - The base rescale stage is the only place that is allowed to create resized pixels (write new image bytes).
  - If a source image already matches the target `(width, height)`, the rescale implementation MAY copy it as-is (still a new file under the preset), but it is still conceptually part of the rescale stage.
- Derived presets do not rescale images.
  - They reuse the base preset's already-resized images by creating hardlinks for the same relative `images/...` paths.
  - This requires base + derived presets to live on the same filesystem (hardlinks cannot cross devices).
  - The pipeline MUST precheck this and fail fast with an actionable hint if the filesystem contract is violated.
  - If the base preset is missing an expected resized image, the derived preset MUST fail fast and instruct the user to repair/rebuild the base preset (no "rescale inside derived preset").

Hardlink materialization contract (derived presets):
- For every `images/...` path referenced by the derived preset JSONL, the derived preset MUST contain a file at that relative path.
- That file MUST be a hardlink to the base preset's corresponding resized image file at the same relative path.
- Materialization is append-only and idempotent (safe to rerun):
  - if the destination path does not exist: create the hardlink,
  - if the destination path exists and is already the same inode as the source: do nothing,
  - otherwise: fail fast (do not overwrite or "repair" in place).

5) **Unified validation boundary**

- Decision: consolidate validation logic into a single shared validator implementation and reuse it from:
  - pipeline `ValidationStage`
  - training preflight (`scripts/train.sh`, `scripts/train_stage2.sh`)
- Rationale: avoid policy drift; keep one authoritative contract implementation.

6) **Parallelism policy**

- Decision:
  - Heavy tasks (rescale/materialize images, per-record JSONL transforms) may use multiprocessing with a shared worker normalization helper.
  - Metadata scans (counting lines, collecting unique image specs) remain sequential.
- Rationale: multiprocessing overhead is only justified for CPU-heavy or IO-heavy operations with sufficient granularity.

Validation determinism:
- Spot-check sampling MUST be deterministic and MUST NOT reduce sensitivity by skipping failures.
- Definition:
  - "Structurally-parsed record" means the JSON line successfully parses and the record passes always-on structural checks (required keys present, numeric fields parseable, etc.).
  - It explicitly does NOT include any IO-based checks (file existence, image open, image size), which are handled by the separate image-check phase.
- Required control flow:
  - Structural parse + always-on structural checks happen first (JSON parse, required keys, numeric field parsing).
    - Any structural failure is a validator failure (do not "sample around" invalid JSON / invalid schema).
    - The validator MUST still scan the full JSONL to completion to report stable summary counts, but MUST return failure if any structural failures are observed.
  - When image spot-checking is enabled with `N`:
    - Select exactly the first `N` structurally-parsed records in JSONL line order (no RNG, no shuffling).
    - Run image checks on those records in that exact order.
    - Do not skip, replace, or resample any of those `N` records due to image-open/size mismatches; any such failure is a validator failure.

## Invariants (Immutability + Overwrite Semantics)

- Preset `images/` are immutable once written by the rescale stage.
- Pipeline tools MUST NOT overwrite existing image files under preset `images/`:
  - allowed: create-missing only,
  - disallowed: truncate/overwrite/replace existing paths in-place.
- Rescale parameter mismatch detection MUST be driven by persisted metadata (not inference):
  - `pipeline_manifest.json` MUST record the rescale-affecting parameters used to create the preset images, at least:
    - `max_pixels`, `min_pixels`, `image_factor`,
    - and any other knob that can affect the output `(width, height)` for a given source image.
    - Canonical location: `pipeline_manifest.json` MUST include these keys under `stage_stats.rescale` (it may also include non-param fields like `num_workers` or per-split counts).
  - The rescale stage MUST compare the requested rescale params against the recorded params in `pipeline_manifest.json` before doing any work.
    - If the recorded params are missing or differ, rescale MUST fail fast with an actionable rebuild hint.
    - If `pipeline_manifest.json` is missing (or does not contain `stage_stats.rescale`) but the preset directory already exists, treat that as "params missing" and fail fast (do not infer params by scanning files).
- If a preset directory already exists with prior outputs and the requested rescale parameters differ (max_pixels / min_pixels / image_factor), rescale MUST fail fast with an actionable rebuild hint:
  - rebuild is manual: the user either picks a new preset name or deliberately deletes the existing preset directory,
  - rebuild MUST NOT be an in-place “overwrite existing files” operation.

## Risks / Trade-offs

- **Hardlink semantics**:
  - Risk: tools that “repair” or overwrite image files could mutate the shared inode across presets.
  - Mitigation: treat resized preset images as immutable; do not write into preset `images/` outside the rescale stage; keep materialization tools append-only (create-missing only, never overwrite).
  - Risk: hardlinks may fail when base + derived presets are on different filesystems (cross-device link) or when the filesystem disallows hardlinks.
  - Mitigation: require base + derived presets under a single storage root; precheck `st_dev` compatibility and fail fast with a hint to move the output root.
- **Breaking naming changes**:
  - Risk: existing scripts/configs may reference legacy locations.
  - Mitigation: provide an explicit migration doc + fail-fast errors that include actionable rename hints; update all in-repo configs/tests/docs.

## Acceptance Checks (Executable)

- Base preset creation (no max filter):
  - `./public_data/run.sh coco rescale --preset rescale_32_768_bbox`
  - Expect: `public_data/coco/rescale_32_768_bbox/train.jsonl` and `public_data/coco/rescale_32_768_bbox/images/` (real directory).
- Derived preset creation (max filter applied after the fact):
  - `PUBLIC_DATA_MAX_OBJECTS=60 ./public_data/run.sh coco coord --preset rescale_32_768_bbox`
  - Expect: `public_data/coco/rescale_32_768_bbox_max60/images/` exists and contains hardlinks to base preset images.
- Training preflight validation uses the unified validator:
  - Run any stage-2 launcher that triggers JSONL preflight (for example `bash scripts/train.sh config=configs/stage2_ab/prod/desc_first_a_only.yaml`).
  - Expect: a single canonical validator enforces `max_pixels` + `multiple_of` constraints on `*.coord.jsonl`.
