## Context

`public_data` already provides a practical end-to-end flow (`download -> convert -> rescale -> coord -> validate`), but core responsibilities are split across:
- runner orchestration in `public_data/run.sh`,
- dataset-specific shell plugins in `public_data/datasets/*.sh`,
- multiple large converter/transform scripts in `public_data/scripts/*.py`,
- partial converter abstraction in `public_data/converters/*` (used primarily by LVIS).

This creates three technical problems:
- shared transform behavior is duplicated (path rewriting, geometry normalization, sorting, filtering),
- onboarding a new dataset requires touching multiple non-uniform entrypoints,
- dataset-specific logic can leak into orchestration (`run.sh`) instead of isolated adapter modules.

We must refactor to a unified, extensible pipeline without breaking current reproducible workflows. User decisions for this change are explicit:
- Keep `public_data/run.sh` and `public_data/datasets/*.sh` as stable external interfaces.
- Keep max-object filtering OFF by default; when enabled, auto-append suffix `max_{N}` to output naming.

## Goals / Non-Goals

**Goals:**
- Introduce a unified shared pipeline/factory abstraction for dataset preparation internals.
- Add pluggable dataset adapters for `lvis`, `coco`, `vg` with a clean registry extension path.
- Centralize shared transforms (rescale, optional max-object filtering, coord-token conversion, formatting/writing).
- Preserve JSONL contract and Qwen3-VL coord-token compatibility.
- Preserve external CLI contract and dataset plugin shell contract during migration.
- Keep geometry invariants stable (no dropped/reordered coordinates beyond existing deterministic canonicalization rules).

**Non-Goals:**
- No redesign of training/eval runtime in `src/`.
- No new public CLI flag surface in `public_data/run.sh` for this refactor phase.
- No changes to upstream HF model internals.
- No change to default data semantics beyond architecture cleanup and explicitly configured optional filtering.

## Decisions

### Decision 1: External interface remains stable; internal execution is unified

Decision:
- Keep `public_data/run.sh` command grammar and `public_data/datasets/*.sh` plugin contract intact.
- Move internal execution to a shared Python pipeline/factory implementation that those entrypoints call.
- Define boundary explicitly:
  - `run.sh` and `datasets/*.sh` are compatibility wrappers/integration surfaces,
  - "core orchestration" means the internal Python pipeline planner + stage executor.

Rationale:
- Minimizes migration risk for existing scripts, docs, and cluster jobs.
- Allows phased rollout with behavior parity checks.

Alternatives considered:
- Full replacement of shell entrypoints now: rejected due to immediate compatibility risk.
- Keep architecture as-is: rejected due to growing duplication and extensibility cost.

### Decision 2: Registry-based adapter factory for source ingestion

Decision:
- Create a dataset adapter registry (`lvis`, `coco`, `vg` initially) and factory instantiation by dataset id.
- Adapter contract owns source-specific concerns only:
  - download hooks,
  - raw annotation parsing,
  - source normalization into canonical intermediate records.

Rationale:
- Encapsulates dataset variance behind explicit interfaces.
- New dataset onboarding becomes additive (new adapter + registry entry) without core orchestration edits.

Alternatives considered:
- Continue one-script-per-dataset conversion: rejected for maintainability.
- Hard-coded `if dataset == ...` factory branches in core pipeline: rejected (violates extensibility).

### Decision 3: Shared transform stages with composable pipeline plan

Decision:
- Define reusable stage interfaces for post-ingestion transforms:
  - image path/extraction normalization,
  - smart resize (pixel budget),
  - optional max-object filter,
  - norm1000 numeric normalization,
  - coord-token expansion,
  - output formatting/writing,
  - validation hook stage.
- Each dataset pipeline is a stage plan composed from shared stages plus adapter.

Normative stage order:
1. Adapter emits canonical intermediate records in pixel coordinates.
2. Image/path normalization and smart resize are applied.
3. Optional max-object filter runs.
4. Numeric normalization emits norm1000 integer JSONL.
5. Coord-token expansion emits coord-token JSONL.
6. Validation hook stage can run contract checks in-plan.

Canonical intermediate record contract at adapter->stage boundary:
- same top-level/object keys as `docs/data/JSONL_CONTRACT.md`,
- coordinates are still pixel-space at this boundary,
- one geometry key per object (`bbox_2d` or `poly`),
- `images` may be absolute in-memory; persisted outputs must be relative.

Output artifact naming and coordinate-space mapping per split:
- `<split>.raw.jsonl`: pixel-space records after resize/filter.
- `<split>.norm.jsonl`: norm1000 integer coordinate records.
- `<split>.coord.jsonl`: coord-token records.

Legacy runner compatibility mapping:
- legacy rescale-facing `<split>.jsonl` (pixel-space) corresponds to canonical `<split>.raw.jsonl`,
- parity gates must compare legacy pixel artifacts against canonical raw artifacts,
- normalized numeric remains explicit as `<split>.norm.jsonl`.

Rationale:
- Eliminates duplicated transformation logic.
- Creates deterministic, testable stage boundaries and easier profiling.

Alternatives considered:
- Keep transform logic embedded in each converter: rejected due to drift risk.
- Single monolithic orchestrator function: rejected due to poor extensibility/testing granularity.

### Decision 4: Max-object filtering policy and suffixing

Decision:
- `max_objects` stage is OFF by default.
- When enabled, effective preset/output naming auto-appends suffix token `max_{N}` (rendered as `_max_<N>` in paths, e.g., `rescale_32_768_bbox_max_60`).
- If the same suffix already exists, it is not duplicated.
- Legacy naming compatibility is required:
  - resolver recognizes existing `max{N}` directories/files as equivalent to `max_{N}`,
  - when an equivalent legacy artifact directory exists, implementation reuses it and must not create a parallel fork solely due to token style.
  - when no equivalent legacy directory exists, fresh runs emit canonical `_max_<N>` naming.

Filter semantics:
- max-object stage drops full records where `len(objects) > N`,
- no object truncation is allowed.

Rationale:
- Preserves backward-compatible defaults.
- Makes filtered artifacts self-describing and reproducible by path name alone.

Alternatives considered:
- Enable max-object filtering globally by default: rejected by user decision and compatibility needs.
- Enable filter without naming changes: rejected due to reproducibility ambiguity.

### Decision 5: Migration by parity gates

Decision:
- Migrate dataset adapters incrementally (COCO -> LVIS -> VG), with parity checks against current outputs.
- Keep old script entrypoints callable during transition and switch runner internals only after parity passes.

Rationale:
- Reduces chance of silent data regressions in large-scale dataset preparation.

Alternatives considered:
- Big-bang migration across all datasets at once: rejected due to verification complexity.

## Risks / Trade-offs

- [Risk] Hidden behavior differences between legacy scripts and staged transforms can alter object counts/order.  
  → Mitigation: per-dataset parity tests on deterministic small slices and full validator checks before switching defaults.

- [Risk] Keeping shell + new pipeline temporarily increases code surface during migration.  
  → Mitigation: establish clear ownership boundaries and remove duplicate internal paths once parity gates pass.

- [Risk] Suffix naming collisions or inconsistent naming policy across commands.  
  → Mitigation: single shared naming utility for preset resolution and suffix normalization.

- [Risk] Large JSONL/image transforms may regress performance.  
  → Mitigation: preserve multiprocessing paths, add stage-level timing/stats manifests, and benchmark pre/post on representative splits.

- [Risk] Deterministic ordering/canonicalization drift can cause silent parity mismatches.  
  → Mitigation: preserve canonical polygon ordering and object sort behavior through shared helpers, and test parity on ordering-sensitive fixtures.

## Migration Plan

1. Implement pipeline core abstractions (registry, adapter base, stage interface, formatter/writer utilities) without changing runner behavior.
2. Introduce COCO adapter + stage plan under new pipeline; run parity tests versus current converter flow.
3. Introduce LVIS adapter migration, reusing existing converter abstractions where possible; run parity tests.
4. Introduce VG adapter migration; run parity tests.
5. Integrate runner internals (`run.sh`) to invoke unified pipeline while preserving existing CLI and dataset shell plugin contract.
6. Enable optional max-object stage wiring (default off) and suffix policy `max_{N}` naming in preset resolution.
7. Remove deprecated duplicate internal logic after parity/stability gates pass.

Parity gate criteria (minimum):
- identical split-level record counts and dropped-record counts,
- identical max-object filtering behavior (drop semantics),
- deterministic object ordering/canonicalization parity on sampled records,
- validator pass on all generated artifacts (`raw`, normalized numeric, coord-token).

Rollback strategy:
- Keep legacy script paths callable during migration window.
- If parity/regression fails for a dataset, fall back that dataset to legacy internal path while keeping others on unified pipeline.

## Open Questions

- None blocking for this spec phase.  
  (Implementation can still decide exact Python module layout under `public_data/` while preserving the interfaces and behavior above.)
