## 1. OpenSpec and Contract Foundation

- [x] 1.1 Add delta spec files for:
  - `gt-vs-pred-visualization`
  - `inference-engine`
  - `detection-evaluator`
- [x] 1.2 Validate the change artifacts with:
  - `openspec validate --type change unify-gt-vs-pred-visualization --strict --no-interactive`

## 2. Shared Canonical Resource Layer

- [x] 2.1 Define a shared canonical visualization-resource schema that remains
  `gt_vs_pred` compatible at the top level rather than introducing a second base
  single-view artifact family.
- [x] 2.2 Keep the canonical top-level contract tight:
  - `schema_version`
  - `source_kind`
  - `record_idx`
  - `image`
  - `width`
  - `height`
  - `coord_mode`
  - `gt`
  - `pred`
- [x] 2.3 Define one canonical visualization object schema:
  - `index`
  - `desc`
  - pixel-space `bbox_2d`
- [x] 2.4 Normalize all supported source object forms into that schema through
  shared geometry helpers:
  - offline `type` / `points`
  - scored artifacts
  - monitor `bbox_2d`
  - `points_norm1000`
  - coord-token / `norm1000` payloads in the `0..999` range
- [x] 2.5 Define one canonical matching sub-schema with:
  - `match_source`
  - `match_policy`
  - canonical index domains
  - `matched_pairs`
  - `fn_gt_indices`
  - `fp_pred_indices`
- [x] 2.6 Define canonical GT ordering and source-index remapping rules so
  `canonical_gt_index` is stable across producers.
- [x] 2.7 Preserve ordered predictions as a contract invariant and thread stable
  object indices through normalized records.
- [x] 2.8 Keep the visualizer interface strict and small:
  - input `gt_vs_pred.jsonl`
  - explicit output path
  - no source-specific path discovery
  - fail-fast on contract violations

## 3. Shared Renderer Semantics

- [x] 3.1 Standardize one shared default GT-vs-Pred renderer with:
  - `1x2` layout
  - GT on the left
  - Pred on the right
  - GT = green
  - FN GT = orange
  - matched Pred = green
  - FP Pred = red
- [x] 3.2 Make `desc` labels error-focused by default:
  - prioritize `FN` / `FP`
  - avoid blanket labeling of matched objects unless requested
- [x] 3.3 Add a deterministic label-placement policy that prefers:
  - outside-box placement
  - staggered conflict resolution
  - truncation before overlap
- [x] 3.4 Reuse existing shared geometry and image-path resolution helpers rather
  than embedding new parallel implementations in each visualization tool.

## 4. Source Adapters and Scene Families

- [x] 4.1 Add / refactor a shared adapter for offline single-run resources:
  - `gt_vs_pred.jsonl`
  - `gt_vs_pred_scored.jsonl`
  - materialize canonical matching as part of the sidecar generation step
- [x] 4.2 Add / refactor a shared adapter for evaluator-selected scenes that can
  carry canonical matching from:
  - `matches.jsonl`
  - `per_image.json`
- [x] 4.3 Keep monitor-dump path layout unchanged; any upstream normalization
  into compatible `gt_vs_pred.jsonl` inputs is outside the visualizer’s path
  ownership.
- [x] 4.5 Support composition of pairwise / multi-run scenes by aligning multiple
  canonical resources on candidate join keys:
  - `record_idx`
  - `image_id` when available
  - `file_name` / `image` for auditability
- [x] 4.5a Fail fast when aligned members disagree on:
  - `width`
  - `height`
  - canonical normalized GT content
- [x] 4.5b Materialize derived canonical visualization resources at a distinct
  sidecar path by default, rather than overloading the raw inference
  `gt_vs_pred.jsonl` path.
- [x] 4.6 Keep backend-compare and Oracle-K workflows as compositions of the same
  canonical single-view resource rather than adding compare-only per-object
  schemas.

## 5. Legacy Tool Convergence

- [x] 5.1 Converge `scripts/run_vis.sh`, `vis_tools/vis_coordexp.py`, and the
  evaluator overlay path on the shared canonical renderer semantics.
- [x] 5.2 Keep `vis_tools/vis_monitor_dump_gt_vs_pred.py` as the semantic
  reference for error-focused review behavior during migration, but remove
  duplicated contract logic once shared adapters exist.
- [x] 5.3 Avoid introducing new parallel visualization entry points unless an
  existing workflow boundary requires a thin wrapper.

## 6. Tests and Docs

- [x] 6.1 Add fixtures covering:
  - offline `gt_vs_pred.jsonl`
  - scored artifact input
  - eval-selected error scene
  - duplicate-GT annotation scene with identical `bbox_2d` + `desc`
  - Oracle-K aligned baseline/oracle scene
- [x] 6.2 Add tests for:
  - canonical GT ordering and source-index remapping,
  - preserved prediction order,
  - canonical object normalization into pixel-space `bbox_2d`,
  - `norm1000` / coord-token inverse scaling through shared helpers,
  - canonical matching normalization and index-domain checks,
  - GT-equivalence validation for multi-run composition,
  - fail-fast when canonical matching is missing for shared review rendering,
  - label-placement determinism under crowded scenes.
- [x] 6.3 Update docs after implementation:
  - `docs/eval/WORKFLOW.md`
  - `docs/eval/CONTRACT.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/ARTIFACTS.md`
  - `progress/diagnostics/visualization_tools_index_2026-03-11.md` or successor guidance.
