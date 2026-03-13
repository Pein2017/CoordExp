## Context

The current repo already exposes the main visualization families we care about:

- offline single-run overlays:
  - `scripts/run_vis.sh`
  - `vis_tools/vis_coordexp.py`
  - `src/infer/vis.py`
- evaluator overlays and selected-scene rendering:
  - `scripts/evaluate_detection.py`
  - `src/eval/detection.py`
- online monitor-dump visualization:
  - `vis_tools/vis_monitor_dump_gt_vs_pred.py`
  - `src/trainers/stage2_rollout_aligned.py`
  - `src/trainers/stage2_two_channel.py`
- pairwise comparison:
  - `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py`
- run-to-run audit and repeated sampling:
  - `src/eval/oracle_k.py`
  - `scripts/analysis/compare_detection_runs.py`
  - `scripts/analysis/report_rollout_stability.py`

The problem is not missing tools.
The problem is missing interoperability.

Today, different producers still encode scene information differently:

- offline artifacts use `gt` / `pred` with `type` / `points` / `desc`,
- monitor dumps often carry `bbox_2d`, `points_norm1000`, `geom_type`,
  `match`, and `stats`,
- evaluator-selected scenes have their own match/export structure,
- comparison tools align scenes with partial identity checks,
- and some renderers still do scaling logic locally.

That means we still pay redundant costs in:

- per-source object decoding,
- per-source matching translation,
- per-source scene identity assumptions,
- and per-renderer scaling logic.

The public ownership boundary for this change is intentionally narrow:

- visualization consumes `gt_vs_pred.jsonl`,
- visualization writes to an explicit output path,
- visualization fails fast on contract violations,
- visualization does not own or rewrite `monitor_dumps` paths.

## Goals / Non-Goals

**Goals**

- Keep one canonical single-view scene contract based on the existing
  `gt_vs_pred.jsonl` worldview.
- Make the object-level schema explicit enough that producers are truly
  interoperable.
- Normalize all visualization scenes to pixel-space bbox review objects.
- Preserve prediction order end-to-end.
- Reuse shared geometry / inverse-scaling helpers rather than renderer-local
  coordinate logic.
- Make canonical matching mandatory for shared GT-vs-Pred review scenes.
- Support:
  - offline infer/eval scenes,
  - evaluator-selected error scenes,
  - online workflows only when they provide a compatible `gt_vs_pred.jsonl`
    input,
  - backend-compare scenes,
  - Oracle-K repeated-sampling scenes.

**Non-Goals**

- No replacement of raw `monitor_dumps/*.json` as the primary training telemetry
  artifact.
- No change to `monitor_dumps` path layout.
- No second parallel base artifact family for the same single-run scene.
- No fully generic arbitrary-panel dashboard spec; the default reusable renderer
  stays a simple `1x2` GT-vs-Pred review panel.
- No promise that canonical visualization records preserve every source-native
  geometry form in top-level review objects. The canonical review contract is
  bbox-only.

## Scene Taxonomy

### 1) Single-run GT-vs-Pred scene

Examples:

- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`

Use:

- inspect one run qualitatively,
- trace ordered predictions,
- review GT, predicted objects, and optional matching/stats.

### 2) Evaluator-selected error scene

Examples:

- evaluator overlays,
- `matches.jsonl`,
- `per_image.json`

Use:

- review `FP` / `FN` heavy scenes,
- preserve threshold/scope semantics from evaluation,
- avoid custom renderer-only match logic.

### 3) Online monitor-dump scene

Examples:

- eval monitor dumps,
- train monitor dumps,
- rollout-aligned and Stage-2 two-channel dump samples.

Use:

- inspect live training/eval rollout failures,
- keep rollout/debug text and diagnostics,
- reuse already computed matching.

### 4) Pairwise comparison scene

Examples:

- HF vs vLLM,
- baseline vs checkpoint,
- baseline vs one Oracle run.

Use:

- align multiple canonical single-view records,
- preserve each member’s ordered predictions,
- compare scenes only after GT equivalence is verified.

### 5) Repeated-sampling / Oracle-K audit scene

Examples:

- baseline plus repeated stochastic runs,
- recoverable/systematic FN audit bundles.

Use:

- analyze recoverability on the same GT scene,
- compose baseline/oracle members by verified alignment,
- reuse the same single-view resource contract.

## Decisions

### 1) Canonical visualization remains gt-vs-pred compatible at the top level

We should not create a second base artifact family when the repo already centers
single-run artifacts on `gt_vs_pred.jsonl`.

So the canonical visualization resource keeps the familiar top-level structure:

- `schema_version`
- `source_kind`
- `record_idx`
- `image`
- `width`
- `height`
- `coord_mode`
- `gt`
- `pred`

Optional identity/provenance helpers:

- `image_id`
- `images`
- `file_name`
- `provenance`
- `stats`
- `debug`

Public visualizer interface:

- input path to canonical or canonicalized `gt_vs_pred.jsonl`,
- explicit output path,
- no implicit source-specific discovery.

### 2) Canonical visualization objects are bbox-only and pixel-space

The audit was right that the earlier draft left too much object-shape ambiguity.
The visualization contract needs one explicit object schema.

Canonical object schema for every item in both `gt` and `pred`:

- `index`: stable integer index within the canonical record,
- `desc`: string,
- `bbox_2d`: `[x1, y1, x2, y2]` as absolute pixel integers.

Canonical visualization records use:

- `coord_mode: "pixel"`

This is intentionally narrower than the full inference/eval artifact contract.
Visualization is a review layer, not the lossless geometry interchange layer.

Implications:

- offline `type` / `points` objects normalize into canonical bbox-only objects,
- monitor `bbox_2d` / `points_norm1000` / `geom_type` objects normalize into the
  same schema,
- polygon or other source geometry may remain in `debug` / `provenance`, but the
  renderer never depends on per-source geometry variants.

### 3) GT order is canonicalized; prediction order is preserved

The audit was also right that GT ordering matters now that:

- `matching.gt_index_domain` is canonicalized,
- evaluator and monitor sources carry source-local GT indices,
- comparison requires GT-scene equivalence checks.

The contract should therefore distinguish GT and Pred ordering rules:

- canonical `gt` order is deterministic and source-independent,
- canonical `pred` order preserves source-relative order.

Canonical GT ordering rule:

- normalize each GT object to canonical pixel-space bbox-only form,
- sort canonical GT objects lexicographically by:
  - `bbox_2d[0]`, `bbox_2d[1]`, `bbox_2d[2]`, `bbox_2d[3]`,
  - then `desc`,
- assign canonical `gt[*].index` from that sorted order.

Implications:

- the same GT scene yields the same canonical GT array across sources,
- adapters must remap source-local GT indices into `canonical_gt_index`,
- comparison can safely reason over canonical GT equality,
- prediction order still remains a faithful trace of model interpretation order.

### 4) Norm1000 and coord-token sources must inverse-scale before rendering

Canonical visualization records must never rely on renderer-local `norm1000`
handling.

Normalization rules:

- if a source carries `norm1000`-style numeric coordinates in the `0..999`
  range, the adapter inverse-scales them to pixel space using per-record
  `width` / `height`,
- if a source carries coord tokens in the same `0..999` domain, the adapter
  resolves them through the shared geometry helpers before deriving `bbox_2d`,
- if a source already carries pixel-space geometry, the adapter only clamps /
  normalizes as needed,
- renderers consume pixel-space `bbox_2d` only.

This keeps scaling logic in one reusable layer and removes one major source of
duplication and drift.

### 5) Matching must normalize into one explicit sub-schema

The earlier `matching` namespace was too loose.
Canonical visualization records need one explicit matching schema so colors,
labels, and scene summaries mean the same thing across sources.

Required fields when `matching` is present:

- `match_source`
- `match_policy`
- `pred_index_domain = canonical_pred_index`
- `gt_index_domain = canonical_gt_index`
- `matched_pairs`
- `fn_gt_indices`
- `fp_pred_indices`

Optional fields:

- `iou_thr`
- `pred_scope`
- `ignored_pred_indices`
- `unmatched_pred_indices`
- `unmatched_gt_indices`
- `gating_rejections`

Adapters must normalize source-specific matching into canonical index domains.

For shared review rendering, canonical matching is mandatory:

- if a source already carries matching, the adapter reuses and normalizes it,
- if a source does not carry matching, canonical matching must be materialized in
  an explicit preprocessing step before rendering,
- renderers never perform ad hoc fallback matching,
- attempts to render a canonical review scene without canonical matching fail
  fast.

### 6) Prediction order is a first-class invariant

The user explicitly wants to trace the order in which the model interpreted the
image.

So:

- the `pred` array preserves the original relative order of surviving predicted
  objects,
- adapters do not sort by score, area, label, IoU, or FP/FN status,
- canonical `pred[*].index` stays stable after normalization,
- if invalid objects are dropped, the survivors keep their original relative
  order and dropped-object details can live under `debug`.

### 7) Comparison composition requires verification, not just join keys

Stable join keys are useful for candidate alignment, but they are not enough to
prove two records describe the same GT scene.

So the comparison contract is:

- candidate alignment may begin with:
  - `record_idx`,
  - plus `image_id`, `file_name`, and/or `image` when available,
- after candidate alignment, the compositor must verify:
  - exact `width`,
  - exact `height`,
  - exact equality of the canonical normalized `gt` arrays,
- comparison composition fails fast if those checks disagree.

This intentionally matches the stronger safety instinct already present in the
Oracle-K flow.

### 8) Materialized visualization resources are derived sidecars

The follow-up audit correctly pointed out that reusing the bare filename
`gt_vs_pred.jsonl` for both raw inference artifacts and normalized visualization
resources is too easy to misread.

So the contract should distinguish:

- raw inference artifact:
  - `<run_dir>/gt_vs_pred.jsonl`
- materialized canonical visualization resource:
  - default `<run_dir>/vis_resources/gt_vs_pred.jsonl`

Workflows may materialize canonical visualization resources elsewhere, but they
must not overwrite or path-alias the raw inference artifact when the object
schema differs.

### 9) Raw monitor dumps remain telemetry, but normalization must be lossless enough

We do not want to replace current monitor dumps with a new primary artifact
family.
We only need a clean normalization boundary for upstream workflows that want to
feed the visualizer.

That means:

- raw dump files stay source-native,
- upstream normalization may derive canonical visualization records from dump
  samples,
- precomputed matching is reused rather than recomputed when that upstream step
  exists,
- rollout/debug/triage payloads stay under `debug` / `provenance`,
- `monitor_dumps` path layout remains unchanged,
- the visualizer itself only consumes `gt_vs_pred.jsonl` and an explicit output
  path.

### 10) The shared default renderer stays simple and error-focused

The current monitor visualizer already has the right review semantics.
That should become the shared default:

- fixed `1x2` layout,
- GT left, Pred right,
- GT green,
- FN orange,
- matched Pred green,
- FP Pred red,
- `desc` labels focused on error objects by default,
- deterministic overlap-avoidance for label placement.

## Example Canonical Record

```json
{
  "schema_version": 1,
  "source_kind": "monitor_dump_eval",
  "record_idx": 12,
  "image_id": 391895,
  "image": "val2017/000000391895.jpg",
  "width": 1024,
  "height": 1024,
  "coord_mode": "pixel",
  "gt": [
    {
      "index": 0,
      "desc": "person",
      "bbox_2d": [124, 208, 312, 521]
    }
  ],
  "pred": [
    {
      "index": 0,
      "desc": "person",
      "bbox_2d": [126, 210, 309, 520]
    },
    {
      "index": 1,
      "desc": "kite",
      "bbox_2d": [710, 120, 830, 260]
    }
  ],
  "matching": {
    "match_source": "precomputed",
    "match_policy": "hungarian_gated",
    "pred_index_domain": "canonical_pred_index",
    "gt_index_domain": "canonical_gt_index",
    "matched_pairs": [[0, 0]],
    "fn_gt_indices": [],
    "fp_pred_indices": [1]
  },
  "stats": {
    "precision": 0.5,
    "recall": 1.0,
    "f1": 0.667
  },
  "provenance": {
    "run_label": "baseline",
    "global_step": 900
  },
  "debug": {
    "monitor_kind": "eval_monitor_dump"
  }
}
```

## Reuse Strategy

The least-redundant implementation shape is:

```text
source artifacts
==============
gt_vs_pred.jsonl
gt_vs_pred_scored.jsonl
monitor_dumps/*.json
matches.jsonl / per_image.json
oracle-k aligned runs

        │
        │ shared adapters
        ▼

canonical visualization resources
=================================
pixel-space bbox-only objects
normalized matching sub-schema
canonical GT ordering
ordered predictions

        │
        ├── shared 1x2 GT-vs-Pred renderer
        ├── comparison composers
        └── scene selectors / audit bundles
```

The key reuse rules are:

- normalize through shared geometry helpers,
- inverse-scale `norm1000` and coord-token sources in the adapter layer,
- materialize canonical matching before shared review rendering,
- keep one review renderer semantics,
- let legacy `vis_tools/` collapse toward thin wrappers over the shared stack.
