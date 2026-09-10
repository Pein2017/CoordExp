---
doc_id: docs.history.evaluation.legacy-eval-reference-20260909
layer: docs
doc_type: historical-reference
status: historical-reference
domain: eval
summary: Preserved legacy MS-Swift/mainline evaluation commands and artifact conventions.
updated: 2026-09-09
---

# Legacy evaluation reference (2026-09-09 extraction)

Extracted from the working-tree `docs/eval/WORKFLOW.md` and
`docs/eval/CONTRACT.md` on 2026-09-09, based on checkout commit `390d8fd48`.
Their headers were dated 2026-07-11. Only legacy-specific sections are retained
here; current Swift commands and outputs remain in the
[current workflow](../../eval/WORKFLOW.md) and
[current artifact reference](../../eval/CONTRACT.md).

The headings and prescriptions below describe historical MS-Swift/mainline
surfaces. Words such as “canonical”, “stable”, and “default” are scoped to that
historical implementation. They do not establish current support, approve a
launch, or describe the current `scripts/evaluate_detection.py` CLI. Resolve
source/config availability and the original run revision before reproduction.
Official test-dev history remains in
[its existing runbook](../../eval/COCO_TEST_SUBMISSION.md).

## Historical YAML commands

The commands in this section are retained for historical reproduction only.
They are not the canonical `main` inference/evaluation route. Use
`python -m src.infer --config configs/coordexp_swift/infer/<config>.yaml`
above for current Swift work.

Run inference:

```bash
PYTHONPATH=. python scripts/run_infer.py \
  --config configs/infer/pipeline.yaml
```

Run confidence post-op:

```bash
PYTHONPATH=. python scripts/postop_confidence.py \
  --config configs/postop/confidence.yaml
```

Non-canonical bbox note:

- do not run confidence post-op for `infer.bbox_format: cxcy_logw_logh` or
  `infer.bbox_format: cxcywh`
- legacy/mainline pipelines may materialize `gt_vs_pred_scored.jsonl` directly
  from canonical standardized predictions with deterministic constant-score
  provenance when COCO/LVIS metrics are requested; the direct CoordExp-Swift
  evaluator in this worktree does not consume that constant-score family in V1
- only use this infer path with checkpoints that were actually trained against
  the matching non-canonical serialization contract
- forcing a legacy `xyxy`-trained checkpoint through
  `infer.bbox_format: cxcy_logw_logh` or `infer.bbox_format: cxcywh` can still
  produce canonicalized eval
  artifacts, but those rollouts are not semantically valid evidence for the new
  parameterization

Raw-text xyxy norm1000 benchmark note:

- use canonical `*.norm.jsonl` data surfaces
- set `infer.mode: text`
- set `infer.pred_coord_mode: norm1000`
- keep `infer.bbox_format: xyxy`
- do not rely on `auto` heuristics for this benchmark path
- confidence post-op remains the scored-eval path here, but it aligns numeric
  bbox spans from the raw norm1000 JSON output instead of coord-token spans
- evaluation/visualization then denormalize through per-record `width` /
  `height` into canonical pixel-space `xyxy`

## Historical wrapper roles

Stable / reportable:

- `scripts/run_infer.py --config ...`
- `scripts/postop_confidence.py --config ...`
- `scripts/evaluate_proxy_detection_bundle.py --config ...`

Compatibility / debug:

- `scripts/run_infer_eval.sh` is a legacy environment-variable wrapper for
  quick inference plus debug evaluation. It defaults away from official-style
  metrics and refuses COCO/LVIS/both metrics entirely because it cannot prove
  scored-artifact provenance for the just-run inference output. Use the
  YAML-first infer -> score -> eval flow for reportable COCO/LVIS/both metrics.
- `scripts/run_vis.sh` is a manual/debug visualization wrapper for an explicit
  artifact and image root. Prefer evaluator overlays or `vis_resources/`
  artifacts tied to resolved config/scoring provenance for reportable evidence.

Historical diagnostics:

- `scripts/pipelines/run_rollout_stability_probe.sh` is a parser/rollout health
  diagnostic that delegates to the legacy/debug `run_infer_eval.sh` wrapper.
  Treat its outputs as local stability evidence, not benchmark claims.

Duplicate-control guard note:

- the direct Swift evaluator is aggregate-only in V1 and does not expose a
  duplicate-control guard, F1-ish matcher, LVIS reducer, overlays, or YAML
  override surface;
- keep those as separate evaluator families until their contracts are rebuilt
  for the Swift artifact shape.

Run Oracle-K analysis:

```bash
PYTHONPATH=. python scripts/evaluate_oracle_k.py \
  --config configs/eval/oracle_k.yaml
```

## Historical artifact expectations

After inference:

- `gt_vs_pred.jsonl`
- `summary.json`
- `resolved_config.json` when using the YAML pipeline
- `resolved_config.path` next to `gt_vs_pred.jsonl` when the YAML pipeline is
  responsible for artifact materialization
- verify `infer.prompt_variant`, `infer.object_field_order`, and
  `infer.object_ordering` in both `summary.json` and `resolved_config.json`
  when comparing prompt/order ablations
- use `resolved_config.path` when a downstream eval or visualization job is
  consuming `gt_vs_pred.jsonl` from outside the original run directory
- if the checkpoint was trained with non-default dense prompt controls, keep
  those infer-time values aligned with training so evaluation does not measure
  prompt drift instead of model behavior
- training-only bbox regression parameterizations such as
  `bbox_geo.parameterization: center_size` do not change this infer/eval
  artifact split: base predictions still write `gt_vs_pred.jsonl`, scored
  predictions still write `gt_vs_pred_scored.jsonl`, and downstream jobs should
  continue using `resolved_config.path` to recover the authoritative
  `resolved_config.json`

After confidence post-op:

- `pred_confidence.jsonl`
- `gt_vs_pred_scored.jsonl`
- `gt_vs_pred_scored.jsonl.provenance.json`
- `confidence_postop_summary.json`

After legacy/mainline non-canonical official-eval compatibility scoring:

- `gt_vs_pred_scored.jsonl`
- `gt_vs_pred_scored.jsonl.provenance.json`
- no `pred_confidence.jsonl`
- no `confidence_postop_summary.json`

Plain `xyxy` infer-only/debug runs do not materialize scored artifacts unless
`confidence:` is configured or official COCO/LVIS/both evaluation is requested.
This keeps raw inference artifacts unscored by default.

Legacy/mainline evaluator families may additionally emit:

- `metrics.json`
- `metrics_guarded.json` when `duplicate_control.enabled: true`
- `per_image.json`
- `per_image_guarded.json` when `duplicate_control.enabled: true`
- `duplicate_guard_report.json` when `duplicate_control.enabled: true`
- `matches_guarded.jsonl` and `matches@<thr>_guarded.jsonl` when match exports
  are enabled under duplicate control
- optional `per_class.csv`, `matches.jsonl`, and overlays
- `vis_resources/gt_vs_pred.jsonl` when the shared GT-vs-Pred reviewer is
  materialized for `scripts/run_vis.sh` or evaluator overlays

Guarded-artifact rule:

- non-COCO raw evaluation continues to consume `gt_vs_pred.jsonl`
- score-aware COCO evaluation continues to consume `gt_vs_pred_scored.jsonl`
- for `cxcy_logw_logh` and `cxcywh`, that scored artifact is constant-score
  compatibility output rather than confidence output
- guarded companions follow the same input family:
  - `gt_vs_pred_guarded.jsonl`
  - `gt_vs_pred_scored_guarded.jsonl`
- treat the raw artifact as the main research/debug surface and the guarded
  artifact as the safety/post-op surface

## COCO + LVIS Proxy Evaluation

For COCO runs trained with LVIS proxy supervision, keep the benchmark headline
explicit:

- report standard COCO metrics on the original COCO GT objects only
- treat LVIS proxy-expanded GT as additive analysis, not as the replacement
  headline benchmark

Recommended flow:

```text
gt_vs_pred_scored.jsonl
  -> materialize proxy GT views
  -> coco_real / coco_real_strict / coco_real_strict_plausible JSONLs
  -> run the standard evaluator on each view separately
```

Use `scripts/materialize_proxy_eval_views.py` on the scored artifact to create:

- `coco_real`
  - original COCO GT only (`proxy_tier = real`)
- `coco_real_strict`
  - COCO GT plus strict LVIS proxies (`same_extent_proxy`)
- `coco_real_strict_plausible`
  - COCO GT plus strict and plausible LVIS proxies

For a one-inference / one-scored-artifact workflow, use
`scripts/evaluate_proxy_detection_bundle.py` to:

- reuse one `gt_vs_pred_scored.jsonl`
- materialize the proxy GT views under the same run directory
- run the standard evaluator once per view
- write side-by-side outputs such as:
  - `eval_coco_real/`
  - `eval_coco_real_strict/`
  - `eval_coco_real_strict_plausible/`
  - `proxy_eval_bundle_summary.json`

Interpretation guidance:

- `coco_real` is the benchmark-aligned number to compare against standard COCO
  baselines
- `coco_real_strict` estimates recoverable misses where LVIS adds same-extent
  annotations
- `coco_real_strict_plausible` is the broadest supervision view and is useful
  for recall analysis, but it is the least comparable to standard COCO

After Oracle-K analysis:

- `summary.json`
- `per_image.json`
- `fn_objects.jsonl`
- optional materialized inference run folders when Oracle-K is asked to generate repeated samples

## Oracle-K Repeated-Sampling Workflow

Use Oracle-K when you want to measure whether baseline false negatives are recoverable under repeated stochastic decoding, and how often they are recovered across `K` rollouts.

```text
baseline artifact or run spec
  + one or more Oracle artifact or run specs
  -> Oracle-K analysis
  -> summary.json / per_image.json / fn_objects.jsonl
```

The Oracle-K workflow is additive:

- standard `scripts/evaluate_detection.py` behavior does not change
- Oracle-K reuses the same F1-ish matching semantics for IoU thresholds, semantic matching, and prediction scope
- cross-run alignment is validated in record order and requires consistent `file_name` provenance; `record_idx` + `gt_idx` remains the normative object key and `image_id` / `file_name` are preserved for downstream visualization analysis

## Historical shared-review and Oracle-K notes

The default repo visualization path now goes through the canonical sidecar:

```text
gt_vs_pred.jsonl
  -> vis_resources/gt_vs_pred.jsonl
  -> shared 1x2 GT-vs-Pred reviewer
  -> vis_*.png
```

Key points:

- Evaluator overlays and `vis_resources/gt_vs_pred.jsonl` are the stable
  provenance-preserving visualization path.
- `scripts/run_vis.sh` and `vis_tools/vis_coordexp.py` are manual/debug helpers
  for explicitly supplied artifacts and image roots; they do not by themselves
  recover YAML `resolved_config.path`, scoring semantics, or benchmark scope.
- evaluator overlays reuse the same shared reviewer semantics instead of a
  second renderer-local box contract.
- post-eval audit materialization may reuse `matches.jsonl` and `per_image.json`
  to preserve canonical matching and join keys.

The YAML config can work in two modes:

- consume pre-generated `gt_vs_pred.jsonl` artifacts for one baseline run plus one or more Oracle runs
- materialize repeated inference runs through the standard infer pipeline before aggregation

Oracle-K v1 is intentionally object-level:

- it records per-run pairing for each baseline-FN GT object
- it preserves trace provenance when available through `pred_token_trace.jsonl`
- it does not require exact token-span-to-object alignment for matched predictions

Recommended validation checks for Oracle-K:

- all runs use the same subset and agree on GT content in the same record order
- the chosen IoU thresholds match the intended F1-ish study
- the output includes both `ever recovered` and `recover_count` / `recover_fraction`
- location-only and semantic+location recovery are inspected separately

## Historical pipeline row shape

- Each record is a JSON object.
- Inline GT is required for evaluation-time consumption; evaluator-facing
  workflows do not use a separate GT file.
- Canonical required keys are:
  - `image`
  - `width`
  - `height`
  - `mode`
  - `gt`
  - `pred`
  - `coord_mode`
  - `raw_output_json`
  - `raw_special_tokens`
  - `raw_ends_with_im_end`
  - `errors`
  - `error_entries`
- Geometry objects live under `gt` and `pred`; legacy top-level prediction
  aliases are not canonical.
- `raw_output_json` is the parsed best-effort raw payload from the shared
  salvage/parser path, not a verbatim raw-text mirror.
- Compact detection inference parses model rollouts at object-span granularity.
  Valid object spans are salvaged into `raw_output_json.objects` and `pred`;
  invalid spans are dropped into `dropped_pred_objects` with a concrete
  `reason`, optional `detail`, and the offending `raw_text`. Compact rows also
  expose `parse_status`, `raw_object_spans_total`,
  `valid_pred_object_count`, and `dropped_pred_object_count`.
- `empty_pred` is reserved for outputs with no valid prediction objects. A
  compact rollout with some valid objects and some invalid spans should report
  `dropped_invalid_object` rather than erasing the whole prediction row.

## Historical coordinate handling

- `coord_mode: "pixel"` means evaluator consumers use `gt` and `pred` points as
  pixel coordinates directly.
- `coord_mode: "norm1000"` means evaluator consumers denormalize via per-record
  `width` and `height`, then clamp and round.
- Records missing `width` or `height` are skipped and counted because geometry
  validation is undefined without image size.

## Historical visualization sidecar

- The shared reviewer consumes canonical `vis_resources/gt_vs_pred.jsonl`
  records with:
  - top-level `schema_version`, `source_kind`, `record_idx`, `image`, `width`,
    `height`, `coord_mode`, `gt`, and `pred`
  - bbox-only per-object payloads: `index`, `desc`, `bbox_2d`
  - canonical `matching` with `pred_index_domain=canonical_pred_index` and
    `gt_index_domain=canonical_gt_index`
- Canonical visualization records preserve prediction order.
- Shared-review rendering fails fast if canonical `matching` is missing.
