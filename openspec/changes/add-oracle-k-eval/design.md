# Oracle-K Eval Design

## Scope

This change proposes an additive Oracle-K workflow. It does not change training, online eval-step behavior, or the existing single-artifact detection evaluator contract.
It may add a dedicated Oracle-K runner that materializes repeated inference runs before aggregation.

## Inputs

The Oracle-K workflow consumes:

- one baseline run entry,
- one or more Oracle run entries,
- one YAML config under `configs/eval/`

Required invariants:

- all artifacts correspond to the same input subset,
- record order is the only normative join key across artifacts,
- width, height, and GT object content agree across artifacts,
- each run entry has an explicit label,
- the workflow may either:
  - consume pre-generated `gt_vs_pred.jsonl` artifacts,
  - or materialize them through a thin repeated-sampling orchestrator.

Recommended run-entry shape:

- `label`
- `pred_jsonl`
- optional `pred_token_trace_jsonl`
- optional `resolved_config_json`

## Matching Semantics

Oracle-K reuses current F1-ish semantics rather than defining a second metric family:

- prediction filtering uses `f1ish_pred_scope`,
- location matching uses the existing greedy 1:1 IoU matcher,
- semantic correctness uses the existing exact-or-embedding threshold rule,
- primary threshold selection follows the current evaluator rule:
  - use `0.50` when present,
  - otherwise use the largest requested IoU threshold.

This keeps Oracle-K auditable against the existing `matches.jsonl` interpretation.

## Recovery Definition

For a GT object that is a baseline FN at the primary threshold:

- location-only recovery:
  - whether any Oracle sample location-matches the GT object,
  - how many Oracle samples location-match the GT object,
  - the resulting location recovery fraction across `K`
- semantic+location recovery:
  - whether any Oracle sample full-matches the GT object,
  - how many Oracle samples full-match the GT object,
  - the resulting full recovery fraction across `K`

Derived labels:

- `ever_recovered_loc`
- `ever_recovered_full`
- `systematic_loc`
- `systematic_full`

V1 prioritizes "can a missed object ever be recovered?" and "how often is it recovered?".
Broader continuation-level attribution remains secondary.

## Object-Level Pairing

V1 uses object-level pairing rather than token-span-to-object alignment.

For each baseline-FN GT object and each Oracle run, the workflow records:

- `loc_hit`
- `full_hit`
- `matched_pred_idx` when present
- matched prediction geometry / desc
- IoU
- semantic similarity / pass-fail

If `pred_token_trace_jsonl` is available, the workflow should preserve run-level trace provenance so later analysis can inspect continuations.
Exact token-span-to-object alignment is a follow-up, not a v1 requirement.

## Outputs

The Oracle-K evaluator writes:

- `summary.json`
  - baseline vs Oracle-K recall-style summary at each configured threshold
- `per_image.json`
  - per-image baseline FN counts and recoverable/systematic breakdowns
- `fn_objects.jsonl`
  - one row per baseline FN object with recovery evidence and per-run object-level pairing across Oracle runs

Expected summary surfaces:

- baseline and Oracle-K `tp` / `fn` counts,
- baseline and Oracle-K recall for location and full matching,
- `oracle_run_count`,
- `recoverable_fn_count_loc`,
- `recoverable_fn_count_full`,
- `systematic_fn_count_loc`,
- `systematic_fn_count_full`,
- `recover_fraction_loc`,
- `recover_fraction_full`.

Expected `fn_objects.jsonl` surfaces:

- record-order join key:
  - `record_idx`
  - `gt_idx`
- diagnostic identity:
  - `file_name`
  - GT desc / geometry
- recovery aggregates:
  - `ever_recovered_loc`
  - `ever_recovered_full`
  - `recover_count_loc`
  - `recover_count_full`
  - `recover_fraction_loc`
  - `recover_fraction_full`
- per-run evidence:
  - run label
  - `loc_hit`
  - `full_hit`
  - matched `pred_idx`
  - matched desc / bbox
  - IoU / semantic score
  - optional trace/config provenance

## Non-Goals

- No COCO Oracle-K metric in v1.
- No change to the existing inference-pipeline contract for standard infer/eval runs.
- No change to the existing `metrics.json` schema produced by `scripts/evaluate_detection.py`.
- No required token-span-to-object alignment in v1.
- No full clustering of every predicted object across all rollouts in v1; baseline-FN-focused pairing is the priority.
