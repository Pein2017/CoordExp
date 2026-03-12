## 1. OpenSpec Contract

- [x] 1.1 Add a detection-evaluator delta spec for Oracle-K repeated-sampling recovery analysis.
- [x] 1.2 Define the recovery contract precisely:
  - baseline FN objects are categorized at the primary F1-ish IoU threshold,
  - recovery is reported separately for location-only and semantic+location,
  - both `ever recovered` and `recover_count` / `recover_fraction` are required,
  - Oracle artifacts align in record order and preserve required `image_id` / `file_name` provenance for downstream visualization analysis,
  - Oracle run entries carry explicit labels.

## 2. Oracle-K Evaluator

- [x] 2.1 Add a dedicated YAML-first Oracle-K entrypoint under `scripts/`.
- [x] 2.2 Add a reusable evaluator module under `src/eval/` that:
  - validates aligned GT across artifacts,
  - validates record-order alignment, required `file_name` provenance, and GT equality,
  - reuses current F1-ish matching semantics,
  - computes baseline vs Oracle-K recall-style summaries for location-only and semantic+location,
  - emits `ever recovered` and recovery frequency statistics,
  - emits per-run object-level pairing for baseline FN objects.
- [x] 2.3 Add a config template under `configs/eval/` with:
  - one baseline run entry,
  - one or more Oracle run entries,
  - explicit run labels,
  - optional trace/config provenance paths.
- [x] 2.4 Add a thin repeated-sampling orchestration path that can materialize Oracle runs from YAML without changing the standard `scripts/evaluate_detection.py` contract.

## 3. Validation

- [x] 3.1 Add focused tests for:
  - location-only `ever recovered` labeling,
  - semantic+location `ever recovered` labeling,
  - recovery counts / fractions,
  - systematic FN labeling for both views,
  - GT mismatch failure,
  - required `file_name` provenance mismatch failure,
  - record-order mismatch failure,
  - no partial Oracle-K metrics on misalignment,
  - threshold selection consistency with the current evaluator.
- [x] 3.2 Verify the Oracle-K outputs remain additive and do not change the existing single-artifact evaluator behavior.

## 4. Docs

- [x] 4.1 Update `docs/eval/WORKFLOW.md` with the Oracle-K analysis workflow.
- [x] 4.2 Update `docs/ARTIFACTS.md` with Oracle-K artifact descriptions.
- [x] 4.3 Review `docs/eval/README.md` for routing updates.
- [x] 4.4 Document the v1 boundary clearly:
  - object-level pairing is included,
  - token-span-to-object alignment is a follow-up.

## 5. Requested Run After Audit

- [x] 5.1 Sample a fixed 200-example subset from:
  - `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- [x] 5.2 Run standard evaluation plus Oracle-K analysis for:
  - `output/stage2_ab/prod/ul-res_1024-ckpt_300_merged`
  - `output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged`
  - `output/stage2_ab/prod/ab_mixed/eff_size_96-b_ratio_0.75-n_softctx_iter_2-epoch_2_merged-ckpt-2442`
- [x] 5.3 Inspect whether baseline-FN objects that are `ever_recovered` also show useful per-run contrastive signal in their paired predicted objects and continuations.
