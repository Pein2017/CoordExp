## Why

CoordExp's current offline evaluator measures one prediction artifact at a time. That is enough for standard COCO and F1-ish evaluation, but it cannot answer the research questions behind Oracle-K:

- when a GT object is missed in the baseline decode,
- can the same checkpoint recover that object under repeated stochastic sampling,
- or is the miss systematic across all sampled continuations?
- how often is that object recovered across `K` stochastic rollouts,
- and does repeated sampling reveal useful signal for contrastive/object-level follow-up work?

We need a reproducible, YAML-first workflow that can:

- run or consume repeated stochastic decodes for the same subset,
- measure object recovery separately for:
  - location-only,
  - semantic+location,
- report both:
  - binary "recovered at least once",
  - empirical recovery frequency across `K` rollouts.

This distinction matters for follow-on optimization work such as contrastive / GRPO-style objectives, where the training signal depends on whether the model already has a latent successful mode.
It also matters for estimating model upper capacity beyond conservative single-decode performance.

## What Changes

- Add an additive Oracle-K workflow under the detection-evaluator surface.
- Define Oracle-K as GT-centric repeated-sampling analysis measured from one baseline decode plus `K` aligned stochastic decodes for the same input subset.
- Allow a thin YAML-first repeated-sampling orchestrator for Oracle-K runs while keeping the existing single-artifact evaluator contract intact.
- Reuse the existing F1-ish semantics for:
  - IoU thresholds,
  - semantic matching,
  - prediction-scope filtering (`annotated` vs `all`),
  - primary-threshold selection.
- Add Oracle-K artifacts that make FN audit practical:
  - summary metrics,
  - per-image breakdowns,
  - one row per baseline FN object with per-rollout object-level pairing and recovery labeling.
- Report recovery separately for:
  - location-only,
  - semantic+location.
- Report both:
  - `ever recovered`,
  - `recover_count` / `recover_fraction`.
- Keep the existing single-artifact evaluator contract intact:
  - no behavior change for `scripts/evaluate_detection.py`,
  - no COCO Oracle-K metric in v1.

## Capabilities

### Added Capabilities
- `detection-evaluator`: Oracle-K repeated-sampling recovery analysis over aligned stochastic prediction artifacts.

## Impact

- Existing standard evaluation remains unchanged.
- Oracle-K is additive and analysis-oriented.
- The first implementation stays YAML-first:
  - users may provide pre-generated baseline / stochastic `gt_vs_pred.jsonl` artifacts,
  - or use a thin Oracle-K runner that materializes repeated inference runs before aggregation.
- The live contract becomes explicit about what "recovered" means:
  - same matching semantics as the current F1-ish evaluator,
  - recovery is tracked separately for location-only and semantic+location at the primary IoU threshold,
  - baseline FN categorization and recovery frequency are derived from those same semantics.
- Broad token-span-to-object alignment is not required in v1:
  - v1 focuses on object-level pairing,
  - deeper continuation attribution remains a follow-up if the recovery signal is promising.
