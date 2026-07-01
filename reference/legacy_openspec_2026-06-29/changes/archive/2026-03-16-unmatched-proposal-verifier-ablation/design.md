## Context

The study still sits at the intersection of four existing repo surfaces:

- inference collection:
  - `scripts/run_infer.py`
  - `src/infer/pipeline.py`
  - `src/infer/engine.py`
- matched / unmatched offline evaluation:
  - `scripts/evaluate_detection.py`
  - `src/eval/detection.py`
- proposal parsing:
  - `src/trainers/rollout_matching/parsing.py`
- teacher-forced scoring:
  - `src/trainers/teacher_forcing/forwards.py`
  - existing scorer / template logic in `src/analysis/unmatched_proposal_verifier.py`

The original implementation proved that the clean teacher-forced slice can be
run offline and that `counterfactual` has signal on GT-vs-hard-negative data.

However, that is not enough for a trustworthy unmatched-proposal conclusion,
because:

- rollout proposal collection quality varies by run,
- successful runs can still fail to produce analyzable unmatched populations,
- matched-vs-unmatched metrics are only meaningful when collection itself is
  first validated,
- and no small manual audit currently anchors what high-scoring unmatched
  proposals actually mean.

So the design should now emphasize evidence quality rather than runtime breadth.

## Design Principles

1. Keep the model path unchanged.
2. Keep the study offline and config-first.
3. Separate clean verifier validation from rollout proposal validation.
4. Do not compare temperatures unless rollout collection has passed a health
   gate.
5. Add a small manual unmatched audit before making any pseudo-label promotion
   recommendation.

## Revised Study Structure

### Layer A: Clean Verifier Benchmark

Purpose:

- isolate whether the verifier proxies carry visual grounding signal.

Inputs:

- GT positives
- GT-derived hard negatives

Outputs:

- AUROC / AUPRC per proxy
- negative-family slices
- score distributions

Interpretation:

- this layer answers “does the verifier work at all?”
- it does not answer “is unmatched rollout promotion safe?”

### Layer B: Rollout Proposal Benchmark

This layer must be split into two explicit stages.

#### B1. Collection Validity Gate

Before any scoring comparison, each checkpoint × temperature run must emit
collection-health statistics.

Minimum required fields:

- `temperature`
- `checkpoint`
- `pred_count_total`
- `pred_count_per_image_mean`
- `nonempty_pred_image_rate`
- `matched_count`
- `unmatched_count`
- `ignored_count`
- `invalid_rollout_count`
- parser failure counts
- duplicate-like rate

Suggested gate fields:

- `collection_valid`
- `collection_invalid_reason`

Recommended initial gate:

- `nonempty_pred_image_rate >= 0.30`
- `pred_count_total >= 100`
- `unmatched_count >= 50`

Reason:

- a run with no usable proposal population should not participate in the
  temperature comparison or pseudo-label discussion.

#### B2. Rollout Proposal Scoring

Only collection-valid runs should be scored into the main rollout comparison.

Outputs:

- matched-vs-unmatched separation
- unmatched top-k proxy analysis
- score distributions by checkpoint and temperature
- proposal-level failure accounting

Interpretation:

- this is the main layer for the pseudo-label question.

### Layer C: Small Manual Audit Benchmark

This layer should be small but required for the final recommendation.

Target size:

- roughly `96` to `128` unmatched proposals total

Sampling design:

- stratify by:
  - checkpoint
  - temperature
  - score quantile
  - nearest-GT weak-overlap bucket

Suggested labels:

- `real_visible_object`
- `duplicate_like`
- `wrong_location`
- `dead_or_hallucinated`
- `uncertain`

Final binary interpretation:

- positive: `real_visible_object`
- negative: the others
- `uncertain`: tracked separately

Reason:

- nearest-GT IoU is not a trustworthy realism label for unmatched proposals,
- so a small audit layer is the most efficient way to make the final conclusion
  credible.

## Temperature Design

Use exactly four temperatures in the main study:

- `0.0`
- `0.3`
- `0.5`
- `0.7`

Reason:

- `0.0` is the greedy baseline,
- `0.3 / 0.5 / 0.7` cover a useful low-to-high stochasticity band,
- removing dense intermediate points reduces runtime and sharpens interpretation.

Interpretation rule:

- temperature effects should be read mainly through Layer B and Layer C,
  not Layer A.

## Staged Workflow

The implementation should be refactored toward these explicit stages:

1. subset / GT table preparation
2. rollout collection
3. collection-health summarization and gating
4. rollout scoring for collection-valid runs
5. report aggregation
6. manual audit artifact preparation and audit ingestion

Each stage should emit its own manifest so runs can be resumed without
repeating earlier work.

Suggested manifests:

- `subset_manifest.json`
- `collection_manifest.json`
- `scoring_manifest.json`
- `report_manifest.json`
- `manual_audit_manifest.json`

## Output Tables

The full trusted study should produce at least these top-level tables:

- `collection_health_by_temp.csv`
- `gt_clean_proxy_metrics_by_temp.csv`
- `rollout_proxy_metrics_by_temp.csv`
- `manual_audit_by_temp.csv`

And these should be clearly separated:

- collection-invalid runs
- collection-valid runs

## Final Decision Standard

The study should only claim a strong unmatched-promotion recommendation if all
of the following hold:

1. `counterfactual` is clearly stronger than `commitment` on the clean GT slice.
2. rollout collection is valid for most checkpoint × temperature combinations.
3. matched-vs-unmatched or unmatched top-k analysis remains informative on
   collection-valid runs.
4. manual audit shows that high-scoring unmatched proposals are often real
   visible objects rather than duplicates, wrong-location boxes, or dead
   proposals.

If any of these fail, the report should explicitly downgrade the conclusion to:

- promising,
- but not yet pseudo-label-promotion ready.
