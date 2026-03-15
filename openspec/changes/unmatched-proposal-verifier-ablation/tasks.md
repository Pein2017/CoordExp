## 1. Spec Reset

- [x] 1.1 Reconcile the active change artifacts with the authority-first study
  framing.
- [x] 1.2 Validate the updated change artifacts with:
  - `openspec validate unmatched-proposal-verifier-ablation --type change --strict --no-interactive`

## 2. Staged Workflow Refactor

- [x] 2.1 Refactor the study into explicit stages:
  - subset / GT preparation
  - rollout collection
  - collection-health summarization and gating
  - rollout scoring for collection-valid runs
  - report aggregation
  - manual audit artifact preparation / ingestion
- [x] 2.2 Emit stage manifests or equivalent resumable provenance artifacts.
- [x] 2.3 Ensure later stages can reuse frozen earlier-stage outputs.

## 3. Clean Verifier Benchmark

- [x] 3.1 Preserve the clean GT-positive / GT-hard-negative benchmark.
- [x] 3.2 Keep the negative-family metadata explicit and sliceable.
- [x] 3.3 Report `commitment`, `counterfactual`, and combination metrics on the
  clean benchmark.

## 4. Rollout Collection Validity Gate

- [x] 4.1 Add collection-health outputs per checkpoint × temperature with at
  least:
  - `pred_count_total`
  - `pred_count_per_image_mean`
  - `nonempty_pred_image_rate`
  - `matched_count`
  - `unmatched_count`
  - `ignored_count`
  - `invalid_rollout_count`
  - parser failure counts
  - duplicate-like rate
- [x] 4.2 Add:
  - `collection_valid`
  - `collection_invalid_reason`
- [x] 4.3 Exclude collection-invalid runs from the main rollout comparison
  tables while preserving them in artifacts / appendices.
- [x] 4.4 Explicitly verify the gate behavior with artifacts:
  - collection-valid runs enter the main comparison tables
  - collection-invalid runs remain visible in collection-health outputs /
    appendices but not in the main comparison tables

## 5. Rollout Proposal Benchmark

- [x] 5.1 Score rollout proposals only for collection-valid runs.
- [x] 5.2 Report:
  - matched-vs-unmatched separation
  - unmatched top-k proxy analysis
  - proposal score distributions
  - checkpoint × temperature proposal statistics
- [x] 5.3 Make rollout-facing temperature conclusions depend on this layer, not
  the clean GT slice alone.

## 6. Manual Unmatched Audit

- [x] 6.1 Prepare a small manually auditable unmatched subset.
- [x] 6.2 Stratify the audit subset by:
  - checkpoint
  - temperature
  - score quantile
  - nearest-GT weak-overlap bucket
- [x] 6.3 Define and document the audit label schema:
  - `real_visible_object`
  - `duplicate_like`
  - `wrong_location`
  - `dead_or_hallucinated`
  - `uncertain`
- [x] 6.4 Collect / ingest manual audit labels into a canonical artifact.
- [x] 6.5 Regenerate the final report using the audit labels.
- [x] 6.6 Use the manual audit layer in the final recommendation logic.

## 7. Temperature Sweep

- [x] 7.1 Restrict the authoritative sweep to exactly:
  - `0.0`
  - `0.3`
  - `0.5`
  - `0.7`
- [x] 7.2 Record temperature in all per-run manifests and aggregate tables.
- [x] 7.3 Compare collection-health, rollout proxy metrics, and audit outcomes
  across these four temperatures.
- [x] 7.4 Keep any auxiliary / failed / exploratory temperatures out of the
  authoritative main conclusion tables and label them appendix-only.

## 8. Final Report

- [x] 8.1 Produce separate summary tables for:
  - clean GT-vs-hard-negative evidence
  - rollout collection health
  - rollout proposal scoring
  - manual unmatched audit
- [x] 8.2 Explicitly answer:
  - strongest single proxy
  - whether combination helps
  - whether rollout evidence is valid enough for interpretation
  - whether pseudo-label promotion is actually justified
- [x] 8.3 Downgrade the final conclusion to “promising but not yet
  promotion-ready” whenever rollout or audit evidence is insufficient.
