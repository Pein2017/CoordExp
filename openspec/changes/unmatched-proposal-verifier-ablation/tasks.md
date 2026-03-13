## 1. OpenSpec Foundation

- [x] 1.1 Add a delta spec for `unmatched-proposal-verifier-study`.
- [x] 1.2 Validate the change artifacts with:
  - `openspec validate unmatched-proposal-verifier-ablation --type change --strict --no-interactive`

## 2. Study Configuration And Subset Control

- [x] 2.1 Add a study config that records:
  - checkpoint list
  - dataset JSONL path
  - sample count
  - seed
  - rollout backend and decode knobs
  - `run.root_image_dir` when the subset lives outside the source dataset root
  - `prompt_variant`
  - `object_field_order`
  - evaluator semantic model path
  - verifier scoring knobs
- [x] 2.2 Support deterministic subset sampling from:
  - `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
  - `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- [x] 2.3 Default to:
  - `val.coord.jsonl`
  - `N=200`
  - the two merged UL checkpoints supplied in this change

## 3. Rollout Proposal Collection

- [x] 3.1 Reuse the existing infer pipeline to collect proposals with:
  - `backend.type: vllm`
  - `temperature: 0.1`
  - `repetition_penalty: 1.1`
  - `infer.generation.batch_size: 16`
  - `infer.backend.server_options.vllm_gpu_memory_utilization: 0.9`
- [x] 3.1a Treat vLLM proposal collection as best-effort repeatable and freeze
  the collected rollout artifacts before final scoring/reporting.
- [x] 3.2 Reuse the existing detection evaluator to derive matched vs unmatched
  proposal buckets with:
  - `f1ish_pred_scope: all`
  - a primary IoU threshold suitable for the report
- [x] 3.3 Materialize a proposal-level table that preserves:
  - checkpoint
  - record / image identifiers
  - proposal index
  - desc
  - bbox
  - matched / unmatched status
  - nearest-GT analysis fields where available

## 4. GT Positives And Hard Negatives

- [x] 4.1 Build a GT-positive table from the sampled subset.
- [x] 4.2 Build GT-derived hard negatives with deterministic families:
  - same-desc wrong-location jitter
  - desc / box cross-swap
  - same-class wrong-location
  - optional oversized / group-box negatives
- [x] 4.3 Keep the negative-construction metadata explicit enough for audit and
  report slicing.

## 5. Teacher-Forced Proxy Scoring

- [x] 5.1 Reuse the existing teacher-forced encode / forward utilities to score
  desc-only commitment from one fixed teacher-forced assistant sequence on the
  original image.
- [x] 5.2 Add bbox-masked counterfactual scoring by reusing that same fixed
  teacher-forced sequence and batching masked images where feasible.
- [x] 5.3 Materialize one normative fixed-sequence source mode for v1:
  - `canonicalized_fixed_sequence_v1`
  - fail fast on sequence canonicalization mismatches.
- [x] 5.4 Emit per-proposal score columns for:
  - commitment
  - masked commitment
  - counterfactual
  - simple combined score
  - optional logistic score
- [x] 5.5 Emit scoring provenance and failure metadata:
  - `scoring_status`
  - `failure_reason`
  - exclusion-from-metrics accounting

## 6. Reporting And Audit Artifacts

- [x] 6.1 Compute and save:
  - AUROC
  - AUPRC
  - score distributions
  - matched-vs-unmatched separation
  - commitment / counterfactual correlation
  - top-k unmatched precision-style analysis
  - calibration / reliability bins, or an explicit skip reason
- [x] 6.2 Produce a concise markdown report that states:
  - exact subset used
  - checkpoints compared
  - prompt/control provenance used for scoring
  - proxy definitions
  - quantitative results
  - qualitative observations
  - recommendation for pseudo-label promotion
- [x] 6.3 Prepare an optional small unmatched audit pack with overlays / crops
  for later manual review.
- [x] 6.4 Document the study entrypoint, config keys, and artifact layout in a
  small runbook or canonical docs update.

## 7. End-To-End Verification

- [x] 7.1 Run a smallest end-to-end smoke on:
  - 1 checkpoint
  - 8 sampled images
- [x] 7.2 Verify the smoke emits:
  - subset metadata
  - infer / eval artifacts
  - GT-positive and hard-negative tables
  - scored proposal table
  - aggregate report
- [ ] 7.3 Run the default two-checkpoint matrix after the smoke passes.

## 8. Temperature Sweep Extension

- [ ] 8.1 Extend the study config to accept an explicit list of rollout
  temperatures for the full run while keeping the other collection/scoring
  knobs fixed.
- [ ] 8.2 Materialize per-temperature collection / scoring artifacts separately
  and record temperature in the checkpoint manifests and proposal outputs.
- [ ] 8.3 Compare verifier metrics and proposal distributions across
  temperatures in the aggregate report.
- [ ] 8.4 Run the full temperature sweep on the merged `main` branch and save a
  concise cross-temperature summary.
