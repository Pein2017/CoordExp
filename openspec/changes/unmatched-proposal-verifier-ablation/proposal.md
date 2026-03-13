## Why

CoordExp already exposes several ways to score generated detections, but the
current tooling still leaves one important research question unresolved:

- when a rollout prediction is unmatched to GT, is it usually a dead / duplicate
  / bad proposal,
- or is it often a real visually grounded object that the annotations miss or
  underspecify?

This matters directly for the next pseudo-labeling step.
If unmatched proposals already carry a reliable verifier signal, we can promote
them softly without training a new detector head or changing the model
architecture.

The repo already has the core ingredients we need:

- existing checkpoint families with different behavior,
- inference and evaluation artifacts that preserve proposal order and
  matched-vs-unmatched buckets,
- teacher-forced forward utilities,
- and confidence post-op code that proves object-level offline scoring is a good
  fit for this stack.

What is missing is one explicit, reproducible offline ablation that compares
proposal-verification proxies on a small shared subset using the existing model,
vision tower, and teacher-forced forward path.

## What Changes

- Add a new capability, `unmatched-proposal-verifier-study`, for a reproducible
  offline ablation over unmatched proposal verifiers.
- Keep the study intentionally narrow and config-first:
  - no new detector head,
  - no DETR-style branch,
  - no large-scale training,
  - no changes to upstream HF model files.
- Define one study workflow that:
  - samples a deterministic subset from a COCO 1024 JSONL,
  - records the source dataset root explicitly when the sampled subset is
    materialized outside the source JSONL directory,
  - runs rollout proposal collection with the existing inference pipeline using
    `backend.type: vllm`,
  - evaluates proposals with the existing detection evaluator using
    `f1ish_pred_scope: all`,
  - builds GT positives and GT-derived hard negatives offline,
  - freezes the collected proposal artifacts before scoring/reporting,
  - scores proposals with one fixed-sequence teacher-forced baseline forward on
    the original image and batched masked teacher-forced forwards on the same
    fixed assistant sequence,
  - aggregates metrics, plots, and a concise markdown report.
- Make the initial matrix explicit and editable:
  - checkpoints default to:
    - `output/stage2_ab/prod/ul-res_1024-ckpt_300_merged`
    - `output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged`
  - dataset defaults to:
    - `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
  - allow:
    - `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
  - default sample count:
    - `200`
  - initial rollout decode settings:
    - `temperature: 0.1`
    - `repetition_penalty: 1.1`
    - `infer.generation.batch_size: 16`
    - `infer.backend.server_options.vllm_gpu_memory_utilization: 0.9`
  - extend the collection matrix to support an explicit temperature sweep over
    multiple rollout temperatures while keeping the rest of the study controls
    fixed
  - prompt/control provenance must be recorded per checkpoint:
    - `prompt_variant`
    - `object_field_order`
    - evaluator semantic model path
- Keep proxy scope small and auditable:
  - primary commitment = desc-only teacher-forced average log-probability,
  - primary counterfactual = commitment drop under bbox masking,
  - primary combination = transparent linear combination,
  - optional logistic calibration only as a secondary comparison.

## Capabilities

### Added Capabilities
- `unmatched-proposal-verifier-study`: deterministic subset selection, vLLM
  proposal collection, GT-positive and hard-negative table construction,
  teacher-forced proxy scoring, cross-checkpoint aggregation, and concise study
  reporting for unmatched proposal verification.

## Impact

- No training objective or model architecture changes are required.
- The study reuses existing infer / eval / teacher-forced infrastructure rather
  than introducing a parallel runtime stack.
- Proposal collection is accelerated with vLLM, but verifier scoring remains an
  offline teacher-forced analysis path over frozen proposal artifacts.
- vLLM collection is treated as best-effort repeatable rather than byte-identical
  deterministic; deterministic claims apply to subset selection, study config,
  and fixed-sequence teacher-forced scoring after collection artifacts are
  frozen.
- The study treats duplicate suppression as an external concern:
  duplicate-like metadata may be carried for analysis, but the core question is
  realism / groundedness of unmatched proposals.
- The output is intended to support a later decision about soft pseudo-label
  promotion, not to claim a final production policy by itself.
- The full run should also answer whether rollout temperature materially changes
  unmatched proposal quality and the usefulness of commitment / counterfactual
  as verifier proxies.
