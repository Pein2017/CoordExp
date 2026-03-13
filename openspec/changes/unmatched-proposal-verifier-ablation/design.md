## Context

The requested study sits at the intersection of four existing repo surfaces:

- inference collection:
  - `scripts/run_infer.py`
  - `src/infer/pipeline.py`
  - `src/infer/engine.py`
- matched / unmatched offline evaluation:
  - `scripts/evaluate_detection.py`
  - `src/eval/detection.py`
- proposal parsing and rollout serialization:
  - `src/trainers/rollout_matching/parsing.py`
- teacher-forced forward utilities:
  - `src/trainers/teacher_forcing/forwards.py`
  - `src/trainers/teacher_forcing/rollout_meta.py`

The goal is not to build a new detector.
The goal is to measure whether proposal-verification proxies can identify
visually real unmatched proposals using the current model stack and a small
offline matrix.

Two practical constraints shape the design:

- rollout proposal collection should use vLLM for speed,
- verifier scoring still needs exact teacher-forced logits, which are easier and
  safer to obtain through the HF / ms-swift path.

So the study deliberately uses a mixed backend:

- vLLM for proposal generation,
- HF teacher-forced forward for commitment / counterfactual scoring.

This is acceptable because:

- the rollout artifact contract is already backend-agnostic,
- the verifier study cares about proposal realism and ranking,
- and exact vLLM teacher-forced logit parity is not a requirement here.

The clarified v1 scoring semantics are:

- rollout once per sampled image / checkpoint,
- derive one fixed assistant sequence for scoring,
- run one teacher-forced baseline forward on the original image,
- run one batched masked teacher-forced forward over selected proposals using
  the same fixed sequence,
- do not perform masked rerollout.

## Goals / Non-Goals

**Goals**

- Build a deterministic, reproducible offline study over existing checkpoints.
- Be precise about what is deterministic:
  - deterministic subset sampling,
  - deterministic GT/hard-negative construction,
  - deterministic teacher-forced scoring over frozen collected proposal
    artifacts.
- Compare three proposal-verification proxies:
  - commitment
  - counterfactual
  - commitment + counterfactual
- Use a small but auditable subset from COCO 1024 with default `N=200`.
- Preserve the current model / vision tower / teacher-forced forward pipeline as
  much as possible.
- Produce study artifacts that are useful for later pseudo-label policy
  decisions.

**Non-Goals**

- No large-scale training.
- No new detector head, region-proposal network, or DETR-style branch.
- No attempt to redesign duplicate suppression.
- No dependence on manual labeling to finish the study.
- No change to upstream HF model files.

## Study Matrix

### Initial checkpoints

The initial default checkpoint list is intentionally small:

- `output/stage2_ab/prod/ul-res_1024-ckpt_300_merged`
- `output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged`

The implementation should keep the list editable via a YAML config rather than
hard-coding it.

### Initial datasets

Supported initial inputs:

- `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`

Recommended default:

- `val.coord.jsonl`

Reason:

- the study is diagnostic rather than training-time,
- evaluation semantics are easier to explain on held-out data,
- and `N=200` already keeps runtime bounded.

### Initial rollout settings

Proposal collection defaults:

- `backend.type: vllm`
- `temperature: 0.1`
- `repetition_penalty: 1.1`
- `infer.generation.batch_size: 16`
- `infer.backend.server_options.vllm_gpu_memory_utilization: 0.9`

The study config should keep these as explicit knobs, but the report must record
their resolved values.

### Temperature sweep extension

After the base smoke and single-temperature study path are valid, the full run
should support a temperature sweep over a small explicit list such as:

- `0.0`
- `0.1`
- `0.3`
- `0.5`

The temperature sweep should:

- keep checkpoint list, subset, repetition penalty, prompt controls, and
  verifier scoring logic fixed,
- materialize per-temperature collection artifacts separately,
- preserve temperature in per-run manifests, proposal tables, and aggregate
  reports,
- report how proposal counts, matched-vs-unmatched separation, and verifier
  proxy quality change as temperature increases.

### Subset placement and image resolution

Subset sampling is deterministic, but image resolution depends on how the subset
JSONL is written.

Normative v1 rule:

- if the sampled subset is materialized outside the source dataset directory,
  the study config must set `run.root_image_dir` to the original dataset root.

This keeps infer/eval/vis image resolution reproducible even when study artifacts
live under a separate analysis directory.

## Data Flow

```text
sampled COCO-1024 subset
  -> vLLM infer per checkpoint (best-effort repeatable)
  -> gt_vs_pred.jsonl
  -> detection eval with pred_scope=all
  -> matches.jsonl
  -> freeze proposal artifacts for scoring/reporting
  -> proposal table
       - matched rollout proposals
       - unmatched rollout proposals
       - optional duplicate-like flags
  -> teacher-forced scorer
       - one baseline forward over one fixed assistant sequence
       - one batched masked forward over the same fixed assistant sequence
       - counterfactual = delta
  -> metrics / plots / markdown report
```

In parallel, the same sampled subset yields a second labeled table:

- GT positives
- GT-derived hard negatives

That table provides the cleanest positive / negative supervision for proxy
quality.

## Proposal Sets

### 1) GT positives

GT positives come directly from the sampled dataset records.
The study should use the canonical desc text already present in the dataset /
artifact path and preserve the GT bbox.

Primary scoring context:

- canonical prompt with no rollout prefix.

Reason:

- this keeps the positive-vs-negative benchmark interpretable,
- and avoids smuggling rollout-specific decode behavior into the cleanest slice.

### 2) GT-derived hard negatives

Hard negatives should stay visually plausible rather than random.
The initial negative families are:

- same-desc wrong-location jittered boxes with intentionally low overlap to the
  GT box,
- desc / box cross-swaps within the same image,
- same-class wrong-location boxes when an image contains multiple same-class
  instances,
- optional oversized / group-box negatives when they can be synthesized
  deterministically.

This is intentional.
Desc-only commitment may still score some same-desc wrong-location negatives
highly, which is useful because it tests whether counterfactual adds location
signal beyond plain descriptor plausibility.

### 3) Rollout proposals

Rollout proposals are collected from the current infer / eval artifact pair:

- `gt_vs_pred.jsonl`
- `matches.jsonl`

The proposal table must retain at least:

- checkpoint id
- dataset split
- record index / image id / file name when available
- proposal index
- desc
- bbox
- matched / unmatched status
- nearest matched GT metadata where available
- nearest-GT IoU and desc if computed offline
- duplicate-like heuristic flags when computed

The evaluator must use:

- `f1ish_pred_scope: all`

Reason:

- the default `annotated` scope intentionally ignores some open-vocab proposals,
- which would hide exactly the unmatched population this study is trying to
  inspect.

## Proxy Definitions

### Commitment

Primary definition:

- desc-only average log-probability of the proposal desc tokens under
  teacher-forced forward on the original image.

Conditioning context:

- for rollout proposals: one fixed teacher-forced assistant sequence per sampled
  image / checkpoint rollout,
- for GT positives and GT-derived hard negatives: canonical prompt with no
  rollout prefix.

Full object-span commitment is optional and secondary.
The initial study should not depend on it.

Operationally, v1 commitment should be computed as:

- 1 rollout to collect proposals,
- 1 teacher-forced baseline forward on the fixed assistant sequence,
- desc-span extraction for all proposals from that same forward.

### Counterfactual

Primary definition:

- counterfactual = commitment(original image) - commitment(masked image)

Mask region:

- predicted / candidate bbox

Mask policy:

- deterministic fill or occlusion policy recorded in config and report
- the implementation should prefer one simple auditable fill policy over
  attention-driven masking.

Operationally, v1 counterfactual should be computed as:

- keep the exact same fixed teacher-forced assistant sequence,
- change only the image input,
- batch the masked images together for the selected proposals,
- do not rerollout after masking.

### Combination

Primary combination:

- transparent linear combination after per-checkpoint score normalization

Secondary comparison:

- 2-feature logistic calibration on `[commitment, counterfactual]`

The report should treat the linear combination as the default simple answer and
the logistic variant as optional evidence of incremental value.

## Backend And Runtime Decisions

### Why vLLM for rollout collection

Proposal collection is embarrassingly parallel across checkpoints and sampled
images, and the infer pipeline already supports a vLLM backend.
Using vLLM keeps the collection stage cheap enough to include two checkpoints on
`N=200` without stretching the study into a long-running experiment.

However, vLLM collection should be treated as best-effort repeatable, not
strictly deterministic.
The study should therefore freeze collected proposal artifacts before computing
report metrics.

### Why HF for verifier scoring

Teacher-forced commitment and counterfactual need exact logits on a controlled
assistant span.
The safest current repo path is the existing teacher-forced forward utilities and
ms-swift template encode path.

This mixed approach minimizes new machinery:

- vLLM generates proposals,
- HF scores them.

## Fixed-Sequence Canonicalization

This remains the highest-risk implementation detail, but the clarified v1 path
is narrower than the original draft.

The study needs one fixed assistant sequence for each sampled image / checkpoint
rollout, but the vLLM infer artifact does not materialize exact raw completion
text or token ids by default.

Normative v1 approach:

- derive one canonicalized fixed assistant sequence from parsed rollout objects
  while preserving proposal order,
- use that same canonicalized sequence for:
  - baseline commitment scoring,
  - masked counterfactual scoring,
- extract desc spans for each proposal within that one fixed sequence.

This means v1 does not require one append-ready prefix per proposal.
It requires one frozen sequence plus auditable desc-span mapping.

Exact raw emitted sequence recovery may be added later through an optional
study-sidecar, but it is not required for v1.

The study should explicitly record:

- fixed-sequence source mode (`canonicalized_fixed_sequence_v1`),
- desc-span extraction mode,
- prompt/control provenance used for scoring.

## Prompt And Scoring Provenance

Desc-only commitment is prompt-sensitive.
So the study must carry the prompt controls used for both proposal collection and
teacher-forced scoring.

Per checkpoint, the study should record at least:

- `prompt_variant`,
- `object_field_order`,
- scorer template settings needed to rebuild the assistant span,
- evaluator semantic model path used for matched/unmatched bucketing.

Default behavior should prefer checkpoint-native resolved config values when they
are available.

## Scoring Failure Policy

The study should not silently drop rows when verifier scoring fails.

Each scored row should preserve:

- `scoring_status`,
- `failure_reason`,
- whether the row is excluded from primary metrics.

Primary metrics should exclude failed rows from AUROC/AUPRC denominators, while
the aggregate summary reports failure counts by reason.

## Output Artifacts

The study should emit, at minimum:

- sampled subset JSONL + sampling metadata
- checkpoint list config
- per-checkpoint run manifests
- GT-positive / hard-negative table
- rollout proposal table
- scored proposal table with:
  - commitment
  - masked commitment
  - counterfactual
  - simple combination
  - optional logistic score
  - scoring status / failure reason
- aggregate metrics tables
- score histograms / distributions
- score correlation summaries
- concise markdown report
- optional small unmatched audit pack with overlays / crops

## Validation Strategy

The implementation should fail fast on these conditions:

- missing checkpoint path,
- missing sampled subset,
- evaluator run not configured with `pred_scope=all`,
- inability to construct one fixed teacher-forced assistant span,
- sequence canonicalization mismatch,
- non-finite verifier scores,
- desc-span extraction mismatch,
- subset image-resolution mismatch caused by missing `run.root_image_dir`.

The smallest end-to-end validation target is:

- 1 checkpoint
- 8 images
- proposal collection with vLLM
- GT/hard-negative table build
- one fixed-sequence baseline forward,
- one batched masked forward,
- desc-only commitment and counterfactual on at least a few objects
- aggregate report generation

Once that works, the default matrix can scale to the requested two checkpoints
and `N=200`.
