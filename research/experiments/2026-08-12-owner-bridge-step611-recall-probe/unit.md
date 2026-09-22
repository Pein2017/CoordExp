---
title: Owner Bridge Step-611 Natural-Decode Recall Probe
description: Natural-greedy behavioral and lifecycle probe of the interrupted permanent-owner-bridge checkpoint with its required bridge composition.
type: investigation
role: research-unit
authority: user-authorized-checkpoint-probe
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
implementation_status: complete_within_unit
unit_id: 2026-08-12-owner-bridge-step611-recall-probe
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_behavioral_hold
updated: 2026-08-12
---

# Owner Bridge Step-611 Recall Probe

## Question and decision boundary

What does the interrupted permanent-owner-bridge checkpoint at production step
611 do during natural greedy decode, and does its complete bridge-enabled
composition recover a useful fraction of GT objects without collapsing
validity, grounding, uniqueness, or native STOP behavior?

This unit evaluates an existing checkpoint. It performs no training, optimizer
step, resume, architecture promotion, checkpoint promotion, decoder change, or
rewrite of the sealed 2026-08-05 through 2026-08-07 research evidence. The
checkpoint is temporary research evidence from an interrupted run, not a final
production model.

## Exact identities

- owner-bridge checkpoint:
  `/data/CoordExp/.worktrees/permanent-owner-bridge/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_xy_owner_bridge_stage1_dora_r16a32_llm_12000_accelerate8_ebs24_4epoch_warmup0p1/checkpoints/step-611`
- source-tuned checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- base model:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- owner-bridge payload fingerprint:
  `c74a93ce0a7c09c590fb320d1e71a1abc3c2201ada040355a4024c89b22f316d`
- source input:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/inputs/val200/coco_val200_len12000.rebased_images.xy_sorted.coord.jsonl`
- source input SHA-256:
  `fa1404991380ac0be90c13a25341182d13fb763784254ea6b2b3ee2cf3398a15`
- prospective human-refined 13-image input:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl`
- decode-compatible 13-image copy, with only deterministic `geo_sorted_xy`
  object reordering and one extra relative-path traversal:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe/artifacts/inputs/human-refined-13.xy-sorted.coord.jsonl`
- decode-compatible 13-image input SHA-256:
  `77584e25f65d78a59a3a434cd0adbb016002cb53a8b7044325636c721fd26e5f`

The source-tuned checkpoint is lineage context only. It is not an execution arm
in this unit and is not rerun or promoted into the main conclusion.

## Competing explanations

- **H1: useful owner enumeration.** The complete step-611 composition attains
  nontrivial matched GT coverage while keeping parser validity, uniqueness,
  grounding, and STOP behavior usable.
- **H2: continuation without recall.** Step-611 emits many rows or stops late,
  but unique matched-owner recall remains weak; row count is not owner recovery.
- **H3: route without consumption.** Lifecycle receipts show inventories and
  route selections, but latches/writes do not align with newly recovered GT
  owners or do not survive to complete rows.
- **H4: active bridge consumption.** Admission, latch, row-write, clear, and
  reroute events execute coherently and are enriched on correct newly emitted
  owners relative to failures within the same checkpoint.
- **H5: interrupted-stage harm or null.** The checkpoint has weak recall,
  duplicate/unmatched concentration, grammar failures, premature STOP, or no
  meaningful bridge lifecycle variation at this stage.

## Fixed decode composition

The only model arm is the complete step-611 composition: base model, step-611
DoRA adapter, step-611 selected embeddings, and step-611 OwnerBridge. Decode
uses dynamic HF, BF16, FlashAttention-2, no resize, `desc_first`,
`geo_sorted_xy`, the exact production prompt, native greedy generation,
`max_new_tokens=512`, `temperature=0`, `top_p=1`, `n=1`, canonical batch size
2, and `repetition_penalty=1.0`.

The bridge-native path rejects repetition penalty other than 1.0, external
logits processors, forced tokens, beam search, sampling, static/non-dynamic
cache, compile, chunked prefill, and speculative decoding. Those controls are
not varied in this unit.

## Execution ladder and protocol correction

The first execution used a mechanical first-16 plumbing smoke and then expanded
to val200. That order was valid for runtime qualification but inefficient for
the scientific decision: the frozen, human-refined 12+2299 panel was already a
more informative mechanism screen. After user correction, the unit therefore
treats the independent 13-image run as the primary behavioral/mechanism screen
and the already-completed val200 run only as a supplementary prevalence check.

The original human-refined panel failed before model load because its authored
object order is not `geo_sorted_xy`. One mechanical repair was used: copy all 13
rows, preserve image ids, object multisets, non-object metadata, and the
legacy-12 versus image-2299 partition exactly, sort each row's objects by
`(x1, y1)`, and adjust relative image paths for the new directory depth. The
repaired input has 13 rows, 392 objects, zero missing images, identical object
multisets, and identical non-object metadata.

## Endpoints

Primary endpoint:

- `bbox_AR100` from the canonical COCO bbox evaluator on the matched val200
  panel.

Secondary endpoints:

- `bbox_AR1`, `bbox_AR10`, `bbox_AP`, `bbox_AP50`, and size-stratified AR;
- GT count, prediction count, per-image row count, and empty predictions;
- parser failures, dropped predictions, truncation, and terminal reason;
- class-aware one-to-one GT matches at IoU 0.50 and 0.75, unique matched GT,
  false negatives, false positives, and repeated prediction content;
- bridge inventory, route, admission residual, latch, row-write, clear, reroute,
  unprocessed-tail, and terminal lifecycle fields;
- deltas by GT density, object size, class, and baseline FN family.

The unit will first describe aggregate deltas, then select divergent cases from
those aggregates. It will not preselect only successes.

## Meaning-bearing invariants

- Image path, dimensions, GT order, coordinate conversion, prompt, wrapper,
  row parser, scorer, token budget, repetition penalty, and evaluator must match
  across compared arms.
- Prediction coordinates are pixel `xyxy`; GT coordinate tokens use the
  canonical conversion path.
- Generic extra rows, later STOP, higher token count, teacher-forced loss, route
  probability, or owner auxiliary loss are not recall improvements by
  themselves.
- A bridge lifecycle receipt proves bridge execution, not beneficial owner use.
- COCO AR on 200 images is a bounded panel result, not a full-val or deployment
  claim.

## Resource bound and stop rule

The smoke uses one process and one free A100, at most 16 images and 512 new
tokens per image. Full execution uses the same shape on the 200-image panel.
No unrelated process is evicted.

Stop and classify the affected arm `invalid/uninterpretable` on composition
drift, missing companion identity, row/image mismatch, non-finite output,
parser/scorer failure, truncated decode, incomplete lifecycle artifacts, or
evaluator receipt mismatch. Repair a mechanical config/artifact issue once.
Do not enter hidden-state or causal-intervention probes unless valid behavioral
results leave at least two mechanism explanations that predict different next
observations.

## Artifact root

All new inputs, inference runs, evaluations, comparisons, receipts, and the
eventual result are rooted under:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe/`

## Completion

The complete bridge-enabled composition ran successfully on the independent
human-refined 13-image panel and on the supplementary val200 panel. The primary
result is a behavioral hold: bridge execution is proven, but step-611 does not
show useful dense recall. See [results.md](results.md). No source checkpoint was
rerun, and no relative recall-improvement claim is made.
