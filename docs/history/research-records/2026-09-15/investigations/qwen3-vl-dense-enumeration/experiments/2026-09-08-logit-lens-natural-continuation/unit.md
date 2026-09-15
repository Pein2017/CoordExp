---
title: One-shot direction graft followed by natural continuation
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-09-08-logit-lens-natural-continuation
status: complete
evidence_status: verified
architecture_promotion_status: not_promoted
updated: 2026-09-08
---

## Question and authorization

Completed and lead-accepted as a negative owner-benefit pilot; see
[results](results.md) and [acceptance](lead-acceptance.json). No successor.

From the frozen Source checkpoint and identical generated prefix, does one
block-27 overfit-direction graft change natural-continuation unique-owner
coverage relative to no graft? The user explicitly requested a quick check
after the prior branch closed. This is a separately authorized bounded
continuation, not an automatic expansion or a deployable-method claim.

## Frozen contrast

Reuse all 13 Human13 training images and checkpoint/input identities from
[radius-direction](../2026-09-08-logit-lens-radius-direction/unit.md).
For each image choose the first coordinate token of the middle completed row
of its already saved overfit trajectory (Image2299 middle group; other images
their existing middle-row group). Choose by position, never outcome or whether
endpoints differ. Both Source arms start from the identical prefix ending
before that token. Donor sees precisely that prefix/image, never future tokens.

Baseline: native Source greedy continuation. Treatment: preserve the Source
current-position radius and replace its direction with the overfit direction
at block 27 for one prefill only; remove the graft thereafter. Each arm follows
its own generated tokens with the native KV cache, without teacher forcing or
additional donor assistance. Keep prefix prefill cache consequences: this is
one state intervention, not an intervention designed to erase its consequences.

FP32/SDPA, RP1.0, deterministic greedy, same prompt/media as parent. Total
generated-output budget 768 tokens including the frozen prefix, so remaining
budget is 768 minus prefix length. Preserve EOS/cap identity and raw IDs/text.

## Decision-owning evidence

Primary: category-compatible one-to-one unique GT owner matching at pixel
IoU >=0.50, paired treatment-minus-baseline TP/FP/FN and gained/lost GT IDs.
Use the existing human-refined 13-image annotations and canonical coordinate
conversion/parser/matcher. IoU >=0.60 and >=0.80 are descriptive, not alternate
success gates. Score the complete prefix-plus-continuation output in both
arms; shared prefix predictions are included equally, not credited as new
generation. The selected partial row is completed freely in each arm.

Frozen GT file: parent `human-refined-13.geo_sorted_xy.coord.jsonl`, SHA256
`5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`;
13 images, 392 annotated owners. Owner identity is `(image_id, coco_ann_id)`;
negative human-added annotation IDs are preserved.

Guardrails: annotation-relative FP, strict duplicate candidates (unambiguous
same-category GT attribution IoU >=0.5, earlier prediction-pair pixel IoU
strictly >0.95), invalid/dropped outputs and token-cap hits. Unmatched
predictions are not automatically hallucinations. Preserve per-image outcomes
and exact gained/lost IDs; do not substitute token flips for owner gains.

A clean positive pilot requires positive net primary owner gain without an
increase in aggregate FP, strict duplicates, invalid/dropped predictions or
cap hits. Otherwise no clean owner benefit is established. No statistical
generalization claim from 13 fitted images is allowed even if positive.

Strongest alternative: local coordinate change is canceled by later generation
or merely redistributes owners; donor training-set fitting can also explain a
positive result. This tests local-to-continuation propagation, not a usable
held-out intervention or full generation from the original prompt.

## Execution and stop

One reused runner owner (`dynamics_runner`), one GPU0 process, <=48 GiB device
allocation, <=20 minutes GPU execution, at most two technical attempts.
Image2299 is the first production-shaped pair and is reused in the panel.
Require same-prefix identity, intended radius/direction, exactly one graft,
native/cache/self parity, and immutable input/code/raw-output receipts before
scientific interpretation. Lead independently scores and checks acceptance.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-natural-continuation/run-v1`.

No training, additional images, reverse arm, layer/dose sweep or automatic
successor. Stop after the fixed panel and synthesis or resource limit.
