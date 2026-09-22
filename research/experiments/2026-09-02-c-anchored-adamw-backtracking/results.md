---
title: C-Anchored AdamW Backtracking Results
description: Quarter-dose backtracking preserves all registered complete-row token margins and produces one natural IoU50 owner gain without a protected or total-coverage loss.
status: complete; mechanically valid; GO_C_ANCHORED_DDP_MULTISTEP
---

# C-Anchored AdamW Backtracking Results

## Verdict

`GO_C_ANCHORED_DDP_MULTISTEP` is mechanically valid on the frozen twelve-image
training/effect panel.  The first registered passing dose was `alpha=1/4`.
Its unmerged shared language-DoRA both passed the exact finite gate and, after
cold reload, gained one selected owner at IoU50 while retaining all three
protected owners and all 137 baseline IoU50 owners.

This is the first result in this lineage that joins shared multi-image finite
descent, complete-row token preservation, saved unmerged readback, and a
positive natural-greedy owner event.  It is still selected-panel evidence, not
held-out generalization or COCO-scale quality.

## Immutable evidence

- update receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-adamw-backtracking/authoritative-v1/receipt.json`,
  SHA-256
  `20e5e41aaaf71bf72cb7429da8b720a77815179f7efd741f768b2a3a28c55648`;
- update log:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-adamw-backtracking-authoritative-v1.log`,
  SHA-256
  `a82684dac4706f3bb0fb4e8f6c710a4362521ae55ce3fd6deebb6ed21bc5eb1f`;
- cold natural rows:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-adamw-backtracking/authoritative-v1/natural-eval/qwen3-vl-2b-c-anchored-adamw-backtracking-panel12/gt_vs_pred.jsonl`,
  SHA-256
  `6ed67da172a442920a3196089d1ce2066fa8c163bca2d69a145637a3cf904d9e`;
- evaluation receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-adamw-backtracking/authoritative-v1/evaluation.json`,
  SHA-256
  `399d090787e46dbdd58e6909c7ea7ee75d1ac87ccea4e12773cc0b2cd76de213`;
- successful cold-decode log:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-adamw-backtracking-natural-panel12-attempt3.log`,
  SHA-256
  `eec38293b039250aad943ebe4a6d7d7897349ba33dfe22d3c936ed3d55ce2a64`;
- update runner/config commit:
  `67bc75d7d8b85d85f1286a4224004dfca795d9ad`;
- safe hardlink materialization/final decode commit:
  `5b4ea2afec9602b537c64466edc9d0522765ac11`.

The selected adapter fingerprint is
`e2f8b1c5f395a7fab629bf5e7f3ed4c857635939bb04785e597c373124eafd57`;
its safetensors payload SHA-256 is
`7d52ba8f48192524dd29dc0299461dadc8e87cbe6f4ea57e8f2969b615354344`.
It remains a 588-tensor, rank-16 DoRA payload over the universal base; no
merged checkpoint was written.

## Deterministic finite selection

The full-dose proposal identity exactly reproduced the predecessor:
`q^T d=0.08304185`, norm `0.04236874`, and proposal SHA-256
`12628cf7269c3e51b32b746ae1708b97d091b52f9b7dcb97baf7aa9bf9039a76`.
Only two registered doses were evaluated:

| dose | mean gain loss | Armijo | minimum protected token margin | decision |
|---:|---:|---|---:|---|
| `1/2` | 1.690180 | pass | -0.037331 | reject |
| `1/4` | 1.710669 | pass | +0.014900 | accept |

The anchor mean was `1.731361`.  At the accepted dose all eleven individual
gain losses decreased; their changes range from `-0.03823` to `-0.00812`.
The half-dose failure is the same protected
`359310:coco_ann:1172698` description position and `umb` competitor exposed by
the full-dose result.  Backtracking crosses its finite safety boundary between
one half and one quarter without changing direction, objective, or parameter
surface.

The accepted realized delta has norm `0.01059019` and
`q^T delta=0.02075831`.  Its maximum storage-level deviation from one quarter
of the full proposal is `4.77e-7`.

## Cold natural behavior

Fresh inference proved exact equality of all 588 saved and materialized FP32
adapter tensors, zero dtype casts, `merged_adapters=[]`, frozen embedding-delta
identity, and the registered RP1/temperature-zero/max-3084 policy.  All twelve
rows ended naturally at `im_end`; there were zero invalid, malformed, parser,
image-validation, or truncation failures.

At the decision-owning IoU50 gate:

- selected-target uptake: `1/11`;
- protected retention: `3/3`;
- total matched owners: `137 -> 138`;
- gained owner: `270570:coco_ann:1140374`;
- lost owners: none;
- predictions: `204 -> 207`.

The gained owner is a `book` with target coordinate bins
`[569,20,711,110]`.  The updated natural row emits the selected complete row at
generated order 6 with bins `[571,20,715,112]` (pixel box
`[658,17,824,97]`), giving category-consistent IoU `0.938338`.  Thus the gain
is a direct natural realization of the trained row, not a global-matcher
exchange.

IoU60 also moves `127 -> 128` with the same sole gain and no loss.  IoU80 is a
monitor: `77 -> 78`, with two gains and one different loss.  Natural ordering
violations remain legal model behavior and are reported as 14 transitions on
6 rows; they are not an admission failure.

## Mechanical path incident

Two pre-decode attempts failed before model loading because the copied panel's
relative image paths first lacked a local image root and then correctly failed
the loader's symlink-escape guard.  Both failed run directories and logs are
preserved under the authoritative root.  The successful attempt used twelve
inode-identical hardlinks, retaining the path-containment invariant without
copying image bytes or weakening the loader.  No failed attempt produced model
behavior and neither enters the scientific denominator.

## Runtime and boundary

Finite selection used 42 forwards, 11 backwards, two tentative fresh AdamW
steps, and one retained update in `42.24 s`, with `26.56 GB` peak CUDA
allocation.  Cold decode generated 1,933 tokens in `130.36 s`, with `10.05 GB`
peak CUDA allocation.

The result supports a production-shaped two-rank, two-step DDP smoke through
the existing rollout-calibration trainer.  It does not yet support an 8-GPU or
full-COCO launch, a population recall claim, missing-label precision, or an
architecture promotion.  Projection remains unnecessary until a concrete
multistep preservation failure makes it decision-relevant.
