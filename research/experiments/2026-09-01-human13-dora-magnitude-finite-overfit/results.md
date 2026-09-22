---
title: Human13 DoRA Magnitude Finite Overfit Results
description: One unmerged shared magnitude-only DoRA adapter passes the complete fixed-panel Human13 canonical and natural-greedy gate.
type: investigation
role: research-result
authority: non_normative_research
status: completed_human13_magnitude_overfit_pass
evidence_status: immutable_n13_primary_acceptance
updated: 2026-09-01
---

# Human13 DoRA Magnitude Finite Overfit Results

## Decision

`N13_PASS` fires the registered success stop. From the exact step-2444 Source,
changing only the 196 shared language-DoRA magnitude vectors produced one
standard unmerged adapter whose fresh natural greedy decode covered all
`392 / 392` annotated owners on the fixed Human13 panel without hard debt.

The primary RP1.0 result is `392 / 392` at each of IoU50, IoU60, and IoU80,
with zero duplicate, unmatched, or malformed rows, natural EOS on all thirteen
images, and zero natural ordering violations. Canonical cold readback remained
full-vocabulary exhaustive and had minimum target margin
`0.09504318237304688`, above the frozen `0.00998` gate.

This is fixed-panel internal-network finite programmability only. It is not
held-out generalization, semantic sharing, population recall, DDP
qualification, architecture promotion, or production readiness.

## Frozen identity

- Source checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- Source adapter SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`
- selected-token delta SHA-256:
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`
- Human13 panel SHA-256:
  `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`
- data authority: COCO `resize=1024`, global max length `12000`, SHA-256
  `81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894`
- trainable surface: 196 magnitude tensors / 573,440 scalars; all DoRA A/B,
  base, vision, multimodal, selected-token-delta, embedding, and output-head
  tensors frozen.

Every stage started independently from this Source. N2 and N4 adapters were
never used to initialize N4 or N13.

## Stage evidence

| Stage | Images / owners / decisions | AdamW steps | magnitude `L2` | cold minimum margin | RP1.0 IoU50 / 60 / 80 | debt / EOS | RP1.10 IoU50 / 60 / 80 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| N2 | 2 / 65 / 592 | 70 | 41.5285 | 0.188885 | 65 / 65 / 65 | 0 / 2 of 2 | 65 / 65 / 65 |
| N4 | 4 / 123 / 1,147 | 90 | 52.8832 | 0.061604 | 123 / 123 / 123 | 0 / 4 of 4 | 123 / 123 / 123 |
| N13 | 13 / 392 / 3,637 | 140 | 76.2576 | 0.095043 | 392 / 392 / 392 | 0 / 13 of 13 | 384 / 383 / 382 |

RP1.10 and natural row ordering are monitors, not gates. RP1.0 had zero
ordering violations at every stage. RP1.10 reported two violation rows at N2,
two at N4, and seven across three images at N13.

The observed magnitude norm grows much more slowly than the number of
canonical decisions across these three nested points. A descriptive log-log fit
gives `||delta m||_2 approximately 5.00 * P^0.333`, where `P` is the decision
count. With only three non-independent nested stages, this is a monitor for a
future registered scaling study, not a scaling law or theorem.

## N13 artifact receipt

- artifact root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-human13-dora-magnitude-finite-overfit/20260901T-n13-adamw-ce-v1`
- candidate receipt file / content SHA-256:
  `57ee541189d37848af9fb2f6d9e11d9f78d3f9424d1dd7fb8b3d22cc958a7acb` /
  `bd11b41597c0308e209078e71da42d33dbe91da312e82ba943e8dc90e2bdbff8`
- magnitude candidate payload SHA-256:
  `4bad9e8d67f0a519f0c2443d450d41695d2035b2ea8661b4327cd339ea155680`
- materialization receipt SHA-256:
  `91e29bdf86bcca435ed40968a0627d4934d6630fccb0f4f922d6aeea67b61969`
- unmerged adapter model SHA-256 / fingerprint:
  `8a5ebfcacfa92be4b873fea4439fc25570a9c415be2245e20fd1da94c9ff4070` /
  `4c36daa74c8d7d113a56fbece284c7fd016a5f5360899d776752c707f15dd5e9`
- adapter payload: 588 tensors, 72,111,984 bytes; Source A/B hash preserved;
  cold live state exactly matched the saved unmerged payload.
- acceptance receipt file / content SHA-256:
  `f3de22186dcbeec1682ad7db431c414f525b0a292c857038c95d8ba17b4045a0` /
  `20a80f2d04bbd908ea9ab7874057018a9b62acad742b79348af1273cf5f4b804`
- frozen train / readback launch packet SHA-256:
  `630a1460e97b22f0a0816018497c91f0a14061c658a38e6d30f312bf95e9f7ba` /
  `809bb5cf61a5bfef2fc25d9c6b60245ec901190d8bebd493e8748b9f878caf47`

The N13 training used one model load, 140 optimizer steps, 1,820 training
language forwards, and 32,312,918,016 peak GPU-reserved bytes. Cold readback
used one model load, 13 canonical forwards, 26 natural generations, and
16,131,293,184 peak GPU-reserved bytes.

## Stop boundary

The same-panel overfit question is answered. The exact-strata permutation,
matched-swap, QP/Farkas, and distributed solver routes remain deferred and do
not need to be built to sustain this result. A COCO-scale C-versus-D0 training
and image-disjoint evaluation is a separate successor with its own generalization
estimand. `HOLD_PRODUCTION` and architecture non-promotion remain in force.
