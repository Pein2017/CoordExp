---
title: Image2299 G46 Payload Legacy-12 Transfer Results
description: Immutable result of the frozen R versus R+Q natural-greedy contrast on the twelve images excluded from Q authoring.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed_once
unit_id: 2026-08-31-image2299-g46-payload-legacy12-transfer
topic: qwen3-vl-dense-enumeration
status: completed_negative_transfer
evidence_status: immutable_paired_receipt
updated: 2026-08-31
---

# Image2299 G46 Payload Legacy-12 Transfer Results

## Decision

The fixed Image2299-authored payload `Q` caused strong negative natural-greedy
behavioral transfer on the exact twelve-image cohort excluded from Q authoring,
conditional on the exact Image2299-trained checkpoint `R`.

Strict same-category global one-to-one IoU50 coverage changed from `138/346`
under `R` to `18/346` under `R+Q`, so the registered primary statistic is
`delta50=-120`. Only 3 previously unmatched owners were gained, 15 native
owners were retained, and 123 native owners were lost. Aggregate hard debt
changed from `434` to `4004`; eleven of twelve `R+Q` legacy decodes reached the
token cap, compared with one of twelve under `R`.

This is a fixed-panel, fixed-checkpoint negative-transfer result. It is not
evidence about `Q` portability to step-2444, population or distributional
generalization, semantic identity representation, or the separate Human13
same-panel overfit experiment.

## Immutable evidence

- artifact root:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-g46-payload-legacy12-transfer/20260831T-image2299-g46-legacy12-transfer-v1`
- receipt: `receipt.json`
- receipt SHA-256:
  `ee4a2367d35927299a52798cb0053f8efb559cfbd6f1eda198acddbd324e53ca`
- status / verdict: `completed` / `degraded`
- model loads / natural generations: `1 / 27`
- treatment-hook calls: `34439`
- elapsed: `3224.8463635370135` seconds
- peak host RSS: `11555288` KiB
- exact Source A-B-A restoration: passed

The native Image2299 sentinel route matched
`5df0ac25aa871ddc0550298e70cd6012ce4dc3b6c6068b97dfa0cddca2f02a67`.
The `R+Q` Image2299 positive-control route matched
`96c2cbe15fdb4c09822d6a472f2510605c5701cfafcc9e33324f14ce90e0cb49`,
with `46/46` strict IoU50 owners, zero hard debt, and natural EOS. Its IoU60/80
diagnostics were `42/25`. After hook removal, the final Image2299 route,
parser-text hash, stop reason, and generated length exactly matched the initial
native sentinel.

## Aggregate readouts

| Arm | IoU50 | IoU60 | IoU80 | hard debt | all natural EOS |
| --- | ---: | ---: | ---: | ---: | --- |
| `R` | 138 | 120 | 73 | 434 | no |
| `R+Q` | 18 | 15 | 4 | 4004 | no |

The primary contrast is exhaustive over all 346 frozen owners; no image or
owner was removed after decoding.

## Per-image readouts

| image | owners | R IoU50 | R+Q IoU50 | delta | gained | retained | lost | R debt | R+Q debt | R EOS | R+Q EOS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1584 | 19 | 10 | 1 | -9 | 0 | 1 | 9 | 3 | 417 | yes | no |
| 2685 | 29 | 10 | 3 | -7 | 1 | 2 | 8 | 6 | 63 | yes | no |
| 4134 | 37 | 18 | 0 | -18 | 0 | 0 | 18 | 10 | 343 | yes | no |
| 5001 | 23 | 15 | 2 | -13 | 0 | 2 | 13 | 5 | 509 | yes | no |
| 6040 | 15 | 10 | 4 | -6 | 0 | 4 | 6 | 0 | 7 | yes | yes |
| 7511 | 44 | 0 | 2 | +2 | 2 | 0 | 0 | 344 | 249 | no | no |
| 10707 | 19 | 14 | 3 | -11 | 0 | 3 | 11 | 3 | 426 | yes | no |
| 13348 | 15 | 3 | 0 | -3 | 0 | 0 | 3 | 7 | 344 | yes | no |
| 13923 | 21 | 11 | 0 | -11 | 0 | 0 | 11 | 1 | 514 | yes | no |
| 14038 | 47 | 8 | 1 | -7 | 0 | 1 | 7 | 44 | 513 | yes | no |
| 14439 | 27 | 19 | 2 | -17 | 0 | 2 | 17 | 3 | 275 | yes | no |
| 16228 | 50 | 20 | 0 | -20 | 0 | 0 | 20 | 8 | 344 | yes | no |

The treatment diverged from native `R` within the first 1-10 generated tokens
for every legacy image. Eleven `R+Q` outputs generated the full 3084-token cap;
only Image6040 stopped naturally. This broad early divergence and exact
post-hook restoration favor payload-induced output-readout interference over
runtime-state contamination. That mechanism is an inference, not a proven
semantic explanation.

## Stop rule

The registered paired run is complete. No retry, RP sweep, scale/sign change,
row selection, refit, or favorable subset is admitted. A cross-checkpoint
`S/S+Q` portability study would be a new experiment with a new compatibility
contract, not a continuation of this result.
