---
title: Image2299 G46 Payload Legacy-12 Transfer
description: Frozen R versus R+Q natural-greedy transfer contrast on the 12 Human13 images excluded from Q authoring.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed_once
unit_id: 2026-08-31-image2299-g46-payload-legacy12-transfer
topic: qwen3-vl-dense-enumeration
status: completed_negative_transfer
evidence_status: immutable_paired_receipt
updated: 2026-08-31
---

# Image2299 G46 Payload Legacy-12 Transfer

## Frozen question and claim boundary

Conditional on the exact Image2299-trained checkpoint `R`, does adding the
immutable Image2299-authored output payload `Q` improve fresh natural-greedy
strict-owner coverage on the exact 12 legacy images excluded from Q authoring?

This is a transfer probe from one authored image to a fixed legacy cohort. It
is not an `S`/`S+Q` contrast, an RP sweep, a scale/sign/subset search, a refit,
or evidence of distributional generalization. The strongest permitted positive
result is improved strict IoU50 coverage on these frozen 346 owners under the
registered intervention. IoU60/80 and debt are diagnostics from the same
decodes.

## Frozen identities

- `R` checkpoint:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-step-v1/checkpoint-selected-r32-step`
- parent receipt SHA-256:
  `a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c`
- adapter tensor / config SHA-256:
  `8b448435b0162de6a11ffa10b30da76cdd714dd9f9e74d77b4c70109d63b0d53` /
  `3f6f4f7da0e63c3fdffd4904c0540c9ba6ee5a6e7af38deb6a4d8c078c1a33b9`
- special embedding tensor / metadata SHA-256:
  `9d6e253f810d9074ef8552fa132926bb197c792ec7de06c2ecaec1a73aae6b25` /
  `7a465c6b7d43c5a46066d99a1076e96fa26dc0fe30f7ca69fc289ed347b30983`
- `Q` payload:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/20260831T-image2299-canonical-g46-global-qp-v3/g46_global_qp_protected_null_output_residual.safetensors`
- `Q` payload / receipt / metadata SHA-256:
  `22d9df392f586979e8b191a89068ea929943356d8bfce86856e8203c0e0f51fd` /
  `ff04300c19eb15ac0f6193ecace0a475e4e023b398eece913e6b7dda568a8a5d` /
  `b0fc581e7ce55987c8678651efff2ff50a4a783d0ccccfcc47a5be6fe6645369`
- `Q`: FP64 `51 x 2048`; selected-set receipt SHA-256:
  `2c7f82ee089f146308cc9f6e45dd5e7a78b5c18c73fc27cd442c600006ba5ab5`
- current `geo_sorted_xy` panel / legacy-12 / ordering receipt SHA-256:
  `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23` /
  `cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85` /
  `cd1273f627f7bdfcb16e9ca6e4a50e9d51f0163081ea7458d6d3deabe9bb82c0`

The ordered cohort is `1584, 2685, 4134, 5001, 6040, 7511, 10707,
13348, 13923, 14038, 14439, 16228`, with exactly 346 owners.

## Registered protocol

Use one Hugging Face FP32/SDPA model load, physical batch size one, RP `1.0`,
temperature `0`, top-p `1`, and the current experiment-local decode/evaluator
seams. Run exactly 27 natural decodes in this order:

1. Image2299 under native `R` as the sentinel;
2. the ordered legacy-12 under `R`;
3. install immutable `Q` through `SelectedOutputRowsHook`;
4. Image2299 under `R+Q` as the positive control;
5. the same ordered legacy-12 under `R+Q`;
6. remove the hook; and
7. Image2299 under restored `R`.

The initial Image2299 route SHA-256 must be
`5df0ac25aa871ddc0550298e70cd6012ce4dc3b6c6068b97dfa0cddca2f02a67`.
The positive-control route SHA-256 must be
`96c2cbe15fdb4c09822d6a472f2510605c5701cfafcc9e33324f14ce90e0cb49`,
with `46/46` strict IoU50 owners, zero hard debt, and natural EOS. The final
restoration identity must equal the initial `R` identity exactly.

## Evaluation and decision

The primary statistic is

`delta50 = sum(strict IoU50(R+Q)) - sum(strict IoU50(R))`.

The immutable receipt also reports IoU60/80 totals, per-image arm counts,
gained/retained/lost IoU50 owner IDs, duplicate/unmatched/malformed/cap debt,
EOS, first token divergence, and generated lengths. Owner IDs come from the
stable global matcher's owner-index mapping; count disagreement fails closed.

- `delta50 > 0`: improved on the frozen legacy-12 cohort.
- `delta50 = 0`: unchanged.
- `delta50 < 0`: degraded.

Any binding, denominator, initial sentinel, positive control, hook-removal, or
restoration failure is `MECHANICAL_INVALID` and suppresses the legacy verdict.
The receipt is no-clobber. No retry, alternate payload, or post-hoc variant is
registered.

## Registered commands

CPU-only admission:

```bash
conda run -n ms python scripts/research/run_image2299_g46_payload_legacy12_transfer.py --check-bindings
```

The one registered model run used:

```bash
conda run -n ms python scripts/research/run_image2299_g46_payload_legacy12_transfer.py run --output-dir <new-output-dir>
```

No retry or parameter variant is admitted after this completed run.

## Execution receipt

On 2026-08-31 the CPU-only binding admission passed all registered hashes,
panel/image-byte identities, the 346-owner denominator, and the FP64 `51 x
2048` payload shape. The focused test file passed `8/8`; `py_compile`, Ruff,
and the scoped whitespace/diff check also passed.

The one registered model run then completed all 27 decodes with one model load.
Its immutable receipt is summarized in [results.md](results.md). The frozen
legacy-12 contrast was strongly negative: strict IoU50 coverage changed from
`138/346` under `R` to `18/346` under `R+Q` (`delta50=-120`), while aggregate
hard debt changed from `434` to `4004`. The Image2299 positive control passed,
and exact `R -> R+Q -> R` restoration passed. This supports negative behavioral
transfer of this payload on this cohort, conditional on `R`; it does not speak
to step-2444 portability or distributional generalization.
