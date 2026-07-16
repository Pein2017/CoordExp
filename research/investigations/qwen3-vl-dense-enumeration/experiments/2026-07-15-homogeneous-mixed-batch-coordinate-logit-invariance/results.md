---
title: Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-Logit Invariance Results
description: Verified one-event evidence that bfloat16 batch shape changes the complete coordinate distribution while full-model float32 restores practical batch invariance.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-Logit Invariance Results

## Verdict

For the exact image-7574 coherent-bowl recipient, Brain Floating Point 16-bit
(`bfloat16`) execution is strongly batch-shape dependent. Physical batch size
four moves probability mass from the original white bowl to the distinct orange
bowl even when all four requests are byte-identical. Neighboring prompt meaning
and target batch position are not required. Full-model Institute of Electrical
and Electronics Engineers 754 32-bit floating-point (`float32`) execution makes
the same recipient practically batch invariant and selects the orange-bowl mode
in every tested layout.

This is a verified execution-semantic result for one recipient. It is not
evidence that dense-scene low recall in general is caused by reduced precision.

## Conclusion-Owning Artifacts

Source-lineage `bfloat16` receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/
  source-bfloat16-natural-cached-direct-20260715b/receipt.json
```

Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
634ff2ecacd1ac84e43226e6925d73d33c5d623ec3ec3a0d62b19193c9bda36c
```

Full-model `float32` diagnostic receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/
  full-model-float32-natural-cached-direct-20260715a/receipt.json
```

`SHA-256` digest:

```text
74c9b47aa9329587719c3e9cf7a95916226b2af80d427c93adce56ac19f4cf68
```

Both receipts preserve the same base model, DoRA adapter, special-token
embedding delta, tokenizer, image, Scaled Dot-Product Attention (`SDPA`)
implementation, prompt hash, and generation policy. The declared and executed
model dtype is stored separately because dtype is intentionally not part of the
shared model-identity object.

## Trust-Gate Correction

The first attempted replay incorrectly assumed that the predecessor's physical
batch size four consisted of four homogeneous target prompts. It did not. The
original first batch was:

1. one 1,320-token no-row prompt;
2. the 1,330-token coherent target row;
3. the 1,330-token coherent other-object row; and
4. the 1,330-token target-description/other-geometry row.

The failed receipt is retained as negative execution evidence:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/
  source-bfloat16-cached-direct-20260715a/receipt.json
```

It stopped interpretation exactly as intended. The corrected primary path uses
natural cached generation for 16 tokens, requires the target recipient to emit
the frozen five-token row prefix without a constraint, and reproduces both
source coordinate bins:

- Single Target: coordinate bin `181` in both repeats;
- exact predecessor mixed-length batch at original target position one:
  coordinate bin `438` in both repeats.

## Source-Lineage bfloat16 Result

All same-layout repeats were exactly equal at the serialized float32-logit
artifact level.

| Cached layout | Selected first horizontal coordinate | White-bowl window mass | Orange-bowl window mass | White-minus-orange log-mass margin |
|---|---:|---:|---:|---:|
| Single Target | `181` | `0.441884` | `0.430928` | `+0.025106` |
| Predecessor Mixed-Length Batch Rotation | `438` | `0.115940` | `0.669296` | `-1.753157` |
| Homogeneous Target Copies | `406` | `0.125033` | `0.652119` | `-1.651647` |
| Equal-Length Mixed Rotation | `406` | `0.125033` | `0.652119` | `-1.651647` |

The exact coordinate bin is not the robust conclusion for batch size four.
Multiple coordinate logits are tied at `bfloat16` resolution, so ordinary
argmax tie-breaking selects different bins inside the same orange-bowl mode.
The complete vector and frozen-window mass are conclusion owning.

Relative to Single Target, Homogeneous Target Copies produce:

- centered root-mean-square coordinate-logit shift: `1.166623`;
- centered root-mean-square shift outside both frozen object windows:
  `1.139275`;
- centered maximum absolute shift: `3.220437`;
- Jensen-Shannon divergence: `0.117722`; and
- white-minus-orange log-mass shift: `-1.676753`.

This is a broad distribution change, not a tiny near-tie perturbation. The
homogeneous and equal-length mixed target vectors are exactly equal at every
target position. The predecessor mixed-length layout produces one additional,
smaller shift but remains invariant under all four cyclic batch rotations.

## Full-Model float32 Result

Every cached and direct recipient in every layout selected coordinate bin
`439`, including both repeats and all four target positions. Cached Single
Target versus Homogeneous Target Copies differs by only:

- centered root-mean-square coordinate-logit shift: `0.000012139`;
- centered maximum absolute shift: `0.000035286`; and
- white-minus-orange log-mass shift: `0.000000954`.

Cached Single Target versus the predecessor mixed-length batch differs by:

- centered root-mean-square coordinate-logit shift: `0.000053860`;
- centered maximum absolute shift: `0.000169754`; and
- white-minus-orange log-mass shift: `-0.000025749`.

All `float32` cached layouts put approximately `0.1053` conditional mass in the
white-bowl window and `0.6672` in the orange-bowl window. The selected bin has a
positive top-one margin of approximately `0.0575`, rather than the exact ties
seen in batch-four `bfloat16` execution.

## Supported

1. **Physical batch shape changes this bfloat16 model state.** Four identical
   target recipients are sufficient to move the complete coordinate
   distribution from the white-bowl basin to the orange-bowl basin.
2. **Semantic cross-request contamination is not required.** Homogeneous and
   equal-length semantically mixed batches produce the same target vector.
3. **Target batch position is not active in this panel.** Both mixed layouts are
   invariant under all four target positions.
4. **The shorter predecessor companion adds a smaller execution effect.** It
   changes the batch-four vector beyond homogeneous execution but does not
   change the selected object basin.
5. **Full-model float32 removes the material batch dependence.** All layouts
   converge to the same orange-bowl distribution within approximately
   `1e-4`-scale logit differences.
6. **The predecessor's apparent white-to-orange successor is not a stable
   semantic commit signature.** It depends on numerical execution precision
   and batch shape.

## Held or Rejected

Held:

- which subsystem first amplifies the `bfloat16` difference: vision encoding,
  multimodal projection/scatter, or the language decoder;
- whether this precision sensitivity recurs at other object decisions or dense
  scenes;
- whether higher precision improves end-to-end detection metrics; and
- whether the stable `float32` orange-bowl mode is semantically preferable to
  the `bfloat16` single-target white-bowl mode.

Rejected for this recipient:

- neighboring semantic content as the main cause;
- physical target batch position as the main cause;
- a narrow white-versus-orange near tie as the complete explanation; and
- the exact coordinate-bin identity as a reliable summary under tied
  `bfloat16` logits.

## Next Discriminator

Do not train or design an architecture from this result. The smallest next
experiment is [Single-Target Visual-Feature Replay into Homogeneous Batch
Four](../2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/unit.md).
Capture the exact batch-one primary and all three DeepStack outputs from
`get_image_features`, repeat them across downstream physical batch four, and
rerun natural cached generation plus direct full-prefix scoring.

If the replayed batch-four coordinate vector returns to the batch-one white-
bowl distribution, the vision output causally carries the batch split. If it
remains at the batch-four orange-bowl distribution, the source is after
`get_image_features`, and only then is a post-scatter replay justified. This
single causal replay is smaller and more decisive than an immediate
correlational all-layer activation sweep.
