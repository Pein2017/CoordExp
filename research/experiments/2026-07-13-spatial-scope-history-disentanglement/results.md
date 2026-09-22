---
title: Masked Spatial Policy and Accepted-Row Prefix Policy Results
description: Dual-root evidence closeout for masked spatial restriction, accepted-row prefix accumulation, native-scale tiling, and equal-call full-image bagging on Dense-Union-51.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-13-spatial-scope-history-disentanglement
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Masked Spatial Policy and Accepted-Row Prefix Policy Results

This record closes the frozen protocol in [the research unit](unit.md). The
[readiness amendment](readiness-amendment.md) remains the immutable pre-output
contract; this file owns the executed facts and bounded interpretation.

## Post-Closeout Annotation Provenance Correction

The later [annotation review provenance and interpretation
correction](annotation-review-provenance-correction.md) establishes that
Independent Reviewer One was `gpt-5.6-sol` with `xhigh` reasoning effort, not a
human annotator using the planned local annotation tool. Its image-only labels
are high-recall model proposals that require human confirmation. Existing
metrics remain reproducible relative to the sealed audit ledger, but their
absolute precision and annotation-gap interpretation is reference-quality
sensitive. The legacy artifact key `manual_precision` is called
**audit-ledger precision** in current discussion.

## Terminology and Evidence Scope

- **Qwen3-VL — Qwen3 Vision-Language**: the pretrained multimodal model family
  under investigation.
- **COCO-80 — Common Objects in Context 80-category ontology**: the closed set
  of reportable categories used by this unit.
- **Dense-Union-51 — Annotation-Derived Dense Union of 51 Images**: the sealed
  dense-scene cohort selected without model outputs.
- **Bounding-box Intersection over Union (`IoU`)**: the overlap threshold used
  to match one prediction to one reference object.
- **Full-Image Single Rollout (`FULL_SINGLE`)**: one sampled complete-image
  rollout; it owns each root's missed-object rescue denominator.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: 16 independent
  complete-image rollouts followed by the frozen merge policy.
- **Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)**: 16
  full-size masked canvases, each decoded from a fresh prompt.
- **Full-Canvas Masked Region with Cumulative Accepted-Row Prefix
  (`MASK_CUMULATIVE`)**: the same 16 masked canvases decoded while carrying the
  frozen accepted-row prompt state forward.
- **Native-Scale Tile with Per-Tile Reset (`TILE_RESET`)**: 16 unresized
  core-plus-halo tiles, each decoded from a fresh prompt.
- **Seed root**: the root pseudo-random seed from which every request seed is
  deterministically derived. The two roots below are analyzed separately and
  are never pooled.
- **Owning-Seed Raw Rescue Difference**: the rescue rate of an object's one raw
  owning masked call minus the rescue rate of its one seed-matched raw
  full-image call, conditioned on `FULL_SINGLE` missing that object.
- **Policy-Utility Rescue Difference**: the difference between two arms'
  post-merge Local Rescue Rates, conditioned on `FULL_SINGLE` missing the
  object.
- **Audit-augmented ledger**: the sealed official annotations plus accepted,
  model-proposed and adjudicated image-only additions. It owns the executed
  scientific headline but is not human-confirmed ground truth.
- **95-percent image-clustered bootstrap confidence interval**: the uncertainty
  interval from 10,000 resamples of whole images.

At the model-execution layer, both roots used the same Dense-Union-51 cohort,
Qwen3-VL checkpoint, four-by-four spatial grid, five-arm request topology,
decode contract with sampling temperature `0.4`, neutral repetition penalty
`1.0`, parser, merger, matcher, and audit ledger; only the seed root changed
from `2026071301` to `2026071302`. Each root executed 3,315 requests, and all
3,315 attempts completed. The post-run assemblies used source commits
`211949f3e2c7c919201e242b8ea675e022686944` and
`6c12b43220d082c54147e71e15a14b7c43198c80`, respectively. Their only relevant
source difference derives every bootstrap sampling stream from the schedule
root seed; point-estimator formulas were unchanged. The two roots therefore
retain separate, root-specific bootstrap streams rather than sharing the old
default stream.

The metric named `raw_any_call_union_rescue_rate` in the artifacts means joint
one-to-one reference matching over all pooled pre-merge raw predictions. It is
not a literal union of separately matched call-level reference sets. The
Owning-Seed Raw Rescue Difference is the separate one-call matched-opportunity
estimand used below.

## Canonical Artifact Handles

### Seed root `2026071301`

Canonical post-run root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/postrun/dense-union-51-primary-after-wave-local-tail-contract
```

- [Post-run assembly receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/postrun/dense-union-51-primary-after-wave-local-tail-contract/postrun-assembly-receipt.json): status `complete`; receipt identity Secure Hash Algorithm 256-bit (`SHA-256`) digest
  `9473d2169a65b89fae56013f7c7b7e42e584c74b4657c3c2627ffa5597722ca3`;
  receipt-file SHA-256 digest
  `257a99ed475c7f6c2da4ae194b9031eae4a287edef522b9a20f9e24b00894c2a`.
- Schedule fingerprint:
  `e306f2daf3a290fb366bb3f05a6ab878e2c541ad371b0d841ab7474b245289ba`.
- Attempt-ledger SHA-256 digest:
  `7cc0760a85f988cb192b451cb4219dab1f5e199861471bce2ac41ca438ef4711`.
- Terminal-bundle-set SHA-256 digest:
  `98d0da795bc5cab923b71a83e3df51f3039f0a0eb151137f3e419449ed4bed55`.
- Metric artifacts: aggregate reports
  `a2e0eb93f256d445d17ca6d9adf4baf66c9d8667348fbec47a2bd7c77598aec3`;
  arm merge
  `8bd5f292e9cc5fa0e9c089f63432ee38de41d8eeefb106f8977fd16f33890ad8`;
  bootstrap reports
  `83f20cfb70fad91e1f8b2fef4b965b3f19f1851a3c038209f6eb65c30ca7a256`;
  image primitives
  `f671bb13fa64308c2278a46188e0c22fbb441fbdf444dbc587f06a402ef6f2b7`.

### Seed root `2026071302`

Canonical post-run root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/postrun/dense-union-51-second-root-2026071302-after-wave-local-tail-contract
```

- [Post-run assembly receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/postrun/dense-union-51-second-root-2026071302-after-wave-local-tail-contract/postrun-assembly-receipt.json): status `complete`; receipt identity SHA-256 digest
  `3e57319485fdea78ed8df4b91c15b641d4c857f8ab03e74410feacb2d79ba53f`;
  receipt-file SHA-256 digest
  `a815fa52001db0788871d00c8803c166d13ab13d871347834d70c7d2fb769e36`.
- Schedule fingerprint:
  `3e4d59ea1650a7bf927e0d0c5d43cc5a27309277d1ce5022ac6f50677fdd4fe9`.
- Attempt-ledger SHA-256 digest:
  `8720a150ae2e281acb008e52ea112ee19411c3c1b113cab29077b97bffa6f73f`.
- Terminal-bundle-set SHA-256 digest:
  `902f934bd7b1a259f500386be1488fe74ef3824004a725b0c20eaa05a45c4a3b`.
- Metric artifacts: aggregate reports
  `153dbc1608d2f44a8e91ea7beb737561c7eb1798998e1c779e6a73e3b8de1c3c`;
  arm merge
  `56c70ceaf8c7edf5364623e74d92934e0cb132181dada48f5777f89f45856fd0`;
  bootstrap reports
  `176a2c71dfbbea33ac9b137ad7a52601eb01474c04b1ffa582e84151d32b6cb9`;
  image primitives
  `e16d71ebc482204f20ce497f8cdda06addc5d15cb588b9cca03ca6cd43ea7535`.

Both roots bind to the same sealed audit reference ledger SHA-256 digest
`52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df`
and readiness seal SHA-256 digest
`c177bc2b12f06fed559660bb5e1d7ff24fa19ee5eb898e642cae27d6991ebb68`.

## Observed Headline Contrasts

All values below use the audit-augmented ledger. Positive values favor the
first named arm. Intervals are 95-percent image-clustered bootstrap confidence
intervals. Roots remain separate.

### Bounding-box Intersection over Union `0.50`

| Contrast | Seed root `2026071301` | Seed root `2026071302` |
|---|---:|---:|
| Owning-Seed Raw Rescue Difference: `MASK_RESET` minus seed-matched `FULL_BAG_K` call | `+0.193764` [`+0.127071`, `+0.256250`] | `+0.207373` [`+0.121738`, `+0.290124`] |
| Policy-Utility Rescue Difference: `MASK_RESET` minus `FULL_BAG_K` | `-0.057906` [`-0.108110`, `-0.013376`] | `-0.029954` [`-0.100000`, `+0.029480`] |
| Policy-Utility Rescue Difference: `MASK_RESET` minus `MASK_CUMULATIVE` | `+0.202673` [`+0.122507`, `+0.288719`] | `+0.193548` [`+0.115195`, `+0.275864`] |
| Policy-Utility Rescue Difference: `TILE_RESET` minus `MASK_RESET` | `-0.069042` [`-0.116424`, `-0.022420`] | `-0.096774` [`-0.140704`, `-0.054283`] |

The `FULL_SINGLE` missed-object denominators were 449 and 434, both above the
frozen minimum of 150.

### Bounding-box Intersection over Union `0.75` sensitivity

| Contrast | Seed root `2026071301` | Seed root `2026071302` |
|---|---:|---:|
| Owning-Seed Raw Rescue Difference: `MASK_RESET` minus seed-matched `FULL_BAG_K` call | `+0.101329` [`+0.064281`, `+0.135783`] | `+0.092127` [`+0.052224`, `+0.129252`] |
| Policy-Utility Rescue Difference: `MASK_RESET` minus `FULL_BAG_K` | `-0.023256` [`-0.059524`, `+0.012918`] | `-0.035176` [`-0.076772`, `+0.003498`] |
| Policy-Utility Rescue Difference: `MASK_RESET` minus `MASK_CUMULATIVE` | `+0.121262` [`+0.079744`, `+0.162205`] | `+0.102178` [`+0.066264`, `+0.133794`] |
| Policy-Utility Rescue Difference: `TILE_RESET` minus `MASK_RESET` | `-0.063123` [`-0.100504`, `-0.024911`] | `-0.061977` [`-0.095810`, `-0.026446`] |

The direction pattern therefore reproduced at both matching thresholds and
both seed roots: masked spatial restriction helped one seed-matched owning
opportunity, but did not improve the final merged policy over equal-call
full-image bagging; cumulative accepted-row prompting was worse than reset; and
native-scale tiling was worse than the masked full canvas.

## Safety and Mechanics

The local rescue signal did not pass the final-policy safety contract.

| `MASK_RESET` audit-augmented safety measure at Intersection over Union `0.50` | Frozen rule | Seed root `2026071301` | Seed root `2026071302` |
|---|---:|---:|---:|
| Overall-retention lower confidence bound | at least `0.85` | `0.791841` | `0.785288` |
| Mask-harm-retention lower confidence bound | at least `0.80` | `0.740000` | `0.746662` |
| Audit-ledger precision point estimate (legacy artifact key: `manual_precision`) | at least `0.70` | `0.466503` | `0.459459` |
| Prediction-count-inflation upper confidence bound | at most `2.0` | `2.253747` | `2.348620` |

`FULL_BAG_K` also failed the absolute audit-ledger-precision and
prediction-count-inflation rules in both roots. Post-merge strict duplicates,
invalid rows, and natural closure did not bind the decision. Relative to the
sealed audit ledger, the observed pattern is unsafe output expansion and
retention harm, not improved enumeration. The absolute precision and
annotation-gap interpretation remains conditional on reference quality.

The primary root's `FULL_BAG_K` Prediction-Set Diversity was `0.179404`, above
the frozen `0.15` weak-diversity floor. Its [derived diversity artifact](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/postrun/dense-union-51-primary-after-wave-local-tail-contract-derived/primary-full-bag-prediction-set-diversity.json)
has artifact identity SHA-256 digest
`2d6eb24b518a23608d05ea9343385928115dfa27ba498f13174bb4669e88c371`
and file SHA-256 digest
`be1353a65ec6dc2f42c00eac2cc2e45554aaecc4b4e4e92e19b6d35b0a5630c5`.
Thus equal-call bagging was a valid stochastic comparator, not a repeated-
deterministic-call control.

The primary root's first 48 images used physical batch size four throughout.
Its [cardinality-matched sensitivity artifact](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/postrun/dense-union-51-primary-after-wave-local-tail-contract-cardinality-matched-first48/cardinality-matched-sensitivity.json)
preserved every headline direction. The artifact identity SHA-256 digest is
`1e9a503e18cc68081342ad3df7fd3a3bb2bb1668999a8fba36e9d254ac00047a`;
the file SHA-256 digest is
`3e143bab231d706c7fed225dfd6d136b1f8ad43a6c51f58ad2e0fa4185e06e75`.
This lowers concern that the primary directional result was created by its 17
natural batch-size-three partition tails; it is not a replacement for the
complete dual-root result.

## Bounded Verdict

### Observed

- Full-canvas spatial restriction reproducibly changes which object is rescued
  in one seed-matched raw opportunity.
- Repeating full-image sampling recovers more or comparable final post-merge
  utility than the masked policy despite the masked one-opportunity advantage.
- Carrying accepted rows through the complete frozen cumulative prompt policy
  substantially reduces rescue relative to reset.
- Native-scale tiles are consistently worse than full-canvas masking under the
  declared core-plus-halo protocol.
- The multi-call policies expand prediction counts, and spatial policies lose
  objects already found by `FULL_SINGLE`.

### Supported

- **A local input-level spatial-restriction effect is supported.** This is a
  behavior-level effect of a pixel-masked input, not evidence that competition
  occurs after the vision tower.
- **Harm from the complete cumulative accepted-row prefix policy is
  supported.** The intervention jointly changes length, content, correctness,
  order, and visibility consistency; the result is not evidence for a
  token-length-only mechanism.

### Ruled Out Under This Protocol

- `MASK_RESET` did not safely outperform equal-call `FULL_BAG_K` as a final
  merged policy. The promotion criterion for improved enumeration is not met.
- `TILE_RESET` did not outperform `MASK_RESET`; a native-tile advantage is not
  present in either seed root.
- The result is not explained by a weak-diversity bagging comparator.

These statements are local to the frozen checkpoint, Dense-Union-51 cohort,
four-by-four grid, mask construction, sampling policy, and merger. They do not
rule out every spatial intervention.

### Unresolved

- Whether candidate competition occurs after one fixed full-image visual
  encoding.
- Whether the harmful cumulative-policy effect comes from history length,
  semantic content, prior-row errors, ordering, visibility inconsistency, or
  their interaction.
- Whether a compact ledger, commit state, or other architectural mechanism is
  necessary.
- Whether a language-only mechanism explains dense-scene degradation.
- Whether entity-level and geometry-level human confirmation of the model-
  proposed review additions changes the absolute safety estimates or headline
  effect sizes.

### Not Claimed

No model architecture, slot representation, ledger, visual cursor, training
objective, or production policy is promoted. The result does not establish
that the vision tower already represents every rescued object, nor that the
language decoder is the sole bottleneck.

## Next Smallest Discriminator

The next unit should first compare pixel masking with spatial restriction
applied after one fixed full-image visual encoding. This directly asks whether
the reproducible local effect survives when the vision-tower representation is
held fixed. It should then, as a separate unit or separately controlled panel,
compare reset with length-matched correct, irrelevant, shuffled, and corrupted
prefixes under the same visual condition. No architecture design is authorized
before these two discriminators localize the effect.
