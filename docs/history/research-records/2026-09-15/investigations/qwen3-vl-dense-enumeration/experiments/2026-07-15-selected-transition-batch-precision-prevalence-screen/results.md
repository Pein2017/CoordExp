---
title: Selected-Transition Batch-Precision Prevalence Screen Results
description: Six-case evidence separating recurrent low-precision token jitter from a material next-object or coordinate-basin switch.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-15-selected-transition-batch-precision-prevalence-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Selected-Transition Batch-Precision Prevalence Screen Results

## Verdict

Physical batch shape produces recurrent, precision-conditioned numeric
differences, but this selected six-case screen did **not** reproduce a material
next-object, stop-versus-continue, description, or coordinate-basin switch at
the first free action.

Under Brain Floating Point 16-bit (`bfloat16`), three of six recipients differed
exactly between physical batch one and four homogeneous copies. All three kept
the same action type, object description, and physical instance; their boxes
differed by only one or two source-image pixels. Full-model Institute of
Electrical and Electronics Engineers 754 32-bit floating point (`float32`)
made the promoted three cases byte-identical across batch shapes for the entire
executed `64`-token continuation.

The literal exact-token recurrence gate therefore fires, but the
research-routing materiality gate does not. A language-layer onset sweep on
these local coordinate jitters would not address the dense-enumeration
question. One bounded full-coordinate-logit and same-layout-repeat discriminator
remains before this numeric branch is either closed or promoted.

## Conclusion-Owning Artifacts

Six-case `bfloat16` screen:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-selected-transition-batch-precision-prevalence-screen/
  source-bfloat16-selected-six-20260715a/receipt.json
```

Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
3d3ebbd6197e54ec3cc21475dfa4e91793027c275ed2e6061c09107f8173c0e3
```

Promoted three-case full-model `float32` control:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-selected-transition-batch-precision-prevalence-screen/
  full-model-float32-promoted-three-20260715a/receipt.json
```

`SHA-256` digest:

```text
a66cebfa5fca80fbc3f7269444c334f78279dea79b8a91144810e965bccbf74f
```

An earlier one-image smoke receipt is superseded by the six-case run:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-selected-transition-batch-precision-prevalence-screen/
  source-bfloat16-smoke-image-7574-20260715b/receipt.json
```

## Executed Scope

- Qwen3 Vision-Language (`Qwen3-VL`) 2B, step-4887
  Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter;
- geometry-sorted adapter and source trajectories;
- canonical Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`) cell zero,
  where `K` is the frozen set of `16` independent full-image sampled calls;
- first complete model-native source row appended to its exact source prompt;
- greedy decode, repetition penalty `1.0`, maximum `64` generated tokens;
- physical batch one versus four homogeneous copies;
- selected images `7574`, `8629`, `9891`, `12576`, `13659`, and `17714`;
- `float32` promoted only for the three `bfloat16` first-action divergences.

All six source bundles passed model, tokenizer, generation-config, source-image,
prompt-reconstruction, source-arm, and source-cell checks. Within every
executed physical batch four, all four copies produced exactly the same first
action and exactly the same complete `64`-token continuation.

## First-Action Result

| Image | Exact `bfloat16` first-action divergence | Batch-one action | Batch-four action | Material interpretation |
|---:|---|---|---|---|
| `7574` | no | `bowl [187,137,268,162]` | same | invariant first action |
| `8629` | yes | `pizza [686,31,997,302]` | `pizza [686,31,996,302]` | same pizza; one-pixel right-edge jitter |
| `9891` | no | `person [371,167,621,753]` | same | invariant first action |
| `12576` | no | `tv [751,69,863,228]` | same | invariant first action |
| `13659` | yes | `person [927,131,1151,543]` | `person [925,131,1151,543]` | same person; two-pixel left-edge jitter |
| `17714` | yes | `cup [358,213,553,368]` | `cup [357,213,553,367]` | same cup; one-pixel left and bottom jitter |

No case changed action status, description, or visually selected instance at
the first action. These six complete-row cell-zero transitions contain no new
material switch. They do not replicate or refute the exact predecessor, which
used a different cell-three source bundle and a partial-row first-coordinate
recipient.

## `float32` Control

| Image | Batch-one `float32` action | Batch-four `float32` action | Complete `64`-token equality |
|---:|---|---|---|
| `8629` | `pizza [688,33,997,302]` | exact same | yes |
| `13659` | `person [927,131,1151,543]` | exact same | yes |
| `17714` | `cup [358,213,553,368]` | exact same | yes |

Thus full-model `float32` removed both the primary one-to-two-pixel jitter and
every later batch-shape difference in the promoted cases. This is evidence for
precision conditioning, not evidence that any one language layer causes the
effect.

## Secondary Continuation Evidence

The primary endpoint was the first complete action. The raw continuations also
show that `bfloat16` batch shape perturbs later coordinate tokens even when the
first action is invariant.

| Image | Zero-based first differing generated-token index | First-action divergence | Parsed object-description sequence |
|---:|---:|---|---|
| `7574` | `15` | no | same six descriptions |
| `8629` | `6` | yes | same seven descriptions |
| `9891` | `23` | no | same seven descriptions |
| `12576` | `16` | no | same six descriptions |
| `13659` | `4` | yes | six shared descriptions; one capped continuation parsed an additional trailing chair row |
| `17714` | `4` | yes | same seven descriptions |

All six batch-one and batch-four `bfloat16` continuations differed exactly
within `64` tokens. Five retained the same parsed description sequence. Image
`13659` had a different parsed row count at the hard token cap, so it cannot be
interpreted as a natural termination difference. The promoted `float32` cases
were completely identical for all `64` tokens.

This secondary observation supports recurrent low-precision trajectory
sensitivity. It does not show that object selection, coverage, or natural stop
behavior changes materially.

## Implementation Correction

The first smoke attempt incorrectly included request-owned `object_span_id` in
the semantic action hash. Homogeneous requests necessarily have distinct
request identifiers, so the comparison rejected valid equal actions before a
receipt was written. The corrected comparator projects only status,
description, geometry, token span, and action tokens; request-owned identifiers
are retained in raw evidence but excluded from equality. Dedicated tests cover
this case.

The repaired runner passed:

```text
11 passed in 0.08s
```

The correction occurred before the conclusion-owning six-case run and does not
alter any reported model output.

## Bounded Verdict

### Observed

- Exact first-action `bfloat16` batch sensitivity occurred in three of six
  deliberately selected recipients.
- Every first-action difference was local coordinate jitter on the same object.
- All six `bfloat16` continuations eventually differed at a coordinate token.
- Full-model `float32` made the promoted three continuations exactly invariant
  across physical batch shapes.

### Supported

1. Physical batch shape and execution dtype are part of the effective runtime
   contract for conclusion-critical mechanistic probes.
2. Low-precision batch-shape numerical sensitivity recurs beyond the original
   bowl recipient at the exact token level.
3. The selected screen provides no evidence that such sensitivity commonly
   changes the first selected object, action type, or meaningful geometry.
4. Full-model `float32` is an appropriate diagnostic control when a claimed
   mechanism depends on an exact low-margin token transition.

### Ruled Out or Demoted

- The six-case screen does not support treating the predecessor white-to-orange
  bowl switch as representative of dense next-object transitions.
- Exact-token recurrence alone is insufficient reason to localize language
  layers or redesign the model.
- This unit does not rescue the earlier native-commit interpretation; it
  reinforces that the dramatic predecessor transition was execution-sensitive.

### Unresolved

- Whether the three micro-divergences are repeat-stable broad coordinate-logit
  shifts or near-tie numerical noise; this unit has no same-layout repeats or
  complete 1,000-coordinate-logit vectors.
- Whether low-precision batch shape ever changes natural long-rollout object
  selection or termination often enough to affect detection quality.
- Whether the first numeric difference for other images is before or after
  `get_image_features`; exact post-vision ownership remains proven only for the
  predecessor recipient.
- Whether production `float32` inference changes Average Precision, recall,
  runtime, or calibration.
- The primary cause of dense low recall, duplicate bursts, and incomplete
  enumeration.

### Not Claimed

- population prevalence;
- production dtype recommendation;
- language-tower root cause;
- explanation of dense low recall;
- architecture or training authorization.

## Independent Audit

An independent Sol model-diagnosis audit recomputed the source and receipt
hashes, action hashes, coordinate-bin and pixel deltas, intersection-over-union
values, first differing token positions, parser counters, description
sequences, and `float32` equality. It found no artifact or implementation
blocker.

The audit identified two conclusion boundaries:

1. the formal gate detects exact token inequality, not material basin movement;
2. homogeneous copies inside one batch are not independent same-layout repeats,
   and argmax outputs cannot reveal whether the underlying coordinate
   distribution moves broadly.

It also confirmed that image `7574` is only an image-level anchor in this unit,
not the exact predecessor transition.

## Decision Update and Next Discriminator

Do not start a language-layer onset sweep. Preserve physical batch shape and
dtype in future probe receipts, and use full-model `float32` when an exact
low-margin transition owns the scientific conclusion.

Run exactly one bounded **Repeated First-Differing-Slot Full-Coordinate-Logit
Panel** on images `8629`, `13659`, and `17714`. At each exact common prefix
immediately before the first differing coordinate, compare matched `bfloat16`
batch one, `bfloat16` homogeneous batch four, and `float32`, with two
same-layout repeats. Preserve all 1,000 coordinate logits, top margins,
centered root-mean-square shifts, Jensen-Shannon divergence, and a fixed local
instance-window probability mass.

- A repeat-stable broad or instance-window-level shift justifies downstream
  onset localization.
- A repeat-scale, same-window perturbation closes the numeric branch and returns
  the program to the same-encoding spatial-eligibility enumeration probe.
