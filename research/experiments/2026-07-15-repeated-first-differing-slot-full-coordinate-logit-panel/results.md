---
title: Repeated First-Differing-Slot Full-Coordinate-Logit Panel Results
description: Verified three-case evidence that recurrent bfloat16 batch-shape coordinate differences are deterministic local same-object perturbations rather than recurrent broad object-basin switches.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Repeated First-Differing-Slot Full-Coordinate-Logit Panel Results

## Verdict

Close numerical escalation for these three recurrent one-to-two-pixel splits.
Do **not** launch a language-layer onset sweep from them. This scoped decision
does not establish global numerical invariance and does not invalidate the
earlier isolated image-`7574` broad bowl-basin shift.

All three Brain Floating Point 16-bit (`bfloat16`) source splits reproduced on
the ordinary autoregressive cache in both repeats. Their full coordinate-logit
changes were repeat-stable, but small: centered root-mean-square differences
were only `4.17%` to `6.22%` of the verified image-`7574` broad-shift anchor,
Jensen-Shannon divergences were only `0.75%` to `1.05%` of that anchor, and
local-neighborhood conditional mass moved by at most `0.001361`.

Every selected coordinate remained in the frozen radius-`16` neighborhood of
the same physical object. Full-model Institute of Electrical and Electronics
Engineers 754 32-bit floating point (`float32`) reduced exact-recipient
physical-batch root-mean-square differences by approximately `5,342` to
`11,278` times and selected the same coordinate bin in batch one and batch
four.

The result is a deterministic, low-amplitude, broadly distributed
low-precision batch-shape perturbation acting on local coordinate ties, not
evidence that material next-object or coordinate-basin switches recur across
the selected dense-image transitions. Here, "local" describes the behavioral
rank exchange; the nonzero logit delta itself is not confined to the local
window.

## Conclusion-Owning Artifacts

Three-case `bfloat16` receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/
  source-bfloat16-selected-three-20260715a/receipt.json
```

Secure Hash Algorithm 256-bit (`SHA-256`) digest:

```text
370b69cb85222c744fdbd83c19ac9ef9039b4a4a0bd8743612b926cde439056e
```

Matched full-model `float32` receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/
  full-model-float32-selected-three-20260715a/receipt.json
```

`SHA-256` digest:

```text
ddf55b7edcd3f6904b7e6f1c8fe4fd1c17ead6ee2c59b311eed8c59ba38d80ba
```

The earlier image-`8629` smoke is superseded by the three-case receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/
  bfloat16-smoke-image-8629-20260715a/receipt.json
```

## Trust Gates

- The source `bfloat16` first-differing coordinate split reproduced in both
  repeats for all three cases.
- Every physical batch of four homogeneous copies had one exact serialized
  `1,000`-coordinate-logit vector across all four positions.
- Every same-layout repeat had an exactly equal serialized coordinate-logit
  vector; the measured repeat noise floor was zero.
- All source prompts, first rows, images, model identity, tokenizer identity,
  generation configuration, and source Full-Image K-Rollout Independent
  Bagging (`FULL_BAG_K`) cell-zero lineage passed continuity checks.
- The runner preserved all `1,000` raw coordinate logits, conditional log
  probabilities, top-`20` coordinates, top margins, dynamic-window masses, and
  input tensor receipts.
- Direct Full-Prefix Scoring used the exact common token prefix immediately
  before the source first difference. Natural Cached Replay separately owned
  reproduction of the source trajectory.

The implementation and predecessor helpers passed before execution:

```text
25 passed in 0.19s
```

An independent scientific audit recomputed receipt hashes, all `1,000`-logit
vector hashes, source split indices, root-mean-square metrics, local-window
mass, Jensen-Shannon divergence, and `float32` attenuation. It found no blocker
or conclusion-changing defect. After the audit, the automatic classifier was
made to enforce explicitly the already-satisfied condition that same-layout
repeat noise cannot exceed the between-layout effect; this did not change any
receipt or verdict.

Post-audit focused verification passed:

```text
26 passed in 0.18s
```

## Natural Cached Replay Result

| Image | Single bin | Batch-four bin | Centered root-mean-square shift | Fraction of broad anchor | Outside-window root-mean-square shift | Jensen-Shannon divergence | Fraction of broad anchor | Local-window mass shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8629` | `974` | `973` | `0.048652` | `4.17%` | `0.046829` | `0.000980` | `0.83%` | `+0.001361` |
| `13659` | `805` | `803` | `0.072583` | `6.22%` | `0.070273` | `0.001240` | `1.05%` | `+0.001024` |
| `17714` | `311` | `310` | `0.064892` | `5.56%` | `0.063781` | `0.000880` | `0.75%` | `+0.000227` |

The predeclared close rule required every case to remain inside its local
neighborhood, stay below one tenth of the image-`7574` anchor on all three
distribution metrics, change local-neighborhood mass by at most `0.02`, and
show no larger repeat instability. All three cases pass every condition.

The absolute batch effect is real and deterministic. The distinction is one of
materiality: it does not move meaningful probability mass between object
neighborhoods.

## Direct Full-Prefix Result

Direct recomputation confirms that exact argmax identity is path-sensitive
under `bfloat16`, while the distribution remains local:

| Image | `bfloat16` single bin | `bfloat16` batch-four bin | `bfloat16` centered root-mean-square shift | `float32` single bin | `float32` batch-four bin | `float32` centered root-mean-square shift | Attenuation factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `8629` | `974` | `974` | `0.051213` | `974` | `974` | `0.00000542` | `9,457` times |
| `13659` | `806` | `803` | `0.100367` | `805` | `805` | `0.00000890` | `11,278` times |
| `17714` | `310` | `311` | `0.049275` | `311` | `311` | `0.00000922` | `5,342` times |

The `bfloat16` direct top-one margins include exact or quantized ties:

- image `8629`: `0.0` in batch one and `0.125` in batch four;
- image `13659`: `0.125` in batch one and `0.0` in batch four;
- image `17714`: `0.0` in batch one and `0.125` in batch four.

Matched `float32` margins were positive and nearly batch invariant:
`0.05504`, `0.00121`, and `0.17282` for images `8629`, `13659`, and `17714`,
respectively. This is the expected signature of local low-precision rank
swaps rather than distinct object modes.

Under `float32`, image `8629` did not naturally generate the earlier
`bfloat16` common prefix, so its Natural Cached Replay is ineligible for an
exact-recipient comparison. This does not weaken the direct precision control:
Direct Full-Prefix Scoring fixes the recipient tokens by construction. Images
`13659` and `17714` naturally reached their frozen prefixes and were also
batch invariant.

## Supported

1. Exact `bfloat16` batch-shape coordinate sensitivity is recurrent and
   deterministic at these selected slots.
2. The three selected first-action divergences are local rank changes inside
   one object neighborhood, not recurrent broad coordinate-distribution shifts.
3. Full-model `float32` strongly attenuates the exact-recipient batch effect and
   removes selected-bin differences.
4. Natural cached generation and direct full-prefix recomputation can select
   neighboring tied bins differently under `bfloat16`; exact coordinate-token
   identity is therefore not a stable mechanistic label.
5. The dramatic image-`7574` white-to-orange shift remains a valid isolated
   execution phenotype, but this panel provides no evidence that it is a
   representative root cause of dense enumeration failure.

## Ruled Out or Demoted

- The selected three micro-divergences do not justify per-layer onset
  localization.
- Exact first-action token inequality is not sufficient evidence for a
  material object or coordinate-basin change.
- This numerical branch does not explain conservative rollout length, low
  recall, duplicate bursts, or incomplete coverage.
- No training screen, architecture change, slot, ledger, or inference policy
  is authorized by this result.

## Unresolved

- Whether rare broad `bfloat16` execution branches affect production detection
  metrics often enough to matter operationally.
- Whether a production `float32` run changes Average Precision, recall,
  latency, or calibration; this unit is not a production-precision benchmark.
- The primary mechanism behind dense-image candidate competition and why
  sampled full-image bagging recovers objects that one greedy trajectory misses.
- Whether restricting access to a subset of one fixed full-image visual
  encoding redistributes next-object support without the hallucinations caused
  by pixel masking.

## Route Decision

Return to the dense-enumeration mechanism. The next bounded discriminator is a
same-encoding spatial-eligibility probe, not a numerical layer sweep: encode the
full image once, hold the text prefix fixed, and vary only which already
computed visual positions remain eligible to influence the next object. This
tests post-vision candidate competition while avoiding pixel masking,
re-encoding, resizing, training, and architecture commitment.
