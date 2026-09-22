---
title: Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit Test Results
description: Verified strong-null evidence that replacing the committed left-cup pixels before visual encoding does not change one exact right-cup successor transition.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit Test Results

## Scope and Immutable Evidence

This unit tested one exact Common Objects in Context validation image `12576`
at Prefix State 56, with the exact coherent left-cup description-and-geometry
row appended before the first free action. The question was whether the
right-cup successor requires raw pixels inside the committed left-cup
bounding box after complete visual re-encoding.

The immutable paired panel receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/
paired-panel-20260715a/receipt.json
```

Its receipt Secure Hash Algorithm 256-bit (SHA-256) is:

```text
ebe7e20cde77932ac7a41c0b5da9757792703a94312550e8aa813d852f948d5b
```

The frozen panel had three conditions:

1. **Clean Pixel Replay**: exact recipient pixels, passed through the
   compositor without selecting pixels.
2. **Target Raw-Bounding-Box Donor Patch**: donor-image pixels assigned at
   the same coordinates inside the left-cup box.
3. **Equal-Shape Unrelated-Region Donor Patch**: the identical assignment in
   the manually reviewed disjoint wall and door region.

The target and control masks were both 141 by 264 pixels, or 37,224 pixels,
with half-open bounds `[251, 360, 392, 624]` and `[150, 20, 291, 284]`,
respectively. The target and control were disjoint. The donor and recipient
were both 864 by 1,152 red-green-blue images. No resizing, interpolation,
alpha blending, cached visual-feature replay, feature hook, or hidden-state
replacement was used.

The ordered sampled seeds were exactly:

```text
602711627830374173
5641984501295450920
8306462649179848189
8458627694586881429
4423663331540486457
1604446646505700360
6481035254874347622
6885534711486411115
```

The inference configuration, checkpoint, prompt hash, coherent-row hash,
model, adapter, tokenizer, parser, and first-action classifier remained
frozen as declared in [the unit contract](unit.md).

## Trust Gate

The independent artifact audit and receipt recount passed the frozen trust
gate. The evidence includes:

- exact eight-seed ordering and request attribution;
- exact selected-slice donor equality and exact recipient-complement equality;
- target-control disjointness and equal shape/count;
- `do_resize=false`, image grid `[1, 72, 54]`, and 972 merged visual tokens;
- one fresh primary visual stream plus three DeepStack streams for each visual
  request batch, with distinct condition-specific processor and feature hashes;
- clean compositor and standard clean no-op parity for greedy and sampled
  token identifiers and canonical float32 score traces;
- 29 JavaScript Object Notation (`JSON`) artifacts counted by the contract
  audit: 28 execution artifacts plus the top-level receipt.

The target perturbation was not larger than the control perturbation. The
target mean absolute pixel difference was `52.4598` with root mean squared
difference `70.7676`; the control values were `148.3046` and `152.8315`.
The panel therefore does not support an explanation based on the control being
the weaker pixel intervention.

## Observed First Action

| Condition | Greedy right-cup successors | Sampled right-cup successors |
|---|---:|---:|
| Clean Pixel Replay | 1 of 1 | 8 of 8 |
| Target Raw-Bounding-Box Donor Patch | 1 of 1 | 8 of 8 |
| Equal-Shape Unrelated-Region Donor Patch | 1 of 1 | 8 of 8 |

All 24 sampled first actions were valid `right_cup` rows. There were zero
target-selective losses, zero reverse-selective losses, zero invalid or
terminal first actions, and zero supported alternative-object first actions.
The predeclared Pixel-Support Effect, defined as

```text
R(equal-shape unrelated-region donor patch)
  - R(target raw-bounding-box donor patch)
```

was therefore `1.0 - 1.0 = 0.0`.

The intervention was active rather than silently ignored. The complete
sampled trajectories differed for every condition pair. Relative to clean,
the target condition changed the first row in 2 of 8 samples, only in
coordinates; the control condition changed the first row in 1 of 8 samples,
also only in coordinates. Greedy coordinates shifted under both donor
conditions while the first-action identity remained `right_cup`.

## Verdict

The frozen predeclared **Strong Null** criteria pass:

- clean, target, and control each retain the right cup in 8 of 8 samples;
- target-selective losses are 0 of 8;
- reverse-selective losses are 0 of 8;
- no asymmetric invalidity collapse occurred.

### Supported narrow claim

The complete raw pixel content inside the committed left-cup bounding box was
not detectably necessary for this exact high-margin right-cup successor under
the frozen same-position donor-patch intervention and full visual
recomputation.

Combined with the preceding post-vision local-support null, the leading
bounded explanation is a text- or prefix-mediated transaction followed by
visual selection of the still-visible right cup. This remains an explanation
for one exact state, not a general model mechanism.

### Disfavored claim

A local committed-object visual revalidation gate that requires the raw left-cup
box, or the selected post-vision support derived from it, is strongly
disfavored for this exact transition.

### Not established

This result does not prove:

- visual independence of the transition;
- absence of any causal logit effect from the replaced pixels;
- a purely textual geometry-sorted transducer;
- a native ledger, persistent commit state, or order-free coverage mechanism;
- generality across images, prefixes, categories, or crowded scenes;
- a training objective, architecture, or 256-image training screen.

The unchanged right-cup visual evidence remains available, and global visual
contextualization or a different causal seam could still matter. The target
box may also contain overlapping scene evidence whose removal is not
equivalent to deleting only a committed object.

## Branch Closure and Next Locus

This strong-null result closes the committed-object-support branch. Do not run
additional donor, mask, halo, magnitude, or second-image sweeps for this
question.

The next research locus is deliberately recorded only as a question, not as an
authorized unit or architecture: under an own-rollout prefix, does causally
correcting only the just-generated row from an approximate or incoherent event
to a phrase-and-geometry-coherent event redistribute the next action toward a
valid uncovered successor? Such a future discriminator would need to separate
event-construction failure, next-object or broader-state failure, and generic
legal-row progression. It is not designed or authorized by this result.

## Verification

- implementation and focused tests passed before the panel;
- the independent contract audit passed the immutable receipt;
- the raw receipt SHA-256 matches the immutable artifact handle above;
- greedy and all eight paired sampled requests passed no-op parity and
  request-attribution checks;
- the strong-null verdict is based on the predeclared rule, not on a post-hoc
  donor or threshold sweep.
