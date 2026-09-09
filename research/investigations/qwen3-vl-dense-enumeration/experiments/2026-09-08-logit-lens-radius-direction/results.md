# Final radius-versus-direction result

Date: 2026-09-08. Status: lead-accepted bounded local evidence; branch closed.
Architecture/training promotion: none.

## Decision

On the frozen Human13 prefixes, donor **direction** reproduces most of the
full-current-state transfer; donor **radius alone** does not. This rejects
residual length alone as an explanation of the preceding late-state result.
It does not establish a context-independent semantic representation or a
natural-generation improvement.

## Executed contrast

Same Source step-2444 and Human13 magnitude-only CE-overfit checkpoints,
base/head/norm/embedding delta, and previously generated overfit prefixes as
the [parent study](../2026-09-08-logit-lens-causal-transfer/results.md).
All 13 images are training images. Image2299 contributes 12 coordinate sites;
each remaining image contributes four. All 60 sites are retained; the one
equal-endpoint site is excluded from endpoint-ratio/choice summaries (59 eligible).
No new generation or training was performed.

At decoder outputs 24 and 27, current-position state h=r*u was replaced by
donor radius with receiver direction, receiver radius with donor direction,
or the full donor state. Other positions and receiver downstream layers
were preserved. Native/self and direct block-28 final-norm controls passed.

## Results

R is the unclipped native-gap-normalized change in the fixed endpoint margin,
computed within each image and then averaged equally over 13 images.
It is **not accuracy**. Donor-choice counts below are raw eligible-site counts,
not image-equal rates. O denotes overfit and S denotes Source.

| Donor → receiver | Block | Radius-only R | Direction-only R | Full R | Donor top-one: radius / direction / full |
| --- | ---: | ---: | ---: | ---: | --- |
| O → S | 24 | -0.0018 | 0.4866 | 0.5000 | 0 / 35 / 37 of 59 |
| O → S | 27 | 0.0109 | 0.7214 | 0.7560 | 0 / 56 / 58 of 59 |
| S → O | 24 | 0.0902 | 0.9485 | 0.9631 | 0 / 12 / 17 of 59 |
| S → O | 27 | 0.0661 | 1.0032 | 1.0115 | 0 / 24 / 32 of 59 |

Normalized factorial interaction (full − radius − direction) is respectively
0.0153, 0.0237, -0.0756, and -0.0578 in table order. Radius is secondary,
not identically irrelevant: full-state replacement still changes donor-choice
counts relative to direction-only, especially in the reverse direction.

The reverse asymmetry survives norm matching. At block 27, direction-only
S→O gives R≈1 but donor choice on only 24/59 sites; its image-equal third-token
rate is 55.13%. Margin erasure must not be called faithful donor transfer.

## Interpretation and future relevance

1. **Observation:** Direction-only nearly reproduces full O→S transfer; radius
   alone never selects the donor endpoint on eligible sites in either direction
   or block. **Inference:** The local effect is direction-dominant, not merely
   a consequence of the smaller overfit residual norm.
2. **Observation:** Reverse third-token outcomes and direction/radius
   interactions persist. **Inference:** Receiver context and downstream layers
   remain part of the mechanism; a universal symmetric state-portability claim
   is unsupported.
3. **Scope:** DoRA parameter-magnitude-only training can change activation
   direction. Parameter magnitude is not residual radius. These findings do
   not require a new architecture or imply that norm regularization is the
   appropriate intervention. Useful state direction/decision content is a
   better supported descriptive focus than residual length alone.

This is a local fixed-prefix intervention, not direct training-time dynamics,
unseen-image generalization, natural owner coverage, EOS improvement, or a
causal identification of a unique semantic concept. No such claim is promoted.

## Acceptance and reproducibility

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-radius-direction`.

- Execution: `final-v1/receipt.json` and 13 per-image receipts/raw traces.
- Independent reduction: `analysis-v1/summary.json`; local `analyze.py`.
- Figure: `analysis-v1/radius-direction.png`.
- Runner: `scripts/research/probe_logit_lens_radius_direction.py`.
- Tests: `tests/research/test_logit_lens_radius_direction.py` (3 passed on lead replay).
- Lead checked all execution artifact hashes, all 13 image receipts/checks,
  runner identity, independent raw reduction, and figure.
- 720 patch traces; 798 teacher forwards (26 native, 26 direct, 26 self,
  240 each radius/direction/full); 26 vision forwards; 52 block-28 head calls;
  all 240 prior full-current anchors matched.
- Native/text replay, selectors, endpoints, intended radius/direction,
  recipient identity, earlier selected logits, non-target residuals, self
  replay, prior anchors and final-norm controls passed the frozen checks.
- Runtime 398.35 seconds; peak GPU allocation 17.969 GB; peak RSS
  14,180,688 KiB. No GPU rerun, retry, training or new generation.

Exact evidence hashes and acceptance disposition are in
[lead-acceptance.json](lead-acceptance.json).

## Stop

The user-requested final round is complete. Close the interpretability branch
with direction-dominant local transfer and persistent context/suffix asymmetry.
No additional study, conditional continuation, training, or architecture change
is launched or queued by this closeout.
