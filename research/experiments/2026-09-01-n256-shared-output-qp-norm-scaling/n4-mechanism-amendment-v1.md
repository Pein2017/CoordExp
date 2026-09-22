# N4 low-norm mechanism decomposition amendment v1

## Question and boundary

The terminal N4 train-only result establishes a norm gap but not its source.
This CPU-only successor asks whether the semantic residual is mainly a trivial
format/EOS calibration, or whether coordinate/category output rows and a small
set of shared singular directions materially carry the registered margins.

It reuses the sealed N4 semantic payload and the first 273 positions of the
sealed N256 semantic capture. It does not solve another QP, open screen-dev,
decode, load the model, or change the stopped transfer conclusion. Row removal
and rank truncation are diagnostics only; they remain forbidden as substitutes
for the exact decision-bearing payload.

## Frozen decomposition

The immutable 1,136-row surface is partitioned by the bound tokenizer and COCO
category authority into three disjoint families:

- 1,000 coordinate tokens, `<|coord_0|>` through `<|coord_999|>`;
- every subtoken used by the 80 COCO category names;
- the five native format/EOS tokens: object-ref start/end, box start/end, and
  `im_end`.

The tokenizer-derived union must exactly equal the frozen row list. For each
family, report row-wise squared-norm share plus full, drop-one-family, and
keep-only-family exhaustive margin replay. Replay reports both registered
constraint satisfaction and the stricter position-level condition that every
competitor clears the frozen FP32 threshold `0.00998`.

Compute the residual SVD once and replay the predeclared ranks
`1, 2, 4, 8, 16, 32, 64, 128, 256`. Report energy retained, position
satisfaction, and the smallest tested ranks reaching 50%, 90%, 99%, and 100%
position satisfaction. No post-hoc rank is added.

## Interpretation and stop rule

- Call the mechanism `FORMAT_ONLY_COMPATIBLE` only if format/EOS alone retains
  at least 90% of positions at `0.00998` and neither coordinate nor category
  removal lowers its own target-family position satisfaction by 10 percentage
  points.
- Otherwise call it `NON_FORMAT_ROWS_NECESSARY` when either non-format removal
  lowers its own target-family satisfaction by at least 10 percentage points.
- Add `LOW_DIMENSIONAL_CONCENTRATION` only if rank 16 retains at least 90% of
  residual energy and at least 90% of positions.

Stop after one immutable mechanism receipt and its interpretation. A positive
result may recommend a separately frozen train-only N8 scaling discriminator;
it does not itself authorize a transfer claim or architecture promotion.
