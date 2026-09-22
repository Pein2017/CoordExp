# A16/K4 owner-return coverage/F1 tradeoffs

Status: CPU diagnostic complete; candidate interpretation for root acceptance.
No official metric, reward, mask, model, or rollout artifact was changed.

## Result

Completed-rollout category-agnostic owner coverage and annotation-relative F1
are not interchangeable on this bank. Across the `96` within-image K4 pairs,
they strictly prefer opposite actions in `6` pairs over `2` images. Another
`14` pairs over `5` images tie on coverage but split on F1 because their valid
prediction counts differ. They agree directionally in `23` pairs and are tied
on both in `53`.

This is a post-ruling diagnostic using pixel IoU `>=0.50`, global
cardinality-first one-to-one matching, and a uniform category. It does not
replace the producer's category-consistent official metric. Its frozen recount
recovers the expected official `16/96` equal-immediate/different-final pairs
over `6` images and the new category-agnostic `13/96` over `5`; `6` of those
new pairs over `2` images have two annotation-matched current-owner actions.

## Definitions and nested support

- **Coverage:** category-agnostic matched annotated owners divided by GT owners.
- **Annotation-relative F1:** `2 TP / (2 TP + FP + FN)`, where `FP` is an
  unmatched valid parsed prediction. Dropped/invalid predictions are reported
  separately. This is not physical-owner precision under incomplete COCO GT.
- **Current-owner action:** the appended sampled action prediction is globally
  matched to an annotated owner not matched at the prefix. This is a geometric,
  annotation-relative mask, not human-perfect groundedness.
- **LOO credit:** `A_j = R_j - mean(R_k, k != j)`. Coverage return is
  `(final TP - prefix TP)/GT`; F1 return is `final F1 - prefix F1`. The common
  prefix subtraction cancels inside each sibling comparison.

The bank has `16` historically selected images, `4` actions per image, `64`
actions, and `96` within-image pairs—not `64` iid images. All `64` actions have
box-end/single-valid-row support. `47` actions across `15` images match a new
current annotated owner; `46` of them form comparison groups of at least two
actions across `14` images, while one is a singleton.

| LOO sibling set | Return | positive | zero | negative | images with nonzero credit |
| --- | --- | ---: | ---: | ---: | ---: |
| all valid K4 (`64` actions) | coverage | 14 | 38 | 12 | 7 |
| all valid K4 (`64` actions) | F1 | 25 | 20 | 19 | 11 |
| current-owner only (`46` comparable actions) | coverage | 4 | 38 | 4 | 2 |
| current-owner only (`46` comparable actions) | F1 | 9 | 32 | 5 | 4 |

Thus downstream signal remains after restricting comparisons to actions that
already match a current owner, but it is small and nested: among `58`
current-owner/current-owner pairs, `47` tie on both final criteria; `3` agree,
`5` tie on coverage but split on F1, and `3` strictly conflict. Coverage alone
distinguishes `6/58` pairs over `2` images; F1 distinguishes `11/58` over `4`.
A current-owner-only RLOO is a different estimator and support, not a harmless
post-hoc loss mask.

## Three decision-bearing cases

1. **287484: strict conflict between two current-owner actions.** Book-first
   branch `2` matches owner `1648226`, then ends with `3/7` owners but `31`
   valid predictions (`28` annotation-relative unmatched), two dropped rows,
   and F1 `0.158`. Bed-first branch `0` matches owner `319739`, ends immediately
   with `1/7`, `1` prediction, and F1 `0.250`. Coverage credits book-first
   positively and bed-first negatively; F1 reverses those signs. Book-first's
   additional final owners versus bed-first are `1648226` and `1658449`.
2. **478148: clean signal where both criteria agree.** Sheep-first branch `0`
   matches owner `62881`, then adds bird `38143`, reaching `4/4` owners with
   `4` predictions and F1 `1.0`. Bird-first branch `2` matches `38143` and
   stops, reaching `3/4` with `3` predictions and F1 `0.857`. This is the
   category-agnostic counterpart of the unit's sheep-first-versus-bird-first
   contrast; it shows downstream differentiation is not supplied only by the
   known 1x7 action.
3. **486123: the 1x7 branch exposes the objective tradeoff.** The unit's
   visually adjudicated ungrounded branch `3` action is annotation-unmatched;
   its final category-agnostic result is `9/11` owners, `15` predictions,
   `6` annotation-relative unmatched, F1 `0.692`. Current-owner branch `1`
   ends at `8/11`, `9` predictions, `1` unmatched, F1 `0.800`. The 1x7 branch
   gains owner `1783860` and loses none versus branch `1`, so coverage prefers
   it while F1 prefers branch `1`. The extra unmatched predictions are not all
   certified hallucinations.

As a sanity boundary, image `345388` is `6/6` owners, `6` predictions, F1
`1.0` for every branch under this diagnostic. Its former category-sensitive
`5` versus `6` suitcase/handbag count is not a geometry mechanism.

## Credit advice and strongest alternative

For the already frozen first learning probe, preserve its category-agnostic
completed-owner-increment K4-LOO credit. It directly represents the declared
owner-coverage estimand and does not silently redefine every unmatched
prediction as a physical false owner. Do **not** add an F1 reward/filter or a
current-owner action mask post hoc. Judge that probe's natural decode with both
owner gains/losses and F1; a gain concentrated in long-output cases like
287484/486123 would not by itself be clean evidence.

The strongest next-protocol alternative is completed-rollout
annotation-relative F1 K4-LOO on the same all-valid action support. It supplies
nonzero credit to `44/64` actions versus `26/64` for coverage and directly
responds to prediction burden. Its weakness is equally concrete: on incomplete
annotations it penalizes unmatched predictions without establishing that they
are false physical owners. That semantic tradeoff requires a separately frozen
trial, not a retrofit to v1. No proxy classifier or new calibration panel is
needed to state this bounded choice.

## Reproduction and evidence

```bash
cd /data/CoordExp/.worktrees/self-rollout-behavior
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/tradeoffs/reduce_tradeoffs.py
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/tradeoffs/verify_tradeoffs.py
```

The fresh consumer verifies all four source/code hashes, `16` images, `64`
actions, `47` current-owner actions, both LOO formulas, zero-sum advantages,
and a formula-sensitivity counterexample. Machine-readable evidence:

- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/tradeoffs/analysis.json`
  (`fe88bb2e3483d4bfd7f61e61ae12f9b99a9719a7d1bbd346496423a4d446c24b`)
- `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/tradeoffs/verification.json`
  (`93079a04b54215fb92866a8fb3a2dfadcccd29c21de8a230b4d78aec695c7ca6`)

Scientific disposition: finite-bank reward disagreement and current-owner
signal are established. Training benefit, population prevalence, and physical
owner precision remain unestablished. Stop rule reached; no new generation,
training, visual adjudication, or reward adoption follows from this package.
