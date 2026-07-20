---
title: Matched Random-Order-Trained versus Geometry-Sorted-Trained Prefix-Order Screen Results
description: A bounded matched-checkpoint pilot finds that random row-order training changes route sensitivity but does not create useful covered-set invariance; geometry-sorted pure cross-entropy remains the pragmatic base for a small transition-calibration screen.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: small_screen_supported_separate_authorization_required
unit_id: 2026-07-20-matched-random-sorted-prefix-order-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_checkpoint_conditional
updated: 2026-07-20
---

# Matched Random-Order-Trained versus Geometry-Sorted-Trained Prefix-Order Screen Results

## Verdict

Randomizing complete object-row order during pure cross-entropy training did
not produce a useful order-invariant covered-object state in this pilot. It
changed the learned transition policy, but its apparent stability on the
dense same-class image was a collapse onto one habitual successor rather than
successful tracking of which people were already covered.

The geometry-sorted checkpoint remained path-sensitive, but retained more
route diversity and produced the clearest interpretable intervention: on image
`18380`, changing only the earlier order first flips `person` versus `cup`,
then makes the uncovered cup's `x1` continuation strongly more likely. The
random-order checkpoint ignores the same intervention. Conversely, several
random-checkpoint switches on image `19109` occur at extremely small local
description margins: a prefix perturbation is enough to move `person` above
`motorcycle` or vice versa at the greedy boundary.

The best bounded model is therefore:

1. the prefix shapes a narrow set of habitual successor routes;
2. local category or coordinate margins can be small enough for earlier row
   order to tip the greedy branch;
3. geometry is resolved after, and partly independently from, entity
   discovery; and
4. previously emitted rows affect later routing, but neither checkpoint has
   demonstrated an order-invariant physical-object ledger.

Terminal output, wrapper syntax, and row closure are not the operative failure
in these selected states. Both checkpoints strongly prefer starting another
valid row over terminal output. Further randomization is not the treatment
suggested by this evidence.

Use the geometry-sorted pure-cross-entropy checkpoint as the pragmatic base
for a small transition-calibration training screen, and retain the
random-order checkpoint as an ablation. This is not a general claim that
geometry-sorted training is superior: there is only one checkpoint per policy,
and the two training runs used different source commits and packing-cache
versions.

## Compared Checkpoints and Evidence Scope

Both step-`4,887` checkpoints use the same base Qwen3 Vision-Language model,
literal system prompt, literal user prompt, description-first row format,
Weight-Decomposed Low-Rank Adaptation rank, token-type gate, data source,
nominal training length, and optimizer-scale settings. The intended changed
factor is the order of complete assistant object rows:

- geometry-sorted pure cross-entropy; and
- random-order pure cross-entropy.

The literal prompt is not a confound. Remaining limitations are one training
seed per policy, different source commits, different packing-cache versions,
selected images, and checkpoint-dependent automatic owner matching. The
result is a mechanism pilot, not a publication-grade estimate of the causal
effect of training-row order.

All paired inference and scoring used the same image, frozen prefix token
identifiers, prompt, generation settings, and case definitions. Complete-row
scoring loaded all `2,149,097,472` model parameters in 32-bit floating point
and accumulated scores in 32-bit floating point.

## Panel One: Common Physical Objects under Different Earlier Orders

The panel contains 24 prefix-pair comparisons per checkpoint over images
`18380`, `9400`, `9590`, and `19109`. Each comparison has one greedy pair and
four temperature-`0.4` paired samples, with one complete generated row per
arm.

| Checkpoint | Greedy physical-owner switches | Sampled physical-owner switches | Greedy exact-row differences | Sampled exact-row differences | Promoted pairs |
|---|---:|---:|---:|---:|---:|
| Geometry-sorted | 1 / 24 | 3 / 96 | 8 / 24 | 45 / 96 | 1 |
| Random-order | 4 / 24 | 4 / 96 | 6 / 24 | 23 / 96 | 5 |

These counts must not be read as a clean sensitivity ranking. Strict physical
ownership was available for only `103 / 240` geometry-sorted outputs versus
`201 / 240` random-order outputs. In dense scenes, an unmatched complete row
can be a shifted box, an ambiguous same-class assignment, or a real object
omitted from official annotations. It is not automatically malformed or
hallucinated. The different matching rate can censor owner switches
differently across checkpoints.

The defensible conclusion is narrower: both checkpoints remain path-dependent,
and random-order training does not remove that dependence. The random
checkpoint has fewer exact serialized-row changes but more observed greedy
owner or covered-recurrence changes in the automatically matchable subset.

## Panel Two: Symmetric Coverage Controls on Image 2299

Four person tuples compare the same covered set under `A then B then C` and
`B then A then C`, plus both leave-one-out controls `B then C` and `A then C`.
A tuple is coverage-qualified only if removing `A` reactivates `A` and removing
`B` reactivates `B`.

No tuple passes that symmetric coverage gate under either checkpoint.

Under the random-order checkpoint, every arm in every tuple routes to person
rank `18`. The two order arms never switch physical owner in the 32 paired
samples, but neither leave-one-out control reactivates the omitted owner. This
is route collapse: the model ignores both changed order and changed covered
set because one habitual successor dominates.

The geometry-sorted checkpoint retains some route response: the four tuples
produce `0`, `0`, `2`, and `3` paired order switches, respectively, for
`5 / 32` total sampled owner switches. One omitted `A` is reactivated, but its
paired omitted `B` is not. This is unilateral transition evidence, not a
symmetric covered-set mechanism.

The key distinction is:

```text
same output under several prefixes
does not imply
the model represents the same covered set correctly
```

Invariance is useful only when the output also changes appropriately when one
covered owner is removed.

## Complete Candidate-Row Score Decomposition

Six behaviorally promoted prefix pairs were scored under both checkpoints.
Five are from image `19109`, so prevalence and same-class binding are not
estimated.

### Image 18380: a clean category and `x1` route switch

Under the geometry-sorted checkpoint, the shuffled earlier prefix changes the
observed alternate cup's complete-row score by `+1.181` summed natural-log
units. Relative to the competing person row, the cup gains `+1.165`. Its
description contributes only `+0.057`; `x1` contributes `+1.107`. The supplied
candidate ranking flips from person first and cup second to cup first and
person second.

Under the random-order checkpoint, the same cup changes by only `-0.078` for
the complete row and `-0.003` at `x1`; candidate ranks do not change. This is
the cleanest checkpoint contrast in the unit: geometry-sorted training learned
a prefix-conditioned category and location route that random training did not
retain.

### Image 19109: tiny local category margins and geometry mismatch

The random checkpoint's observed greedy switches can cross very small
description margins. In selected prefixes, the first distinguishing
description tokens for `motorcycle` and `person` exchange vocabulary ranks
`1` and `2` after small score changes. This is a local branch competition, not
a terminal decision.

Exact canonical ground-truth rows do not reliably reproduce the generated
greedy owner. A generated motorcycle can have high overlap with the physical
object while using a locally preferred boundary token that differs sharply
from the canonical annotation. Low probability for one exact four-coordinate
ground-truth row therefore does not imply that the object was visually
inaccessible.

Complete-row scores are useful only as within-candidate intervention measures.
They are not a normalized object-choice distribution, and unequal row lengths
make cross-category sums especially unsuitable as the sole branch predictor.
Interpretation must combine:

- the earliest token at which candidate routes diverge;
- within-candidate score change under the prefix intervention;
- the actual generated row; and
- separate entity and geometry matching.

### Terminal is not the selected-case bottleneck

The raw margin between object-row start and terminal output remains strongly
positive for every scored boundary:

- geometry-sorted checkpoint: approximately `10.36` to `11.46`;
- random-order checkpoint: approximately `12.58` to `14.02`.

The paired prefix changes to those margins are much smaller than the margin
itself. Suppressing terminal output would not identify the correct successor
in these cases.

## Hypothesis Decisions

### Supported within this bounded panel

1. Earlier row order can alter the next route even when the physical covered
   set, row count, and final row are held fixed.
2. Random-order training changes this dependence but does not remove it.
3. A stable output across prefix permutations can be caused by habitual-route
   collapse rather than correct covered-set tracking.
4. Prefix effects can enter at the category decision and then be reinforced or
   redirected during coordinate generation.
5. Entity discovery and exact physical extent must be evaluated separately.

### Rejected or reduced

1. **Random row order automatically teaches order-free coverage.** The
   symmetric image-`2299` gate fails completely.
2. **Early terminal choice explains these transitions.** Row start dominates
   terminal at every scored boundary.
3. **Exact canonical coordinate-row likelihood measures whether the object is
   seen.** Generated geometry can match the object while differing from the
   canonical coordinate path.
4. **Fewer output changes imply better memory.** The random checkpoint's
   apparent image-`2299` invariance is one-route collapse.

### Still unresolved

- whether matched multi-seed retraining reproduces these checkpoint
  differences;
- whether same-category instance decisions show the same local mechanism;
- whether a training intervention improves final unique-object coverage rather
  than merely exchanging one valid route for another;
- whether a compact explicit state is needed after a direct loss-only
  treatment; and
- how the mechanism transfers beyond the selected dense images.

## Recommended Small Training Screen

The evidence now supports designing a small loss-only treatment before adding
slots, object queries, or an explicit ledger. Launch remains a separate
implementation and resource decision.

Use geometry-sorted pure cross-entropy as the base and add two local
comparisons built from native prefix states:

1. **Same-covered-set order consistency.** Under two prefixes containing the
   same complete physical-object rows in different earlier orders, a verified
   uncovered successor should remain competitive. Do not require identical
   serialized output or one canonical next owner.
2. **Leave-one-object-out response.** Removing a known object's row from the
   prefix should raise that object's transition score relative to its score
   when covered, while unrelated verified candidates should remain stable.

A local ranking term can compare a verified uncovered candidate `u` with a
verified covered recurrence or an explicitly reviewed unsupported successor
`c` at prefix `P`:

```text
softplus(margin - score(u | P) + score(c | P))
```

The score should begin at the earliest route-distinguishing token or short
distinguishing prefix:

- description token for cross-category candidates;
- first discriminating coordinate or short coordinate prefix for same-category
  instances.

Ordinary cross-entropy remains the transcription objective. Geometry targets
must tolerate accepted high-overlap generated extents rather than treating one
exact canonical coordinate sequence as the only accessible object path.

The 256-image screen should test whether the treatment:

- raises greedy unique-object recall;
- moves sampled-only verified objects into greedy rollout;
- reduces the bagging-versus-greedy unique-object gap;
- avoids increasing covered recurrence, malformed rows, and unsupported
  entities; and
- changes the targeted local margins instead of only increasing rollout
  length.

Random-order training remains an ablation. If this local treatment fails while
object support and covered-history information remain readable, the next
question is phrase-to-instance and coordinate binding. If covered-history
information is absent or cannot be used even under direct local supervision,
an explicit compact state becomes more credible.

## Stop Decision

Stop this research unit. Do not add more samples of the same selected cases,
scan decoder layers, or infer a final architecture. The next valuable action is
to freeze a small training-screen specification with matched controls and a
human-reviewed evaluation cohort.

## Artifacts

- Stage-One scoring selection receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen/stage1/matched-random-sorted-prefix-order-selection.json`.
- Stage-Two geometry-sorted result:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen/stage2/image2299-coverage-controls/sorted/result.json`,
  SHA-256 `c45ecd6c8f983361656cb4a2a2c0285a2488b84ea3fe080b88e0a51cae6416af`.
- Stage-Two random-order result:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen/stage2/image2299-coverage-controls/random/result.json`,
  SHA-256 `ea8fb81098f311e72bd3dbe30f8061e5faa21bc97e65b1a80628ebeac53e80c2`.
- Candidate scoring manifest:
  `candidate-row-scoring-manifest.json`, SHA-256
  `c21014a8cb620f29c7f145325b353a9aa046d316fe7f5cbe753f9324cb3b2388`.
- Geometry-sorted scoring receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen/stage3/candidate-row-scores/sorted/receipt.json`,
  SHA-256 `67fc40f8dda8cde9fe2af1e68fd341e6d46a4660f7ebdad74e3bd2325cb99143`.
- Random-order scoring receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen/stage3/candidate-row-scores/random/receipt.json`,
  SHA-256 `c1d46a00487e3ce2183e889a75ec302b0b97e9140da1f91dfdb141f1ebef46ae`.
- Mechanical paired report:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-matched-random-sorted-prefix-order-screen/reports/candidate-row-score-delta-summary.md`.

The experiment-local converters, summarizers, manifests, and focused tests are
retained in this worktree. No architecture or production inference contract is
promoted.
