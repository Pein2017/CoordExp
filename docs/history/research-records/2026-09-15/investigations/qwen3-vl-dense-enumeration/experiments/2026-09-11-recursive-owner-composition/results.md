# Result and next-experiment recommendation

## Outcome

The fixed24-image, eight-GPU acquisition completed mechanically and its cold
parser/metric replay agrees. It supplies **conditional annotation-relative
completion**, not a learned or autonomous missing-instance result.

| Fixed24 outcome | Images |
|---|---:|
| Both GT-row interventions needed; complete final annotated set |21|
| First GT-row intervention already releases the other owner |2|
| Second intervention loses an old owner; final set incomplete |1|

The one failure is image412516 and remains in the denominator. The two
single-correction closures are307814 and323322. No first-correction rejection
occurred. All24 cold natural trajectories reproduced the saved Stable50
actions. Seventy continuations generated2720 free tokens; all8 processes exited0.
Controller wall time was70.37s; summed worker elapsed GPU allocation was335.24s,
not a GPU-utilization percentage. Peak allocated CUDA memory was below9.40GB
per replica. There were **zero training steps**.

The21 two-correction successes contain13 cases where both insertions precede
existing rows,6 with the second insertion at EOS, and2 with both at EOS.
Only11 have an old matched owner generated freely **after the second forced
row**. These are diagnostic strata, not a changed primary denominator or
proof of an internal ledger. The full assisted cohort gains48 annotated owner
matches and loses1 old match; the forced48 must not be called autonomous gains.

## Why these21 are not an automatic training set

The lead used `view_image` on the first8 prospective success comparisons and
the failed comparison, then on three already-acquired originals to distinguish
class conflict, reflection and tiny-background cases. All12 visual judgments
are preserved; no GT was edited and no new image/GPU search was opened.

Three decisive examples:

- **531929:** the natural box already covers the full tennis player, whereas
  the target GT box largely covers the upper body. Adding the GT row gains a
  TP, but does not find a new player. The racket also has an existing
  near-threshold box. This is not a clean two-instance omission example.
- **354398:** the allegedly missing `car` is already predicted as `truck`
  with pixel IoU0.80144, on an antique fire vehicle. “No same-class row” is
  demonstrably insufficient for declaring a missing instance.
- **391492 / 184490:** broad natural broccoli/couch regions versus multiple
  GT subregions leave grouping/instance granularity unresolved. Oracle row
  insertion cannot resolve that semantics by itself.

Other reviewed cases involve partial/occluded people, strongly overlapping
bag annotations, mirror views or tiny background donuts. These are **holds
or distinct repair strata**, not certifications that the GT is false. No
reviewed pair was promoted as an unequivocal two-new-instance training label.
This is not a claim about every unreviewed member of24, nor a general null
result for recursive learning.

The visual geometry is grounded in the shared renderer. Its panel matcher is
a visualization aid; official numbers above come from the frozen native global
assignment, not colors or renderer matching.

## Recommendation: change the immediate priority, not the architecture

**First: the already agreed GT-free later-row deduplication learning pilot.**
The observed issue is not a reason to build a universal FP classifier. It is
a reason not to supervise all annotation-relative FNs as new-instance rows.
The strict prediction-to-prediction duplicate rule has an independently
defined signal and can be tested without settling noisy class/owner GT.

Proposed bounded contrast, **not launched in this unit**:

- Stable50 anchor versus Stable50 plus later-row negative learning; no added
  missing-owner CE and no architecture/special-token/KV changes.
- A fixed small train-only duplicate panel; acquire actual current student
  natural rollouts at fixed refresh points. Apply the user's exact rule:
  IoU strictly `>0.95` versus any earlier valid row, any class, every qualifying
  later row once. No GT or blanket unmatched-negative labels.
- Reuse language-only DoRA and the accepted eight-rank execution path. Keep
  a fixed preservation panel and the seven already learned targets. KL must
  not directly preserve the duplicate actions the objective is suppressing;
  explicit masks and normalization require a fresh real two-step check.
- Proposed maximum32 updates with a fixed schedule; no automatic extension,
  adaptive extra preservation images or extra coefficients after seeing the
  result. Freeze the exact panel, negative-loss token credit, coefficient and
  total execution budget before that launch rather than pretending those
  implementation/semantic details are already settled here.
- Evaluate unforced native behavior on the existing exposed384, retaining
  the full denominator: old/gained/lost owners, IoU50/80, F1, strict repeats,
  drops, EOS/caps and lengths. Inspect a fixed set of changed duplicate
  examples and a looser overlap diagnostic: crossing from0.951 to0.949 is
  not successful duplicate removal. Lower repeat counts achieved by losing
  true owners or premature stopping do not pass. No confirmation512 access.

**Second: joint full-row/path supervision versus recursive assistance removal,
on truly admitted owner pairs—not automatically these21.**

This remains the direct discriminator for the user's conditional-composition
question. Use independent byte-identical Stable50 clones, each shared across
its cases. A learns complete reliable witness rows jointly. B first learns the
later correction under the actual earlier-assisted history, tests a b-only
release from the original input, then learns the earlier correction while
protecting the **verified learned successor**, not an old reference that
still skips it. Refresh only real failed histories. Both arms use the same
on-policy duplicate rule and comparable weighted positive exposure; report
raw tokens and actual counts as well. This is an algorithm-package comparison,
not an isolated proof that reverse order is optimal. Tie favors the simpler A.

The next missing-owner panel must explicitly distinguish image occurrences
(including reflection policy), genuine omission, localization/extent repair,
class conflict and unresolved grouping. A small manually accepted panel is
enough; a general matcher/judge is not a prerequisite. No automatic expansion
of this closed candidate search is implied.

**Hold:** KV/register/ledger surgery. This phase provides no evidence that a
new information route is the required fix. Oracle conditional success and
annotation repair must not be promoted to an architectural conclusion.

## Evidence and status

- [Raw root](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-recursive-owner-composition)
- [Frozen inputs](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-recursive-owner-composition/inputs.json)
- [Native reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-recursive-owner-composition/reduction.json)
- [Computed interpretation](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-recursive-owner-composition/interpretation.json)
- [Visual judgments](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-recursive-owner-composition/visual-review.json)
- [Comparison manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-recursive-owner-composition/visualizations/manifest.json)
- [Worker quality trial](worker-trial.md)

Scientific status: **closed conditional-supply result; pure new-instance labels
not promoted; trainability unanswered**. Technical status: verified eight-GPU
producer and cold readback. The original Stable50 remains unchanged. The next
training contrasts above are recommendations, not executed results.
