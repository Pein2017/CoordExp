# Sampling acquisition diagnosis and bounded proposals

Status: **CPU diagnostic lead-accepted; proposals advisory, not launch grants**.
Root replayed the full reduction to a temporary output and matched all fields
except creation time. This package performs one
CPU-only reduction of sealed samples. It changes neither the frozen strict
`D := class-blind native-pixel IoU > 0.95` rule nor any model, checkpoint,
threshold, training, decoding, or launch authorization.

## Decision-bearing diagnostic

Estimand: among the saved raw-softmax first actions under each literal `h`, how
often is a parser-valid sampled row (a) equal in normalized description to the greedy
repeat description, but (b) spatially near either that exact greedy-repeat row
or any earlier `h` row? Class-blind overlap makes this insensitive to description
aliases. IoU cutoffs below `>0.95` are descriptive spatial-nearness probes only,
not physical-owner gold.

| sampling surface | valid / all | exact greedy description | median max-IoU to any `h` | any-`h` IoU `>0.50`; `>0.90` | exact-greedy-row IoU `>0.50` |
|---|---:|---:|---:|---:|---:|
| unchanged Stable50, step 1 | 23/24 | 16/23 (69.6%) | 0.195 | 1/23; 0/23 | 0/23 |
| all changing steps 1--32 | 751/768 | 466/751 (62.1%) | 0.0095 | 8/751; 1/751 | 3/751 |

Across all steps, 751 valid rows occupy 738 distinct native-pixel boxes. Only
9/751 overlap the exact greedy-repeat row above even the loose diagnostic
`0.25`; none exceed `0.90`. The all-step sample is not stationary because the
positive update changes parameters: notably, 351017 produces 209 table rows
but only 16 bottle rows, while 417044 remains 247/253 donut and 477415 remains
203/250 chair among valid rows.

**Observation:** Stable50 samples often preserve the greedy row's category but
almost never its location. **Supported inference:** “the sampler mostly drew a
slightly jittered version just below 0.95” is not a good account of the missing
event. The greedy exact row can be the tokenwise mode while its complete-action
sampling incidence is tiny and coordinate mass is diffuse. **Still live:** a
severely translated, contracted, or overextended box may denote the same
physical owner at low IoU; descriptions cannot resolve that. The admitted
0.934 same-bottle counterexample already proves that strict `D=0` is not
same-owner gold. No KV/copying/owner-memory mechanism follows.

Authoritative reduction:
[`sampling-diagnostic-candidate.json`](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-branch-vs-repeat-event/sampling-diagnostic-candidate.json).

## Ranked falsifiable proposals

The pending shared-checkpoint endpoint remains the first gate. If original
prompt quality is already acceptable, stop; none of the proposals below is
needed or authorized.

### 1. History-conditioned positive refresh, not stronger punishment

**Hypothesis/proposal.** If final `h`-only decoding selects the learned `c` but
the original prompt still reaches a different looping history, the main defect
is exposure to a changed prefix. Refresh the same already admitted positive row
at the observed natural pre-loop history; do not add labels or a new negative
term.

**Estimand.** Versus the current positive-only checkpoint, change in original-
prompt recovered-owner set and burden after learning the same `c` under the
actual natural history, conditional on fixed-`h` success. Admission must also
verify that the owner represented by `c` is not already covered in that new
history: a later loop after successful recovery does not justify teaching the
same owner again. Matching uncertainty requires bounded visual evidence, not
an automatic missing-owner label inferred from strict `D=0`.

**Strong alternative.** The supplied row is useful only off-policy, or failure
is general validity/termination rather than prefix mismatch.

**Smallest test / cost / stop.** The no-new-cost discriminator is already in the
endpoint: fixed `h` versus original prompt. Continue this proposal only for the
pattern “fixed `h` repaired, original prompt not repaired.” A future minimal
three-case refresh would reuse existing `c` and protection, then read those
three original prompts once. Stop on no natural branch change, protected
successor loss, or worse invalid/malformed or premature-stop owner-loss burden;
no dose sweep. More EOS terminations alone are not failure: escaping a length
cap by completing a useful output may be desirable.

**Already falsified / quality relation.** Older-history multiplicity moved row
scores opposite a simple count theory, so history is consequential but not a
counter. Fixed-route all-token argmax falsifies “no learnable positive direction”
only teacher-forced at `h`; 6/7 local releases falsify “a supplied row can never
open continuation.” Only original-prompt owner gains/losses can accept refresh.

### 2. Row-boundary best-of-K valid-nonduplicate inference

**Hypothesis/proposal.** The loop is a narrow greedy mode with abundant valid
nonduplicate alternatives. At a completed greedy row with strict `D=1`, sample
a small row set and choose the highest-scoring parser-valid `D=0` row; otherwise
retain the existing decode. This is an inference alternative to any stronger
negative CE and uses no GT for selection.

**Estimand.** Original-prompt owner/burden change caused by row-boundary
selection, not by weight changes.

**Strong alternative.** `D=0` candidates are hallucinations or severe reboxes
of the same owner, so selection trades measured repeats for owner loss, invalid
continuation, or premature EOS.

**Smallest test / cost / stop.** Reuse one highest-mean-logprob valid `D=0`
candidate from each case's eight saved Stable50 step-1 actions, requiring only
three future forced-row greedy continuations. Stop before scaling if fewer than
2/3 yield a visually plausible distinct row and no worse suffix burden. A later
original-prompt comparison must still report paired owner gains/losses,
invalid/malformed rows and cap/EOS; repeat reduction alone fails.

**Already falsified / quality relation.** 23/24 initial draws being valid `D=0`
falsifies “there is no alternative action supply,” not their owner correctness.
The native release panel supports downstream escape after a supplied row, but
does not validate these sampled rows or original-prompt quality.

### 3. Greedy-hard-negative versus admitted-positive row margin

**Hypothesis/proposal.** If the endpoint shows failure even at literal `h`, mine
the actual greedy `D=1` row and optimize an explicit row-level preference for
existing `c` over that row. This changes acquisition/ordering, rather than
increasing a zero Monte Carlo event coefficient.

**Estimand.** On the three fixed histories, probability that natural greedy
chooses `c` or another valid `D=0` row, followed by original-prompt owner/burden
change.

**Strong alternative.** Positive CE already gives the required ordering; added
negative pressure merely diverts probability into invalid/EOS alternatives.

**Smallest test / cost / stop.** First use the pending `h`-only read: fixed-`h`
success stops this proposal. Only fixed-`h` failure motivates a tiny paired
three-case update/read, with no coefficient sweep. Stop on zero branch changes,
invalid burden increase or owner loss, including losses from premature EOS;
an EOS-count increase by itself is not a rejection criterion.

**Already falsified / quality relation.** Multiplying the current event term is
falsified mechanically by `D=0` and exact zero gradient. Coordinate-only UL32
also falsifies treating fewer strict repeats as success: repeats fell while
geometry-invalid drops rose by 100 and TP50 fell. Row margin remains a distinct
hypothesis, but original-prompt joint quality is its only promotion surface.
