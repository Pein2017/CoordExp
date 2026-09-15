# Autonomous owner-outcome research: bounded closeout

Status: **lead-accepted; closed for low expected marginal value; no promotion**.
The user authorized autonomous iteration until marginal value became low or a
major user-owned decision was needed. This closeout uses the former stop rule.
No new user decision, protected confirmation, architecture change, external
paid resource or shared skill/config modification was consumed.

## Outcome first

Four actual Source-started one-step updates did not increase the total natural
greedy development owner count. The best observed annotation-relative F1 has
the same owner count as Source, one owner gain/one loss, six fewer predictions,
and a training-side owner loss. It is not a successful net-owner recovery model.

All rows remain in the category-agnostic pixel-IoU>=0.50 one-to-one denominator.
Dev64 is already-used development evidence, not independent confirmation.

| Arm | Dev owners / GT | Valid predictions | Annotation-relative F1 | Owners gained / lost vs Source |
| --- | ---: | ---: | ---: | ---: |
| Source, reused baseline | 295 / 498 | 574 | 0.550373 | — |
| Immediate coverage, coordinate credit | 293 / 498 | 583 | 0.542091 | 0 / 2 |
| Completed coverage, coordinate credit | 295 / 498 | 575 | 0.549860 | 0 / 0 |
| Completed F1, coordinate credit | 295 / 498 | 571 | 0.551918 | 1 / 1 |
| Completed F1, full-action credit | 295 / 498 | 568 | 0.553471 | 1 / 1 |

F1 is `2 TP/(GT+valid predictions)`; unmatched predictions are annotation-
relative, not certified physical false objects. Full-action F1 improves this
score by0.003098 over Source, not recall or unique-owner count. On train16,
Source/immediate/completed-coverage/coordinate-F1/full-action-F1 owners are
93/94/93/92/92 out of124. No arm is selected as a robust or generalizing winner.

Both F1 arms gain owner1987423 in dev image070033 and lose owner1656471 in
image131580, retaining294 Source owners. Immediate additionally loses
owner1194977 in image255904. On train, both F1 arms lose owner1926376 in
image532132; immediate's separate training gain is owner1926253. Complete
paired identities, category-consistent diagnostics and prediction burden are
owned by the linked trial results below, not an aggregate-only acceptance.

## What was learned

1. **Immediate validity does not determine final return.** The independent
   [tradeoff census](tradeoffs/result.md) verifies coverage/F1 ranking conflict
   in6/96 sibling pairs over two images. Image287484's book-first route reaches
   more annotated owners but produces31 valid predictions versus one on its
   bed-first route, reversing F1 preference. The sheep-first/bird-first case
   shows useful downstream differentiation without relying only on the known
   ungrounded1x7 action. This is finite-bank signal, not training benefit.
2. **The coverage null is not complete behavioral immobility.** The independent
   [transition census](transition_census/result.md) verifies that downstream
   coverage changes27/80 natural token sequences, all first diverging at a
   coordinate literal, while preserving every Source owner set. The remaining
   53/80 are exact-token identical. This is a mixture of discrete-greedy inertia
   and geometric movement that does not recover owners.
3. **Changing reward and then broadening credit did not solve recovery here.**
   F1 credit reduces annotation-relative burden slightly. Adding direct credit
   to description/schema tokens reduces it a little more, but reproduces the
   coordinate-F1 owner sets. This does not establish a description-choice
   mechanism. Root's contextual visual inspection identifies real shelf
   book/media regions in the gained/lost cases, not informationless phantom
   regions; local assignment changes do not imply new semantic discovery.

## Why stop, and what remains unknown

The matched bank, source, one-step dose and natural endpoint have now tested
the two most immediate alternatives: outcome-aware reward and broader action
credit. There is no net development owner gain, and the F1 arms retain a train
regression. More same-bank reward/mask/dose variants would mainly add selection
opportunities on the same reused development panel rather than resolve a new
well-supported mechanism. Root judges their expected marginal value too low.

Larger or fresher on-policy support, repeated learning, another dose and genuine
out-of-panel stability remain **untested**, not disproved. They would need a
new bounded research question; they are not silently scheduled here. The small
annotation-F1 change does not justify consuming the protected confirmation512.
This is not a general rejection of RL, future credit or full-action training.

The known1x7 action stayed in all training arms under explicit user permission;
its positive or negative relative credit depended on the declared reward.
There was no dedicated remove-only ablation, so its isolated contribution to
the learned policies remains unidentified. Feedback remains pending.

## Mechanical acceptance and cost boundary

All four arms independently start from original step2444, use the same16-image
K4 bank and one fresh AdamW step at2.5e-6, and save588 updated language DoRA
tensors. Eight-rank updates, unmerged checkpoint persistence and actual cold
natural decode were qualified end-to-end. Direct credit is256 coordinate tokens
for the first three arms and626 full-action tokens for the last, not relabeled
coordinate evidence. Root freshly reduced each trial's persisted artifacts;
all `lead-results.json` files exactly match the owning `results.json` hashes.

There are320 newly generated natural rows in total, all ending `im_end`, no
caps. One v2 evaluator-arm whitelist failure occurred after a useful saved
update; the failed invocation is preserved, the real consumer failure was
reproduced and repaired, and only evaluation was recovered. No update was
repeated or original failure hidden.

Eight-GPU allocation windows sum672.551867 seconds, approximately1.49456
allocated GPU-hours, including that failure/recovery. This is not measured
active-kernel time or a model API price ledger. All owned GPU processes were
released. The authoritative machine-readable closeout is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/closeout.json`

SHA256 `0ff09b512140413d6fbce063b332ebed58ea729294c3b6e5a63c2f658decd832`.

## Experimental team observations, not a model ranking

- **Astra learning owner:** integrated the native eight-rank update, save and
  cold-evaluation package. V1 used independent L2 training/evaluation lanes
  (worker-reported Astra-low training and Luna-xhigh evaluation). Root closed
  the exact-denominator, real-consumer and shared-cleanup-bound risks.
- **Sol scientific owner:** completed two distinct CPU questions: the reward
  tradeoff census and natural-transition census. Both passed root's fresh
  consumer checks with no substantive scientific rework. The second took
  roughly12 minutes by the worker's task receipt. These were not duplicate
  runs of the Astra engineering task.
- **Variants stayed single-owner:** Astra handled v2/v3 without L2. Rough
  reported implementation-to-final times were15/12 minutes, respectively.
  V2 missed a new-arm consumer whitelist, requiring one explicit eval-only
  recovery; v3 had no CPU/runtime correction. These times include coordination
  and runtime, are not matched workloads, and cannot establish model superiority
  or comparable API cost.

Provisional orchestration lesson: use L2 for genuinely separable initial
training/evaluation ownership, then reuse a single informed owner for small
variants. Test the newly serialized object at its actual consumer rather than
assuming old green leaf tests cover a new arm. Keep independent scientific CPU
questions with a capable economical worker instead of retiring a model family
from unrelated failures. These observations remain local research notes; no
shared guidance was changed.

## Evidence route

- [Frozen authority and checkpoint history](unit.md).
- [V1 immediate versus completed coverage](learning/result-v1.md).
- [V2 completed F1 and preserved eval-only recovery](learning/result-v2-f1.md).
- [V3 full-action versus coordinate-only F1](learning/result-v3-full-action-f1.md).
- CPU diagnostics and their consumer commands are linked above.

Artifacts and local commits are preserved on `probe/self-rollout-behavior`.
No remote push, worktree retirement, checkpoint deletion or experiment
continuation is part of this closeout.
