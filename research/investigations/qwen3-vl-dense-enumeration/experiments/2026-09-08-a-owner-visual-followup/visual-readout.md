# Contextual owner review and the 1x7 action

Root personally inspected all 24 tight/context pairs, six wider contexts, the
whole A486123 source, and all four A action tight/context pairs via `view_image`:
63 distinct views. This follow-up is **not blind** and is not human gold. Earlier
blind model labels and all official metrics remain unchanged.

## What changed in the interpretation

Standalone tight-crop recognizability is not an owner requirement. Surrounding
pixels can establish an object whose tiny crop is otherwise uninterpretable.
Description mismatch is not an owner veto. Separate contextual binding, box
extent, and novelty/duplication; do not collapse them into one validity bit.

The root's qualitative grouping is 12 clear bindings with reasonable visible
extent; 5 context-supported bindings with partial/shifted extent; 7 uncertain
single-instance bindings. These are **not TP counts or population estimates**.
In particular, the earlier strict 4/16 conjunction is neither the valid-owner
count nor an upper bound. Examples include the book spine visible in wider
bookshelf context, a projector called tv, and a plate called bowl. Candidate-24's
white chair back is judged adequate in context rather than rejected for extent.

Candidate pairs 07/09 and 10/19 visibly refer to the same spoon and same plate,
respectively. Their boxes need not exceed the frozen strict pixel IoU >0.95
geometric-repeat threshold. Thus zero geometric repeats does not establish zero
physical-owner duplicates. Conversely, an annotation-relative unmatched row is
not automatically hallucinated or a newly discovered owner.

## A486123: separate action validity from continuation return

All four sampled actions have the same 10-token schema/description (traffic
light); only the four coordinate tokens differ. Actions 0/1/2 are grounded in
the real green traffic light, although action 0 captures only a narrow slice and
fails the official IoU match. Action 3 is [796,607,797,614], exactly 1x7 native
pixels. Tight pixels and surrounding context do not support an independent
object there. The observed issue is an ungrounded/degenerate detection, not a
deduction from size alone.

| Branch | Immediate added TP | Final TP | Added TP from common 5-owner prefix | Final tokens |
|---|---:|---:|---:|---:|
| 0: grounded partial light | 0 | 6 | 1 | 78 |
| 1: grounded light | 1 | 7 | 2 | 87 |
| 2: grounded light | 1 | 7 | 2 | 87 |
| 3: ungrounded 1x7 | 0 | 8 | 3 | 142 |

Relative to either best grounded branch, branch 3 gains **one** annotated car
owner, 1783860, and loses none. Relative to the native baseline (6 TP), it gains
traffic light 407445 and car 1783860, losing none. The person 554354 is common
to the natural baseline and every branch; it must not be credited as the
1x7-specific gain. Branch 3 later emits the actual traffic light again, then a
longer car sequence; only one of those five car predictions adds a matched
owner. It also has more annotation-relative unmatched predictions (7 versus
2 in branches 1/2); they are not all certified hallucinations.

**Observation:** on this fixed prefix, the visually ungrounded action has the
highest realized suffix owner return. Local action validity and downstream
return are therefore not interchangeable on this bank.

**Hypotheses, not established mechanisms:** the odd coordinates may perturb
spatial order/continuation or leave the true light unenumerated, eliciting its
re-emission and further cars. A useful non-owner internal cue is possible, but
the example does not establish visual scratchpad reasoning or that hallucination
generally helps. Only the action was changed; interpreting why the deterministic
suffix changed requires more than this post-selected case. No new generation,
training reward, penalty, or desc masking policy is adopted here.

Machine-readable labels and 63-view hash receipt live under the output root's
`a-owner-followup-v1/root-visual/`. Exact case comparisons are in
`a-owner-followup-v1/root-visual/a486123/case-attribution.json`.

Feedback remains pending. A + owner remains the active research focus.
