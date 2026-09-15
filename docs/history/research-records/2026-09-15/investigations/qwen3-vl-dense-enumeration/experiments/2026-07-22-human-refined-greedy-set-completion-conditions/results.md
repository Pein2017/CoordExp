---
title: Human-Refined Greedy Set Completion under Controlled Ground-Truth Prefix Assistance Results
description: Executed twelve-image evidence on native stopping, same-covered-set prefix order, one-row causal intervention, and progressive coordinate release for Qwen3-VL dense object enumeration.
type: investigation
role: research-results
authority: executed_evidence
architecture_promotion_status: not_promoted
unit_id: 2026-07-22-human-refined-greedy-set-completion-conditions
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-22
---

# Human-Refined Greedy Set Completion under Controlled Ground-Truth Prefix Assistance Results

## Verdict

The source checkpoint does not expose a reliable native procedure that
redistributes probability across all remaining physical objects until the set
is complete. It does, however, contain substantial local capabilities that are
hidden by unstable row-to-row routing:

1. a valid uncovered row is sometimes immediately behind a native terminal
   decision;
2. earlier prefix order changes later object selection even when the covered
   owner set, remaining owner set, prefix length, and immediately preceding row
   are all fixed;
3. supplying one discriminative coordinate can sometimes select a same-class
   owner and let the model recover the rest of its box;
4. there is no universal coordinate at which complete instance binding occurs;
   the earliest sufficient coordinate depends on which candidates remain, and
   a later extent boundary can still fail independently.

The deepest supported diagnosis is therefore not simply "the model stops too
early" or "the model has no memory." The prefix creates a distributed route
state that affects terminal choice, category choice, spatial owner selection,
box completion, and the subsequent suffix. Pure cross-entropy training has not
made that state reliably set-completing or invariant to harmless permutations
of already covered objects.

The result does not yet require object slots or an explicit covered-set memory.
It instead motivates a smaller training test: at self-generated prefixes,
increase the aggregate score of at least one valid uncovered physical owner
over terminal, duplicate, covered-owner, and invalid alternatives, while
preserving the model's freedom to choose its own next owner. Same-class
coordinate supervision should be applied progressively at the first spatial
boundary that actually separates the candidates, rather than assuming that
description or `x1` is always sufficient.

## Executed Evidence

The experiment used the frozen description-first, geometry-sorted,
pure-cross-entropy plus token-type-gate Weight-Decomposed Low-Rank Adaptation
checkpoint at step `4887`:

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_
accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json
```

Conclusion-bearing decoding used the Hugging Face backend, Scaled Dot Product
Attention, full-model 32-bit floating point, physical batch size `1`, repetition
penalty `1.0`, greedy decoding, and a `4,096` new-token ceiling. No arm reached
the token ceiling; the longest generated suffix used `455` tokens.

The twelve human-refined validation images were:

```text
1584, 2685, 4134, 5001, 6040, 7511,
10707, 13348, 13923, 14038, 14439, 16228
```

The authoritative rows came from:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

These are selected mechanism cases with 346 trusted Common Objects in Context
80-category annotations. They are not a blind population benchmark.

The immutable output root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-human-refined-greedy-set-completion-conditions/
```

## 1. Native Completion and Terminal Suppression

The main completion grid contains 63 unique image-by-prefix-depth states. Each
state was decoded under native greedy stopping, one-time terminal suppression,
and repeated terminal suppression. This gives 189 logical condition arms;
equivalent conditions shared the same model call when their executed prompt
and decode policy were identical.

| Decode condition | Complete rows generated | Unique remaining-owner credits |
|---|---:|---:|
| Native greedy | 444 | 248 |
| Suppress only the first terminal choice | 478 | 254 |
| Suppress terminal choices until the row budget | 622 | 260 |

Of the 63 native states, 34 encountered a terminal decision before exhausting
their row budget. One-time suppression added exactly one row in all 34, but
only 6 of the 63 total states gained any additional trusted owner. In 28 of the
34 early-terminal states, the added row did not discover a new owner.

Repeated suppression added 178 complete rows over native decoding but only 12
additional owner credits. It improved the trusted match count in 9 of 63
states, while 25 states generated extra rows without any new trusted owner. It
also introduced duplicates, unresolved rows, and invalid geometry. It created
no new fully completed state.

With no ground-truth row prefix, repeated suppression increased generated rows
from 246 to 346 across the twelve images, while conservative unique-owner
coverage increased only from 130 to 133. None of the twelve no-prefix runs
completed its trusted set.

Native completion occurred at some tested assistance depth for only five of
the twelve images:

| Image identifier | Tested remaining-object depths with native completion |
|---:|---|
| 5001 | 1 |
| 10707 | 1 |
| 13923 | 4, 2, 1 |
| 14038 | 1 |
| 16228 | 1 |

These results establish that early stopping is real and occasionally hides a
valid row. They also show that it is not the main global obstruction. Removing
the terminal choice does not tell the model which uncovered owner should
receive the released probability.

The no-override controls were exact: all 29 one-time-suppression controls and
all 29 repeated-suppression controls matched native raw token identifiers when
no terminal choice was actually overridden.

## 2. Same Covered Set, Different Prefix Order

The matched-order study held fixed:

- the image;
- the set of already covered physical owners;
- the set of remaining physical owners;
- prefix length;
- the final forced row immediately before generation.

Only the order of earlier covered rows changed. Five variants were compared:
geometry order, reverse geometry order, category-grouped order, and two fixed
random orders. The panel contains 36 matched states: twelve images at remaining
depths `8`, `2`, and `1`.

| Observable changed under at least one order | Matched states |
|---|---:|
| Raw generated suffix | 30 / 36 |
| Set of discovered remaining owners | 13 / 36 |
| Number of discovered remaining owners | 12 / 36 |
| Complete versus incomplete outcome | 2 / 36 |

The owner-set changes were not confined to long prefixes. By remaining depth:

| Remaining depth | Raw suffix changed | Owner set changed | Owner count changed | Completion changed |
|---:|---:|---:|---:|---:|
| 8 | 12 / 12 | 7 / 12 | 6 / 12 | 0 / 12 |
| 2 | 9 / 12 | 4 / 12 | 4 / 12 | 0 / 12 |
| 1 | 9 / 12 | 2 / 12 | 2 / 12 | 2 / 12 |

Four conclusion-bearing states were independently re-executed once under each
of the five order variants. This produced 20 paired replays, one new execution
for each original state-by-variant condition, and all 20 matched the original
raw token output exactly. This is a one-repeat deterministic check for each
condition, not 20 repetitions per condition. Within that bounded check, the
observed order differences reproduced under greedy decoding rather than arising
from sampling noise.

Representative cases show several mechanisms:

- Image `5001`, one remaining owner: geometry and category order recover the
  scissors; reverse and random orders stop immediately.
- Image `14038`, one remaining owner: geometry order recovers the cell phone;
  reordered prefixes produce book rows instead.
- Image `10707`, two remaining bottles: geometry order generates category
  `cup` at the two bottle locations; reordered prefixes can recover the trusted
  `bottle` category.
- Image `16228`, eight remaining owners: geometry and category order reach a
  trusted bench, while reverse and random orders additionally reach person
  owner `1703142`.

Because the last forced row and prefix length were fixed, these effects cannot
be reduced to immediate last-row inertia or a simple length penalty. The
history before the last row continues to alter later routing.

## 3. Matched Terminal Cases

Two matched-order states isolate when terminal choice is locally decisive.

### Image 5001

With one owner remaining, geometry and category order natively generate the
scissors. Reverse and random order stop immediately. Suppressing only that
first terminal decision under the reverse and random prefixes recovers a valid
scissors row with coherent geometry.

This is a clean local terminal-gate failure: the correct row is directly
available, but it does not beat terminal under those prefix states.

### Image 16228

With eight owners remaining, the geometry-order suffix reaches trusted bench
owner `575192` and then stops. One-time suppression adds person owner `-57`.
Repeated suppression additionally reaches person owners `-45` and `1703142`,
but also generates lower-quality rows.

This shows that several valid owners can remain locally reachable after a
native stop. It simultaneously shows why repeated terminal suppression is a
diagnostic rather than a treatment: quality degrades as the model is repeatedly
forced past its native decision.

## 4. One-Row Causal Replay

Every admitted causal case passed two validity gates:

1. appending the exact native row through the same branch path reproduced the
   natural suffix exactly at the raw-token level;
2. generated rows were assigned globally one-to-one across the branch and
   suffix, so a forced branch could not also be counted as a newly discovered
   owner.

### Image 5001: category access without stable extent

The remaining trusted owners were a handbag and scissors. Native greedy
decoding generated the scissors and then stopped.

- Supplying only description `handbag` generated a handbag-shaped row
  `[860,549,887,704]`, but its intersection over union with the trusted handbag
  was only `0.2408`; the suffix still retained the native scissors row.
- Supplying the complete trusted handbag row retained the scissors suffix, so
  the assisted trajectory represented both owners.
- Supplying an already covered person row caused immediate terminal output and
  lost the scissors route.

The valid remaining row can add to, rather than necessarily replace, the
native suffix. The covered-owner control also rejects the explanation that any
extra object row merely triggers generic continuation. Description can select
the handbag category here, but it does not recover the trusted physical extent.

### Image 10707: category and same-class owner are different decisions

Two trusted bottles remained: left owner `-187` with box
`[226,786,266,902]`, and right owner `-186` with box
`[282,788,326,903]`.

Native decoding generated category `cup` at both bottle locations. Thus the
spatial supports were accessible even though the category did not follow the
trusted ontology.

- Supplying only description `bottle` for intended right owner `-186`
  generated the right bottle with intersection over union `0.8737`.
- Supplying the identical description for intended left owner `-187`
  generated the same right bottle, with intended intersection over union `0`.
- Supplying the complete left-bottle row caused the released suffix to recover
  the right bottle, jointly covering the pair.

Description resolves category but is not a same-class instance pointer.
Exact geometry changes the successor route and can expose the other instance.

### Image 14038: a category-route failure

Under a reverse prefix, only a cell phone remained, but native decoding and an
already-covered-book control continued along book rows. Supplying description
`cell phone` generated box `[906,955,956,999]`, with intersection over union
`0.9378`, and then stopped. Supplying the complete phone row produced the same
behavior.

Here the target geometry was already available after the correct category was
selected. The failure occurred before localization, in category or route
selection.

### Image 16228: description cannot select among people

- Supplying description `person` for intended owner `1703142` generated that
  owner and then retained the trusted bench route.
- Supplying the same description for intended owner `-57` generated owner
  `1703142` again rather than `-57`.
- Supplying the complete row for either person retained the trusted bench
  later in the suffix.

Description again fails as a same-class owner pointer. Exact person geometry
changes the route without universally consuming one fixed unit of later
rollout capacity. This rejects a simple model in which every emitted object
always replaces exactly one later object.

## 5. Progressive Coordinate Release

The experiment then supplied only the beginning of a trusted row and allowed
the model to generate all remaining coordinate tokens and the subsequent
suffix.

### Horizontally separated same-class owners

For the left bottle in image `10707`:

| Supplied trusted coordinates | Generated complete box | Direct intended intersection over union |
|---|---|---:|
| `x1=226` | `[226,789,263,900]` | 0.8851 |
| `x1=226, y1=786` | `[226,786,263,900]` | 0.9091 |
| `x1=226, y1=786, x2=266` | `[226,786,266,900]` | 0.9828 |

Every arm subsequently recovered the right bottle. For this pair, trusted
`x1` was enough to select the left instance and induce a coherent remaining
box.

For right-side person owner `-57` in image `16228`:

| Supplied trusted coordinates | Generated complete box | Direct intended intersection over union |
|---|---|---:|
| `x1=739` | `[739,492,822,641]` | 0.8272 |
| `x1=739, y1=484` | `[739,484,822,641]` | 0.8670 |
| `x1=739, y1=484, x2=813` | `[739,484,813,641]` | 0.9691 |

These two cases establish that a minimal spatial cue can causally select an
owner and let the native model transcribe most of its box. They do not make
`x1` a universal binding coordinate.

### Vertical-book negative control

Image `14038` contains two vertically separated trusted books with the same
left boundary `x1=787`:

- upper owner `-122`: `[787,501,863,518]`;
- lower owner `-130`: `[787,771,834,803]`.

All other 45 trusted owners were placed in the matched prefix, leaving exactly
these two books. The description-only and `x1`-only arms for the upper and lower
targets had identical prefix hashes and produced the same lower-stack book.
They therefore contained no information capable of distinguishing the two
owners.

Adding `y1` separated the trajectories, but did not guarantee the complete
extent:

| Intended owner | Supplied trusted coordinates | Generated complete box | Direct intended intersection over union |
|---|---|---|---:|
| Upper book | `x1,y1` | `[787,501,822,502]` | 0.0271 |
| Upper book | `x1,y1,x2` | `[787,501,863,868]` | 0.0463 |
| Lower book | `x1,y1` | `[787,771,829,806]` | 0.8245 |
| Lower book | `x1,y1,x2` | `[787,771,834,806]` | 0.9143 |

The upper-book `y2` either collapsed or expanded across much of the stack even
after the other three boundaries were supplied. This is direct evidence that
the final extent boundary can fail independently of category, `x1`, `y1`, and
`x2`.

The dense book stack contains overlapping and visually ambiguous instances.
The direct intended intersections over union above are descriptive. Automatic
one-to-one matching conservatively leaves some lower-stack assignments
ambiguous and does not promote those values to authoritative owner credits.

## 6. Visual Review

Crop-enlarged review was used for the conclusion-bearing ambiguous cases:

- image `5001`: the handbag and scissors are small and partly occluded;
- image `10707`: the physical objects look like cans or containers, making
  trusted `bottle` versus generated `cup` partly an ontology question rather
  than entity hallucination;
- image `14038`: the cell phone is visually clear, while the vertical book
  stack has ambiguous individual extents;
- image `16228`: the person is real but low-pixel and near the image edge.

The review artifacts are under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-human-refined-greedy-set-completion-conditions/visual-review/
```

This review reinforces the need to keep two judgments separate:

1. whether a physical entity and category were discovered;
2. whether all four boundaries recover an acceptable physical extent.

## Mechanistic Model

At object-row granularity, the observed process is better represented as a
sequence of conditional decisions than as one atomic object prediction:

\[
\begin{aligned}
p(y_t\mid I,P_t)
={}&p(\mathrm{continue}\mid I,P_t)\\
&\cdot p(d_t\mid \mathrm{continue},I,P_t)\\
&\cdot p(x_1\mid d_t,I,P_t)\\
&\cdot p(y_1\mid d_t,x_1,I,P_t)\\
&\cdot p(x_2\mid d_t,x_1,y_1,I,P_t)\\
&\cdot p(y_2\mid d_t,x_1,y_1,x_2,I,P_t),
\end{aligned}
\]

where `I` is the image, `P_t` is the entire serialized prefix, and `d_t` is
the description. Appending the completed row changes the next state:

\[
p(y_{t+1}\mid I,P_t,y_t).
\]

The prefix-order experiment shows that `P_t` is not reducible to the set of
covered owners plus the last row. Earlier serialization order changes this
conditional distribution.

The coordinate-release experiment is consistent with progressive candidate
filtering. For a candidate physical owner `o`, a conceptual posterior is:

\[
q(o\mid I,P_t,d,x_1,\ldots,x_k)
\propto
q(o\mid I,P_t)
\mathbf{1}[\operatorname{class}(o)=d]
\prod_{j=1}^{k}\operatorname{compatibility}(x_j,o).
\]

This is an explanatory model, not a claim that the checkpoint contains an
explicit symbolic filter. It explains why `x1` is sufficient when candidates
are horizontally separated, why `y1` is needed when they share `x1`, and why a
remaining boundary can still drift after owner evidence has narrowed.

The evidence supports a weak form of commit: writing a row changes which later
rows and terminal decisions are likely. It does not support a reliable
set-valued commit operation that always transfers probability from a completed
owner to some valid uncovered owner.

## Competing Hypotheses after the Experiment

| Hypothesis | Result | Evidence boundary |
|---|---|---|
| Early terminal choice is the main cause of low recall | Rejected as a global explanation | Repeated suppression added many rows but few owners and no completed state |
| Prefix is only a record of the covered owner set | Rejected | Matched covered-set permutations changed later owner sets and completion |
| Only the immediately preceding row matters | Rejected | The last row was fixed in the matched-order panel |
| Description universally fixes the physical instance | Rejected | Bottle and person descriptions selected the same favored same-class owner |
| `x1` universally completes instance binding | Rejected | Vertically separated books shared `x1`; `y1` was necessary, and `y2` still failed |
| Coordinate tokens are independent boundary regressors | Rejected literally | Supplying an earlier coordinate causally changed later coordinates and suffix routes |
| Every emitted row consumes one fixed unit of later rollout capacity | Rejected | Assisted valid rows sometimes retained the native suffix and increased represented owners |
| The model has no native object or coverage state whatsoever | Not established | Rows causally alter later selection, but the state is neither reliably readable nor set-completing |
| A compact explicit covered-set carrier is required | Still open | This unit did not compare a learned or oracle carrier against native state |
| The vision tower lacks the relevant entities | Not supported for the selected rescue cases | Minimal category or coordinate cues often exposed coherent native rows; some low-pixel cases remain uncertain |

## Training Consequence

The next training screen should not force equal probability over all remaining
owners and should not prescribe one canonical successor. A narrow local route
is acceptable if repeated transitions eventually cover the set. The direct
failure is that valid uncovered routes often lose to terminal, a covered owner,
an invalid row, or a habitual but incomplete successor.

Let `S(y | I, P)` be the length-normalized score of a complete candidate row
`y` at a self-generated prefix `P`. Let `U(P)` be valid uncovered physical
owners and `C(P)` covered owners. Let `Y(o)` contain the acceptable row
realizations for owner `o`, including approved geometric ambiguity where
necessary. First aggregate alternative annotations within each owner without
rewarding an owner merely for having more acceptable annotations:

\[
S_o(P)
=
\operatorname{logmeanexp}_{y\in Y(o)} S(y\mid I,P).
\]

Then define:

\[
A_U(P)
=
\operatorname{logsumexp}_{o\in U(P)} S_o(P).
\]

Define the competing harmful mass:

\[
A_B(P)
=
\operatorname{logsumexp}
\left(
S_{\mathrm{terminal}},
\{S_o(P):o\in C(P)\},
S_{\mathrm{invalid\ branch}}
\right).
\]

A minimal set-transition objective is:

\[
\mathcal{L}_{\mathrm{transition}}
=
\operatorname{softplus}
\left(m + A_B(P) - A_U(P)\right).
\]

Here `m` is a positive safety margin. This objective asks that at least one
valid uncovered route collectively beat the harmful alternatives. It does not
make all uncovered owners equally likely and does not require the same next
owner under every valid prefix order.

Three additions are justified by the evidence:

1. **Use self-generated prefixes and physical owner identifiers.** Canonical
   teacher prefixes alone do not expose the route states that fail in rollout.
2. **Add matched prefix permutations.** For prefixes with the same covered set,
   require sufficient aggregate valid-uncovered mass under each order, without
   aligning their exact next-owner distributions.
3. **Use progressive same-class coordinate supervision.** Supply or score the
   earliest coordinate prefix that distinguishes a target from current
   same-class distractors; do not hard-code `x1` as the binding point. Retain a
   complete-box consistency term because a later extent boundary can remain
   wrong after the owner is mostly determined.

Terminal suppression remains useful only for mining positive routes that were
locally available. Description-only contrastive binding is insufficient for
same-class cases. An explicit object-slot or covered-set architecture should be
tested only if the loss-based screen cannot improve greedy set coverage while
native prefix states still preserve the relevant owner evidence.

The recommended next scale is a 256-image treatment screen with fresh
self-rollouts, trusted or reviewed physical-owner identities, and a source
checkpoint control. Promotion should depend on clean greedy unique-owner
recall, entity-versus-geometry breakdown, duplicate and invalid rates, and the
gap between low-temperature bagging and one greedy rollout. This experiment
does not authorize that training.

## Relation to Earlier Units

This unit reconciles three earlier observations:

- The 2026-07-16 coordinate-release factorial established a causal
  `x1`-to-`x2` transport effect in one dense-chair state but could not separate
  physical-owner state from box grammar.
- The 2026-07-18 fixed-prompt replication rejected a universal completed owner
  at description or `x1`; in a nearby-person control, `y1` began the route
  change and later coordinates separated the people.
- The 2026-07-21 forced-owner-path intervention showed that the earliest useful
  supplied information varies across cases and that current-row correctness
  and downstream set value are distinct.

The new matched candidate geometry explains the apparent tension: there is no
single universal binding token. The earliest discriminative coordinate depends
on the geometry of the remaining candidate set, and complete physical extent
can remain unresolved afterward.

## Stop Decision

The unit meets its stop rule:

- all twelve human-refined images have a sparse completion surface;
- zero conclusion-bearing runs ended by token truncation;
- terminal, prefix-order, same-class owner, and extent failures each have a
  matched diagnostic or causal case;
- broad terminal suppression and broader one-row probing now have diminishing
  information value;
- the next material question requires training or a qualitatively new state
  intervention.

No architecture is promoted. The next decision is whether to authorize the
256-image set-transition and progressive-coordinate training screen described
above.

## Primary Artifacts

Main completion and terminal grid:

```text
geometry-depth-grid-*.json
smoke-*-geometry-depths-N-8-2-1.json
```

Same-covered-set order and deterministic replay:

```text
matched-order-grid-a.json ... matched-order-grid-h.json
replicate-5001-depth1.json
replicate-14038-depth1.json
replicate-10707-depth2.json
replicate-16228-depth8.json
```

Matched terminal panels:

```text
matched-stop-panel-5001-depth1.json
matched-stop-panel-16228-depth8.json
```

One-row and coordinate causal panels:

```text
causal-micro-panel-5001.json
causal-micro-panel-10707.json
causal-micro-panel-14038.json
causal-micro-panel-16228.json
partial-coordinate-panel-10707.json
partial-coordinate-panel-16228.json
partial-coordinate-panel-14038-vertical-books.json
```
