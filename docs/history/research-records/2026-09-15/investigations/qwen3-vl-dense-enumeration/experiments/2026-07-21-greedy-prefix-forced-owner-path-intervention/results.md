---
title: Greedy-Prefix Forced Object-Path Intervention Results
description: Bounded evidence about when sampled object paths become portable to exact greedy prefixes and whether they improve fixed-budget unique-instance coverage.
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-07-21-greedy-prefix-forced-owner-path-intervention
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: bounded_mechanistic_evidence
updated: 2026-07-21
---

# Greedy-Prefix Forced Object-Path Intervention Results

## Verdict

Three of the four frozen cases pass the predeclared mechanical gate for a
small training screen. A low-temperature sampled object's path can therefore
be transported into an exact greedy state without injecting the complete row.
The required supplied prefix is not universal: it ranges from the description
token to one or two coordinate boundaries, and local entity acquisition alone
does not guarantee beneficial downstream set coverage.

This result supports testing a prefix-conditioned, coherent-row training
objective. It does not support a final architecture, a universal object-commit
token, or beam search as the next research step.

## Execution Receipt

- Source checkpoint: geometry-sorted, pure cross-entropy, token-type-gated
  Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter at step 4887.
- Release policy: greedy decoding with repetition penalty `1.0`.
- Total generated-token budget per arm: `512`.
- Source annotation file digest:
  `81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894`.
- Frozen case-manifest digest:
  `75af70466eeff6966c2fc70730113c22c0f4909ba3b2977cce677942c6a5c0e0`.
- All four artifacts contain the same manifest digest.
- All four native horizons reproduce the recorded greedy behavior exactly.
- Every non-terminal case passes exact-token native forced-prefix no-op parity.
  The terminal case instead reproduces the native terminal token exactly.
- Target and control staircases use the same token depth. A completely forced
  row is excluded from decoder-acquisition and training-promotion decisions.
- The final fixed-point rerun gives every control arm an explicit row-and-token
  budget receipt. Direct and nested current-row and suffix target-acquisition
  fields agree in all four artifacts.
- Focused tests: `6 passed` with the repository's unrelated shared test
  configuration bypassed through `--noconftest`.

Final artifacts:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-greedy-prefix-forced-owner-path-intervention/
forced-owner-path-step4887-20260721a/final/
```

## Case Results

| Case | Earliest strict automatic acquisition by released decoder | Earliest useful fixed-budget rung | Matched control | Interpretation |
|---|---|---|---|---|
| `person-5001-row3` | first horizontal boundary (`x1`) | none | a non-target control path also reaches the target later | The intervention changes when the person appears, but does not add unique fixed-budget coverage. This is reordering, not a training-positive rescue. |
| `wine-glass-2685-row6` | first horizontal boundary (`x1`) | first vertical boundary (`y1`) | control never reaches target | `x1` identifies the glass locally but loses two native owners and gives a unique-owner delta of `-1`. Adding `y1` yields the target plus two later owners, removes one native owner, gives a unique-owner delta of `+2`, retains `7/8` native non-target owners, and creates no new duplicate owner. |
| `person-7511-row4` | second horizontal boundary (`x2`) under the frozen automatic matcher | `x2` under the frozen gate | same-category person control never reaches target | Automatic matching needs the reproduced donor extent, but crop review shows that the `x1` and `y1` hybrids already point to the same physical person. Geometry remains loose, so this case supports entity-path training but not coordinate-loss supervision. The strict `x2` rung adds one unique owner, retains all five native non-target owners, and creates no new duplicate owner. |
| `truck-13348-after-stop` | description token `truck` | description token `truck` | syntax-matched `person` control never reaches target | Supplying the category after the native terminal is enough for the released decoder to generate a complete box for one real truck, add one unique owner, retain all native owners, and create no new duplicate. Because the control category differs and several trucks exist, this establishes category-conditioned coherent continuation, not a universal instance-binding rule. |

For `wine-glass-2685-row6`, owner `2685:-83` is not present in the six-row
recipient prefix. Native row 6 introduces it and a later native row repeats it.
The target arm replaces that native branch with owner `2685:-78`; losing
`2685:-83` from the fixed-budget final set is therefore a real opportunity
cost, not an omitted parent-ledger entry.

The three strict promotion-positive cases are therefore:

```text
wine-glass-2685-row6 at y1 or later
person-7511-row4 at x2 or later
truck-13348-after-stop at description or later
```

The complete-row arm is excluded in all cases because the decoder contributes
no token to the intervened row.

## Crop-Assisted Review

Entity existence and box quality were reviewed separately. The enlarged
comparison crops are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-greedy-prefix-forced-owner-path-intervention/
forced-owner-path-step4887-20260721a/final/review/
2685-decisive-crop-x5.png
7511-decisive-crop-x5.png
13348-decisive-crop-x5.png
```

The `2685` target and control are neighboring wine glasses. The target hybrid
is correctly attached to the intended glass, while the control remains on the
neighbor. The `7511` donor and the one-bin-wider `x1`/`y1` hybrid cover the same
tiny person near the right image boundary; both are looser than the refined
reference box. The `13348` description-only release reconstructs the same
leftmost truck as the sampled donor, while the control identifies a distinct
person. No crop changed a real entity into an entity hallucination.

## Mechanistic Interpretation

### 1. There is no single demonstrated commit position

The earliest sufficient supplied information depends on the state and scene:

- a category token can cross a native terminal and let the model write a
  coherent box;
- `x1` can be enough to identify an entity, but not enough to preserve later
  set coverage;
- a later boundary can improve both current geometry and the subsequent path;
- an automatic overlap threshold can make binding appear later than visual
  entity identity actually does.

The experiment therefore rejects a universal claim such as "the model always
binds an instance at `x1`" or "the description always fixes the instance."

### 2. Current-row correction and later set value are distinct

The `2685` result is the clearest example. Forcing `x1` makes the current glass
correct, yet produces a worse fixed-budget owner set. One additional vertical
boundary changes the later greedy trajectory and improves unique-owner
coverage. A useful treatment must therefore evaluate both:

1. whether the current row belongs to a real uncovered entity; and
2. what that row does to the remaining fixed-budget rollout.

### 3. A sampled rescue is often already supported by the greedy state

Three partial-prefix interventions succeed without importing the sampled
trajectory's earlier history. The model can complete the remaining row under
the exact greedy prefix. This reduces the probability that every rescue
requires an inaccessible sampled-history state. It is consistent with a
ranking or credit-assignment problem in which a valid row path exists but does
not win under native greedy decoding.

This is not proof that the model has a complete native coverage ledger. The
experiment only shows local transportability for three selected rescue rows.

### 4. Entity discovery is stronger than exact extent estimation

The `7511` crop demonstrates that a row may identify the intended person while
missing an automatic overlap threshold because one or more boundaries are
loose. Training and evaluation must keep these judgments separate:

- physical entity and category;
- center or location;
- four boundary coordinates and full-box extent;
- downstream unique-instance coverage.

## Training Consequence

The stop rule is satisfied, so the next justified step is a small training
screen rather than another decoding search method. The minimal treatment
should compare coherent candidate rows under the same model-visited prefix:

\[
L_{\text{row preference}}
=
\log\left(1+
\exp\left[m-S(Y_{\text{new owner}}\mid I,P)
+S(Y_{\text{bad branch}}\mid I,P)\right]
\right),
\]

where:

- `P` is a frozen own-rollout prefix;
- `Y_new owner` is a reviewed row for a real uncovered physical entity;
- `Y_bad branch` is the native repeat, unresolved harmful branch, or terminal;
- `S` scores the complete row, not only its first token; and
- positive events must have non-negative fixed-budget downstream set value.

Coordinate-token cross-entropy or a coordinate correction loss may be added
only when geometry is trusted. Entity-only rescues such as `person-7511-row4`
must not supervise exact boundaries. The required prefix depth should remain a
diagnostic and weighting signal; it should not become one hard-coded universal
training boundary.

The first screen should remain small and answer whether this treatment moves
sampled-only physical owners into greedy rollout without increasing new
duplicates, malformed rows, or unsupported entities. It should not yet add an
external slot, detector, or explicit covered-set carrier.

## Beam Search Decision

Beam search is deferred. Standard beam search expands several high-probability
continuations that usually share early prefixes and also inherits sequence
length bias. In this task it may spend more computation refining the same
conservative route rather than expose different valid object orders. Existing
low-temperature independent sampling already supplies diverse, reviewed rescue
rows and directly serves the current training question.

Beam search can later be an equal-compute diagnostic baseline if the question
becomes whether structured search recovers more unique physical owners than
independent sampling. It is not the highest-value next treatment.

## Limitations

- There are only four deliberately selected cases; the evidence is causal but
  bounded, not a population estimate.
- The `7511` entity verdict is crop-supported while its geometry is explicitly
  untrusted.
- The `13348` control does not hold category constant, so the result cannot
  separate a latent instance pointer from category-conditioned traversal.
- `person-5001-row3` demonstrates timing change but not added coverage.
- Unresolved suffix rows remain unknown. They were not counted as entity
  hallucinations or silently promoted to unique owners.
- No result establishes a full native commit or coverage-memory mechanism.
