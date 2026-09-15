---
title: Sampled-Rescue Object Transition Distribution and Causal Replay Results
description: Verified fixed-prefix and forced-token evidence for object-mode fragmentation, within-row binding, and cross-row transition behavior in two dense-scene cases.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-14-sampled-rescue-object-transition-causal-replay
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-14
---

# Sampled-Rescue Object Transition Distribution and Causal Replay Results

## Scope and Evidence Identity

This exploratory case study used the frozen step-4,887 Qwen3 Vision-Language
(`Qwen3-VL`) checkpoint and configuration declared in [the owning research
unit](unit.md). The primary sampled decode condition used temperature `0.4`,
top-p nucleus threshold `0.95`, and repetition penalty `1.0`. Greedy controls
disabled sampling and used the same repetition penalty. All fixed-prefix
conditions preserved the same image, processor, prompt, tokenizer, checkpoint,
and exact recipient token prefix within their comparison.

The resolved artifact root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-14-sampled-rescue-object-transition-causal-replay/
```

The executed evidence is mechanics-only and case-bounded. Primary handles are:

- Wave 0 greedy-anchor receipts and frozen object-ledger summary:
  `wave0-greedy-anchors/image-{12576,15254,19432,2299}/receipt.json` and
  `wave0-greedy-anchors/wave0-greedy-vs-sampled-ledger-summary.json`;
- Wave 1 first real fixed-prefix smoke:
  `wave1-fixed-prefix-smoke/image-12576/greedy-terminal/receipt.json`;
- Wave 2 image `12576` rescue-entry and greedy-terminal receipts:
  `wave2-fixed-prefix-distribution/image-12576/rescue-entry/receipt.json`,
  `rescue-entry-expansion-{2,3,4}/receipt.json`,
  `greedy-terminal-expansion-{2,3,4}/receipt.json`, and the Wave 1 receipt;
- Wave 2 temperature-sensitivity receipts:
  `wave2-fixed-prefix-distribution/image-12576/{rescue-entry,greedy-terminal}-temperature-{0p2,0p6}/receipt.json`;
- Wave 2 independent image `19432` receipts:
  `wave2-fixed-prefix-distribution/image-19432/rescue-entry-chair-383277/receipt.json`
  and `greedy-terminal/receipt.json`;
- Wave 3 nested forced-token receipts:
  `wave3-token-causal-replay/image-12576/{shared-row-opener,target-pizza-first-description,target-pizza-complete-description,target-pizza-first-coordinate,competing-cup-first-description,competing-cup-complete-description,competing-cup-first-coordinate}/receipt.json`;
- Wave 3 complete-row receipts:
  `wave3-token-causal-replay/image-12576/{commit-target-pizza,commit-competing-cup,commit-covered-duplicate-pizza,commit-unsupported-chair}/receipt.json`;
- current runtime attestation:
  `runtime-attestation/request-scoped-sampling-three-policy-current-cuda.json`.

The primary image-`12576` fixed-prefix state is called **prefix state 56
(`P56`)**, meaning the exact legal assistant continuation containing 56
generated tokens immediately before the sampled target pizza row. The frozen
object ledger identifies:

- target pizza: `coco-ann:1571077`;
- competing left cup: `coco-ann:678023`;
- later right cup: `coco-ann:678923`.

## Observed

### Wave 2: fixed-prefix one-row distributions

At exact `P56`, temperature `0.4`, top-p nucleus threshold `0.95`, and
repetition penalty `1.0`, 32 unique sampled continuations produced:

| First action | Count |
|---|---:|
| Target pizza `coco-ann:1571077` | 11 of 32 |
| Competing left cup `coco-ann:678023` | 21 of 32 |
| Any other first action | 0 of 32 |

The target pizza appeared anywhere in 12 of 32 continuations. Greedy decoding
at the identical state selected the left cup. Every first sampled action was a
valid object row.

At the exact greedy-terminal prefix for the same image, 32 sampled
continuations produced 19 immediate terminal no-more-objects actions and 13
near-identical knife fragments overlapping an already covered knife. The
target pizza and every other newly accepted object appeared in 0 of 32 draws.
The knife fragments are classified as repeated or localization-mismatched
covered-object outputs, not new accepted objects.

The independent image-`19432` case used rescued chair
`coco-ann:383277`. At its rescue-entry state, greedy decoding selected the
chair and seven of eight temperature-`0.4` samples matched it; the eighth was a
same-class localization miss. At its greedy-terminal state, eight of eight
samples terminated immediately.

Temperature `0.2` and `0.6` were retained as separate sensitivity conditions.
They changed frequency but did not reverse the state contrast: the image-12576
target pizza appeared in one of eight and three of eight rescue-entry draws,
respectively, and in zero terminal-state draws at either temperature.

### Wave 3: nested token-level replay

The **row opener only** condition forced exactly the object-reference opening
token and then released greedy decoding. It produced the left cup. Forcing one
source-specific first description token produced a complete valid row whose
phrase and box matched that source:

| Forced cumulative span | Released greedy result |
|---|---|
| Row opener plus `pizza` | Target pizza phrase and target pizza box |
| Row opener plus `cup` | Left cup phrase and left cup box |
| Complete `pizza` description | Target pizza phrase and target pizza box |
| Complete `cup` description | Left cup phrase and left cup box |
| `pizza` description through first coordinate | Target pizza phrase and target pizza box |
| `cup` description through first coordinate | Left cup phrase and left cup box |

Thus, the first source-specific description token is the earliest tested span
that separated these two complete-row outcomes. The intervention is textual:
it does not by itself show that the decoder selected a visual instance before
the description token. In particular, the token `pizza` combined with `P56`
may let the learned traversal state resolve the canonical same-class geometry.

### Wave 3: complete-row-conditioned next transitions

Each condition forced exactly nine tokens forming one complete row and then
ran one greedy continuation with sampling disabled plus eight
temperature-`0.4`, top-p nucleus-threshold-`0.95` next-action continuations:

| Forced complete row | Next first action |
|---|---|
| Target pizza `coco-ann:1571077` | Left cup `coco-ann:678023`, 9 of 9 |
| Left cup `coco-ann:678023` | Right cup `coco-ann:678923`, 9 of 9 |
| Already covered earlier pizza | Left cup, 8 of 9; target pizza, 1 of 9 |
| Unsupported chair, syntax- and length-matched | Left cup, 9 of 9 |

All 36 next actions were valid object rows. No condition produced an immediate
terminal action or invalid first action. After forcing the left cup, the target
pizza did not reappear in any of the nine next actions.

## Supported

1. **Fixed-state object-mode fragmentation is supported for image `12576` at
   `P56`.** The identical state contains two recurring valid next-row modes:
   the left cup and target pizza. Greedy decoding selects only the stronger
   left-cup branch.
2. **Earlier-trajectory state dependence is supported in two bounded cases.**
   The rescued object is repeatedly available at an earlier rescue-entry state
   but absent at the paired greedy-terminal state for images `12576` and
   `19432`.
3. **Strong within-row description-conditioned binding is supported.** One
   source-specific first description token is sufficient for the released
   decoder to complete a matching phrase and geometry in both the pizza and
   cup directions at `P56`.
4. **The dominant complete-row transition in this panel is compatible with a
   geometry/order/frontier-like successor process plus generic complete-row
   advancement.** The target-pizza, duplicate-pizza, and unsupported-chair rows
   all predominantly advance to the left cup, while the left-cup row advances
   to the right cup.

## Ruled Out

**Strict object-specific commit-to-uncovered redistribution is ruled out for
the image-`12576` `P56` case under this panel.** The required donor-object
crossover did not occur. Committing the target pizza did not redistribute the
next action to the left cup more specifically than an unsupported chair or an
already covered pizza did, and committing the left cup did not recover the
target pizza. The evidence instead shows a strong row-conditioned successor
transition whose exact state variable is unresolved.

The generic claim that fixed-prefix bagging rescues are only parser noise is
also ruled out for the two audited rescue-entry cases: the recovered pizza and
chair rows are valid and map to frozen ledger objects.

## Unresolved

- The exact factor driving the cross-row successor remains unresolved. The
  live candidates are last-row geometry, lexical or category inhibition,
  geometry-and-description conjunction, generic row progression, and a
  distributed property of the complete prefix state.
- Description-token forcing localizes a behaviorally decisive phase but does
  not establish a visual instance pointer, a pre-description object lock, or
  an endogenous object-selection variable.
- The selected cases do not determine how often fixed-state object-mode
  fragmentation occurs across the validation population or whether every
  bagging rescue has the same mechanism.
- Incomplete dense-scene annotations remain a safety obstacle for any
  termination or uncovered-object training target.

## Not Claimed

- no population-level probability, recall, mean average precision, or
  architecture conclusion;
- no claim that sampling is a desirable final detection policy;
- no claim that a persistent ledger, object slot, external detector, or visual
  cursor is necessary;
- no claim that the first description token is where visual instance identity
  is originally formed;
- no claim that the unsupported-chair control is visually meaningless beyond
  its frozen case classification;
- no authorization for training, an OpenSpec change, or a final forward-pass
  design.

## Execution Stop Decisions

No 32-sample Wave 3 expansion was run. The complete-row arms already produced
unanimous or near-unanimous directional results at one greedy plus eight
sampled requests, and the missing crossover was not a marginal frequency
question.

Wave 4, **Conditional Late-Middle Residual Replay**, was not entered. Full
phrase-and-box rows already succeeded after first-description-token forcing;
the remaining discriminator concerns cross-row prefix state rather than
within-row residual persistence.

## Training-Screen Gate

A **256-image training screen**, meaning a small matched-budget fine-tuning
comparison intended only to test learnability, is **not authorized**.

Training Gate 1 passes because audited target objects have stable support at
controlled rescue-entry states. Training Gate 2 identifies a source-specific
first-description phase, but that textual intervention does not yet define a
nontrivial endogenous visual-selection or commit target. The safe 256-image
label cohort and one frozen mechanism-level training arm required by Gates 3
and 4 are also absent. No training launch is implied by this unit.

## Exactly One Next Discriminator

Run one bounded **Prefix-State Phrase-Geometry Factorial**, operationally a
fixed-`P56`, equal-token-length causal panel that independently varies the
complete-row description token and complete-row geometry while holding the
recipient image, model, prefix length, row grammar, and decode policy fixed.
Include no-row, covered-duplicate-row, and irrelevant-row controls.

This single factorial must distinguish:

1. **geometry frontier**: the next action follows the committed geometry even
   when the description is counterfactual;
2. **category or lexical inhibition**: the next action depends mainly on the
   committed description category;
3. **phrase-geometry conjunction**: only a semantically and spatially coherent
   row produces the successor transition;
4. **generic row advancement**: every equal-length legal row advances to the
   same successor regardless of phrase and geometry.

This next unit is analysis and causal inference only. It does not authorize an
architecture, training objective, or training run.
