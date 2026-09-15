---
title: Visual-Support Counterfactual Commit Test Results
description: Verified one-case evidence that a coherent phrase-and-geometry prefix transition does not require the selected local post-vision support under same-position donor substitution.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-visual-support-counterfactual-commit
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Visual-Support Counterfactual Commit Test Results

## Scope and Evidence Identity

This unit used one exact Qwen3 Vision-Language (`Qwen3-VL`) state on Common
Objects in Context validation image `12576`. **Prefix state 56 (`P56`)** plus
the exact coherent nine-token `cup` description and left-cup geometry row was
held fixed. The preceding factorial showed that this state advances the first
free row to the right cup in all nine requests.

Only the encoded visual features were changed. The primary image-embedding
stream and all three DeepStack streams either replayed the clean recipient
features, replaced the 86-cell left-cup support with same-position features
from donor image `17436`, or applied the identical 86-cell substitution to a
disjoint upper-left control region. Tokens, positions, image-token count,
prefix, row text, model, adapter, decode policy, and paired seeds were fixed.

The verified artifact is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-visual-support-counterfactual-commit/
  paired-panel-20260715b/receipt.json
```

## Trust Gate

The executed visual grid was `1 x 72 x 54`, with merge size `2`, a merged
`36 x 27` grid, and 972 tokens in each visual stream. The target and control
masks each began with 88 cells and used the frozen symmetric pruning rule to
produce 86 translation-equivalent, disjoint cells.

For both substitution conditions, greedy and sampled controllers recorded all
four visual streams. Every stream established:

- exact equality between the selected donor slice and the substituted
  recipient slice;
- equal selected-slice Secure Hash Algorithm 256-bit (`SHA-256`) hashes;
- canonical float32 maximum absolute difference of `0.0`;
- byte-identical recipient features outside the selected indices;
- one feature call, zero grid mismatches, and restored hooks.

The clean standard path and cloned-feature replay path also matched exactly.
Greedy and all eight sampled requests preserved generated token identifiers and
canonical float32 score traces. The causal panel therefore passed the frozen
runtime and artifact trust gate.

## Observed

### First free object identity

| Encoded visual condition | Greedy successor | Sampled right-cup successors |
|---|---|---:|
| Clean Feature Replay | right cup | 8 of 8 |
| Target-Support Donor Substitution | right cup | 8 of 8 |
| Equal-Area Unrelated-Support Donor Substitution | right cup | 8 of 8 |

Every paired seed produced the same object-identity pattern:

```text
clean replay -> right cup
target-support substitution -> right cup
unrelated-support substitution -> right cup
```

The predeclared **Visual-Support Commit Effect**, defined as the unrelated
control right-cup fraction minus the target-substitution right-cup fraction,
was therefore `0.0`. There were zero selective target losses and zero reverse
selective patterns. No first action was terminal, invalid, a phrase-geometry
chimera, a left-cup revisit, a target-pizza fallback, or another object.

### The intervention was active but not target-selective

The object identity was stable even though the intervention was not a
behavioral no-op. Relative to clean replay:

- target substitution changed the exact first-row token vector in 1 of 8
  sampled requests;
- unrelated-control substitution changed it in 2 of 8;
- only 2 of 8 complete sampled continuations remained token-identical under
  either substitution.

The differences were small coordinate or later-continuation changes, not a
selective loss of the right-cup successor. First-row log-probability changes
also had mixed signs under both target and control substitution. The executed
operator can perturb decoding, but the selected committed-object support did
not uniquely control this transition.

## Supported

1. **The selected local post-vision left-cup support is not necessary for this
   exact successor transition under the executed operator.** Replacing all 86
   selected cells across the primary and three DeepStack streams produced no
   right-cup identity loss in greedy decoding or any paired sample.
2. **A text-mediated commit transition is now the leading bounded
   explanation.** Once the coherent left-cup phrase and geometry are present
   in the prefix, the model can advance to the right cup without a detectable
   revalidation requirement at the selected local feature positions.
3. **The local feature substitution was not silently ignored.** It changed
   coordinate and downstream token trajectories, but those changes were not
   selectively stronger for the committed-object support than for the
   equal-area control.

## Disfavored

For this exact state and operator, a **local post-vision visual
transaction-consistency gate** is strongly disfavored. Its predeclared positive
signature required at least six of eight target-specific right-cup losses while
clean and control retained at least seven. The observed target loss was zero of
eight.

This does not reject all visual use. The right cup itself remained visually
available, and globally contextualized image tokens may preserve left-cup
evidence outside the substituted cells.

## Unresolved

- Whether the coherent row transition is purely textual or whether left-cup
  evidence was globally compiled before the post-vision intervention seam.
- Whether the right-cup successor is selected from its unchanged visual support
  after a text-mediated left-cup commit.
- Whether a pixel-space replacement that is re-encoded through the full visual
  tower produces a selective target effect.
- Whether the same mechanism holds at another exact prefix, image, category,
  or crowded same-class scene.
- Whether a self-generated approximate row produces the same high-margin
  transition as the exact canonical row.

## Not Claimed

- no proof of a purely textual geometry-sorted transducer;
- no proof that visual evidence is absent from the transition;
- no population prevalence, mean average precision, recall, or detector-level
  capability conclusion;
- no stable object ledger, order-free commit state, visual cursor, slot, or
  final forward-pass architecture;
- no authorization for a 256-image training screen or large-scale training.

## Exactly One Next Discriminator

Run one bounded **Pre-Vision Visual-Support Counterfactual Commit Test**. Keep
exact `P56`, the coherent left-cup row, donor, paired seeds, and decode policy
fixed, but apply the same-position donor replacement in pixel space before
contextual visual encoding:

1. clean recipient image;
2. donor patch replacing the left-cup pixels without resizing;
3. an equal-shaped unrelated-region donor replacement.

Recompute the complete visual encoding for every condition. A selective target
collapse would show that target-region pre-vision input contributes causally
somewhere through the recomputed visual encoding. It would not by itself prove
committed-object revalidation, because target-location contextual disruption,
geometry or traversal effects, and downstream successor corruption remain
alternatives. Similar target and control collapse would indicate generic image
perturbation. If all three conditions retain the right cup, the evidence for a
text-mediated geometry-sorted commit state becomes materially stronger.

Pre-contextualized visual-patch replacement is a different intervention seam
and is outside this discriminator. It must not be substituted for the frozen
pixel-space condition during implementation.

After that one stronger erasure test, stop this committed-object-support line
unless it produces a selective effect. Do not add architecture or training to
this line beforehand.

## Verification

- focused unit tests: `30 passed`;
- Python compilation, command-line help loading, and scoped diff checks passed;
- the repaired immutable execution receipt records exact selected-slice and
  complement equality for all four visual streams;
- no-op parity passed for greedy and all eight paired sampled requests;
- independent scientific diagnosis reproduced the `8/8`, `8/8`, `8/8`
  identity result and the one-sided claim boundary;
- independent final artifact audit approved the repaired receipt after the
  selected-slice attestation was added.
