---
title: Prefix-State Phrase-Geometry Factorial Results
description: Verified one-case evidence that a geometry-sorted prefix transition depends jointly on row phrase and geometry rather than either factor alone.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-prefix-state-phrase-geometry-factorial
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Prefix-State Phrase-Geometry Factorial Results

## Scope and Evidence Identity

This unit used one exact Qwen3 Vision-Language (`Qwen3-VL`) state on Common
Objects in Context validation image `12576`. **Prefix state 56 (`P56`)** is the
legal 56-token assistant continuation immediately before the sampled target
pizza row. Each complete-row intervention appended exactly nine canonical
tokens at token span `[56,65)` and then released the model to generate its
first free row.

The four factorial arms independently crossed one description token (`pizza`
or `cup`) with four coordinate tokens (target-pizza geometry or left-cup
geometry). Two equal-length controls appended an already covered pizza row or
an unsupported-chair row. A no-appended-row condition retained exact `P56` as
an intentionally unequal-length baseline.

Each condition used one greedy request and eight paired sampled requests.
Sampling used temperature `0.4`, top-p nucleus threshold `0.95`, repetition
penalty `1.0`, and the same eight seeds across all seven conditions. All 63
first free actions parsed as valid complete rows. The model, tokenizer,
configuration, installed runtime, prompt, recipient token prefix, and source
image were identical across compared conditions.

The resolved artifact root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-prefix-state-phrase-geometry-factorial/
  paired-seed-panel-20260715b/
```

The cross-condition machine-readable summary is `summary.json` under that
root. Individual condition receipts are under `conditions/<condition>/`.

## Observed

### Two-by-two factorial

The table counts the greedy request plus eight sampled requests:

| Appended description | Appended geometry | First free successor |
|---|---|---|
| `pizza` | target pizza | left cup, 9 of 9 |
| `cup` | target pizza | left cup, 9 of 9 |
| `pizza` | left cup | left cup, 9 of 9 |
| `cup` | left cup | right cup, 9 of 9 |

Only the coherent `cup` phrase plus left-cup geometry advanced past the left
cup to the right cup. Supplying the `cup` phrase without left-cup geometry, or
left-cup geometry without the `cup` phrase, was insufficient: both crossed
rows returned the left cup in all nine requests.

The target-pizza half of the matrix is asymmetric and less identifying. A
coherent target-pizza row and either crossed row all returned the left cup.
Therefore the experiment supports a joint phrase-and-geometry requirement for
recognizing the left-cup transition; it does not establish that every object
is committed by a symmetric conjunction rule.

### Controls

| Appended condition | First free successor |
|---|---|
| No appended row | target pizza, 5 of 9; left cup, 4 of 9 |
| Already covered pizza row | target pizza, 5 of 9; left cup, 4 of 9 |
| Unsupported-chair row | left cup, 8 of 9; target pizza, 1 of 9 |

The covered-pizza control matched the no-row object label on all nine paired
requests. Among the eight sampled requests, its exact first-row token vector
matched the no-row condition in seven. It therefore behaved like a near-no-op
at the first-action level despite adding a complete legal row.

The unsupported-chair row biased the distribution toward the already dominant
left-cup mode but never advanced to the right cup. It is compatible with a
generic row-progression or distribution-concentration effect, but it cannot
explain the unique coherent-left-cup transition.

## Supported

1. **A simple description main effect is rejected in this case.** Holding the
   target-pizza geometry fixed, changing `pizza` to `cup` did not change the
   successor. Holding left-cup geometry fixed, the description mattered only
   as part of the coherent `cup` plus left-cup combination.
2. **A simple geometry main effect is rejected in this case.** Holding the
   `pizza` description fixed, changing target-pizza geometry to left-cup
   geometry did not change the successor. Left-cup geometry alone did not
   advance past the left cup.
3. **Uniform generic complete-row advancement is rejected.** The equal-length
   rows did not all produce the same transition. The coherent left-cup row
   uniquely advanced to the right cup, while the covered duplicate behaved
   like the no-row baseline.
4. **A bounded phrase-by-geometry interaction is supported.** At `P56`, the
   left-cup transition behaves operationally like a conjunction: both its
   description and geometry must coexist in one coherent row before the next
   action advances to the right cup.
5. **The prefix is a content-sensitive executable state, not merely a token
   counter.** Complete rows of identical length and grammar produce different
   successor states depending on their joint semantic-spatial content.

## Mechanism Update

The smallest mechanism consistent with the panel has two components:

```text
complete legal row
  -> weak generic concentration toward the dominant next mode

phrase-and-geometry-compatible row
  -> stronger object/order-specific successor transition
```

The second component resembles a **transaction-consistency gate**: phrase and
geometry must describe a compatible row before the prefix state treats the
left cup as completed and moves to its geometry-sorted successor. This is more
specific than a generic continuation pulse and more structured than a pure
coordinate frontier.

However, the same observations are also explained by a learned canonical
serialization validator. The model may recognize the exact phrase-geometry
pattern expected by its geometry-sorted training policy without maintaining an
order-free visual object ledger. The experiment does not distinguish a true
visual object commit from a sequence-level canonical-row transition.

## Ruled Out

For this exact state, the following standalone explanations are ruled out:

- description category alone controls the next object;
- last-row geometry alone controls the next object;
- every complete legal row advances one identical traversal step;
- the earlier complete-row findings can be explained solely by row length or
  wrapper completion.

## Unresolved

- Whether the phrase-geometry conjunction is a native visual object-ownership
  check or a memorized geometry-sorted serialization pattern.
- Whether the same interaction appears at other prefixes, images, categories,
  same-class instances, and crowded scenes.
- Whether a coherent object row suppresses that object because it is visually
  committed, because its canonical rank was consumed, or both.
- Whether actual self-generated rows create the same transition as exact
  teacher-forced rows.
- Whether this transition can be shaped into higher greedy coverage without
  damaging one-shot detection, precision, or termination.

## Not Claimed

- no population prevalence, mean average precision, recall, or architecture
  conclusion;
- no stable order-free ledger, explicit object slot, visual cursor, or final
  forward pass;
- no general symmetric phrase-geometry commit rule;
- no claim that an unsupported-chair region is visually empty outside the
  frozen case review;
- no authorization for a training screen or large-scale fine-tuning.

## Exactly One Next Discriminator

Run a bounded **Visual-Support Counterfactual Commit Test**. Hold exact `P56`
and the exact coherent `cup` plus left-cup row tokens fixed, but vary only the
encoded visual support available for the left cup:

1. original fixed full-image features;
2. targeted removal or replacement of left-cup support;
3. a same-area unrelated-region intervention.

If the transition is a textual geometry-sorted serialization transducer, all
three conditions should still advance to the right cup. If the row is
revalidated as a visually grounded object event, removing left-cup support
should selectively weaken the right-cup transition while the unrelated-region
control preserves it.

The test should remain a small inference case study and must preserve the
fixed encoding, image-token count, prefix, row tokens, and decode policy. One
positive `P56` interaction is not sufficient to promote a 256-image training
screen. A coherence-versus-canonical-order replication becomes useful only
after the visual-grounding question is resolved.

## Verification

- focused unit tests: `16 passed`;
- all condition receipts and the cross-condition summary parse as valid JSON;
- all six appended conditions report exact recipient context and a verified
  nine-token complete-row grammar;
- all eight sampled seeds are paired across every condition;
- independent contract audit: approved with no severity-zero or severity-one
  findings;
- independent read-only recount reproduced the full factorial and control
  counts from raw receipts.
