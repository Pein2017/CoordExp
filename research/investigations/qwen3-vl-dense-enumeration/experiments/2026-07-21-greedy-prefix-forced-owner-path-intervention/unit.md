---
title: Greedy-Prefix Forced Object-Path Intervention
description: Measure how much of a sampled rescue row must be supplied at an exact greedy prefix before Qwen3-VL acquires the physical owner and preserves later unique-owner coverage.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: completed
unit_id: 2026-07-21-greedy-prefix-forced-owner-path-intervention
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: bounded_mechanistic_evidence
updated: 2026-07-21
---

# Greedy-Prefix Forced Object-Path Intervention

## Question

Low-temperature sampling sometimes emits a real physical object that native
greedy decoding misses. At the exact greedy prefix immediately before that
decision, how much of the sampled object's row must be supplied before the
released greedy decoder:

1. completes a row for that physical owner;
2. preserves or improves fixed-budget unique-owner coverage; and
3. avoids duplicate, unresolved, or malformed output?

The unit tests whether the sampled rescue is a shallow decision-calibration
failure, a later phrase-to-instance or coordinate-binding failure, or a path
that is not portable to the greedy state. It does not assume that any fixed
token position is an object-commit boundary.

## Intervention

For each frozen case, let `P_g` be the exact token prefix formed by the first
`k` complete native-greedy rows. Let `Y_s` be one reviewed or resolver-trusted
complete row from a low-temperature sampled trajectory that reaches a physical
owner not covered by `P_g`.

Run the following arms from `P_g`:

1. **Native greedy**: release generation without supplying any donor token.
2. **Cumulative donor-prefix staircase**: find the first token where `Y_s`
   differs from the native next row. Supply `Y_s` from row start through that
   token, then repeat while extending the supplied prefix one donor token at a
   time through description, row syntax, and the four coordinate tokens.
3. **Complete donor row**: supply all of `Y_s`, count it as one output row, and
   release the remaining greedy suffix.
4. **Non-target control staircase**: repeat the same endpoint schedule with a
   real, syntax-matched row for another same-image owner. For the three
   non-terminal cases the control uses the same category and row grammar where
   available. For the terminal case, the row-opener endpoint is byte-identical
   between target and control, while later endpoints distinguish `truck` from
   an already covered `person`. Target and control rows must have the same token
   length so every intervention depth has a real matched control.

For a premature native stop, the first difference is the sampled row opener.
The observed first staircase depth that yields the target owner is an empirical
result. It must not be named or interpreted as a universal owner-binding site.

Every arm receives the same maximum number of complete rows for that case. The
supplied row consumes one row of this budget, and every supplied or generated
token consumes the same total token budget. No arm gains extra suffix capacity
because part of its current row was forced.

## Frozen Cases

| Case | Greedy state | Sampled donor | Target physical owner | Evidence role | Complete-row budget |
|---|---|---|---|---|---:|
| `person-5001-row3` | first 3 greedy rows; native row 3 repeats owner `5001:1201627` | seed `21015`, row 3 | `5001:1329465` | human-reviewed greedy duplicate; donor is a high-intersection-over-union provisional owner match | 8 |
| `wine-glass-2685-row6` | first 6 greedy rows; native row 6 first selects owner `2685:-83`, which is repeated later in the native trajectory | seed `21008`, row 6 | `2685:-78` | crop-reviewed native owner that becomes a later duplicate; donor is a provisional distinct-owner match | 16 |
| `person-7511-row4` | first 4 greedy rows; native row 4 has unresolved displaced geometry | seed `21012`, row 4 | `7511:-169` | both the ambiguity and donor owner were crop-reviewed; donor geometry is loose and must be reported separately | 8 |
| `truck-13348-after-stop` | all 6 native greedy rows followed by natural termination | seed `21008`, row 6 | `13348:1372048` | tests whether a high-intersection-over-union sampled owner row can cross the native stop state | 8 |

The first three cases compare a sampled donor row against the greedy decision
at the same row index, but their earlier sampled and greedy trajectories are
not token-identical. Only `P_g` is the intervention recipient. The sampled
prefix is provenance for discovering `Y_s`, not a matched causal control.

## Primary Measurements

For every staircase endpoint, record:

- exact supplied and released token identifiers;
- endpoint role: row opener, description token, description end, box start,
  first horizontal boundary (`x1`), first vertical boundary (`y1`), second
  horizontal boundary (`x2`), second vertical boundary (`y2`), or box end;
- whether the intervened row resolves to the target physical owner;
- whether a non-target control at the same forcing depth also acquires the
  target or merely opens another valid row;
- whether the target first appears in the intervened row or later suffix;
- unique physical-owner set at the fixed row budget;
- unique-owner coverage both including and excluding a completely injected
  donor row, plus retention of downstream non-target owners;
- duplicate physical owners, unresolved rows, malformed rows, and unmatched
  predictions kept as unknown rather than called hallucinations;
- box geometry separately from entity discovery; and
- exact-token no-op parity for the native row and suffix.

## Competing Outcomes

| Observation | Supported interpretation | Consequence |
|---|---|---|
| The first differing token is sufficient and later coverage is safe. | The rescue is primarily a local decision-ranking problem. | A small own-prefix row-level preference treatment becomes justified. |
| Phrase or syntax is insufficient, but one or more coordinate tokens acquire the owner. | Description identifies a class while geometry progressively resolves the instance. | Train a coherent owner-plus-box path; do not reduce treatment to the first token. |
| Only the complete donor row works. | Current-row generation is internally unstable, but a committed row can redirect the suffix. | Test complete-row counterfactual training before adding an external state carrier. |
| The complete row acquires the owner but later unique coverage falls or duplicates rise. | The row is locally valid but has negative downstream set value. | Reject it as a positive training event in that exact state. |
| No donor depth works under `P_g`, although the sampled trajectory emitted the owner. | The rescue depends on the sampled history or on a state absent from `P_g`. | Local path-margin training is insufficient; investigate prefix-state transport or visual routing. |
| Any donor row merely defeats STOP without owner-specific acquisition. | The intervention is generic continuation, not object-specific control. | Do not train from this event. |

## Controls and Refusals

- Source checkpoint, prompt, image encoding, repetition penalty `1.0`, and
  greedy release policy are fixed.
- Rows are split from recorded integer token identifiers at the canonical box
  end token. Decoded text is never re-tokenized to construct a prefix.
- The target owner must be absent from `P_g` under positive-only matching.
- New generated or hybrid rows use the frozen positive-only, same-category
  matcher with intersection-over-union threshold `0.5` and ambiguity margin
  `0.05`. Any ambiguous or unmatched rung that could change the verdict must
  receive crop-assisted review before it can support the conclusion.
- The `person-7511-row4` donor is a frozen human crop-reviewed entity match with
  intersection-over-union below `0.5`. That verdict may be inherited only by a
  byte-identical reproduced row. A changed hybrid row remains unresolved until
  it receives its own crop-assisted review; human evidence is never relabelled
  as an automatic intersection-over-union match.
- Uncertain geometry may diagnose entity acquisition but cannot supervise a
  coordinate loss.
- Where a native next row exists, a native-row forced replay must reproduce the
  native row and suffix exactly before causal interpretation is allowed. The
  terminal case instead requires exact reproduction of native termination.
- Beam search is deferred. Existing low-temperature trajectories already
  provide the necessary rescue donors; beam search would add a new inference
  surface and confound route diversity with length and shared-prefix bias.

## Stop Rule

Close this unit after the four frozen cases and do not add more examples unless
one implementation ambiguity prevents interpretation. Proceed to a small
training screen only if at least two cases show a partial donor-prefix rung
where the released decoder contributes at least one token, acquires the target
owner, the matched non-target control does not acquire that target, and
fixed-budget downstream value is non-negative. A completely injected donor row
can show how the suffix responds to a changed state, but it cannot count as
decoder owner acquisition or satisfy the training gate. Otherwise record which
failure branch was observed and choose the next mechanism probe from the table
above.

## Evidence Sources

The execution manifest freezes exact paths and Secure Hash Algorithm 256-bit
digests. Interpretation additionally uses:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-individual-trajectory-versus-union-support-audit/
support-audit-step4887-20260721a/analysis/reviewed-support-v3.json

/data/CoordExp/.worktrees/research-probes/research/investigations/
qwen3-vl-dense-enumeration/experiments/
2026-07-21-individual-trajectory-versus-union-support-audit/
review-decisions.json
```

The completed result and interpretation are recorded in [results.md](results.md).
The final machine-readable artifacts are under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-greedy-prefix-forced-owner-path-intervention/
forced-owner-path-step4887-20260721a/final/
```
