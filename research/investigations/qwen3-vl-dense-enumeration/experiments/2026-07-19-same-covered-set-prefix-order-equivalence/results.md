---
title: Same Covered Physical-Object Set under Different Earlier Prefix Orders Results
description: A bounded fixed-prefix panel finds both local covered-owner suppression and path-dependent next-owner changes when the covered physical set, row count, and final row are held fixed.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-19-same-covered-set-prefix-order-equivalence
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded
conclusion_status: hybrid_coverage_and_order_sensitive_state_supported
updated: 2026-07-19
---

# Same Covered Physical-Object Set under Different Earlier Prefix Orders Results

## Verdict

The missing causal comparison has now been executed. With the image, model,
prompt, complete prefix rows, covered physical-object set, row count, and final
row held fixed, exchanging only two earlier rows can change which physical
person is generated next. The effect is real but conditional: two cases retain
the same next owner for every paired seed, while two cases switch between
distinct people.

The same panel also contains a clean local covered-owner effect. In case 2,
person rank 6 is generated in all 9 runs when its row is omitted, and in none
of 18 runs when its row is present, regardless of whether that row appears
first or second among the exchanged rows.

The best bounded interpretation is therefore neither a pure set-only state nor
a pure order-only continuation rule. The decoder can suppress at least one
previously emitted physical owner in an order-robust way, while retaining a
path-dependent residual that can tip competition among valid uncovered people.
This result does not show that path dependence harms final set coverage, and it
does not yet justify a new training loss or model component.

## Evidence Scope

- Image: `2299`, the dense school group photograph with a near-complete human
  relabel in
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`.
- Model: the 2-billion-parameter Qwen3 Vision-Language model with the
  geometry-sorted, description-first, pure cross-entropy plus token-type gate
  Weight-Decomposed Low-Rank Adaptation checkpoint at step `4,887`.
- Runtime: Hugging Face, Scaled Dot Product Attention, full-model 32-bit
  floating point, physical batch size 4, repetition penalty `1.0`.
- Main prefixes: `A then B then C` and `B then A then C`.
- Held fixed: complete row token multiset, covered physical-object set, row
  count, final `C` row, image, prompt, model, generation policy, and paired
  sample seeds.
- Executed control: `B then C`, which removes `A` from the covered set.
- Generation: one greedy row and either 8 or 24 paired samples per arm at
  temperature `0.4` and top-p `0.95`.
- Total: 204 successful one-row generations across four cases and three arms.
- Resolved inference configuration fingerprint:
  `6af0ff8b638a4464cfb993e74d0f2fa2fb87789f81b83f0f7fd851e62d861df5`.

The executed runner tokenized each frozen canonical row exactly once and then
appended the stored token identifiers. Generated rows were not decoded and
retokenized. Every run passed the same-row-token-multiset, same-covered-set,
same-row-count, and same-final-row invariants.

## Observed Results

Person ranks are zero-based positions among the 38 human-relabeled people in
the geometry-sorted serialization.

| Case | Exchanged `A`, `B`; final `C` | Greedy next owner | Paired-seed owner agreement | Exact generated-row agreement | `A` after `B then C` | Interpretation |
|---|---|---|---:|---:|---:|---|
| 1 | ranks 2, 0; rank 3 | rank 6 versus rank 6 | 8 / 8 | 7 / 8 | 0 / 9 | Owner order-robust, but `A` was not behaviorally active. |
| 2 | ranks 6, 0; rank 2 | rank 1 versus rank 1 | 8 / 8 | 6 / 8 | 9 / 9 | Clean local covered-owner suppression that is robust to the exchanged order. |
| 3 | ranks 3, 0; rank 4 | rank 7 versus rank 16 | 20 / 24 | 11 / 24 | 2 / 25 | Physical-owner order sensitivity; the removal control is weak. |
| 4 | ranks 10, 0; rank 14 | rank 18 versus rank 18 | 17 / 24 | 15 / 24 | 0 / 25 | Sample-level physical-owner order sensitivity, but `A` was not behaviorally active. |

### Case 2: the clean covered-owner result

When person rank 6 is omitted from the prefix, all nine `B then C` runs produce
person rank 6. When person rank 6 is included, neither `A then B then C` nor
`B then A then C` produces that person in any of 18 runs. Both order arms have
the same physical-owner distribution: person rank 1 in seven runs and an
unmatched or low-overlap person-shaped row in two runs.

This is direct local evidence that emitting a row can change later access to
that physical person without requiring that row to occupy one specific earlier
position. It is not proof of a general object ledger because the symmetric
removal of `B` was not executed and the panel covers one image.

### Cases 3 and 4: order changes the next physical owner

Case 3 changes the greedy next owner from person rank 7 to person rank 16. Four
of 24 paired sample seeds receive different automatic owners. Enlarged-crop
inspection confirms that the rank-7 to rank-16 switches at seeds 103 and 117
are genuine changes between distinct people. Two outputs automatically labeled
`none` partially cover person rank 16 and fail the fixed overlap threshold;
they must not be interpreted as no-object hallucinations.

Case 4 changes the paired distribution from 12 rank-13 and 13 rank-18 outputs
under `A then B then C` to 7 rank-13 and 18 rank-18 outputs under
`B then A then C`. Seven of 24 paired seeds switch owner. Enlarged-crop review
confirms that rank 13 and rank 18 are distinct, spatially separated people and
that all seven switches are genuine owner changes with acceptable geometry.
Six switches go from rank 13 to rank 18 and one goes in the reverse direction.
This directional imbalance is descriptive at this sample size, not a
population-level effect estimate.

## What the Panel Supports

1. **Earlier order remains causally relevant after the same final row.** The
   physical next owner changes in cases 3 and 4 even though the covered set and
   final row are identical.
2. **The prefix is not reduced to only the last row.** Moving an earlier row
   changes behavior without changing the final row.
3. **A local order-robust commit effect can exist.** Case 2 suppresses the
   emitted owner equally under both earlier orders.
4. **Physical owner and exact coordinates have different stability.** Exact
   row agreement is lower than owner agreement in every case. Coordinate-token
   differences must not be treated as physical-owner changes.
5. **The natural state is plausibly hybrid.** A compact record of what has been
   emitted and a residual record of how the prefix was traversed can coexist in
   ordinary decoder hidden states.

## What the Panel Rules Out

- A strict set-only explanation in which the covered physical-object set and
  final row completely determine the next-row distribution.
- A strict final-row-only explanation in which all earlier order information
  is causally absent.
- The claim that pure cross-entropy produces no commit-like effect at all.
- Automatic classification of every unmatched generated box as a hallucination.

## What Remains Unresolved

The panel does not establish:

- whether the order-sensitive route improves or harms eventual unique-object
  coverage;
- whether a different valid next owner later converges to the same uncovered
  set or causes omission, duplication, or early stopping;
- how common the effect is across images, prompts, checkpoints, or natural
  self-rollout prefixes;
- whether both exchanged earlier owners are independently active, because the
  symmetric `A then C` control that removes `B` was not executed;
- whether the random-order checkpoint learns less path-sensitive behavior; or
- whether the relevant state is represented as coordinates, semantics,
  low-dimensional traversal state, or a distributed combination.

The four `A` tuples were selected from earlier evidence rather than
pre-screened on this exact checkpoint. Only case 2 passes a strong behavioral
activation control, and case 3 passes it weakly. This limits any estimate of
the interaction between coverage and order.

## Decision for the Pending Training OpenSpec

Keep the training infrastructure and algorithm discussion pending. This result
does not support adding a slot, explicit covered-set carrier, exact-row
invariance loss, or order-invariance objective.

The result does justify four narrow evaluation requirements for the later
training study:

1. compare prefixes with the same physical covered set and different earlier
   orders;
2. pre-screen both exchanged owners with symmetric leave-one-out controls;
3. score physical-owner and unique-set behavior separately from exact
   coordinate-token equality; and
4. evaluate a short future horizon, so a different valid next owner is judged
   by later unique coverage, duplication, and stopping rather than by whether
   it follows one canonical route.

An exact serialized-row invariance target would be actively misleading here:
the same physical owner often survives while one or more coordinate tokens
change. The scientific target is robust set completion, not identical text.

## Highest-Information Next Experiment

Do not add more seeds to the existing cases. The main uncertainty is case
identifiability, not sampling variance.

The smallest useful follow-up is:

1. pre-screen new tuples on the current checkpoint;
2. require both `B then C` and `A then C` to reactivate the omitted owner;
3. retain two to four activation-qualified cases across two to four
   human-reviewable dense images; and
4. extend the two order arms for only two to four generated rows.

The decisive question is whether different immediate valid routes converge to
the same unique uncovered set. If they do, path dependence is largely benign
and training should preserve route freedom. If one order systematically loses
objects, repeats owners, or stops earlier, then a coverage-aware training
signal has a concrete behavior to correct.

After this symmetric short-horizon comparison is interpretable, repeat the
same exact intervention on a matched random-order checkpoint. The current new
random-order training run has not yet produced a comparable checkpoint, so a
historical adapter should not be presented as an exact matched ablation.

## Artifacts

- Combined summary:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-same-covered-set-prefix-order-equivalence/paired-summary-v1/summary.json`
  with SHA-256 checksum
  `ce2ab7853cb5923e8f6354dc13dee40f60d8d7e5427f530d6b8e6788f0997f19`.
- Case 1 result checksum:
  `9b958931e4de9a3bc9c18b8fc07d71e8ff310665721ca5c8d7f1e392a2d2333a`.
- Case 2 result checksum:
  `f051fd775f1b84f98549ddf86ee1e45a0e3f4e3a45e19cfef251e4a35804b66e`.
- Case 3 result checksum:
  `134206c167cca22cdaa0662d8e563b16a78bb7e08cac1958c3c43bceb1fe9d91`.
- Case 4 result checksum:
  `27fdc03db8d4928985da5f3cd9884eb73407d073db09777fce85d103dd7abb38`.
- Frozen case specification: `cases-image2299.json` in this experiment
  directory.
- Experiment-local runner:
  `scripts/research/run_same_covered_set_prefix_order_probe.py`.
- Experiment-local summarizer:
  `scripts/research/summarize_same_covered_set_prefix_order_probe.py`.

The run artifacts preserve the resolved model, prompt, image, row-token,
generation, and runtime identities. They do not preserve a source commit or
dirty-diff checksum for the experiment-local code. The result is sufficient
for this bounded mechanism probe, but the missing code-state receipt should be
fixed before promoting this runner into a reusable evaluation contract.
