---
title: Image 2299 Near-Complete Human Relabel Successor Transition
description: Test whether appending one naturally generated person row selectively suppresses that physical person and redistributes the next-row distribution to other unreported people.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-18-image2299-near-complete-human-relabel-successor-transition
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Image 2299 Near-Complete Human Relabel Successor Transition

## Question

At one exact, naturally reached prefix containing two `person` rows, does
appending a third naturally generated `person` row cause the next generated row
to behave as though that physical person has been committed, or does it merely
advance a geometry-sorted traversal pattern?

## Why This Image

Image `2299` is a dense school group photograph. The user manually relabeled
the image in:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

The record contains 38 `person` objects and 8 `tie` objects. Thirty of the 32
existing sampled third-row predictions can be assigned uniquely to a relabeled
person at Intersection over Union (`IoU`) at least `0.5` with a top-versus-
second match margin at least `0.05`. The previous audit ledger assigned only 23
of 32. This relabel therefore removes annotation incompleteness as the dominant
explanation for this case while retaining two unresolved predictions as
unsupported or geometrically weak outcomes.

The exact parent is the original prompt plus two naturally emitted `person`
rows. Existing discovery samples at this exact parent admit four distinct
physical people with sampled support counts `3`, `10`, `7`, and `3`.

## Competing Explanations

1. **Physical-instance commit.** Appending person `i` selectively lowers the
   chance of immediately generating person `i` again and moves probability to
   other not-yet-emitted people.
2. **Geometry-sorted traversal.** The latest row acts mainly as a spatial
   frontier. Successors move rightward or downward according to the trained
   ordering policy, regardless of whether a physical person was committed.
3. **Category anti-repetition.** Any appended `person` row suppresses the whole
   `person` category, moving probability to `tie`, terminal output, or invalid
   output rather than to a different person.
4. **Generic row progression.** Any complete row changes continuation behavior
   similarly; the identity and geometry of the appended person have little
   effect.
5. **Annotation or matching artifact.** Apparent switching is produced by
   incomplete labels or ambiguous ownership rather than a model state change.

## Primary Observation

For each of four naturally admitted third-row people, append one high-IoU
natural exact row to the same parent and sample exactly one successor row under
the same paired seed vector. Classify the successor as:

- the just-appended person;
- either of the two people already present in the parent;
- another relabeled, not-yet-emitted person;
- `tie`;
- terminal;
- invalid; or
- unresolved under the frozen matching rule.

The first decision is based on the full owner-by-owner successor table, not on
a single mean metric.

## Pilot Outline

- Reuse `scripts/research/run_native_sibling_branch_replay.py` with two narrow
  execution additions required by the selected natural prefixes:
  exact donor prompt token reuse and an explicitly receipted local sampling
  context.
- Use the geometry-sorted Gaussian-supervision Weight-Decomposed Low-Rank
  Adaptation (`DoRA`) adapter at step `4,887`.
- Keep Scaled Dot-Product Attention, Brain Floating Point 16-bit model
  parameters, physical batch size one, temperature `0.4`, top-p nucleus
  threshold `0.95`, repetition penalty `1.0`, and a nine-token maximum
  successor horizon.
- Select three existing natural exact rows for each admitted person, including
  the highest-IoU row.
- Run the same eight fresh seeds across all twelve rows. This gives 96 calls
  while balancing every physical owner and directly measuring exact-coordinate
  variant sensitivity.
- Match against the relabeled 46-object record. Do not fall back to nearest
  unused annotations and do not force unresolved outputs into an owner.

## Interpretation Rules

- Selective self-person suppression plus increased mass on other people
  supports an executable physical-instance transition at this state.
- Primarily rightward or downward successor movement supports geometry-sorted
  traversal rather than an order-free covered set.
- Similar successor distributions for all four appended rows support generic
  row progression.
- Loss of the whole `person` category supports category anti-repetition.
- A result that depends on one exact coordinate variant remains variant-local.
- This single image cannot establish a general covered-set mechanism or justify
  an architecture or training change.

## Stop Rule

Stop before expansion if any arm fails exact prompt reconstruction, produces no
complete successor row for most smoke seeds, or cannot be matched reliably
under the relabeled record. Stop after the 32-seed panel if the four arms are
indistinguishable within paired sampling noise or are explained by a monotone
geometry frontier. Do not enter hidden-state work or training from a null or
geometry-only result.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-18-image2299-near-complete-human-relabel-successor-transition/
<immutable-run-identifier>/
```

## Result

The completed panel supports an immediate physical-owner-sensitive transition:
the just-emitted person is repeated in zero of 96 calls, all four emitted people
remain reachable under another-owner treatment, and 93 successors match a
specific relabeled person. It rejects whole-category anti-repetition, generic
row progression, and a strict one-way geometry frontier for this case.

It does not distinguish a multi-row covered-object state from a last-row spatial
successor rule. The next discriminator must hold the final owner fixed while
changing an earlier covered person.

See [results.md](results.md).

## Non-Goals

- no population Average Precision or recall estimate;
- no terminal-token suppression;
- no attention, residual, hidden-state, or vision-feature intervention;
- no forced object row;
- no slot, query, covered-set carrier, loss, or training proposal; and
- no claim that the relabel is a universal dense-scene benchmark.
