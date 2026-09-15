---
title: Human-Resolved Dense Branch Value and Calibration Screen
description: Test whether a valid sampled-only branch yields safer downstream unique-person coverage than the greedy branch and therefore justifies a small own-prefix training screen.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-18-human-resolved-dense-branch-value-and-calibration-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Human-Resolved Dense Branch Value and Calibration Screen

## Research Question

At one exact parent state in the manually relabeled dense-person image `2299`,
does choosing a valid third-row person that sampling can reach produce more
safe, unique people over the next four rows than choosing the greedy third-row
person?

This is a treatment-linked question. A sampled branch is useful for training
only if it improves later set coverage, not merely because it differs from
greedy decoding.

## Mechanism and Alternatives

The working mechanism is that the native decoder contains several valid
object-transition routes, but greedy decoding repeatedly chooses one narrow
route. A small training correction might make a better existing route stable.

The strongest alternative is that sampling only changes traversal order. If
all valid branches reach the same number and quality of later people, preferring
the sampled branch would add arbitrary supervision rather than improve set
enumeration. Other failure modes are earlier-person repetition, repeated people
within the four-row suffix, malformed rows, early termination, and poor boxes
that cannot be assigned to one physical person.

## Existing Evidence Used as Context

The unit does not repeat earlier yes-or-no commit probes.

- Image `12576` establishes fixed-state sampled rescue: pizza appears in 11 of
  32 samples while greedy selects cup, but terminal-state sampling does not
  rescue pizza.
- Image `7574` shows trajectory change without reciprocal object commit.
- Image `15254` is a clean null case with no bagging-only rescue.
- Image `2299`, now manually relabeled with 38 people and 8 ties, shows
  owner-sensitive next-row redistribution and no immediate self-repeat in 96
  prior calls.

Together these cases justify testing downstream branch value on the only case
with sufficiently complete physical-owner labels. They do not justify training
before that value is measured.

## Minimal Execution

Use the step-4,887 geometry-sorted Gaussian-supervision Weight-Decomposed
Low-Rank Adaptation (`DoRA`) checkpoint. Hold the image, exact two-row parent,
sampling policy, and eight paired seeds fixed. Append one of four naturally
generated third-row people:

| Natural branch | Relabeled person rank | Role |
|---|---:|---|
| owner `0003` | 3 | sampled-only branch |
| owner `0004` | 2 | sampled-only branch |
| owner `0006` | 4 | sampled-only branch |
| owner `0012` | 14 | greedy branch |

Generate at most four complete successor rows with temperature `0.4`, top-p
threshold `0.95`, repetition penalty `1.0`, and the same seeds in every arm.
Match each person row to the manual relabel at Intersection over Union (`IoU`)
at least `0.5` with a top-versus-second margin at least `0.05`. Do not force an
ambiguous row into an owner.

The primary outcome is the number of unique new people in the four-row suffix,
excluding the two people in the shared parent and the appended branch person.
Safety outcomes are parent or branch repeats, within-suffix repeats, ties,
terminal output, malformed rows, and unresolved geometry. Compare each sampled
branch with the greedy branch by paired seed and report deterministic bootstrap
confidence intervals.

## Decision and Training Gate

A sampled-only branch opens the training gate only if it shows a reproducible
positive paired gain in unique new people without a material safety regression.
The gain must be attributable to physical owners rather than longer output.

If the gate opens, the next step is a 256-image short training screen with:

1. a matched continued-cross-entropy control; and
2. one own-prefix branch-ranking treatment that increases the score of a
   verified uncovered-object row relative to the model's covered, repeated,
   terminal, or lower-value branch at the same state.

The screen keeps the existing Qwen3-VL backbone, DoRA trainable surfaces, row
format, and data pipeline. It adds no slot, detector, object query, or explicit
covered-set carrier. Promotion requires higher greedy unique-object recall,
smaller sampling-over-greedy rescue gap, and no material increase in duplicate,
malformed, unsupported, or low-quality geometry outcomes.

If no sampled branch has downstream value, do not run that treatment. The
result would mean that sampling exposes alternative routes but has not yet
identified a better training target. If safety fails through unresolved boxes,
the next intervention belongs inside phrase-to-geometry binding rather than
next-object selection.

## Stop Conditions

- Stop if exact parent or natural branch reconstruction fails.
- Stop training promotion if no sampled branch safely beats the greedy branch.
- Stop interpretation if physical-owner matching is ambiguous for most rows.
- Do not infer population performance from this single-image causal screen.

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/
```

## Non-Goals

- no suppression of the terminal token;
- no new detector, slot, query, or memory architecture;
- no claim that one preferred object order is universally optimal;
- no large training run before the 256-image screen; and
- no reliance on official incomplete annotations for image `2299`.

## Result

The three sampled-only branches did not beat the greedy branch on downstream
unique-person coverage or safety over four later rows. The sampled-branch
preference training gate is closed. Alternative branches remain behaviorally
real, but they more often enter weakly localized `tie` output rather than a
better short-horizon physical-object route. See [results.md](results.md).
