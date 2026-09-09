---
title: S2 Observed-Alias Handbag Continuation Rescue
description: Test whether one annotated continuation update after the exact naturally generated umbrella alias restores the displaced handbag without losing the umbrella or any C owner.
status: complete; mechanically valid scientific stop; paired-profile implementation not promoted
---

# S2 Observed-Alias Handbag Continuation Rescue

Result: [results.md](results.md).

## Frozen question

> Starting from the failed-but-target-bearing first-crossing step-2 adapter,
> can one continuation-only annotated-row update at the exact natural history
> through the umbrella alias restore the displaced handbag while retaining the
> umbrella and every C-covered owner?

This is a single-image mechanism discriminator, not the scalable training
algorithm.  It is cheaper and more informative than first adding a new paired
StateBank profile across eleven events: the canonical `p + u` state was reached
in none of the 22 predecessor event/exposure reads, whereas this exact observed
alias state exists.

No OpenSpec or production-schema change is warranted.  Reuse the existing
`annotated_complete_row_imitation` family, loss, trainer, DDP reducer,
checkpoint writer, and unmerged loader.

## Frozen source and exposure

Source is the unmerged step-2 adapter from the completed first-crossing DDP
unit, used only as a mechanistic probe checkpoint:

- adapter fingerprint:
  `fea8c9af55a25657a0684a05ac1771f887ffd617e7fb24874c3c08bbd6bb85a3`;
- derived source checkpoint ID:
  `f19e6af07adc5a7e3369a2b3afecb7e12ef2c4fffeee15a8fbc7a9dbaa5e6274`;
- S2 cold rows SHA-256:
  `264ad533e6fb03bb0d170aa442b0cc5e70c32562c06cb55e443528ef85196067`;
- S2 generated-token trace SHA-256:
  `09e2abcbb0903ba031229e78de3502566e34142d5fefc0a9b02ac69135e1e6b9`;
- predecessor authoritative analysis SHA-256:
  `95b68d8c11a291491541e953dceb7fc157ae02d3f78d7dccdfe9d25c2076ced0`;
- universal base and frozen step-2444 selected-token embedding delta remain
  unchanged.

On image `359310`, take the exact S2 generated response prefix through the
fourth complete row, the naturally generated umbrella:

- prefix length: 38 response tokens;
- prefix SHA-256:
  `8fc19eb788679a4edb5895590a34ab9328678720d46904961bb8a557182a7814`;
- umbrella row:
  `[151646,3551,33042,151647,151648,151819,151998,151876,152026,151649]`;
- umbrella coordinate bins: `[149,328,206,356]`;
- umbrella-versus-GT IoU: about `0.8045`.

The prefix must come from the trace, not reconstructed text.  Require exactly
three preceding complete rows and bind the umbrella to owner
`coco_ann:1426509` through the frozen global one-to-one IoU50 match.

## Correct annotated continuation target

Supervise only the canonical COCO handbag owner `coco_ann:1172698`, bbox
`[151,364,186,431]`, immediately after that prefix:

```text
[151646,10661,21250,151647,151648,151821,152034,151856,152101,151649]
```

Its token SHA-256 is
`fa874f5b7ae73e83615a98bb9e714424b993f2c990fac8a17bc7f90ea8f99515`.
The tokenizer must independently encode `handbag` as `[10661,21250]` and the
four coordinate tokens must decode to the frozen annotation bbox.

Two similar rows are explicitly not the target:

- `0fad5460cfb5a426f2512297d4694d977da4694f4cefbf5ec301494a32adbf43`
  is the C-generated handbag alias `[151,372,186,425]`; it remains a behavior
  monitor but is not relabeled as annotation;
- `2eb19cac85988384369042caee00166037dd2809b5d9df6bbaa64545c69e5186`
  is an invalid hybrid combining the word `handbag` with umbrella coordinates
  `[145,326,209,355]`; fail closed if it appears anywhere.

The full `prompt + observed prefix + handbag` length is 1,410, below the
12,000-token contract.

## One-update objective

Materialize two identity-equivalent copies of this one annotated event, one
per DDP rank.  Their event IDs must be distinct but their image, prompt,
prefix, owner, target tokens, and weight must match exactly.  StateBank
normalization and the global denominator 2 make the objective equal to the
single-event annotated-row CE; duplication changes distribution only, not the
gradient.

Run exactly one canonical BF16 update:

- world size 2, effective batch size 2, one microstep per rank;
- language DoRA only, warm-started from the S2 adapter in place as one
  unmerged adapter relative to the universal base;
- AdamW, LR `2.5e-6`, weight decay 0, global clip 1;
- frozen selected-token embedding identity-copy only;
- save one unmerged final checkpoint; never merge base weights.

Image `270570` book-to-mouse is an untouched monitor, not a trained anchor.
Its dense repeated-book annotation does not justify diluting this one-hop
mechanism test.

## Cold behavior and decisions

Cold-load the unmerged proposal on the same twelve-image panel with HF greedy,
temperature 0, RP1, and max 3,084.  Report IoU50/60/80 owner coverage,
selected/protected owners, all S2 and C gains/losses, duplicates, parser drops,
ordering violations, invalid/capped rows, and natural EOS.

Return `GO_OBSERVED_ALIAS_HANDBAG_CONTINUATION` only if all conditions hold:

1. all 137 C IoU50 owners are present;
2. umbrella `359310:coco_ann:1426509`, handbag
   `359310:coco_ann:1172698`, and book `270570:coco_ann:1140374` are IoU50;
3. the handbag prediction occurs immediately after the matched umbrella in
   emission order;
4. the exact 38-token frozen prefix is revisited before that handbag row;
5. total IoU50 coverage is at least 139;
6. all 12 rows reach natural EOS with no invalid or cap;
7. all identity, finite-update, trainable-surface, checkpoint, and cold
   unmerged-readback gates pass.

If owner-level conditions 1, 2, 3, 5, 6, and 7 pass but the exact prefix is not
revisited, return `BEHAVIOR_POSITIVE_ATTRIBUTION_HOLD`: the behavioral rescue
is useful but does not identify the frozen alias-conditioned transition.

Otherwise return `SCIENTIFIC_STOP_OBSERVED_ALIAS_HANDBAG_CONTINUATION`.  Stop
after this one update.  Do not sweep dose, add source-row aliases, train a full
suffix, or implement the eleven-event paired profile.

## Successor logic and claim boundary

A GO licenses only design of the smallest truthful paired target/continuation
profile from C across multiple images.  A valid stop closes that schema work
for now.  Neither outcome establishes held-out transfer, COCO-scale benefit,
generalization, or production readiness.
