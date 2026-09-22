---
title: C-Anchored Tail-Boundary Two-Step DDP Discriminator
description: Move the same eleven annotated rows from first-crossing prefixes to the complete C-natural tail before EOS and test target uptake without Source-owner replacement.
status: completed; mechanically valid scientific stop
---

# C-Anchored Tail-Boundary Two-Step DDP Discriminator

Result: [tail-boundary DDP result](results.md).

## Frozen question

> From the same C adapter, does conditioning each of the same eleven annotated
> target rows on its complete C-natural transcript with only terminal
> `<|im_end|>` removed produce a reliable new owner without losing any owner
> already covered by C?

This is the shortest discriminator for the replacement seen in the completed
[first-crossing DDP unit](../2026-09-02-c-anchored-ddp-two-step-smoke/results.md).
At a first-crossing prefix, target CE directly raises the target against the
next Source row; image `359310` consequently replaced a protected handbag with
the selected umbrella.  At the complete-C tail, the direct incumbent is EOS,
not an existing owner.  Shared DoRA can still disturb earlier rows, so natural
owner retention remains decision-owning.

No OpenSpec is warranted: this unit reuses the existing annotated-complete-row
StateBank family, loss, trainer, DDP reducer, checkpoint writer, and unmerged
loader.  Only an experiment-local CPU materializer and child configs are
needed.

## Frozen identities and contrast

The predecessor and treatment share:

- C adapter fingerprint
  `5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95`;
- source checkpoint ID
  `f75949ad231c1846399a1a8c5470bf7bda71c930a0a76474dbccedb551a16036`;
- small-dual plan SHA-256
  `f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`;
- C natural rows SHA-256
  `9f8aa4f478883468ace45daa4a4de9b90d055938057ca8e2a1add83eff2077ed`;
- C generated-token trace SHA-256
  `cb712451b24f32b63df753154fbcbe53a8b35bcf5170cba5a0ac7789ca5efeb6`;
- exact `1024 * global max len 12000`-lineage training JSONL SHA-256
  `86d34cc2efbce9814847dd12fc12cab2f46d04168ce39905bfd62a93ced783fd`;
- universal base, step-2444 frozen selected-token embedding delta,
  processor/tokenizer, `geo_sorted_xy` prompt preference, eleven target rows,
  per-image credit, optimizer, two exposures, and twelve-image cold panel.

The sole treatment change is the conditioning boundary.  For each selected
row, read exact generated token IDs from the frozen C trace, require one final
non-pad stop token equal to `<|im_end|>` (`151645`), remove that terminal token,
and use all preceding generated IDs as the prefix.  Do not reconstruct the
prefix by tokenizing rendered text.

All eleven selected owners are C-unmatched at IoU50 and none of their exact
canonical rows occurs in its C tail.  The largest
`prompt + tail + target` sequence is 1,598 tokens, below the frozen global
limit 12,000.

Image `398214` remains included.  Its actual C trace is terminal-valid but has
20 row openers while the parser retains 19 predictions.  Record
`source_tail_dropped_row_count=1`; do not silently substitute the parsed rows
or exclude the image.

## Objective and execution

Train only the canonical annotated target row after each tail prefix.  Source
rows are conditioning context, not supervised labels; EOS, unmatched rows,
and unannotated objects are not supervised or treated as negatives.

Reuse the predecessor's two lexical 12-record windows, including the duplicate
of `positive:88854:1`.  Public StateBank normalization must again realize
exactly `1/11` objective credit per image per update.

Run the existing canonical path with:

- world size 2, effective batch size 12, six microsteps per rank per update;
- two BF16 updates from C, language DoRA only;
- AdamW, LR `2.5e-6` at both updates, weight decay 0, global clip 1;
- checkpoints after steps 1 and 2;
- no merged checkpoint or base-weight export;
- frozen embedding identity-copy only, byte/tensor-identical to the input.

Eight-GPU scaling is deliberately not part of this discriminator: changing
world size would add no evidence before the conditioning mechanism passes.

Cold-load both unmerged checkpoints in fresh processes and decode the same
twelve images with HF greedy, temperature 0, repetition penalty 1, and the
same 3,084-token cap.

## Monitors

For C, step 1, and step 2 report category-consistent global one-to-one owner
coverage at IoU50/60/80, all eleven selected targets, the three protected
owners, total owner gains/losses, predictions, duplicates, parser drops,
invalid/malformed rows, caps, and natural EOS.

`geo_sorted_xy` is a preference, not an admission invariant.  C already has
natural inversions, and tail-appending the canonical target introduces an
inversion in all eleven events.  Record ordering-violation image/row/pair
counts; do not reject or alter a transcript for ordering.

Do not relax a category globally.  A `book`, fruit, vegetable, or cup owner is
`dense_category_annotation_uncertain` only when many repeated overlapping
instances make individual ownership or boundaries unclear.  Sparse instances
remain ordinary reliable GT.  Small, distant, occluded, or truncated owners
are reported separately as `visual_difficulty`.  Both strata retain their
training weight and remain in every metric denominator.

## Decision and stop

Each cold read is a registered exposure.  Select the earliest of step 1 then
step 2 that satisfies all of:

1. all 137 C IoU50 owners retained, including all three protected owners;
2. at least one selected target outside
   `dense_category_annotation_uncertain` reaches IoU50;
3. total IoU50 coverage is at least 138;
4. all 12 decodes reach natural EOS, with no invalid or capped row;
5. source, StateBank, trainable-surface, checkpoint, frozen-embedding, and
   cold unmerged-readback identity gates pass.

If such a read exists, return `GO_C_ANCHORED_TAIL_BOUNDARY`; a later registered
overdose does not invalidate the earlier checkpoint but must be reported.  If
the umbrella appears on image `359310`, the handbag must also remain IoU50 by
the all-C-owner retention rule.

If neither exposure qualifies, return
`SCIENTIFIC_STOP_C_ANCHORED_TAIL_BOUNDARY` and identify whether the failure is
no reliable target uptake or nonlocal Source-owner interference.  Mechanical
invalidity permits only repair and exact rerun; it is not a scientific result.

Stop after this single two-step run.  Do not sweep LR, dose, regularization,
owners, GPUs, or add suffix replay.  A conditional target-then-displaced-row
continuation is considered only after interpreting a valid negative.

## Claim boundary

Success would show only that moving these eleven same-panel targets to the
actual C tail can yield a finite shared-DoRA update with strict same-panel
retention.  It would not establish held-out transfer, COCO-scale benefit,
missing-label precision, sorted output, generalization, or production
readiness.
