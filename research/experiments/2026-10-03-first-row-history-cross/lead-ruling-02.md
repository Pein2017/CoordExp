# Accepted local negative: aligning the first row does not recover the next bottle

The lead accepts the complete four-condition diagnostic. The single command
exited0 at clean execution commit`2daf0da8dfc6aed13e16e91cdfe591d464c9f7ac`,
using2checkpoint loads,4requests/292actions and zero optimizer/backward/replay/
exports. Native wall was47.605seconds; peak CUDA allocated4.875GB, reserved5.109GB,
and process RSS16.893GB. The process has exited. Both natural-history controls
match all9pre-force greedy choices and all73saved action IDs exactly. No retry,
extra model call or repeated readback occurred. Saved consumer verification is
`outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross/lead-native-consumer-check-01.json`;
[results](results.md) retain command, artifact identities and complete row evidence.

## Observed contrast

All four conditions supply one person row and freely generate64actions. Every
condition has6complete valid free rows and one budget-censored unfinished bottle
span; these are local observation boundaries, not full-horizon stopping failures.

| Checkpoint / supplied first row | First free category | First free box | Target bottle IoU | Strict duplicate events |
|---|---|---|---:|---:|
| A0 / native | person | [197,114,407,514] | 0 for all emitted bottles | 0 |
| A0 / GT | person | [197,114,407,499] | 0 for all emitted bottles | 0 |
| A16 / native | bottle | [0,0,23,86] | 0 | 2 |
| A16 / GT | bottle | [0,0,23,86] | 0 | 2 |

The intended next bottle is annotation-4947389372712316, box[186,30,207,106].
Every one of A16's six emitted bottle rows has zero IoU with every same-category
annotation in this image, under either supplied history. The supplied person
never receives recovery credit. Unmatched proposals remain annotation-relative;
this experiment does not establish physical absence or hallucination.

The GT substitution is not inert. A0 changes4free coordinate tokens, including
a different bottle's best same-category overlap .489996→.504117. A16 changes one
later x2coordinate38→31. These changes do not localize the intended target; the
A0 threshold crossing alone is not proof of a physical-owner gain.

## What this rules out, and what remains open

For this image, these checkpoints and this local target, replacing the three
first-person coordinate tokens with their exact teacher values does not rescue
localization. It weakens the proposed explanation that this specific small
first-row mismatch is the necessary source of the early failure. It does not
show that all history changes are irrelevant or rule out later-history effects.

A16/GT supplies the exact teacher first row and then freely emits the correct
bottle category/header. At zero-based action14, before any wrong generated
coordinate or repeated row, it chooses x1=0 where this teacher successor has
x1=186. Thus a required conditional localization decision is still not realized
under this correct teacher context, despite the learned next-category change.
This is stronger evidence for inspecting conditional fitting than for attributing
this first error solely to accumulated bad history. It does not identify whether
finite optimization, conflicting image contributions, parameter coupling,
normalization or another mechanism causes the failed decision; no per-target
margin, gradient or training-source intervention was measured here.

Close this unit without another query, training intervention, checkpoint choice
or extension. No checkpoint is promoted. This one-image/short-horizon conditional
diagnostic establishes neither sustained coverage nor physical-FN recovery.
