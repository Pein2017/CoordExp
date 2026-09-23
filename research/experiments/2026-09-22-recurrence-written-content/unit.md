# Content of the just-written row at a fixed recurrence exit

Date: 2026-09-22. Lead-defined successor under the user's explicit autonomous
research/GPU grant. Direct execution child: trace_dynamics (sol/xhigh).

From mature untied+axis step2444's native val:7511 row89 x2 state, does replacing
only the preceding complete row's content change z38-z999 and the actual winner,
under identical length, positions, current-prefix tokens and qualified replay?

The [accepted predecessor](../2026-09-22-recurrence-position-history/results.md)
found history-following choices in four crossed history/position cells. At late
positions, adding the native repeated row reduced the margin by1.139614, whereas
advancing the current prefix's positions increased it. This new contrast isolates
whole-row content at fixed count/positions, which that crossing did not identify.

## Sources and two conditions

Bind the predecessor's attempt002 source-to-cell manifest, full-vocabulary L_L
vector and original raw/trace/receipt/image/checkpoint bindings. Preserve its
native batch4, target index2, FP32/SDPA, effective embeddings/adapters and prompt.
All row indices are zero-based and raw token offsets are authoritative.

- R: exact native prefix through row89's y1, predicting x2 at offset807.
  The immediately preceding row88 is person [0,571,38,575].
- B: replace row88 with the exact nine-token row89 observed in the original
  raw output: person [0,571,999,999]. This changes row88's x2/y2 tokens at
  offsets798/799; all other tokens, masks, MRoPE and physical positions remain
  identical. Current S at offsets801:807 remains unchanged.

The replacement is teacher-forced, not natural generation. Both source rows are
user-adjudicated bad predictions: water hallucination and multi-owner long box.
This is not a physical owner or quality-improvement estimand.

## Prediction and acceptance

Measure d=z38-z999, delta=d_B-d_R, both full-vocabulary argmaxes, top candidates
and winner gaps. Persist full logits. Verify R against the predecessor's entire
L_L vocabulary vector within2e-4 absolute error, exact winner999, and original
top-two trace parity. Prove actual input changes are exactly the two named tokens
and consumed positions/causal masks are unchanged between conditions.

Two deliberately restricted models make opposite advance predictions:
equal-strength monotone row-template copying predicts delta<0; symmetric
row-template inhibition predicts delta>0. These predictions additionally assume
no other content-dependent effects. Use absolute delta>4e-4 as the predeclared
numerical discrimination guard; this is not a statistical confidence interval.
An opposite sign challenges the respective restricted model, not every possible
copying/adaptation mechanism. A B winner38 with vocabulary gap>4e-4 additionally
shows local conditional reversal. Report delta relative to the predecessor's
1.139614 added-history effect; overcoming the small native0.006081 margin alone
does not explain that larger effect.

Strong alternatives: different geometry, contextual retrieval, suffix-state
changes and x2/y2 interactions. The intervention changes both coordinates and
cannot be called an isolated x2 feedback test. An accumulator and positional
rereading remain observationally compatible. Do not pool this exposed single
case with the predecessor as independent evidence.

## Execution and stop

One GPU, two planned forwards, cap4 and10 minutes model execution. Reuse the
maintained replay/loading/identity helpers; no execution from retained sources.
Freeze source/input bindings before forward, retain any failure, and use fresh
attempt paths after producer changes. Stop after native qualification plus one
B score, or return a decision-bearing technical failure. No additional edits,
lag/dose/layer sweeps, training, free generation or new images in this unit.

Maintained source: probes/training_set_completion/recurrence_written_content.py.
Output root: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-written-content/.
Root owns this protocol/state/results and acceptance; child owns source and
execution artifacts. A small source-level mutation check must reject a changed
current S or altered token length, alongside actual model readback qualification.
