# Next native history: can the measured logit change predict y1?

2026-09-23. Lead `01a0c831-b332-7e12-b931-6ebe2359c99f`; execution owner
`923-worker` `01a0ce4b-9b55-7392-8a25-6a76f9e12c3a`, `gpt-6-sol/xhigh`.
The user renewed autonomous mechanism research. [State](state.json) owns dispatch;
[prediction manifest](prediction-manifest.json) freezes inputs, formulas and
categorical predictions before either target model evaluation.

## Why this next step

The [accepted chair factorial](../2026-09-23-recurrence-chair-history-position/results.md)
identified content/position interaction, but all strict shared winner-transport
predictions failed. Two-candidate margin direction missed third vocabulary
modes. Earlier history/phase interventions already demonstrated local sensitivity.
Another finer sensitivity grid would not establish predictive understanding.

**Question:** Can a constant logit increment measured across the first added A
row predict the full-vocabulary y1 distribution after one further identical A,
better than keeping the previous distribution unchanged?

This tests a minimal local recursive approximation. It is not a mechanistic
theory of the Transformer, a universal recurrence law or a prediction of the
native exit. The chosen next native history is already known; its **supplied-x1
conditional y1 readouts** have not been evaluated in the preceding packages.
No predicted winner may be changed after looking at these target readouts.

## Exact target and frozen forecasters

Original mature untied+axis source `refined-04` is a **single request**, batch0,
`coco2017_train_000000477415`. Preserve all source/model/input bindings.
E is native history before row4, offset36; L is before row5, offset45.
The next history U is native[:54], exactly E+A+A, where both A rows are the
same nine-token broad-chair output. Native row6 is a different box, so U is
**not a third arrival of A**. Its class/grammar prefix still matches chair.

At U append the four original header tokens and the same supplied x1=510 or418;
read the next y1 once. Only the current x1 changes from native98. Do not append
any artificial history or alter positions, causal masks or cache order.

For each probe use already saved full-vocabulary FP32 vectors, converted to
FP64 for calculations: z_E=E/E, z_L=L_A/L, z_LE=L_A/E. Three fixed forecasts:

| Name | Predicted logits at U | x1=510 winner | x1=418 winner |
| --- | --- | ---: | ---: |
| Persistence | z_L | 413 | 0 |
| Affine native-step | 2 z_L − z_E | 413 | 0 |
| Local position-step | 2 z_L − z_LE | 408 | 0 |

Softmax each full vector without coordinate masking, fitted temperature,
clipping, additional candidates or coefficient selection. Adding a common
logit offset has no effect on these probability forecasts. The formulas are
selected using earlier evidence; only the two U evaluations are prospective.

The affine forecast assumes the next identical row-plus-position step produces
the same logit displacement as the previous one. The position forecast assumes
the prior nine-position displacement at L transfers to the next interval. The
factorial interactions make both assumptions questionable; failure is informative.

## Decision and limits

Primary endpoint: for each probe independently calculate
`KL(P_U || Q_forecast)` over the full vocabulary, with log-softmax in FP64 and
P_U from the actual retained target vector. This is deterministic distribution
approximation error, not population uncertainty or sampled-token evidence.

The strict shared affine prediction requires **both** frozen global winners to
match and affine KL to beat persistence KL by more than0.001 nats on **both**
probes. A failure on either probe rejects that shared local prediction. Report
ties within0.001 as numerically unresolved. This numerical guard is not a
statistical confidence interval. Report position-step KL and its categorical
hits as the predeclared alternative; do not select a new forecast afterward.

Retain actual global winner/runner/gap, each forecast's probability assigned to
the actual winner, and all three KL values. Probe2's identical categorical
predictions are non-discriminating; its distributions can still differ. Do not
count the two probes as independent images. A correct one-step conditional
forecast would support only this local approximation, not physical identity,
valid boxes, natural exit timing or a circuit. A failed forecast closes this
one-step law; no polynomial, coefficient, horizon or neighboring-row scan follows.

## Five-call execution contract

CPU preparation and the following qualified model work are admitted. Worker
owns implementation in `probes/training_set_completion/recurrence_next_history_prediction/`,
its supporting records and `candidate-results.md`; lead owns protocol, manifest,
state and acceptance. Raw outputs go under the matching dated unit in
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/`.

1. Verify native rows4/5 are exactly A and row6 shares the declared four-token
   header. Verify source hashes and both current-x1 offsets. If not, HOLD; never
   replace the target with a convenient later row.
2. Reconstruct and save all six full forecast distributions on CPU, with hashes,
   before any target model call. Match the manifest's categorical predictions.
   Freeze commands, current producer/direct-import captures, effective model
   bindings, full input/position crosswalk and cost forecast before GPU work.
3. Three qualification calls: fresh L reference for each supplied x1, matching
   the accepted full vector within2e-4; and unforced native U y1 with x1=98,
   matching original source chosen/top-two/logsumexp within2e-4 (native y1=579).
   Complete all three before the two target calls.
4. Two target calls: U with x1=510 and418. Save raw full vocabulary vectors and
   actual model inputs/positions before reduction. Verify input changes only
   the target x1 relative to native U; all prior tokens, image and positions
   remain original. No free generation or position intervention.
5. Reduce against the already frozen forecasts and run separate-process cold
   readback. Return all five cells, predictions, actual results, bindings,
   technical limits, costs and terminal jobs directly to the lead.

Reuse maintained primitives and preserve accepted predecessor bytes. Exactly
**five model and five vision forwards**, zero free tokens, one GPU job;
incremental ceiling **0.25 allocated GPU-hours** including setup/diagnostics and
failures, with cumulative ceiling8 and prior charged cost in the manifest.
Stop on a qualification/semantic/cost conflict before repair or additional calls.
Scientific failure is completion. No Git actions, peer/shared-runtime writes,
extra images, candidate search, new main task or automatic successor.

Direct return uses the existing `worker_turn.py --to lead` pair route. Preserve
worker model/effort. The lead independently verifies and accepts; no self-acceptance.
