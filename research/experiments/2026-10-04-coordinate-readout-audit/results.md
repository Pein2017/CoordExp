# Coordinate readout audit: distinct rows, conditional score competition

Lead-accepted and closed. The complete bounded diagnostic rejects exact global coordinate-row collapse at B0/B16 and demonstrates that neighboring coordinate inputs can change actual output scores and winners. Selected invalid decisions are already present in B0 when given the same B16 history. This supports context-dependent coordinate competition as the immediate description; it does not identify a unique instance-binding, slot, precision or training-origin cause.

## Evidence and execution

The [frozen unit](unit.md) owns the question and16-cell matrix. One native invocation completed16 requests /3226 emitted actions, including forced actions, with exactly two model loads and zero optimizer/backward/replay/training/warmup/export. All six checkpoint/image natural controls reproduced every saved token and pre-force median argmax, with zero mismatches. Selected raw log probabilities were identical to the saved originals; maximum median log-probability difference was1.036e-6, reported descriptively. All seven intervention requests also had exactly unchanged raw/median coordinate log-probability vectors at their observed pre-edit positions. Counterfactual winners were observations, never fidelity gates.

Execution commit: `8506ad07ef787665eacf088c6e4055eed86d38c3`, clean at preparation and release; unchanged through completion. CPU qualification passed11 tests, including RED/GREEN for a raw-versus-median final-token likelihood bookkeeping error. Final consumer acceptance independently checked all16 saved raw artifact identities and literal prefixes/budgets. The producer exited0 and its PID settled. Runtime341.12s, peak RSS15.50GiB, peak CUDA allocation4.98GiB, retained native payload8.51MiB; no resource excess.

Artifact root: `outputs/research/physical-fn-recovery/2026-10-04/coordinate-readout-audit/`. Primary evidence: `prepared-01/static-geometry.json`, `prepared-01/input-packet.json`, `native-release-01.json`, `native-01/readback.json`, all16 condition JSONs, `native-01/terminal.json`, `native-01-exit.json`, `native-summary-01.json`, and `lead-consumer-check-01.json`. Readback SHA256: `f0d09e403fdcb9590f2a00ee5910f6ea15b4cb4061f4dcd79e6717d3ddea2db3`. CPU logs and positive/HOLD fixtures remain under `cpu-qualification-01/`; their original receipt fields are preserved.

## Static coordinate rows

All1000 rows are distinct in both checkpoints in each measured representation: analytic input, actual BF16 input addition, analytic output and median-normalized output. Adjacent similarity was already high at B0 and barely changed at B16:

| Measurement | B0 | B16 |
|---|---:|---:|
| Runtime input adjacent cosine, median |0.9826027|0.9825987|
| Analytic output adjacent cosine, median |0.9853539|0.9853472|
| Input cosine for120/121 |0.9830526|0.9830563|
| Output cosine for120/121 |0.9863321|0.9863207|
| Input distance for120/121 |0.2434440|0.2434116|
| Output distance for120/121 |0.2169747|0.2170617|

Median same-coordinate B0-to-B16 displacement was0.002829 for runtime input and0.001698 for analytic output; no row remained exactly identical across checkpoints. Distinct rows and high cosine can coexist. Neither cosine nor row displacement establishes the relevant hidden-state projection margin. Actual logits separately accumulate and round base and delta terms; analytic output geometry is not an exact full-precision reconstruction of those logits.

## Actual conditional decisions

All row/action indices are zero-based. A pre-force winner is the model's choice under the supplied prefix, even when the acquisition then forces a different token. Only the last action of an intervention request is genuinely free.

| Condition | Observed result | Supported boundary |
|---|---|---|
|7511 P46 zero width: x1=684; test683/685 instead |Free x2 stays684 in both;1000-coordinate log-probability RMS differences0.0943/0.0960 |Input distinction survives, but x2 does not track this ±1 change. The x pair becomes ordered for683 and reversed for685; this is not a full-box recovery experiment.|
|13348 normal P25: x1=544; test543/545 instead |Free x2 stays559 for543 and changes559→558 for545; RMS differences0.1696/0.4596 |Neighboring input values can change a neighboring output winner in actual native computation.|
|351017 first bottle x2=23; test22/24 instead |Next bottle pre-force x2/action26 changes23→29 for both edits; x1/y1 remain0; free boundary29 remains object-start |History affects conditional scores, but this is not literal copying of the edited value. Action26 remained forced to23, so this does not demonstrate a changed free row or loop escape.|
|13348 trusted normal worker prefix `[546,633,556]` |Pre-force x2 prefers558; supplied x2 is556; free y2=682 |The known-row context remains numerically usable; the exact annotation corner is not the x2 winner. This is not a repair assigned to invalid P26.|
|B0 under identical B16/7511 history |At action421 x2=684 after x1=684; at439 x2=677 after x1=684, matching B16's selected invalid decisions |These conditional failures were not newly created only at B16. This is a supplied-history comparison, not B0's natural trajectory.|
|B0 under identical B16/13348 history |Free y2=623 versus B16's627, both below supplied y1=629 |Checkpoint changes alter the selected coordinate while leaving this conditional order violation.|
|B0 under identical B16/351017 history |Next bottle pre-force x2/action26 prefers14 versus B16's23 |Checkpoint state affects the conditional repeated-row competition; body, input and output changes are not separated.|

The three selected B16 invalid slots place over0.999994 probability mass in the coordinate family. Best legal minus best illegal median scores are−0.1701 (zero width),−0.2778 (reversed x), and−0.2561 (reversed y). The normal P25 x2 control has+10.6584. These failures concern competition inside the coordinate family and insufficient conditional order preference at those positions; they are not a failure to choose the coordinate token family.

Actual raw-score ties remain important. At zero-width action421, raw maxima tie677/684 at24.375, while median normalization selects684. At reversed-x action439, raw maxima tie673/677 and median selects677. All those raw winners are also illegal under x1=684, so turning normalization off is not established as a geometry repair. Normal P25 x2 has raw maxima556/558 and median winner559; first/next bottle x2 raw winners17/25 become23/23 under median. Distinct coordinate rows therefore do not imply unique or robust raw-score rankings. No precision intervention or hidden-state decomposition was performed.

## Interpretation and stop

Exact global codebook collapse and complete functional indistinguishability of adjacent inputs are falsified for the measured tables/contexts. Newly collapsed rows cannot explain the B16 failures measured here. High similarity, finite-precision ties and small score margins may still contribute to fragile choices; this diagnostic does not isolate their causal contribution.

The strongest supported description is a history- and checkpoint-dependent conditional score distribution whose local winner can violate coordinate ordering or revisit an earlier numerical region. In the zero-width case the right-edge preference remains near684 despite a changed left edge. In the normal case the winner responds to a one-bin change. In the early bottle case, both edits redirect the same conditional competition to29 rather than copying22/24. These are heterogeneous responses, not one established failure circuit.

Instance ambiguity, endpoint/slot use, visual localization and insufficient conditional geometry constraints remain distinguishable hypotheses, not identified causes. Invalid rows and boundary bottles were not assigned GT owners. The protocol did not hold a failing physical owner fixed under a verified correct history, isolate body versus input/output updates, alter precision, or test an unrestricted continuation after the changed bottle winner. It cannot establish natural-loop escape, annotation/physical-owner recovery, generalization or a training remedy. No checkpoint is promoted. This unit is closed with no automatic next query or training trial.
