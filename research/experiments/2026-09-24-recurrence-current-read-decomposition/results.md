# Historical contribution versus attention redistribution

**Lead-accepted and closed.** The frozen historical-contribution-selectivity primary passes; the symmetric redistribution comparator does not. [Lead acceptance](lead-acceptance-v1.json) owns the decision and complete failure-inclusive accounting; [independent verification](supporting/lead-attempt-002-verification-v1.json) binds the evidence.

At original refined-03 target2 train351017, one native A history and six replayed current tokens were identical in all cells. The current y1-position query predicts x2. All 28 layers used their own incoming Q/K/V. H is the native weighted contribution from the preceding nine-token row; R is the native-denominator contribution from other readable keys; O_R is their normalized output. The intervention writes only that query's pre-o_proj head output.

| Output at every layer | TV to native | TV to full current mask | Ratio to native/mask TV |
| --- | ---: | ---: | ---: |
| Native H+R | 0 | 0.969709174089 | 1 |
| Remove history contribution: R | 0.964094437012 | 0.209818945828 | 0.216373064661 |
| Redistribute other weights, keep history: H+O_R | 0.706115110596 | 0.875779598352 | 0.903136344126 |

The ratio difference is 0.686763279464, beyond the frozen 0.1 margin; removal ratio is below 0.5. The native reconstruction identity and O_R bridge pass all-four-vector gates before both partials. Their maximum errors are 5.4121e-5 and 2.1935e-5 versus the unchanged 2e-4 tolerance. Independent NumPy FP64 reconstruction checked all 168 saved local transforms and reproduced full-vocabulary TVs to numerical precision. Actual selected/complement consumers, masks, original source, companions and state gates passed.

This narrows the computation behind the mask effect: retaining the native denominator while removing H is enough to approach the mask endpoint under the registered criterion. Redistribution remains an active influence, and the intervention changes both vector direction and magnitude. It does not isolate semantic coordinate content, prove copying, establish that K alone is abnormal, or demonstrate a recovered free row. F physical identity remains HOLD; R includes image, prompt and current nonhistory keys.

Attempt002 used six model and six vision forwards, zero generated/reused tokens, with 50.671184331 outer seconds and 1,996,222,285 raw bytes. Attempt001 remains terminal technical-invalid (one model/vision forward, 19.749049656 seconds); the one non-model CUDA repair fixture charged 15.502195388 seconds. All jobs are terminal. Accepted cumulative sequence charge is 0.932168453653 GPU-hours. No further model execution is authorized by this closed result.

The immutable [candidate](candidate-attempt-002-results.md) and [cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-current-read-decomposition/attempt-002/readback.json) contain all raw vector and source bindings. The [next CPU-only brief](supporting/lead-direction-norm-feasibility-v1.md) tests feasibility of a fixed direction/norm control without selecting new heads, layers or samples.
