# Vocabulary and comparison boundaries

These definitions are shared by the current synthesis. Each experiment still owns its exact checkpoint, population, parser, matcher and policy; a convenient alias never substitutes for them.

| Term | Operational meaning / warning |
|---|---|
| Owner | A distinct physical instance. An emitted row or GT record is not automatically a distinct owner. |
| GT | Ground-truth annotation supplied by the dataset. Incomplete annotations do not enumerate every real instance. |
| Row | Description plus four coordinate tokens, with the experiment's literal serialization. |
| Native greedy | Original task input, no supplied assistant repair prefix or inference-time oracle, next-token argmax under the declared decode processor and cap. |
| Conditional release | Supply a prefix or intervention and freely continue. It does not establish that the model visits that condition naturally. |
| Teacher-forced | Score/train a specified sequence under its preceding target tokens. Low average loss is not a complete argmax certificate. |
| Source | Usually four-coordinate x→y `geo_sorted_xy` step-2444 in recent work. Earlier step-4887, random-order and fitted descendants are different models. Read the owning record. |
| Human13 | A fixed human-refined panel with 392 annotated owners. A same-panel fit is not a held-out test. |
| N16 | Ambiguous historically: a 16-image QP ladder stage or the later 16-package, 11-image learning endpoint. Always qualify the study. |
| CE / SFT | Cross-entropy / supervised fine-tuning on specified target tokens, with a specified masking and normalization rule. |
| DoRA | Weight-decomposed low-rank adaptation. Magnitude-only and full adapter surfaces are different parameter spaces with non-comparable learning-rate scales. |
| Output QP | A quadratic program fitting a shared output-readout residual to constraints on captured states. Certification applies to that finite constraint system; free decoding is separately tested. |
| RLOO | REINFORCE with leave-one-out baseline across sampled trajectories. Positive relative advantage need not mean better than greedy. |
| IoU50/60/80 | Intersection-over-union thresholds 0.5/0.6/0.8. Summing threshold hits counts an owner more than once. |
| Strict annotated match | Usually same-category global one-to-one matching, but verify the exact experiment. |
| Latest physical review | Class-agnostic, cardinality-first one-to-one IoU≥0.5 inherited matches; only residual unmatched valid rows visually adjudicated. Different from earlier all-row extent review. |
| Unmatched / FP | Matching outcome / annotation-relative false positive, not by itself a physical truth label. |
| Repeat | Re-selection of an already emitted physical owner. An IoU threshold is a selector/proxy and can miss lower-overlap reboxing. |
| Unknown | Insufficient evidence for positive or negative attribution. Keep it separate; neutrality does not imply zero gradient. |
| EOS / cap | Natural end-of-sequence decision / forced token-budget termination. A cap is not successful completion. |
| RP | Repetition penalty, a decode-time transformation that can exchange owners and alter geometry, not a harmless duplicate filter. |
| KL / margin | Kullback–Leibler divergence / score gap between token alternatives. Neither alone is an owner-preservation label. |
| KV / residual state | Attention key/value cache / hidden residual activations. Readability and intervention effects do not establish an abstract owner-indexed memory. |
| Train / development / confirmation | Different information-use populations. Previously exposed panels are not untouched confirmation; review-only confirmation images do not become training data by convenience. |
| Accepted | Evidence accepted for a named scope. Not convergence, generalization, architecture promotion, or current launch permission. |

When reproducing a comparison, bind the original image preprocessing, token order, prompt, checkpoint, parameter surface, numerical backend, denominator, review policy and stop semantics. Do not compare labels alone.
