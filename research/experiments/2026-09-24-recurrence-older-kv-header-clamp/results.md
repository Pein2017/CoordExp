# Fixed current-header Q/K/V collapses the older-joint effect

The [lead acceptance](lead-acceptance-v1.json) accepts all22 technical cells and the frozen **retention primary NONPASS / bidirectional displacement-collapse comparator PASS**. The [candidate](candidate-results.md) and [independent CPU verification](supporting/lead-verification-v1.json) bind the exact source, vectors, captures and actual consumers.

| Base | Adaptive native→joint TV | Clamped native→joint TV | Remaining displacement ratio | Distance to original joint / adaptive TV |
| --- | ---: | ---: | ---: | ---: |
| AF | 0.782429 | 0.030320 | 0.038752 | 0.975709 |
| FF | 0.530136 | 0.054345 | 0.102511 | 0.999270 |

Both remaining-displacement ratios fall below the frozen0.2 bound. Neither clamped joint stays within half the original separation of its adaptive joint endpoint. The secondary remove/add component pattern also fails; all components were executed and remain in the denominator. Native and clamped corner winners remain151670 for AF and151671 for FF, but the full distributions decide the result.

This makes current-header attention-state adaptation consequential for the large older-record effect under the joint intervention. A possible sequence within one forward is: an older-record attention read changes current-header residual representations; later layers derive different Q/K/V from them; subsequent reads change; the final coordinate distribution moves. Fixing those Q/K/V blocks that route while leaving residual/MLP computation live. The result weakens sufficiency of direct older K/V coupling at a fixed read state. It does not separate changing queries from changing current-token keys/values, identify a unique circuit, or measure a natural mediation fraction.

The source is still one image, native AF versus written-history FF, replaying a common four-token person header before x1. No token was generated. It is not a multi-token feedback measurement, free-box recovery, natural onset/exit prediction or population result. F/F2 physical identity remains HOLD; earlier shared two-record and V-only/K-only predictions remain NONPASS.

Fresh adaptive cells match all14 accepted all-four vectors exactly. Both separate native-clamp identities are exact. Lead rechecked all22 inputs/vectors and per-axis donor/capture consumers, source/masks/recorded phases, companion preservation and restoration. All504 saved headout error/bound calculations pass, worst0.10144 of the bound; postK error0. The runtime FP64 SDPA reconstruction was inspected and its saved arithmetic rechecked, without claiming a second independent full-SDPA replay from incomplete saved full inputs.

One terminal job,22model/4vision/0generated,186.280167371 outer GPU seconds; cumulative0.6441715111661737GPUh. Exact evidence and accounting live in the acceptance receipt. This unit is closed. The next bounded question is whether fixing current-header Q alone or K/V alone removes the joint effect; no head/layer subdivision is implied.
