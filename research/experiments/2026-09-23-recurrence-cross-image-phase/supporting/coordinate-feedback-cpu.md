# Retrospective y1-token readout of accepted phase cases

This CPU-only analysis uses exactly the six lead-accepted cases and their saved full 152670-vocabulary vectors. It is exploratory interpretation of the closed cross-image package, not a prospective success test. R4 and D3 remain technical-invalid/unanswered. No model, CUDA, forward or GPU call was made.

[Machine-readable complete results](coordinate-feedback-cpu.json) SHA-256 `bbdbc280558e608d74b80445692c942c156814a85300460c33eecda4fc767a55` contain exact P, logP and 1-based rank for all three role aliases in every native/cross/sham/latest+9/earlier+9 vector, all 30 vector hashes and source bindings. The [brief](coordinate-feedback-cpu-brief.md), lead acceptance, manifest, registry and final ledger are bound there. FP64 full-vocabulary log-softmax was used without coordinate-only renormalization. Rank is one plus the number of strictly greater logits; ties share rank. Signed nats with absolute value at most `0.001` are numerically indifferent for interpretation, not statistically insignificant.

## Source roles and informative denominator

| Case | Class | Earlier row: box, y1 token | Latest row: box, y1 token | Current row: box, y1 token | Role coincidence |
| --- | --- | --- | --- | --- | --- |
| `mature:7511:5` (R) | person | 3: `[0, 539, 41, 553]`, y1=539 (`152209`) | 4: `[0, 539, 41, 553]`, y1=539 (`152209`) | 5: `[0, 539, 41, 553]`, y1=539 (`152209`) | E=L=C |
| `mature:14038:10` (R) | book | 8: `[795, 656, 850, 681]`, y1=656 (`152326`) | 9: `[795, 656, 850, 681]`, y1=656 (`152326`) | 10: `[795, 656, 850, 681]`, y1=656 (`152326`) | E=L=C |
| `mature:351017:5` (R) | person | 3: `[0, 0, 33, 86]`, y1=0 (`151670`) | 4: `[0, 0, 33, 86]`, y1=0 (`151670`) | 5: `[0, 0, 33, 86]`, y1=0 (`151670`) | E=L=C |
| `mature:1584:2` (D) | person | 0: `[128, 615, 158, 705]`, y1=615 (`152285`) | 1: `[151, 641, 170, 695]`, y1=641 (`152311`) | 2: `[177, 623, 209, 704]`, y1=623 (`152293`) | all three distinct |
| `mature:2299:2` (D) | person | 0: `[0, 298, 97, 999]`, y1=298 (`151968`) | 1: `[85, 253, 186, 484]`, y1=253 (`151923`) | 2: `[120, 108, 226, 450]`, y1=108 (`151778`) | all three distinct |
| `mature:4134:7` (D) | person | 5: `[68, 100, 461, 999]`, y1=100 (`151770`) | 6: `[105, 307, 154, 484]`, y1=307 (`151977`) | 7: `[295, 294, 362, 592]`, y1=294 (`151964`) | all three distinct |

Every box and class comes from the frozen selected registry, and each complete nine-token raw row encodes its declared box/class. Each earlier/latest/current y1 token was checked against the original source trace at its own raw offset, including the saved native current y1 ID. The three R cases have **one token under three role names**, giving **0/3 R informative cases** for either exact-token contrast. The three D cases have distinct earlier, latest and current y1 tokens, giving **3/3 D informative cases** for both contrasts (**3/6 total**). Equal-role R values below are aliases, not independent alternatives or zero-effect counterexamples.

## Five saved cells per case

Each E/L/C triple is in earlier/latest/current order. The full-precision values, including tied ranks, are in JSON; compact displayed values are rounded. `L−E` and `L−C` are raw-logit differences in nats, equal to the corresponding log-probability differences within one full-vocabulary cell.

### `mature:7511:5` (R)

| Cell | P(E/L/C) | logP(E/L/C), nats | rank(E/L/C) | L−E | L−C | Winner |
| --- | --- | --- | --- | ---: | ---: | ---: |
| native | 0.0404935829/0.0404935829/0.0404935829 | -3.206612/-3.206612/-3.206612 | 1/1/1 | — | — | 152209 |
| cross S−9 | 0.052904825/0.052904825/0.052904825 | -2.939261/-2.939261/-2.939261 | 1/1/1 | — | — | 152209 |
| native sham | 0.0404934542/0.0404934542/0.0404934542 | -3.206615/-3.206615/-3.206615 | 1/1/1 | — | — | 152209 |
| latest K+9 | 0.049971405/0.049971405/0.049971405 | -2.996304/-2.996304/-2.996304 | 1/1/1 | — | — | 152209 |
| earlier K+9 | 0.0398597526/0.0398597526/0.0398597526 | -3.222388/-3.222388/-3.222388 | 1/1/1 | — | — | 152209 |

Source raw SHA-256 `e1eac8fc2a83ca1759b233354eb827bbcab88eba4047b9f690c8f02830936fdc`; source trace `8efc8bb91e7250e6ef73b389d53be352f8bd8167ba140251470a8ed4b38abdaf`; original reduction `ac406d6fb009a2d2e97e1de7d3a2f7afcfdd6e7ac2ba19a21fbcd2051c0de268`. The five vector hashes are in JSON.

### `mature:14038:10` (R)

| Cell | P(E/L/C) | logP(E/L/C), nats | rank(E/L/C) | L−E | L−C | Winner |
| --- | --- | --- | --- | ---: | ---: | ---: |
| native | 0.0124242336/0.0124242336/0.0124242336 | -4.388106/-4.388106/-4.388106 | 1/1/1 | — | — | 152326 |
| cross S−9 | 0.0158426511/0.0158426511/0.0158426511 | -4.145050/-4.145050/-4.145050 | 19/19/19 | — | — | 152340 |
| native sham | 0.0124242278/0.0124242278/0.0124242278 | -4.388107/-4.388107/-4.388107 | 1/1/1 | — | — | 152326 |
| latest K+9 | 0.0165845992/0.0165845992/0.0165845992 | -4.099281/-4.099281/-4.099281 | 20/20/20 | — | — | 152340 |
| earlier K+9 | 0.0124086885/0.0124086885/0.0124086885 | -4.389358/-4.389358/-4.389358 | 1/1/1 | — | — | 152326 |

Source raw SHA-256 `7d9388c7ca670a5833ea0d57e6fe8ee83e0e3ec97bdfa23951c1aacb4e210a08`; source trace `0dba6a8172c84dc14e5c72442cd6a6c713110be437041c6d4e1222d0a5b8856a`; original reduction `72f4288e0130a7c639720cb11564a66863e1516af149ac87bc54df0cbbd4e737`. The five vector hashes are in JSON.

### `mature:351017:5` (R)

| Cell | P(E/L/C) | logP(E/L/C), nats | rank(E/L/C) | L−E | L−C | Winner |
| --- | --- | --- | --- | ---: | ---: | ---: |
| native | 0.244129751/0.244129751/0.244129751 | -1.410055/-1.410055/-1.410055 | 1/1/1 | — | — | 151670 |
| cross S−9 | 0.126681494/0.126681494/0.126681494 | -2.066079/-2.066079/-2.066079 | 1/1/1 | — | — | 151670 |
| native sham | 0.244130649/0.244130649/0.244130649 | -1.410052/-1.410052/-1.410052 | 1/1/1 | — | — | 151670 |
| latest K+9 | 0.11687845/0.11687845/0.11687845 | -2.146621/-2.146621/-2.146621 | 1/1/1 | — | — | 151670 |
| earlier K+9 | 0.240736233/0.240736233/0.240736233 | -1.424053/-1.424053/-1.424053 | 1/1/1 | — | — | 151670 |

Source raw SHA-256 `77a65cc8dc35220ec58766cca92b9caa9225eb4ce99c0330e340d40b6cfd8cbf`; source trace `1b75265af561e712e74a74f455ae484dde17468fb5e76367b5cb32b6838295d6`; original reduction `310cba60bc9e1940538a3163a080a44042f08afb2e7d5616a489ff7a4ff98db4`. The five vector hashes are in JSON.

### `mature:1584:2` (D)

| Cell | P(E/L/C) | logP(E/L/C), nats | rank(E/L/C) | L−E | L−C | Winner |
| --- | --- | --- | --- | ---: | ---: | ---: |
| native | 0.0237898276/0.00030319172/0.113606459 | -3.738497/-8.101145/-2.175015 | 13/35/1 | -4.362648 | -5.926130 | 152293 |
| cross S−9 | 0.00250181767/0.0567297182/0.00502827316 | -5.990738/-2.869457/-5.292679 | 47/1/35 | +3.121281 | +2.423222 | 152311 |
| native sham | 0.0237896477/0.000303189428/0.1136069 | -3.738505/-8.101153/-2.175011 | 13/35/1 | -4.362648 | -5.926142 | 152293 |
| latest K+9 | 0.00227692547/0.0575396713/0.00468539597 | -6.084929/-2.855281/-5.363305 | 49/1/35 | +3.229649 | +2.508024 | 152311 |
| earlier K+9 | 0.0262990093/0.000355526042/0.109045043 | -3.638224/-7.941912/-2.215994 | 13/36/1 | -4.303688 | -5.725918 | 152293 |

Source raw SHA-256 `cf49a34ded3df998b3f8ddce7cfbe687d65d09b6f09a49b48dcc816554f7f061`; source trace `f77d8fc2ec9e0449b59da1fae942a04ccce5e5427c2c1d4da4b2385071ff7922`; original reduction `0bc96535b76175010e46e2e0ae01e33f91ab11302dd2d453b02acb9c23cfa057`. The five vector hashes are in JSON.

### `mature:2299:2` (D)

| Cell | P(E/L/C) | logP(E/L/C), nats | rank(E/L/C) | L−E | L−C | Winner |
| --- | --- | --- | --- | ---: | ---: | ---: |
| native | 0.000157739521/7.10294415e-05/0.0700268409 | -8.754565/-9.552416/-2.658877 | 318/445/1 | -0.797851 | -6.893539 | 151778 |
| cross S−9 | 0.000346481724/0.000266939002/0.0429620309 | -7.967680/-8.228490/-3.147439 | 349/402/1 | -0.260810 | -5.081052 | 151778 |
| native sham | 0.000157739417/7.10291917e-05/0.0700269286 | -8.754566/-9.552420/-2.658875 | 318/445/1 | -0.797853 | -6.893544 | 151778 |
| latest K+9 | 0.000413228081/0.000368783642/0.0353438574 | -7.791511/-7.905300/-3.342631 | 355/379/1 | -0.113790 | -4.562670 | 151778 |
| earlier K+9 | 0.000148407437/7.11892165e-05/0.068777005 | -8.815549/-9.550169/-2.676886 | 316/418/1 | -0.734620 | -6.873283 | 151778 |

Source raw SHA-256 `cf49a34ded3df998b3f8ddce7cfbe687d65d09b6f09a49b48dcc816554f7f061`; source trace `f77d8fc2ec9e0449b59da1fae942a04ccce5e5427c2c1d4da4b2385071ff7922`; original reduction `ca5f99641c5034a5476665c8872f72649a6ea7e6dfd06a07ddf6a7eb77b7532c`. The five vector hashes are in JSON.

### `mature:4134:7` (D)

| Cell | P(E/L/C) | logP(E/L/C), nats | rank(E/L/C) | L−E | L−C | Winner |
| --- | --- | --- | --- | ---: | ---: | ---: |
| native | 4.75220284e-06/0.0123035579/0.0455701455 | -12.256902/-4.397867/-3.088502 | 569/28/1 | +7.859035 | -1.309364 | 151964 |
| cross S−9 | 3.04733524e-05/0.0302439531/0.0170126947 | -10.398658/-3.498459/-4.073795 | 587/1/19 | +6.900199 | +0.575336 | 151977 |
| native sham | 4.75216226e-06/0.012303535/0.0455703213 | -12.256911/-4.397869/-3.088499 | 569/28/1 | +7.859042 | -1.309370 | 151964 |
| latest K+9 | 2.1835151e-05/0.0349717122/0.0143010708 | -10.731989/-3.353216/-4.247421 | 596/1/24 | +7.378774 | +0.894205 | 151977 |
| earlier K+9 | 4.20438143e-06/0.0122230749/0.0441847944 | -12.379383/-4.404430/-3.119375 | 583/28/1 | +7.974954 | -1.285055 | 151964 |

Source raw SHA-256 `cf49a34ded3df998b3f8ddce7cfbe687d65d09b6f09a49b48dcc816554f7f061`; source trace `f77d8fc2ec9e0449b59da1fae942a04ccce5e5427c2c1d4da4b2385071ff7922`; original reduction `ef88057cd464fb5f4680b9f373014978c8a73bcd5113b3cfd1f69cb3623f27ae`. The five vector hashes are in JSON.

## Signed contrasts

The signs below are **minuend minus subtrahend**. `ΔP(L)` is the latest-role token probability difference; `ΔlogP(L)` and the two contrast differences are in nats. The native-minus-cross and each treatment-minus-native direction is fixed before inspecting values. Sham-minus-native checks are retained in JSON.

| Case | Direction | ΔP(L) | ΔlogP(L) | Δ(L−E) | Δ(L−C) |
| --- | --- | ---: | ---: | ---: | ---: |
| `mature:7511:5` | native − cross | -0.0124112421 | -0.267351 | — | — |
| `mature:7511:5` | latest K+9 − native | +0.00947782207 | +0.210307 | — | — |
| `mature:7511:5` | earlier K+9 − native | -0.000633830328 | -0.015776 | — | — |
| `mature:14038:10` | native − cross | -0.00341841746 | -0.243057 | — | — |
| `mature:14038:10` | latest K+9 − native | +0.00416036554 | +0.288826 | — | — |
| `mature:14038:10` | earlier K+9 − native | -1.55451283e-05 | -0.001252 | — | — |
| `mature:351017:5` | native − cross | +0.117448257 | +0.656024 | — | — |
| `mature:351017:5` | latest K+9 − native | -0.127251301 | -0.736565 | — | — |
| `mature:351017:5` | earlier K+9 − native | -0.00339351818 | -0.013998 | — | — |
| `mature:1584:2` | native − cross | -0.0564265265 | -5.231688 | -7.483929 | -8.349352 |
| `mature:1584:2` | latest K+9 − native | +0.0572364796 | +5.245865 | +7.592297 | +8.434155 |
| `mature:1584:2` | earlier K+9 − native | +5.23343213e-05 | +0.159233 | +0.058960 | +0.200212 |
| `mature:2299:2` | native − cross | -0.00019590956 | -1.323926 | -0.537041 | -1.812488 |
| `mature:2299:2` | latest K+9 − native | +0.000297754201 | +1.647116 | +0.684061 | +2.330870 |
| `mature:2299:2` | earlier K+9 − native | +1.59774909e-07 | +0.002247 | +0.063231 | +0.020256 |
| `mature:4134:7` | native − cross | -0.0179403953 | -0.899408 | +0.958837 | -1.884701 |
| `mature:4134:7` | latest K+9 − native | +0.0226681543 | +1.044651 | -0.480262 | +2.203569 |
| `mature:4134:7` | earlier K+9 − native | -8.04829734e-05 | -0.006563 | +0.115918 | +0.024309 |

## Interpretation

The simple directional copying account predicts that native relative phase should favor the immediately previous y1 over crossed phase on informative exact-token measures. **D1 is the strongest counterexample:** native P(latest y1=641) is `0.0003031917204` versus crossed `0.0567297182194`, while native-minus-cross `L−E = −7.483929` and `L−C = −8.349352` nats. Latest K+9 reverses those signed contrasts relative to native (`+7.592297`, `+8.434155` nats), despite the same token history. This favors an intervention-induced redistribution toward the previous coordinate in D1, not an already active native copying direction.

D2 also has negative native-minus-cross `L−E = −0.537041` and `L−C = −1.812488` nats, and lower native P(latest y1). D4 is mixed: native-minus-cross `L−E = +0.958837`, but `L−C = −1.884701` nats and native P(latest y1) `0.0123035579` is below crossed `0.0302439531`. A favorable comparison against the earlier token can coexist with a lower latest-token probability and a worse comparison against current y1. Across all three informative D cases native P(latest y1) is lower than crossed; the exact latest-versus-earlier sign is mixed. The three R cases cannot distinguish the three roles because their y1 token IDs coincide; their common-token probability changes have mixed signs.

These directions weaken a general claim that correct native phase favors immediate-repeat y1 through simple copying. They do not establish where the model reads from, value copying, physical owner identity, a unique circuit or natural rollout dynamics. The previous full-distribution phase-transfer criterion remains valid and separate. No nearby-bin selection, fitted threshold, coordinate-only normalization or new image was used.

Independent NumPy FP64 max-shift logsumexp recomputed all `30` bound raw vectors, `90` role-alias P/logP/rank measurements and every informative direct logit contrast. Max absolute differences from the primary PyTorch CPU reduction were `2.22e-15` nat for logP, `2.22e-16` for P and zero for contrasts; ranks/ties matched exactly. Cost: **0 model loads, 0 CUDA calls, 0 forwards, 0 GPU seconds**. This is a worker CPU interpretation for lead review, not self-acceptance or a successor experiment.
