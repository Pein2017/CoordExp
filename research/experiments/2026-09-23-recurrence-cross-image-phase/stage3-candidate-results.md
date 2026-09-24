# Cross-image phase: D-only continuation candidate and technical stop

Status: **worker candidate; fixed D queue stopped at D3 technical qualification**. This report is not lead acceptance. D1 and D2 completed the original seven-call/two-vision protocol; D3 stopped after cached native call 3 when an ended companion exceeded the frozen all-row `2e-4` full/cache parity gate. No D3 crossed/sham/treatment cell and no D4 model call occurred. The [cumulative eight-case ledger](supporting/stage3-cumulative-ledger-v1.json) includes every charged failure and terminal job. Original [stage2 candidate](stage2-candidate-results.md), [stage2 ledger](supporting/scale-terminal-ledger-v1.json) and captured scale producer remain byte-identical.

## Authority and CPU boundary

The [D-only admission](lead-D-continuation-v1.json) SHA-256 is `b81588f28bda98c82a949a133be0faa1e4facdf99680b13b8f9cbbc2889c1b59`; [R2/R3 acceptance and R4 ruling](lead-r2-r3-acceptance-and-r4-ruling.json) is `e64d460697e4bf08302eba33ccc0a42820760ee1d679f16b1b110902f0a93d73`. The R4 receipt stays terminal technical-invalid with its `21.416175059974194` seconds charged. No R4 retry, tolerance change or false pass was used. The [separate continuation producer](../../../probes/training_set_completion/recurrence_cross_image_phase/continuation.py) is `185c55f23d9aaa2d06dae4f344a44b0c3bb3ebb2dce067caf5f6e46cf96b455d`; frozen [scale producer](../../../probes/training_set_completion/recurrence_cross_image_phase/scale.py) remains `cff77ec4b6ae3f129fa94c1710d7264810dc134cba4091668a2ab738b78238c4`.

The [CPU caller gate](supporting/d-continuation-cpu-gate-v1.json) bound exact R1/R2/R3/R4 receipts and the `186.57946596294641` second prior package charge. It rejected an unknown R4 error, a missing R4 charge, changed D order, R4 replay, and a missing prior package charge. The actual continuation entry also rejected D2 before D1 and R4 replay without model/CUDA calls. The [new preflight](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/preflight.json>) (`9a7e44bfcbd36c8515a1af4f0a917c2f44177ef1060a93e84857aa253744b513`) reused unchanged qualified source-batch records, captured 23 current sources including the new caller, froze exact D1–D4 commands and accounted the prior charge. The four-case two-times shape forecast was `414.1803341539108` seconds against `3413.4205340370536` seconds remaining before launch.

All D cases use the original refined-00 four-request batch, with target indices 0, 1, 2 and 3 respectively. The full source input is reconstructed for each case at its exact target step; no companion, pad, EOS, position, mask or image row is dropped. Each new case independently checks its full source trace where defined, all four full/cache vectors, sham, all-layer phase/actual consumed K and unchanged V/companion history/suffix K/V, then cold readback if technically complete.

## Frozen eight-case outcomes

`B` is full-vocabulary TV(native,cross); ratios are TV(treatment,cross)/B. An em dash means no valid scientific readout. R1–R3 are lead-accepted; D1/D2 remain worker candidates pending lead review.

| Order | Case | Status | B | Latest ratio | Earlier ratio | Model/vision calls in case | Charged seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R1 | `mature:7511:5` | lead-accepted selective | 0.1321820171 | 0.1723740118 | 1.0654804194 | 9/4 incl first failure | 64.726317823 |
| R2 | `mature:14038:10` | lead-accepted selective | 0.2643368016 | 0.1258403122 | 1.0176040487 | 7/2 | 50.559775978 |
| R3 | `mature:351017:5` | lead-accepted selective | 0.2726771880 | 0.0837612988 | 0.9890948519 | 7/2 | 49.877197102 |
| R4 | `mature:99184:7` | technical-invalid, unanswered | — | — | — | 3/2 | 21.416175060 |
| D1 | `mature:1584:2` | candidate selective | 0.8738936138 | 0.0152785637 | 0.9962173449 | 7/2 | 49.806788586 |
| D2 | `mature:2299:2` | candidate selective | 0.2283125331 | 0.2972526337 | 1.0263886743 | 7/2 | 49.521355689 |
| D3 | `mature:2685:12` | technical-invalid, unanswered | — | — | — | 3/2 | 23.512117691 |
| D4 | `mature:4134:7` | held, unrun after D3 | — | — | — | 0/0 | 0 |

D1 and D2 passed source trace parity, all-row full/cache agreement (`7.2479248e-5` maximum each), native/latest-sham parity, all 56 row-layer phase qualifications, actual consumer gates and separate-process cold readback. Their raw full-vocabulary [D1 reduction](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d1-1584-2/reduction.json>) is `0bc96535b76175010e46e2e0ae01e33f91ab11302dd2d453b02acb9c23cfa057`; [D2 reduction](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d2-2299-2/reduction.json>) is `ca5f99641c5034a5476665c8872f72649a6ea7e6dfd06a07ddf6a7eb77b7532c`. D1 global winner changes from native `152293` to crossed/latest `152311`; D2 winner remains `151778` in every cell. The full distribution is the criterion; these winner observations are secondary.

## D3 stopping evidence

D3 target batch row 2 and other active batch rows 1/3 passed original source trace parity. Companion row 0 had ended before source offset 122; its EOS/pad tail was retained and no after-end trace parity was invented. Saved full/prefill/cached consumers agree on all rows' token, mask and three-axis position splits and cache slots. Four-row full versus cached native maximum absolute logit differences are:

| Batch row | Source status | Max absolute difference |
| --- | --- | ---: |
| 0 | ended companion | **0.0002195835** |
| 1 | active | 0.0000377893 |
| 2, target | active; source trace passed | 0.0000724792 |
| 3 | active | 0.0000391006 |

Row 0 exceeds the unchanged `0.0002` all-row gate. Its top-two IDs match, but D3 remains technical-invalid and scientifically unanswered; the numerical cause is unestablished. The [terminal receipt](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d3-2685-12/receipt.json>) SHA-256 is `8cafd670c02e80c7a0e27b4e96d74010d8412a8438c5fe3c8bb93d6a46338691`. Preserved [full](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d3-2685-12/cells/01-full-native/full-batch-vocabulary.pt>) and [cached](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d3-2685-12/cells/03-native/full-batch-vocabulary.pt>) four-row vectors and [cached consumer](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d-continuation-v1/d3-2685-12/cells/03-native/consumer-raw.json>) support cold inspection; exact hashes are in the cumulative ledger. No extra model call, retry, score-based replacement or D4 launch was made.

## Interpretation and cost

The lead-accepted R threshold remains satisfied by three accepted selective R identities, with R4 explicitly unanswered. D1 and D2 are two candidate selective numerical different-antecedent cases. This is evidence of transfer in those measured D inputs, while the prespecified **at least 3/4 D threshold remains unresolved**: D3 is technical-invalid and D4 unrun. The unmatched numerical D cohort cannot establish healthy physical controls, physical recurrence specificity or population rates.

This continuation charged `122.84026196599007` seconds and 17 model/6 vision calls. The package cumulative charge, including the initial R1 failure and both parity failures, is `309.4197279289365` seconds (`0.0859499244247` allocated GPU-hours), **43 model/16 vision calls**, leaving `3290.5802720710635` seconds under its one-hour cap; sequence cumulative charge is `0.1756112177549965` GPU-hours under eight. Raw package artifacts total `113278768` bytes under 1 GiB. All eight launched jobs have terminal receipts and absent PIDs; the new D jobs used PIDs `1531562`, `1533007`, `1534452`. D1–D3 new peak RSS was `11859276` KiB, peak GPU allocated `11364276224` bytes, reserved `12012486656` bytes. The queue is stopped at the admission's new technical-failure boundary for lead ruling; no self-acceptance or successor was started.
