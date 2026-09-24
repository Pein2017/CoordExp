# Cross-image phase: final D4 worker candidate

Status: **D4 candidate complete; lead acceptance pending**. The exact frozen eight-case package is now fully accounted for: six valid selective numerical readouts and two technical-invalid/unanswered cases. No retry, replacement, new sample or gate change occurred. The [final eight-case ledger](supporting/stage4-final-eight-ledger-v1.json) binds every charged job, failed receipt, vector result and cost.

## Authority and qualification

The [D4-only admission](lead-D4-admission-v1.json) SHA-256 is `269eafd0b620a40f0ae7eecf53b6ef3cc3fdae9c31ee2a21e9cc2bc9d01f5247`; [D1/D2 acceptance and D3 ruling](lead-D1-D2-acceptance-and-D3-ruling.json) is `f500a545b5012347a65fd6039b7032620d1cdf3cd136ff5980df8b3e515cafaf`. The [D4 producer](../../../probes/training_set_completion/recurrence_cross_image_phase/d4.py) is `87dcd37886a017749e7560ceaa3c143944334543ce82e6fbccfb343bc300c1f9`; captured [scale](../../../probes/training_set_completion/recurrence_cross_image_phase/scale.py) and [D continuation](../../../probes/training_set_completion/recurrence_cross_image_phase/continuation.py) remain `cff77ec4b6ae3f129fa94c1710d7264810dc134cba4091668a2ab738b78238c4` and `185c55f23d9aaa2d06dae4f344a44b0c3bb3ebb2dce067caf5f6e46cf96b455d`. Stage2 and stage3 candidate/ledger bytes remain unchanged. The earlier [missing-authority observation](supporting/stage4-authority-gap.md) is retained; the lead subsequently materialized and identified both authoritative files.

The [D4 CPU gate](supporting/d4-cpu-gate-v1.json) bound all eight prior terminal receipts and `309.4197279289365` already charged seconds. It rejected another case, changed R4 or D3 failures, missing D3 or package charge, and a relaxed D3 gate. The actual caller rejected another case and an unlisted output path before model load. The [preflight](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d4-only-v1/preflight.json>) SHA-256 is `3496938eb81a4e14b7486b9e936a1a4075e08abbbad34beaf459edccf754891e`. It captured 24 current source/import files, froze the one exact command and target-index-3 four-request source crosswalk, and reused the unchanged CPU source/phase checks. All four original companion rows were active at offset 73; the 2x shape forecast was `104.4668498607787` seconds versus `3290.5802720710635` seconds left before launch.

D4 executed exactly one full native source forward, one full-batch prefill and five cached suffix readouts: native, crossed S-minus9, native sham, latest-key plus9 and earlier-key plus9. Seven model/two vision forwards, zero generated tokens. All four source-row chosen/top2/logsumexp traces passed. Full/cached maximum absolute logit errors by batch row were `[3.52859497e-5, 4.43458557e-5, 8.91685486e-5, 3.71932983e-5]`, each below `2e-4`. Native/sham, all 56 row-layer phase qualifications, actual selected historical K and unchanged V/companion K/V, consumer masks/slots and separate-process cold readback passed. Raw vectors and consumers were saved before reduction.

## Frozen eight-case outcomes

`B` is full-vocabulary TV(native,cross); treatment ratios are TV(treatment,cross)/B. The original selective criterion is `B>1e-6`, latest ratio `<0.5` and earlier-minus-latest ratio `>0.1`, with the frozen numerical guard. D4 is a worker candidate; R1–R3 and D1–D2 have lead acceptance. Technical-invalid cases stay unanswered in their four-case denominators.

| Case | Status | B | Latest ratio | Earlier ratio |
| --- | --- | ---: | ---: | ---: |
| R1 `mature:7511:5` | lead-accepted selective | 0.1321820171 | 0.1723740118 | 1.0654804194 |
| R2 `mature:14038:10` | lead-accepted selective | 0.2643368016 | 0.1258403122 | 1.0176040487 |
| R3 `mature:351017:5` | lead-accepted selective | 0.2726771880 | 0.0837612988 | 0.9890948519 |
| R4 `mature:99184:7` | technical-invalid, unanswered | — | — | — |
| D1 `mature:1584:2` | lead-accepted selective | 0.8738936138 | 0.0152785637 | 0.9962173449 |
| D2 `mature:2299:2` | lead-accepted selective | 0.2283125331 | 0.2972526337 | 1.0263886743 |
| D3 `mature:2685:12` | technical-invalid, unanswered | — | — | — |
| D4 `mature:4134:7` | **candidate selective** | **0.3852547268** | **0.1796998059** | **0.9984507498** |

D4 latest TV-to-cross was `0.0692301996363`, earlier TV-to-cross `0.3846578708928`, and earlier-minus-latest ratio `0.8187509439`. Native global winner was token `151964`; crossed/latest winner was `151977`; earlier winner returned to `151964`. The complete distribution, not winner restoration, determines the criterion. Its [raw reduction](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d4-only-v1/d4-4134-7/reduction.json>) SHA-256 is `ef88057cd464fb5f4680b9f373014978c8a73bcd5113b3cfd1f69cb3623f27ae`, [pilot](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d4-only-v1/d4-4134-7/pilot.json>) `5b36161b6015a12bbf64a838d757197449fe474c9a509a9ef80adf772ffe4191`, and [terminal receipt](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/d4-only-v1/d4-4134-7/receipt.json>) `037c77e05df03152715a0522b6ec5ee1c2efe5dc1388e7126a6eeb0a5b9a542c`.

The **R at-least-3 threshold is lead-accepted** from R1–R3. D1/D2 accepted plus D4 candidate reach the prespecified **D at-least-3 threshold, pending D4 lead review**; D3 stays technical-invalid and cannot be counted as a scientific nonpass. These are numerical-history transfer observations in source-ordered, unmatched cohorts. They do not establish physical recurrence specificity, healthy controls, natural recovery or a population pass rate.

## Final cost and stop

D4 charged `50.670438170433044` GPU seconds, seven model/two vision calls, peak RSS `11801952` KiB, GPU allocated `11291790336` bytes and reserved `12081692672` bytes. Its PID `1544507` is absent and its receipt terminal. All nine launched jobs, including the initial R1 failure and the R4/D3 parity failures, have terminal receipts and absent PIDs. Package total is **`360.0901660993695` seconds (`0.1000250461` GPU-hours), exactly 50 model/18 vision calls**, leaving `3239.9098339006305` seconds under the one-hour package cap. Cumulative sequence charge is `0.18968633946900568` GPU-hours under eight. Raw package artifacts total `129733682` bytes under 1 GiB. The finite package is stopped here for lead acceptance; no successor or additional call was started.
