# Cross-image phase: remaining-case candidate and technical stop

Status: **worker candidate; queue stopped at R4 technical qualification**. This is not lead acceptance. The admitted R2, R3, R4, D1–D4 order was followed until the R4 gate failed. No R4 crossed, sham, or treatment readout, and no D call, was made. The eight frozen identities remain in the denominator. See the [exact terminal ledger](supporting/scale-terminal-ledger-v1.json) and [CPU gate record](supporting/scale-cpu-preflight-v1.json).

## Binding and qualification

The [original protocol](unit.md) SHA-256 is `30f310c3adb5e2be63a5616e36726d000527bc3fafdb83a91108ae481b7197eb`; [manifest](manifest.json) `1815e9e9cb5640876955d1f6b7b4bf2eb0d7d4ca4d292c60c2be374cef6201e0`; [remaining admission](lead-remaining-admission-v1.json) `74fd8ede6820b0f1c27c2c5f8008343605e27f554d5ab4cbceb9957cb51f96e4`. Accepted R1 [candidate](candidate-results.md) remains `782ff18654fb75e5ff77fc67c62dd1c891740d143b4c60037428f75945d76569`; its captured [producer](../../../probes/training_set_completion/recurrence_cross_image_phase/run.py) remains `1545524d2db5b4e3b62747af6d647f72e8111ab6125d69f5492a0abce7f6d3c9`. The separate [scale producer](../../../probes/training_set_completion/recurrence_cross_image_phase/scale.py) is `cff77ec4b6ae3f129fa94c1710d7264810dc134cba4091668a2ab738b78238c4`.

Before any new GPU call, the [preflight](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/preflight.json>) (`a4fa0966317618e341a57996710c36b4f55e21b9caf8bbf429bce59ebf9b02ca`) reconstructed all seven original four-request batches on CPU and captured 22 producer/direct-import sources. It froze exact commands, full input/position/cache hashes, target indices, six ended companion tails, original source IDs/media, and actual rotary CPU checks. The consumer verifier rejected wrong target, phase sign, S position, historical span, companion content/position, and mutations of ended pad tails. The two-times shape forecast was 738.6697979024 seconds against 3535.2736821771 remaining seconds. Each executed job checked actual model inputs, image routing, full-batch masks/cache slots, all-layer selected K, unchanged V/unselected K and companion historical/suffix K/V. Full vectors and consumer observations were saved before reduction.

## Eight-case ledger

`B` is full-vocabulary TV(native,cross); ratios divide treatment-to-cross TV by B. `—` means no valid scientific readout. R1 is already lead-accepted; R2/R3 are worker candidates pending lead review.

| Order | Case | Group/target | Ended companion indices | Status | B | latest ratio | earlier ratio | New calls model/vision | Charged seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | `mature:7511:5` | refined-01/2 | none | lead-accepted selective | 0.1321820171 | 0.1723740118 | 1.0654804194 | prior 9/4 incl failure | 64.726317823 |
| R2 | `mature:14038:10` | refined-02/2 | 0 | candidate selective | 0.2643368016 | 0.1258403122 | 1.0176040487 | 7/2 | 50.559775978 |
| R3 | `mature:351017:5` | refined-03/2 | 1 | candidate selective | 0.2726771880 | 0.0837612988 | 0.9890948519 | 7/2 | 49.877197102 |
| R4 | `mature:99184:7` | fresh-06/1 | 0,2,3 | technical invalid at full/cache gate | — | — | — | 3/2 | 21.416175060 |
| D1 | `mature:1584:2` | refined-00/0 | none | held, unrun | — | — | — | 0/0 | 0 |
| D2 | `mature:2299:2` | refined-00/1 | none | held, unrun | — | — | — | 0/0 | 0 |
| D3 | `mature:2685:12` | refined-00/2 | 0 | held, unrun | — | — | — | 0/0 | 0 |
| D4 | `mature:4134:7` | refined-00/3 | none | held, unrun | — | — | — | 0/0 | 0 |

R2 and R3 each passed source trace parity for active rows, all four cached/full vector comparisons below `2e-4`, sham agreement, all 56 row-layer phase gates, and separate-process cold readback. Their [reductions](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/02-14038-10/reduction.json>) and [R3 reduction](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/03-351017-5/reduction.json>) have SHA-256 `72f4288e0130a7c639720cb11564a66863e1516af149ac87bc54df0cbbd4e737` and `310cba60bc9e1940538a3163a080a44042f08afb2e7d5616a489ff7a4ff98db4`. R2 global winners: native `152326`, crossed/latest `152340`, earlier `152326`; R3 all four winners `151670`. These are secondary categorical observations, not the criterion. R2 maximum row parity error was `1.39594078e-4`; R3 was `1.71184540e-4`.

## R4 stopping evidence

The original full source target row 1 passed chosen/top2/logsumexp trace parity. Rows 0, 2 and 3 had ended before offset 68 and were explicitly marked trace-undefined after EOS, with original EOS/pad tails retained. The actual full, prefill and cached-native consumers agree on every row's token split, attention mask split, three-axis position split and cache slots; full and prefill carry image inputs, cached suffix is image-free. Full versus cached native maximum absolute logit differences by batch row are:

| Row | Source status | Max absolute difference |
| --- | --- | ---: |
| 0 | ended | 0.0001401901 |
| 1, target | active; source trace passed | 0.0000224113 |
| 2 | ended | **0.0002126694** |
| 3 | ended | 0.0001657009 |

Row 2 exceeds the frozen **0.0002 all-row gate**. Its top two tokens match, and the excess is small, but the mandatory companion parity fails. The [terminal receipt](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/04-99184-7/receipt.json>) is `088cfd822f60cb48b61eb306f35e915c21baf13f2ff11e9f2b698e096ae8098a`. Raw [full](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/04-99184-7/cells/01-full-native/full-batch-vocabulary.pt>) and [cached](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/04-99184-7/cells/03-native/full-batch-vocabulary.pt>) four-row vectors have SHA-256 `564a5d691863324d91f651a3a9be9388e8de5c779352da144c0d6e5f708c02d6` and `8ac7bdd47dba45182f5edb8ed11bcda227f5a580c0f7538265e0fb3a37b1c344`. The saved [cached consumer](</data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/remaining-v1/04-99184-7/cells/03-native/consumer-raw.json>) is `a2c58918146e82dc2cd5b20bb73e5b29326febb2733d112f23a3f6e7932b5ba6`. The cause of the row-2 numerical difference is not established. No tolerance change, retry, replacement or later cell was made.

## Decision and cost boundary

Three R cases have selective numerical outcomes, so the prospective **at least 3/4 R threshold is met as a worker candidate** while R4 remains technically unanswered in its denominator. The D comparison is unresolved because all four D cases are unrun. No physical recurrence, healthy-control specificity, natural continuation or box-recovery claim follows from these numerical labels.

New charged time is `121.8531481400` seconds; package including R1 and its failed attempt is `186.5794659629` seconds (`0.0518276294` allocated GPU-hours), leaving `3413.4205340371` seconds under the one-hour package cap. Cumulative sequence charge is `0.1414889228` GPU-hours under eight. Total calls including the R1 failure are **26 model/10 vision** versus the admitted 58/18 ceiling; zero free tokens. All three new GPU jobs have terminal receipts and absent PIDs (R2 `1519560`, R3 `1521009`, R4 `1522453`); new peak RSS was `11629208` KiB, peak GPU allocated `11315638784` bytes and reserved `12152995840` bytes. Package raw artifacts total `66807956` bytes under the 1 GiB planning limit. R1 bytes and source remain unchanged. The queue is stopped at the admission's technical-failure boundary for lead ruling; no further execution is authorized by this worker report.
