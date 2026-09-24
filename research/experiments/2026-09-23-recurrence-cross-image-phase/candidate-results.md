# Cross-image relative-key-phase R1 candidate, attempt002

**Status:** bounded first-case **candidate for lead review**, not lead-accepted. Only `mature:7511:5` was executed. The seven other frozen numerical-history identities remain **HOLD**, regardless of this effect. Existing person/owner invalidity HOLD on image7511 remains separate; no physical recurrence, scene-health, natural-exit or box-recovery claim is made. The four-R replication and D comparison are unresolved.

## Contract, repair and calls

The frozen [unit](unit.md) and [manifest](manifest.json) are unchanged (SHA-256 `30f310c3adb5e2be63a5616e36726d000527bc3fafdb83a91108ae481b7197eb` and `1815e9e9cb5640876955d1f6b7b4bf2eb0d7d4ca4d292c60c2be374cef6201e0`). [Lead repair ruling](supporting/lead-repair-attempt-002.json) SHA-256 `2f2af5ed04156338d5443fa30236a6635127b1eb6d8ad8b5e8d6dd30e7e748ad` binds failed attempt001 and authorizes only the per-span `+9` correction and one fresh attempt002. Attempt001 terminal receipt remains SHA-256 `f2fff0bc94cdda193068bffed6eb6825732e4721ae82be4e07e0e5a93032ed39`.

CPU RED invoked the old actual destination function with the frozen earlier/latest dictionary and reproduced its `dict + int` TypeError, with no model or CUDA. CPU GREEN invoked the repaired actual function with maintained `Qwen3VLTextRotaryEmbedding` and config on CPU. It returned both nine-position destinations, all three axes exactly `+9`, unmodified source tensors, 9×128 cos/sin per span and a full FP64 oracle maximum error `1.3811381e-05`. The caller-facing verifier rejected wrong sign, clipped latest span, swapped spans and wrong native row. The original full-batch CPU checks also rejected wrong target index, history span, S sign/position and companion content. [RED](supporting/attempt002-cpu-red.json) SHA-256 `51688999b1a382ba8454d5f60005be61e4c87a47aadbf3b90e73ad31dd10b796`; [GREEN](supporting/attempt002-cpu-green.json) SHA-256 `747f0bf05dc99b7db29c402bc25336e38986da0d980e9f3313a6f825da82576f`. Fresh producer SHA-256 `1545524d2db5b4e3b62747af6d647f72e8111ab6125d69f5492a0abce7f6d3c9`; 21 direct source captures are bound in [preflight](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/attempt-002/preflight.json), SHA-256 `ddb77455ceaf1fb225772d30206606a61a5bee535965857b7f8ba7a1475d23a9`.

The source is the exact original-policy untied four-request refined-01 batch: request images 5001, 6040, **7511 at index2**, 10707; target source raw y1 offset50 after native x1 token151670. Full source-step width1370, earlier/latest physical spans `[1347,1356)` / `[1356,1365)`, five-token S `[1365,1370)`, and all-axis rotary starts411/420/429. The full input identity SHA-256 `94c2f8d936ae9366a1d8949c8cfda78f7ff0196db0fcb314ea35bc23e5718a7b` matched the source receipt. Untied model selected-ID hash `6e4ad29711e3d49a8d9b2b4ad714a3f27d584a81feb68c775648516b9e359f01`, input/output delta hashes `36d07fc1fa94efd27db18065714c56d3121eacbfedeb33114fc551f88d921b4c` / `1abba0209e5c5038f98255b0712c7c41628f06ba089ba9c4dabc443ee155e8a2` agreed with frozen source.

Attempt002 completed the exact sequence: full native source-shaped forward, historical prefill, cached native anchor, target-only S minus9 cross, native-S latest identity sham, latest K plus9, earlier K plus9. **7 model / 2 vision forwards, zero free tokens.** Full native used all four original images. The five cached suffix calls were image-free and kept the four physical suffix slots and original masks/companions. No other case or ablation was called.

## Technical qualification

- All four full native source-step chosen/top-two/logsumexp comparisons passed the original trace at `2e-4`; maximum top-two logit error was `3.2424927e-05`. The saved full-batch vocabulary vectors and actual full consumer inputs support this. Cached versus full native maximum vector error by batch index was `[4.00543e-05, 5.43594e-05, 2.57492e-05, 8.28505e-05]`, all below `2e-4`. The latest identity sham versus cached target native error was `1.03116e-05`.
- All 56 all-layer/two-row phase qualifications passed; largest measured/bound ratio `0.239056`. Loaded actual destination phase matched observed latest native phase for earlier and actual S first five for latest with maximum error **0** in each check. Full FP64 destination oracle error was `1.3811381e-05`.
- Each cached cell recorded the actual full-batch model input, three-axis S positions, cache slots and causal mask. All 28 attention consumers matched the intended selected native/sham/treatment K hashes; historical V, unselected K and all companions' historical K/V remained unchanged. Companion suffix K/V hashes also agreed across all target-only interventions. Patch/crop/finally restored the complete historical cache digest after each cell.
- Separate-process cold readback passed: it rebuilt all four source input/token/rotary rows independently from raw plus saved prompts, reloaded every vector, recomputed phase gates/candidate K, cached/full and sham errors, and checked all-layer actual-consumer and companion hashes. A separate FP64 full-vocabulary TV computation reproduced every reported baseline and ratio exactly. `readback` and `reduce` commands exited 0.

## Scientific first-case readout

FP64 full-vocabulary probabilities, no coordinate-only renormalization:

| Contrast | TV to cross | Ratio to native/cross baseline | TV to native |
| --- | ---: | ---: | ---: |
| Native to cross baseline | `0.132182017142` | `1` | `0` |
| Latest K plus9 | `0.022784744577` | `0.172374011759` | `0.115025634370` |
| Earlier K plus9 | `0.140837351055` | `1.065480419352` | `0.010157490778` |

Earlier-minus-latest ratio is `0.893106407593`. Thus this one **numerical R case** meets its frozen per-case selective-transfer criterion: baseline `>1e-6`, latest ratio `<0.5`, and earlier-minus-latest `>0.1`, clear of the `1e-6` numerical boundary guard. This is a distributional result: all full/native/cross/sham/latest/earlier global winners were the same native y1 token **152209**. Runner tokens were 152208 for full/native/sham/earlier and 152211 for cross/latest. Native y1 log probabilities were respectively `-3.206609`, `-3.206612`, `-2.939261`, `-3.206615`, `-2.996304`, `-3.222388`. The category is not a physical-owner label or an R-cohort replication verdict.

The [raw reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/attempt-002/reduction.json) SHA-256 is `ac406d6fb009a2d2e97e1de7d3a2f7afcfdd6e7ac2ba19a21fbcd2051c0de268`; its cell bindings lead to the complete target vectors, original four-request vectors, actual consumer records and all phase evidence. The terminal [receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-cross-image-phase/attempt-002/receipt.json) SHA-256 is `89ebbb0aeb26b7a3a8d3067872da95b881f2c656b264166b938025e67d09eed5`.

## Cost, terminal state and held-case reforecast

Attempt002 charged **50.484704 allocated GPU seconds = 0.014023529 GPU-hour**, including setup. Peak RSS was `11,648,968 KiB`; peak GPU allocated/reserved `11,158,611,456` / `11,983,126,528` bytes. The terminal job exited 0 and PID1504229 is absent. Its final artifact tree contains 39 files, `16,381,375` bytes. Attempt001 was terminal technical-invalid, charged `14.241614` seconds and 2 model / 2 vision calls. **First-case total including failure:** `64.726318` seconds = `0.017979533 GPU-hour`, **9 model / 4 vision calls**. Sequence cumulative is `0.107640826 GPU-hour` against 8; first-case cap is 540 seconds including failure, package cap3600 seconds. The failed attempt's 3,531,841 bytes remain preserved. No Git action, predecessor edit or held-case launch occurred.

The measured full-batch first case replaces the older six-call cost basis. For the seven held cases, applying each frozen pixel-element × padded-width ratio relative to R1 to its measured 50.484704 seconds, then a 2× margin for independent setup and variability, yields this **planning**, not runtime-bound, reforecast:

| Held case | Shape factor vs R1 | 2× planned GPU seconds |
| --- | ---: | ---: |
| mature:14038:10 | 1.086946 | 109.748 |
| mature:351017:5 | 1.058800 | 106.906 |
| mature:99184:7 | 1.067995 | 107.835 |
| mature:1584:2 | 0.998838 | 100.852 |
| mature:2299:2 | 0.998838 | 100.852 |
| mature:2685:12 | 1.069723 | 108.009 |
| mature:4134:7 | 1.034639 | 104.467 |

Seven-case planned additional cost is **738.670 seconds**. Including both R1 attempts, prospective package total is **803.396 seconds = 0.223166 GPU-hour**, below the 1-hour package ceiling; projected sequence cumulative is **0.312827 GPU-hour**. R1's 16.38 MB raw tree and the frozen shape range suggest artifact capacity remains below the 1-GiB plan, but this is not measured for the held seven. Their technical status, outcomes and any cohort threshold verdict stay **HOLD** until lead admission and execution.

Lead decision boundary: review R1's technical qualification, source/companion proof, numerical readout, failure-inclusive spend and measured-shape forecast. No self-acceptance, held-case scale or successor is implied.
