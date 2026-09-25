# Contextual K/V carrier candidate

Status: **candidate cold-readback passed; lead acceptance pending.**

Admission SHA `4b52088233fd8fc5c5e898de31844b990e026fb2b53167c8f192ce103536ac18`; producer SHA `11c262a845a694e0575120bb4b584d854a3bb63dfc940672da3b95eeb867da6a`.
Raw receipt SHA `d3a102fcfe5bd8adac4b8a00f7326d05095e1cf1559b6b1e1267472715f63357`; readback SHA `3291a8f7343b9f100a38135e207bee3ba8d84c1d68b1936125270e2da58703ab`.

## Bound source and conditioning

Original refined-03, four requests, target index 2 train351017; untied step2444 FP32/SDPA, original image SHA `8bb2f91a7a1672a4aacd41092dc9480dbb86c50ac32ba2566f7e5e622e6db1c9`. Original raw SHA `77a65cc8dc35220ec58766cca92b9caa9225eb4ce99c0330e340d40b6cfd8cbf`, trace SHA `1b75265af561e712e74a74f455ae484dde17468fb5e76367b5cb32b6838295d6`, input identity `94d4ee001a7c36a4a598b96a5ace0cf1c04718c83025ee5aea18906c0c856150`. Source images, companion tokens, positions, attention and checkpoint remained bound. Preflight SHA `37c242f875e2eccaa2b3f8008858b6368328a1203e2ec634e5780d8915bda308` captured 29 maintained/import sources and the exact commands before model load.

AF versus FF changed only older-row target raw slots 5/6/7. Both replayed the **same naturally emitted four-token header** `[151646,8987,151647,151648]` at current physical slots `[1380,1384)`; it was conditioning, not free generation. Historical prefill was width1380, latest F occupied physical `[1371,1380)`, and no phase or position was edited. The original two-record shared conjunction remains a distinct NONPASS.

## Frozen endpoint

Anchor TV B = 0.79346795136; shared outcome **older_origin_comparator_pass**.

| Hybrid | TV to latest | TV to older | r_L | r_E | Category |
|---|---:|---:|---:|---:|---|
| hybrid_AF_FF | 0.530136028192 | 0.342365983578 | 0.668125319093 | 0.431480544351 | older_origin |
| hybrid_FF_AF | 0.78242855438 | 0.0964808023915 | 0.986087154547 | 0.121593823955 | older_origin |

Both directions meet `r_E<0.5` and `r_L-r_E>0.1`, away from the `1e-6` numerical guard. Neither meets the prospective latest-origin primary. An independent NumPy FP64 reduction of the saved raw vectors reproduced B, both TVs and all ratios within `5.6e-16` of the cold PyTorch reduction.

| Cell | Winner / runner | Gap | z(151671)−z(151670) | P(151670), rank | P(151671), rank |
|---|---|---:|---:|---|---|
| anchor_AF | 151670 / 151673 | 1.272991 | -1.332893 | 0.096842, 1 | 0.025539, 3 |
| anchor_FF | 151671 / 151670 | 0.302612 | +0.302612 | 0.307634, 2 | 0.416349, 1 |
| hybrid_AF_FF | 151670 / 151671 | 1.001644 | -1.001644 | 0.204731, 1 | 0.075193, 2 |
| hybrid_FF_AF | 151670 / 151671 | 0.077579 | -0.077579 | 0.367901, 1 | 0.340439, 2 |

The second hybrid's categorical winner switches to 151670 while its **full distribution remains much closer to the older FF anchor**. Winner coincidence does not change the frozen TV verdict.

Full-vocabulary FP64 softmax and TV; both directions retained. The common header is replayed conditioning, with zero generated tokens. These x1 distributions do not establish a physical owner or natural mediation percentage.

## Qualification and resources

All four fresh references/prefills, two cached anchors and two explicit K/V identity shams qualified before either hybrid; all-four reference/cache/sham max-absolute logit gates use 2e-4. Actual 28-layer selected K+V, older/prompt/companion complements, native mask/positions and finally restoration were checked and cold-read separately.

Fresh full_AF and full_FF each matched their accepted saved all-four vectors at max error `0`; AF matched the original target and companion source traces, whereas FF used the accepted counterfactual vector and original companion traces. Cached/full max errors were `5.340576171875e-5` (AF) and `5.7697296142578125e-5` (FF). Both explicit sham/anchor max errors were `0`; hybrid companion logits and suffix K/V matched their same-base anchors (`0` error). The two prefill caches had exact equal target prehistory and companion K/V; first-layer latest K **and V** were equal, with later-layer latest K/V differences observed. Prefill origin/hash evidence is SHA `82c7ad4e2296d8aa818b1beee5e150ebe1d34077668f4b6ac2f9cec2211d4b81`; saved donor/older tensors SHA `7f4e495d5bbf43d7d5f8f7eeb92d3345ba6e2d764a23ee55c66ad7f4969e345a`. The previous saved layer-0 hidden states were post-block outputs; they were not used as first-layer K/V evidence.

Actual calls 10 model / 4 vision / 0 generated; parent outer 82.676312357 s; internal 76.996800244 s. Prior sequence 0.537174243789956 GPUh; new cumulative 0.560139886111 GPUh.
Peak RSS 11155960 KiB; GPU allocated/reserved 12476571136/13254000640 bytes; artifact bytes before receipt 34867706. Terminal child PID 2138312, exit 0.

Parent outer receipt SHA `d6404a813f493cc878e6887eafe7170d69b0412ca31f30c2a3405ba29af5b7f2` records the terminal child and charged setup. Exact commands: `python -B -m probes.training_set_completion.recurrence_contextual_kv_carrier.run preflight`, then `run`, then separate-process `readback`. The raw root is `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-contextual-kv-carrier/attempt-001`; total artifact bytes after cold readback are 35,021,374. One terminal GPU job, no failed job or retry.

Individual full vectors, source inputs, all-layer consumers, secondary winner/rank/probability records and hashes are indexed by the cold [readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-contextual-kv-carrier/attempt-001/readback.json).

Original two-record shared conjunction remains NONPASS; F physical owner remains HOLD. This candidate is not self-accepted and admits no successor.
