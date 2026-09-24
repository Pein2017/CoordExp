# Book first-revisit native-reading candidate

Status: **worker candidate; lead scientific and physical acceptance pending**. Both admitted states completed in frozen order. Row4 passed separate-process cold readback before row3 launched. There was no extra cell, retry, forced token, cache split, or model readback.

## Binding and execution

- Frozen [protocol](native-read-protocol.md) SHA-256 `e74467fa451e157c5a8f118eb1069c0fa3d21251a83681df7292674a71a616aa`; [admission](lead-admission-v1.json) SHA-256 `08fc23ff2693d3597263a72fb3f8b7c57348ffed7b2f731e8ee9a328574284ea`.
- [CPU preflight](supporting/native-read-preflight-v1.json) binds the maintained producer and 18 source captures, source identity, installed native SDPA mask, full four-request new-08 batch, model loader SHA/size crosswalk, exact commands, 1413 maximum width, 376.389902-second planning forecast and 1 GiB artifact envelope. A [direct-import supplement](supporting/native-read-direct-import-supplement-v1.json) captures `artifacts.py` before GPU. CPU actual-caller and parser checks: 65.
- Untied step2444 FP32 SDPA source: target index 1, image train151704, batch request IDs 135237/151704/159714/169872, full prompt width 1362, left pads 0/42/78/42, 23,924,736 pixel elements. Raw, trace, runtime receipt, image, source panel, checkpoint/row/delta hashes and current maintained loader bindings are in the [cold summary](supporting/native-read-cold-summary-v1.json). The historical loader capture is absent; only byte-identical maintained `src/qwen/untied_embeddings.py` was imported.
- Raw full-vocabulary vectors, input tensors, per-step checkpoints, all-layer consumer and state observations, terminal receipts, outer process records and overlays are under [native-read-v1](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-book-first-revisit/native-read-v1). The cold summary binds all 52 raw vectors individually and its vector-manifest digests. No raw artifact was reduced before saving.

## Complete freely generated rows

| State and arm | Emitted token IDs | Box, 0–1000 | Stop |
| --- | --- | --- | --- |
| Row4 native | 151646,2190,151647,151648,151877,152499,151940,152522,151649 | [207,829,270,852] | complete book |
| Row4 identity sham | same as native | [207,829,270,852] | complete book |
| Row4 latest-row mask | 151646,2190,151647,151648,151867,152499,151936,152529,151649 | [197,829,266,859] | complete book |
| Row3 native | 151646,2190,151647,151648,151867,152499,151940,152529,151649 | [197,829,270,859] | complete book |
| Row3 identity sham | same as native | [197,829,270,859] | complete book |
| Row3 latest-row mask | 151646,2190,151647,151648,151860,152438,151924,152484,151649 | [190,768,254,814] | complete book |

The first token of each masked arm was the observed native greedy opener, reused at t0 only after exact empty-rectangle input/mask identity; it was not supplied. The mask then remained active at t1–8 while each arm consumed its own greedy output. Native and sham completed their exact nine-token source rows. Both masked rows have valid geometry and free class token 2190 (`book`). At each state the native/sham and masked prefixes match through the four-token opener/class/box-start; coordinates are freely generated.

The [row3 overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-book-first-revisit/native-read-v1/row3_new_B/overlay.png) supports the masked upper book A1652859: annotation IoU 0.673993 with A, 0 with B. The [row4 overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-book-first-revisit/native-read-v1/row4_first_B_revisit/overlay.png) supports the masked lower book B2141669: IoU 0.567515 with B, 0 with A. This is a worker visual proposal for lead review; IoU alone does not assign physical identity.

Conditional on that visual attribution, the frozen primary paired physical forecast is supported: masked-minus-native scores are row3 −2 (B to A) and row4 0 (B to B), so row4-minus-row3 is +2. Both stronger secondary **exact preceding-row** forecasts fail: row3 masked x1=190 versus preceding A x1=189; row4 masked x2=266 versus preceding B x2=270. No exact-row promotion from owner proximity.

## Technical gates and accounting

- Both separate CPU cold readbacks passed: original source and all four active traces at every native step; source and own-prefix inputs and positions; full-vocabulary greedy choices; all 28 actual text attention masks per executed forward; 9t selected native-readable cells, same-rectangle complement hashes; prior-history and companion layer states; all three companion vectors/traces; sham full-batch vectors. Maximum observed sham, history-state and companion-state/vector differences are exactly zero. No t9–15 reference was invoked because both treatments completed at t8.
- Each state used 26 actual model/26 vision forwards, 27 logical tokens including one reused t0, no free-token supply. Total: **52 model, 52 vision, 54 logical tokens, two t0 reuses**. Both child PIDs are absent; both receipts and cold readbacks are terminal.
- Parent-monotonic outer charges: row4 **72.949521162 s**, row3 **72.360749230 s**, total **145.310270391 s**. Prior sequence 0.314055537790022 GPUh; candidate cumulative **0.3544195017876344 GPUh**. Final raw root 570,630,060 bytes, below the 1 GiB planning envelope. The forecast was an estimate, not an elapsed-time gate.

This intervention removes one readable prior row after the native opener. It changes the effective prefix and does not identify a covered-owner ledger, literal copy operation, or natural onset mechanism. The paired outcome is limited to these two different histories/positions and the reviewed A/B books.
