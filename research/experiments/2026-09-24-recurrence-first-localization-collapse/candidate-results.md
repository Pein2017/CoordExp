# First-localization-collapse candidate, attempt 001

Status: **worker candidate; lead acceptance pending**. The original refined-03 four-request source, target 2 (`train351017`), row0 history `raw[:9]`, native positions, and all companions were retained. This is a numerical localization intervention; row1/row2 physical identity remains HOLD under the accepted visual correction.

## Frozen bindings and qualification

- Protocol `unit.md` SHA-256 `3ea786e900a491b32c540f997456ebbfdcc69ed9b123de9e1f94ceca6652c708`; admission `lead-admission-v1.json` SHA-256 `2736fe8e963b2d3aa134157457d550074cc43c4659b34451c7bbce38b8909954`; visual acceptance SHA-256 `920ce5d329a1ba547d4b952e0f0db4fffb1cdd609e660cee2a2cbbde1601b2d3`.
- [CPU preflight](supporting/attempt-001-preflight.json) SHA-256 `bf60e6268ed3c40a041004211062b0bf64fe87d852e214d2d5605bf848df7f73` froze the exact commands and 25 direct source captures. Producer SHA-256 `8470bfb24547dbb0dbeaa24393861445baecdc3ff85bc6681b8986b7ffed3d9b`. The original raw/trace/runtime receipt SHA-256 values are `77a65cc8dc35220ec58766cca92b9caa9225eb4ce99c0330e340d40b6cfd8cbf`, `1b75265af561e712e74a74f455ae484dde17468fb5e76367b5cb32b6838295d6`, and `aebb2eb404e9d0d183976334e28cd56a35ec414c5748d33399023d2cd6ae325e`. Image SHA-256 `8bb2f91a7a1672a4aacd41092dc9480dbb86c50ac32ba2566f7e5e622e6db1c9`; input identity SHA-256 `94d4ee001a7c36a4a598b96a5ace0cf1c04718c83025ee5aea18906c0c856150`. Checkpoint is the bound step-2444 DoRA adapter plus untied special-token delta; maintained loader SHA-256 `b283d22486e9fef91193de3dfbd295abc07f1292a9165a966d1525735140bb08` matches the historical capture.
- CPU preparation reproduced four request IDs and source batch: prompt lengths `[1336,1362,1362,1320]`, raw lengths `[255,37,3084,3084]`, 24,502,272 image elements, target prefix width 1371 and maximum width 1386. Forty-six parser, actual-caller, mask, source, target, position, companion, own-token, and receipt mutation checks passed before model load. Artifact forecast 488,447,776 bytes under the 1 GiB envelope; 186.16 seconds planning forecast, with no elapsed-time ceiling.

## Complete trajectories

All arms generated their own tokens by full-vocabulary greedy argmax, including t0. No token was supplied, no grammar filter was used, and no model/vision call was reused.

| Arm | Complete row token IDs | Decoded box | Stop |
| --- | --- | --- | --- |
| native | `151646,8987,151647,151648,151670,151670,151703,151756,151649` | `person [0,0,33,86]` | complete at 9 |
| independent native-4D sham | same as native | `person [0,0,33,86]` | complete at 9 |
| latest-record-read mask | `151646,8987,151647,151648,151670,151683,152206,152669,151649` | `person [0,13,536,999]` | complete at 9 |

The masked row is **exactly** the original row0 nine-token serialization. Its valid box has IoU 1.0 with frozen broad A `[0,13,536,999]` and 0.004554520962329253 with frozen fragment F `[0,0,33,86]`: the frozen numerical primary passes and the secondary exact-row0 forecast passes. [Overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-localization-collapse/attempt-001/overlay.png) shows the numerical regions on the original image; it does not establish physical recurrence or a shared person for F.

## Consumer and cold-readback evidence

- [Raw receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-localization-collapse/attempt-001/receipt.json) SHA-256 `31f4a39b69b0dbe845889a3c60215ec2a1ff65e4e85197638b4b92e2b5b5ac38` binds every full vocabulary vector, input, actual mask, all 28 layer mask entries, historical/companion states, own greedy token, and source trace. Native source parity passed for all four rows at all nine steps; active companion parity passed at every sham/mask step. Sham full-vector maximum error 0. Historical and companion state/vector maximum error 0 over all matched steps. The t0 mask was native; subsequent target queries `[9,9+t)` alone were denied row0 keys `[0,9)`, and actual same-rectangle complements stayed native.
- Separate CPU [cold readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-localization-collapse/attempt-001/readback.json) SHA-256 `bbb7a44c0054fa1f6ca7243c0b7ca17c0cf7c2c5b66b23cc22ac84669cec4933` reconstructed all 27 inputs, greedy tokens, parsed stops, source/companion traces, 28-layer masks, and endpoint from the saved raw files. It used no language-model load or GPU call. Beyond native step 8 there was no invented native vector reference; the treatment also stopped at step 8.

## Finite cost and boundary

The [outer terminal record](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-localization-collapse/attempt-001/outer.json) SHA-256 `531067beb421f4fe77720fa7373b00a06fe3835d6cd132c20e6b14f31d1b7436` records one model child PID 2059680, exit 0, PID absent after completion, and **79.95819828659296 outer seconds** (74.37151952087879 internal seconds). Counts: 27 model, 27 vision, 27 emitted target tokens, zero reused, versus maxima 34 each. Peak RSS 9,485,612 KiB; peak GPU allocated/reserved 9,964,238,336/10,779,361,280 bytes. Final raw-root bytes 361,619,461. Prior sequence 0.44266888110542796 GPU-hours plus this outer charge gives 0.4648794917405927 GPU-hours. A local parent-wrapper assertion before spawning any child was corrected; it incurred zero model, vision, and GPU calls. [Stdout](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-first-localization-collapse/attempt-001/stdout.log) SHA-256 `50f0cc2973624a268eec6ab61547aa2a1a62b4dd7a3f6f26e2b7141f6a447479`.

The result demonstrates a complete numerical broad-A return under this exact read mask and original source. It does not identify the physical owner of F, prove a general recurrence mechanism, or authorize another arm. The branch stops here for lead review.
