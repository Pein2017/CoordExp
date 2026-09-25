# Older-record routing: bounded worker candidate

Status: **technically qualified candidate; frozen primary paired prediction NONPASS**. Lead acceptance and physical attribution remain pending. The five admitted arms are complete; no retry or successor was run.

## Binding and execution

- Authority: `unit.md` SHA-256 `6735d7b151cb0b3f23b131c38fa5dc9b6145e260ca7d51ae6270b8307f2f4b4d`; `lead-admission-v1.json` `c9f3aa466ce19a37f5ab56ea556722ecae2bd68704832160386c43986b909d4e`; predecessor acceptance `235d852f5af846a53b88abbe6525430f4c17482ef415c766ece8bfa5723665d2`.
- Original new-08 full four-request source, target index 1 (`train151704`), raw history `[:45]`, latest B raw `[36,45)` unchanged, original image/media/positions/causal masks and three companions. Source SHA-256: image `fb4947d5efc5674259b0b9fea8074b0b08803b49a612d5408d2de2ca61203846`; raw `b6747f8950b7a1ed52df40503a3f9cbfe5a494c395ad6181f8857e6d6d220381`; runtime receipt `c8c31700e78b356bab061c36b5cef111a9c960a61178c7eb0b851d87c045d7da`; trace `a99ff250db03455f6afe8ebbca24b88148cfb92b8e894b97a9e71c279ffa3ec7`.
- Untied step-2444 FP32 SDPA. Preflight `9992b2beaa075ba744a56c56cf1ca3650f5198b41078671eb7de0d61fb0818fc` froze the producer `faf12488e41d51fd4159f982476f1de756b4b65607f36a5503c6184aae96f35b`, all 21 directly used source captures, exact commands, 119 CPU caller/parser checks, source identity and full-batch shape. Prompt width 1362; executed widths 1407–1415, within admitted 1422; 23,924,736 pixel elements per forward. The sham and every treatment executed its own t0; no current token was supplied, no cache split or grammar constraint was used.
- Raw root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-record-routing/attempt-001/`. The [machine ledger](supporting/older-record-candidate-ledger-v1.json) SHA-256 `baabff9ba275bf3da8108fe5a675d9168856cc4b5e102730ac6f1afbb45fe698` binds all 45 vector hashes, inputs, receipts, cold readbacks, outcomes, charges and overlay.

## Complete free rows and frozen outcomes

The original Q row is `[151646,2190,151647,151648,151926,152346,152058,152421,151649]`, a complete canonical `book` box `[256,676,388,751]`. The `earlier_A` row is `[151646,2190,151647,151648,151906,152412,151953,152437,151649]`, a complete valid `book` box `[236,742,283,767]`. IoUs below were independently recomputed from the saved boxes, with B*=`[207,829,270,852]` and Q=`[256,676,388,751]`. `q=IoU(Q)-IoU(B*)`; each delta is relative to fresh native.

| Arm | Final three history rows | Free row | IoU(B*) | IoU(Q) | q; Δq | Frozen region |
| --- | --- | --- | ---: | ---: | ---: | --- |
| native | A,B0,B1 | exact original Q nine tokens | 0 | 1 | 1; 0 | Q |
| identity-write sham | A,B0,B1 | exact original Q nine tokens | 0 | 1 | 1; 0 | Q |
| earlier_A | A,A,B1 | distinct nine-token box `[236,742,283,767]` | 0 | 0.02243353028065 | 0.02243353028065; −0.97756646971935 | neither B nor Q |
| earlier_B1 | A,B1,B1 | exact original Q nine tokens | 0 | 1 | 1; 0 | Q |
| swap_earlier | B0,A,B1 | exact original Q nine tokens | 0 | 1 | 1; 0 | Q |

The primary pair fails: `earlier_A` is not B-region, although `earlier_B1` is Q-region. The same-multiset swap emits Q exactly, but the prospective order discriminator required the primary pair first and is **unresolved**. This is descriptive output, not support for total-B evidence or an order-specific circuit. Secondary exact outcomes: `earlier_A ≠ B1`; `earlier_B1 = Q`; `swap_earlier ≠ B1` and `swap_earlier = Q`. All outputs are complete book rows; no cap, malformed, EOS or invalid-geometry outcome occurred.

The [five-arm source overlay](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-record-routing/attempt-001/five-arm-overlay.png) displays the frozen A/B0/B1/Q references and each generated box. The `earlier_A` box falls at the lower-left edge of the Q-area object and above A; physical identity is **HOLD for lead review**. Q remains physical owner **UNKNOWN**. Numerical Q matching does not establish a new owner or a valid book. The treatment result cannot identify a visit ledger or the cause of natural onset.

## Qualification and cost

- First block native/sham/earlier_A: 27 model, 27 vision, 27 free emitted tokens; receipt SHA `dfdee1495d0946440c849f033706883fd3459b2b191e75786c061808dc3e0e9d`; separate-process cold readback SHA `9f840630365190c0a8ab4276f237fefc5a45625607db8bd724b82c6e18fbf404`; parent outer charge **92.17608365416527 s**. Its complete technical readback passed before the second block.
- Second block earlier_B1/swap_earlier: 18 model, 18 vision, 18 emitted tokens; receipt SHA `36b1a8bebac18f765098a236f44f28cf2eefa4a8fc533d3ce8d7189ae3bc8bdc`; cold readback SHA `734aebff1fa491bfb94e430a625292b2856093f0b5b1d3e4f6957822aec532c9`; parent outer charge **65.90398306399584 s**.
- All 45 raw full-vocabulary vector and input hashes were independently rechecked. Actual input IDs, masks, three-axis positions, cache positions and media match saved caller inputs; all 28 text layers consumed the native mask on each forward. The 144 applicable source/companion trace entries pass, maximum selected score/log-normalizer error `6.866455078125e-5` against the `2e-4` gate. Sham full vectors and historical states agree exactly with native in the recorded comparison; companion vectors/states agree exactly at matched steps. Each cold readback reconstructed its declared historical writes, own greedy prefixes and parser outcome from saved evidence. No prior row5 full-vector reference was invented.
- **Total: 45 model / 45 vision / 45 emitted tokens / 0 reuse.** Parent outer charge **158.0800667181611 s** = `0.04391112964393364` allocated GPU-hours. Prior sequence `0.39875775146149434`; candidate cumulative `0.44266888110542796` GPU-hours. Both execution children exited 0 and PIDs `1976558`, `1977942` are terminal absent. Maximum RSS `11,498,268 KiB`, GPU allocated `9,952,458,240 B`, GPU reserved `11,066,671,104 B`. Raw-root artifacts including overlay are `632,847,432 B`, below the 2 GiB planning envelope.

This worker stops at candidate evidence. Lead owns final technical/scientific acceptance and physical attribution.
