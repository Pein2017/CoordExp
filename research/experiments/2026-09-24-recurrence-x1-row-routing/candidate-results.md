# First-x1 row-routing candidate

Status: **candidate cold readback passed; lead acceptance pending.**

Protocol SHA `bb5e59d6b306335ed72f6b1ed811e1e346595126099424f120d2c812c13557b6`; admission SHA `e8bef47468960f0d1259943ffd4c2e62553dd651a25b26da88d065a68b5bd15a`; producer SHA `4efd0abb3888c9a4d400312f966ecf41acb2a75021dedd8436afcb50f89ed914`.
Receipt SHA `06902cb4bef39a5377f1f09884e58deee68388c13adeea8cb874abf3606b1a8f`; cold readback SHA `9a4b442dd6497105cfdbf2e9b410ba158c4febffcfb69c457ac4179dbcb1e57e`. Original refined-03 four requests, target2 train351017. AF/FF history differs only at older raw5/6/7; all current headers were freely emitted.

## Frozen outcomes

Shared reciprocal primary **False**; AF broad-A **False**, FF fragment-F **False**. Exact free-suffix secondary AF **False**, FF **False**, conjunction **False**. Supplied x1 has no prediction credit.

| Arm | Complete selected row | Stop | Box | IoU A | IoU F | Region |
|---|---|---|---|---:|---:|---|
| AF_native | 151646,8987,151647,151648,151670,151670,151699,151756,151649 | complete | [0, 0, 29, 86] | 0.004002851346164391 | 0.8787878787878788 | fragment_F |
| FF_native | 151646,8987,151647,151648,151671,151683,152206,152669,151649 | complete | [1, 13, 536, 999] | 0.9981343283582089 | 0.004424141875563434 | broad_A |
| AF_sham | 151646,8987,151647,151648,151670,151670,151699,151756,151649 | complete | [0, 0, 29, 86] | 0.004002851346164391 | 0.8787878787878788 | fragment_F |
| FF_sham | 151646,8987,151647,151648,151671,151683,152206,152669,151649 | complete | [1, 13, 536, 999] | 0.9981343283582089 | 0.004424141875563434 | broad_A |
| AF_flip | 151646,8987,151647,151648,151671,151670,151717,151756,151649 | complete | [1, 0, 47, 86] | 0.006346698318257247 | 0.6808510638297872 | fragment_F |
| FF_flip | 151646,8987,151647,151648,151670,151683,152206,152669,151649 | complete | [0, 13, 536, 999] | 1.0 | 0.004554520962329253 | broad_A |

Raw greedy and selected tokens, every full-vocabulary vector, input and stepwise log-normalizer are in the cold readback. The original image overlay is bound there. F/F2 physical owner remains HOLD.

## Qualification and cost

Both nine-step natives and both independently executed identity writes passed same-base saved input/vector and source gates before either flip. Flip steps0–4 matched same-base native references before selection. Actual next-forward consumption, native masks at all28 layers, rotary positions, historical and companion states, source trace, parser, terminal jobs and all raw vectors passed separate CPU readback. Beyond nine steps, source companion traces were checked without inventing a native same-step full vector.

Fresh 54 model / 54 vision / 54 logical emitted, of which 50 greedy, 2 identity writes and 2 flips; zero reuse. Parent outer 147.846220292s, internal 141.980068393s. Prior sequence 0.726550840497GPUh, charged cumulative 0.767619235023GPUh.
Peak RSS 11589000KiB; GPU allocated/reserved 9945811456/11075059712B; raw bytes before receipt 842288754; terminal PID 2268353, exit 0.

Raw attempt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-x1-row-routing/attempt-001`. This is a conditional first-token intervention, not a physical new-owner, natural onset or internal-pathway claim. No self-acceptance or successor.
