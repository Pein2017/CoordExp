# Three-case latest-y1 V × key-phase candidate

Status: **worker candidate**. The lead accepted D1 (`mature:1584:2`) in [lead-first-case-acceptance.json](lead-first-case-acceptance.json). The fixed D2 (`mature:2299:2`) and D4 (`mature:4134:7`) calls completed under [lead-remaining-admission-v1.json](lead-remaining-admission-v1.json); those two await lead acceptance. No retry, new case, or successor was run.

## Frozen execution and technical result

The [all-three ledger](supporting/stage2-three-case-ledger-v1.json) is SHA-256 `2e49914de2cd0d38171adb76cf45b378f4d463006342b6ef309224639457991a`. It binds each source, raw vocabulary tensor, actual consumer record, terminal receipt, cold readback, reduction, and cost. The frozen [protocol](unit.md) and [manifest](manifest.json) have SHA-256 `b665c1ad61ce8930f59a278ac43f9f30104be89a3df2641263232795bcca4190` and `68854cbc6c1c81e122d00631b7724ae5891993ea83086014ee9dcffd87d8e015`.

The new [scale source](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_y1_value_phase/scale.py) has SHA-256 `5de12099f0ec99b9700731ab8f0ad72805b48d080ee407e63592b186d9f5ff61`. It binds each admitted case and charge, then invokes the unchanged, captured [first-case operation source](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/recurrence_y1_value_phase/run.py) (SHA-256 `6e606141af0efae208151b07c1f662937743891d2be7f52813da1b32860a309e`) for its patch, source forward, actual consumer observer, cold readback, and reducer. Each raw pilot names that operation source; the wrapper and exact commands are bound separately in [scale-preflight.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/remaining-v1/scale-preflight.json), SHA-256 `69a502ffee3c6969d60eae2fc6557a43313dfdf6acd3f3ec787672e9d34291c3`. No accepted first-case producer, candidate, or output byte was changed.

Before GPU work, both CPU plans bound the original four-request refined-00 source and all four live companions. D2: target index 1, left pad 150, donor/destination cache slots 1377→1386, width 1395. D4: target index 3, left pad 10, slots 1427→1436, width 1445. Each has 23,396,352 pixel elements; query-minus-latest-y1 rotary phase `(8,8,8)` and +9 key treatment offset `(-1,-1,-1)`. Each plan captured the operation source plus 22 direct maintained/dependency imports. The **actual caller** rejected mutated target, V slot, donor, phase, and missing prior charge. The 28-layer inference-tensor cache fixture rejected wrong target/slot/donor and unrelated K/V, with normal and forced-exception restoration. D4's GPU entry additionally required D2's terminal charge, cold readback, and reduction. The revised 2× forecast of 230.305500 seconds for the two cases fit the charged package and sequence caps before launch.

Every case used exactly: full original source, fresh native historical prefill, native anchor, latest-row K+9 anchor, K+9 identity-V sham, native-K donor-V, and K+9 donor-V. All three fresh native anchors met four-row full/cache parity at `2e-4`; every native and K+9 target anchor matched its accepted prior full vector exactly (max error `0`), and every sham matched its same-K anchor exactly. All 28 layers passed native replay, scale-aware phase construction and FP64 oracles. In all five cached cells per case, actual attention observers matched intended selected K and one V slot, native unselected target/companion K/V, causal mask and S rotary inputs; companion suffix K/V and finally-restored historical cache matched the native anchor. Separate-process cold readback passed for D2 and D4. Their six-cell raw records remain under:

- D2: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/remaining-v1/01-2299-2/case`
- D4: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-recurrence-y1-value-phase/remaining-v1/02-4134-7/case`

## Frozen prospective decision

`M = z(earlier-y1) - z(latest-y1)`, `Delta0 = M(native K, donor V) - M(native K, native V)`, `Delta1 = M(+9 K, donor V) - M(+9 K, native V)`, and `I = Delta1 - Delta0`. Strict pass requires `Delta1 > 1.0` nat and `I > 0.5` nat, with a 0.001-nat numerical HOLD band. These are raw full-vocabulary logits, not coordinate-renormalized probabilities. An independent CPU calculation from every raw vector reproduced all margins and the two same-K full-vocabulary TV values per case exactly.

| Frozen case | M native / +9 / native+donor / +9+donor (nat) | Delta0 | Delta1 | I | TV donor native / +9 | One-case category |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1584:2, lead accepted | `+4.362648 / -3.229649 / +4.354855 / -1.822130` | `-0.007793` | `+1.407518` | `+1.415312` | `0.008309 / 0.242158` | Pass |
| 2299:2, candidate | `+0.797851 / +0.113790 / +0.750878 / +0.467956` | `-0.046972` | `+0.354166` | `+0.401138` | `0.015034 / 0.106612` | **Valid nonpass** |
| 4134:7, candidate | `-7.859035 / -7.378774 / -7.892201 / -5.668036` | `-0.033166` | `+1.710738` | `+1.743904` | `0.008520 / 0.190184` | Pass |

**The frozen shared all-three prediction fails** because D2 is technically valid and below both thresholds. The D1 and D4 positives remain per-case observations. In D2 the +9 donor cell still predicts current y1 token 151778 as winner, with earlier/latest y1 probabilities `0.000324 / 0.000203`; D4 changes the winner to token 151966 while earlier/latest y1 probabilities are `0.000071 / 0.020624`. D1's +9 donor winner is likewise neither coordinate. The detailed earlier/latest/current probabilities, log probabilities, ranks, winner, runner, gap, and all cell hashes are in the case reductions bound by the ledger. Broad distribution changes and a contextual earlier-y1 donor preclude a literal-copying or unique-attention claim; the +9 arm moves the whole latest K row.

## Complete job and cost ledger

| Case/job | Status | Model / vision | Charged allocated GPU seconds | Cold | Reduction SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| 1584 first case | Lead accepted, terminal exit 0 | 7 / 2 | `56.562706403` | Passed | `af302612611bca67afa02e660eead27cfd41c7ef159bb3373ffd1e5a75a87257` |
| 2299 remaining 1 | Candidate, terminal exit 0 | 7 / 2 | `56.804913588` | Passed | `d32248b30023d650220b51b4b6f1165c7679de42941eac14a8e9246769e4e8ed` |
| 4134 remaining 2 | Candidate, terminal exit 0 | 7 / 2 | `61.230590738` | Passed | `8d22685bcb9adfef8fc798106cab153029528a5493b62e458b3b7885d4590c6a` |

Wrapper job charges include setup before the reused operation source: D2/D4 operation-only times were 56.577494361 and 60.946325719 seconds. **Total: 21 model / 6 vision forwards, zero generated tokens, 174.598210730 allocated GPU seconds (`0.048499503` GPU-hour)**, below the 900-second package cap. Sequence cost is `0.238185842` GPU-hour after adding the prior `0.189686339`. There were no failed jobs or extra calls. Combined first and remaining output roots contain 49,310,593 bytes, below the 128 MiB plan. Peak allocated GPU memory was 11,291,651,072 bytes; peak reserved 12,081,692,672 bytes; peak RSS 11,481,188 KiB. All three terminal jobs returned exit code 0 and the receipts mark terminal completion.

The package stops here. The worker has not accepted D2/D4, extended the cohort, searched another slot, or started a successor.
