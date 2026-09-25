# Older-record K/V split candidate

Status: **candidate cold-readback passed; lead acceptance pending.**

Protocol SHA `7b4eb5bf1ba144d468440c84960ccfc8b8b73e0db4ed5cafb1fda9c6561e9fe0`; admission SHA `619e7be923579a0225ac3e9dd3ce4f76a65c9c8b0207094643ed7387dccc3620`; producer SHA `ffb975f6c0a30c0adbd9c768e19f48b7566cbda83a43a42167e3acdd7e7533a1`.
Raw receipt SHA `9b7d1df4c4b4c7018c937f5075af20b2729865a187ab2232372501965a8260bb`; cold readback SHA `8f116f5bda4d2df9a90dbd79953365ea153f87111a3802cd59ec5d32faf5b2a7`.

## Bound source and finite cells

Original refined-03 four-request source, target index2 train351017; untied step2444 FP32/SDPA. Original image SHA `8bb2f91a7a1672a4aacd41092dc9480dbb86c50ac32ba2566f7e5e622e6db1c9`, raw SHA `77a65cc8dc35220ec58766cca92b9caa9225eb4ce99c0330e340d40b6cfd8cbf`, trace SHA `1b75265af561e712e74a74f455ae484dde17468fb5e76367b5cb32b6838295d6`, source input identity `94d4ee001a7c36a4a598b96a5ace0cf1c04718c83025ee5aea18906c0c856150`. AF/FF differ only in older raw5/6/7. Latest F physical `[1371,1380)`, original four-request images/companions, attention and three-axis positions were held fixed. Only target older physical `[1362,1371)` K and/or V were edited; no rephasing.

The exact call ledger is 1–2 full_AF/full_FF; 3–4 prefill_AF/prefill_FF; 5–6 anchor_AF/anchor_FF; 7–8 sham_AF/sham_FF; 9–10 joint_AF_from_FF/joint_FF_from_AF; 11–12 V_AF_from_FF/V_FF_from_AF; 13–14 K_AF_from_FF/K_FF_from_AF. Every call executed once, in order. The CPU preflight SHA `5b3aa619c67621fadc10936f893daaade6517402f9bd4a557bd813ffec6dd6dd` records 62 actual patch/caller/serialized-reader checks, 30 direct source captures and the exact command/shape forecast before model load.

## Frozen full-vocabulary endpoint

Shared outcome **mixed_or_neither**. FP64 full152670-way softmax; both AF and FF bases retained.

| Base | D native→joint | TV V→joint | TV K→joint | r_V | r_K | Category |
|---|---:|---:|---:|---:|---:|---|
| AF | 0.78242855438 | 0.113238082828 | 0.19996933193 | 0.144726419038 | 0.255575197007 | selective_V |
| FF | 0.530136028192 | 0.517847945321 | 0.523225997962 | 0.976820887059 | 0.986965552495 | neither |

The shared **V-selectivity primary fails** and the symmetric **K-selectivity comparator fails**. In AF, V and K are individually below 0.5, but V wins the selective margin: `r_K−r_V=0.110848778`, beyond the 0.1 rule and 1e-6 guard. In FF, neither component is below 0.5; V-only and K-only remain near the native FF distribution (TV to native `0.0557316` and `0.0731771`) while their joint swap has `D=0.530136`. Thus FF needs the combined edit for this endpoint under its fixed latest context. The result does not support a shared single-axis sufficiency law. Independent NumPy FP64 reductions of the saved raw vectors reproduced both D and all four ratios within `6.7e-16`.

| Base/arm | Winner / runner | z(151671)−z(151670) |
|---|---|---:|
| AF anchor / joint / V / K | 151670/151673 · 151670/151671 · 151670/151671 · 151670/151671 | −1.332893 · −0.077579 · −0.257324 · −0.083086 |
| FF anchor / joint / V / K | 151671/151670 · 151670/151671 · 151671/151670 · 151671/151670 | +0.302612 · −1.001644 · +0.522860 · +0.586435 |

These categorical values are descriptive. The full-vocabulary TV criterion decides the primary and comparator; token probability/rank details are in readback.

The header `[151646,8987,151647,151648]` was replayed conditioning; zero tokens were generated. This fixed-cache-origin contrast does not identify a physical owner, a literal coordinate copy, or a natural mediation percentage.

## Qualification and resources

Calls1–10 qualified before V/K-only readouts. Fresh full references, cached anchors, independent older K+V shams and both complete older swaps passed the frozen all-four 2e-4 gates, including complementary accepted hybrid references. Actual all28-layer older K/V donor, untouched latest/prehistory/companion cache, native mask/positions, companion suffix, and finally restoration were checked and cold-read separately.

Both full references matched their accepted saved all-four vectors at max error `0`; AF also matched original target and companion source trace while FF used the accepted counterfactual target reference plus original companions. Cached/full and accepted-anchor max errors were `5.340576171875e-5` (AF) and `5.7697296142578125e-5` (FF). Both explicit sham/anchor errors were `0`. Both complete older swaps matched the accepted complementary latest-swap **all-four** vectors at error `0`, with identical suffix input JSON; companion logits and suffix K/V remained equal to the same-base anchor for all component cells. Fresh prefill origin receipt SHA `68a19533fbc49e2bce2b93aae15031cabfcb13fbf76267e6b40db5ec4d06f248`; saved all28-layer donor blocks SHA `7f4e495d5bbf43d7d5f8f7eeb92d3345ba6e2d764a23ee55c66ad7f4969e345a`.

Actual calls 14 model / 4 vision / 0 generated. Parent outer 116.233682826 s; internal 110.520330347 s. Prior sequence 0.560139886111397 GPUh; new cumulative 0.592427020230 GPUh.
Peak RSS 11093772 KiB; GPU allocated/reserved 12476571136/13254000640 bytes; artifact bytes before receipt 45066587. Terminal child PID 2150559, exit 0.

Parent outer receipt SHA `5e60b637e7c0a7c63b49eccb69a8a66f0c2fb921d01386cc7b58d0f1d6855614` includes setup and the terminal child. Exact commands: `python -B -m probes.training_set_completion.recurrence_older_kv_split.run preflight`, then `run`, then separate-process `readback`. Raw root `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-kv-split/attempt-001` contains 45,230,791 bytes after cold readback. One terminal GPU job, no failed job or retry.

All individual raw vectors, source inputs, consumer hashes, component-to-native distances and fixed-token probabilities/ranks are indexed in the cold [readback](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-older-kv-split/attempt-001/readback.json).

F physical owner remains HOLD. This is not self-accepted and admits no successor.
