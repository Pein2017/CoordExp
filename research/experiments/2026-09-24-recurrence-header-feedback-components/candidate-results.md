# Current-header Q versus K/V candidate

Status: **candidate cold readback passed; lead acceptance pending.**

Protocol SHA `1f291c4907e9c0139bbed3dc08d2b55193d5edeea4829e17f9736ee01b15e8a3`; admission SHA `5c98685e3c5e945f81bd28c0b2a7a347d9741f8c1f58c0cd000ad3a299e62300`; producer SHA `b0037cda758e6dc6c3d7549c9526fb366866a786af7bad345587c9215ed82690`.
Receipt SHA `59bfc514c7324b0078cb49e18a27da7f5abe22eaec9643f14e94e4a1572fe109`; cold readback SHA `f18af52cfaa453b55bd7b49e772d478aeb9735eafe319a994abdaacd287e2acc`. Original refined-03 four requests, target2 train351017; AF/FF common header replayed, zero generated tokens.

## Frozen FP64 full-vocabulary decision

Shared category **mixed_or_changed**. Primary both-base query-adaptation signature **False**; symmetric K/V comparator **False**; both-collapse **False**; both-retain **False**.

| Base | Adaptive D | t_Q | e_Q | t_KV | e_KV | Category |
|---|---:|---:|---:|---:|---:|---|
| AF | 0.78242855438 | 0.0936889679526 | 0.939210702663 | 0.898449711126 | 0.202375815464 | query_adaptation |
| FF | 0.530136028192 | 0.115322497468 | 0.98652472573 | 0.142614823353 | 0.923721147478 | both_collapse |

All raw TVs and partial-joint TVs to the same-base all-clamped joint, full-vocabulary winners/runners/gaps, and fixed token probabilities/ranks are in the cold readback. Categorical winners do not affect the criterion.

## Technical qualification and cost

Calls1–18 passed fresh adaptive, whole-clamp bridge and separately executed partial-native identities before all four fixed treatments. Both own-base native Q/K/V captures exactly reproduced the accepted tensors at all28 layers and four positions. All22 raw vectors/inputs, selected older K/V donors, base latest/prehistory/companion complements, native masks/rotary, actual pre- and post-actuation Q/K/V, selected/free axes, appended K/V, scale-aware FP64 pre-o_proj reconstruction and finally restoration were checked. Cold readback recomputed saved-headout arithmetic and full-vocabulary TVs in a separate CPU process; it was not a second full SDPA replay.

Actual 22 model / 4 vision / 0 generated. Parent outer 187.555419259 s; internal 181.735511832 s. Failed attempt001 13 model/4 vision/0 generated, outer 109.010166332 s remains charged. Combined attempts 35 model/8 vision/0 generated, outer 296.565585591 s. Prior sequence before attempt002 0.674452112925 GPUh; charged cumulative 0.726550840497 GPUh.
Peak RSS 11278560 KiB; GPU allocated/reserved 12476571136/13268680704 B; artifact bytes before receipt 120027282; terminal child PID 2220584, exit 0.

Raw attempt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-24-recurrence-header-feedback-components/attempt-002`. Technical bounds and scientific thresholds remain frozen. F/F2 physical identity stays HOLD. The intervention tests within-forward adaptation; it does not establish a natural recurrence loop or mediation fraction. No self-acceptance or successor.
