# Coordinate-margin provenance — candidate

Eight frozen states passed the forward and CPU gates. Evidence remains **unreviewed** pending root acceptance. No new free continuation was generated.

| State | Raw winner | Equal-norm winner | Raw 0−1 | Equal 0−1 | Raw 0−30 | Equal 0−30 |
|---|---:|---:|---:|---:|---:|---:|
| tied-original-offset5 | 0 | 0 | 0.740131 | 0.423821 | 1.171003 | 0.228645 |
| tied-original-offset15 | 0 | 0 | 0.747665 | 0.428844 | 1.294996 | 0.346713 |
| tied-pair01-offset15 | 0 | 23 | 0.703079 | 0.399274 | 0.332750 | -0.582937 |
| tied-pair01-offset25 | 52 | 52 | 0.491249 | 0.250381 | -0.566984 | -1.307429 |
| untied-original-offset5 | 0 | 23 | 0.659433 | 0.326079 | 0.937107 | -0.072328 |
| untied-original-offset15 | 0 | 13 | 0.695526 | 0.360506 | 1.074011 | 0.063694 |
| untied-pair01-offset15 | 23 | 23 | 0.623013 | 0.302235 | 0.131033 | -0.852564 |
| untied-only0-offset15 | 46 | 46 | 0.402422 | 0.131603 | -1.742683 | -2.608937 |

## What the accounting establishes

At tied pair01 offset15, the raw 0−30 margin is +0.332750. Its symmetric direction term is −0.589133 and row-length term +0.921884: length reverses that selected-pair ordering. Equal norms give −0.582937, but the global equal-norm winner is 23, not 30. The 0−1 margin remains positive after equalization in all eight states. These are distinct competitions.

For 0−1, the final MLP (zero-based layer27) is the largest positive component in all eight states; its contribution ranges approximately +0.4315 to +0.6734. Other attention/MLP contributions oppose or reinforce it. This does not establish that layer27 originates or causes recurrence.

For tied 0−30, native offset15 has attention sum +0.64978 and MLP sum +0.64548; pair01 offset15 has +0.51103 and −0.17804. The margin change is distributed, with several late MLP updates changing sign or magnitude. At tied pair01 offset25 the raw margin becomes negative and raw/equal winner is52. Untied pair01/only0 offset15 select23/46 while 0−1 still favors0, showing why a selected pair is not a full-winner explanation.

The strongest surviving account is ordinary history-dependent distributed computation combined with static readout-length advantages. This accounting supplies candidate components, not a unique faulty circuit, training-origin explanation or owner ledger. No patch or successor is authorized here.

## Exact convention and evidence

Let n_i=||W_i|| and q_i=(W_i/n_i)·h. Raw margin is decomposed as ((n0+nj)/2)(q0−qj)+((n0−nj)/2)(q0+qj). Equal-norm scores use the accepted lower median coordinate norm. There is no centering or bias. Effective W includes base plus the actual output delta. Each residual contribution is projected through that state’s actual final RMS scale and gain; cumulative values are accounting views, not logit-lens predictions. FP32 addition, normalization and head-rounding residuals are retained separately. Target inter-layer extra contribution is zero at every measured seam.

Maximum residual-vector reconstruction error: 0.000170031562; maximum selected-pair error: 3.42883831e-06. Both are below frozen2e-4. All source head/coordinate comparisons are exact; hooks preserve full-vocabulary scores bitwise. Independent saved-tensor reconstruction and +0.01-margin corruption sensitivity pass. No model reruns or failed model attempts.

Cost: 236 batch forwards,16 prefix replays, 346.514 allocated GPU-seconds across GPUs0–7; 180939744 retained tensor bytes. All8 producers exited0; no live child or owned job.

The supplied prefixes retain native heterogeneous companions, image/masks and incremental positions. Companion outputs are not a scientific outcome. No attention/KV archive, gradient, physical review or new generation was acquired.

Artifacts: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-coordinate-margin-provenance/ARTIFACTS.md`; result, reconstruction, verification, costs and closure are under the same root. Exact checkpoint/config/input identities are in every `runtime/<state>/receipt.json` and its bound native receipt.
