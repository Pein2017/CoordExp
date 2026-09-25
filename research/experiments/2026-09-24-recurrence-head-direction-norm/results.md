# Head direction versus norm at the fixed x2 endpoint

Lead accepted and closed, 2026-09-24. All six cells technically qualify. The frozen direction-following pair and reverse norm-following pair both **NONPASS**, category `mixed_or_other`. [Acceptance](lead-acceptance-v1.json), [independent CPU verification](supporting/lead-attempt-002-verification-v1.json), and [worker candidate](candidate-attempt-002-complete-results.md) retain the evidence.

Original refined-03 four-request batch, target2 train351017, one A history and the same replayed six-token current prefix. At current y1 query1376 predicting x2, every layer/head computes native N=H+R and remaining contribution R from its own incoming Q/K/V. G=(||R||/||N||)N preserves that local native direction at R's norm; D=(||N||/||R||)R preserves the remaining direction at N's norm. The write is the selected **pre-o_proj input after the actuator**, not an o_proj output. Attention, other positions and companions remain native.

B=TV(N,R)=0.964094437012, full-vocabulary FP64 probabilities.

| Control | TV to N / B | TV to R / B | Frozen interpretation |
|---|---:|---:|---|
| G, local native direction / R norm | 0.929776197476 | 0.937110707594 | Far from both anchors; fails either paired prediction's G clause |
| D, local R direction / native norm | 0.960140751482 | 0.390278974408 | Meets its direction-following half, insufficient for the paired primary |

TV(G,D)=0.853774636417. G's large movement shows that local gain changes can strongly alter the endpoint; it does not reproduce removal. D remains relatively closer to removal after local norm restoration. These observations support neither a shared direction-only signature nor a shared norm-only signature. They do not by themselves identify an interaction mechanism: the norms and directions are recomputed on each arm's adaptive trajectory, not crossed from a single frozen activation pair. Perturbation sizes and post-projection/residual norms are not matched.

Fresh N/R references pass, and independent N/R write identities qualify before the controls. Lead rehashed source/consumer evidence, reconstructed all six inputs and 168 layer transforms with the actual CPU reader, and independently recomputed attention partitions, per-head norms/directions and full-vocabulary TVs in NumPy. The largest reference error is 5.4121e-5 under2e-4; frozen numerical boundaries are clear. The candidate's “post-o_proj” wording is corrected above; actual source and consumers implement the admitted pre-projection intervention.

The prior historical-contribution-selective result remains accepted. This closes its simple direction-versus-norm follow-up; it does not identify copied content, a unique key/circuit, natural recurrence or a physical F/F2 owner. No further gain, layer or head search is admitted. The broader explanation must next earn a prediction about complete localization or independent normal-versus-repeat behavior, rather than another fixed-endpoint sensitivity result.

Six model / six vision forwards, zero generation and reuse. All children terminal. New fixture/model full enclosing charges24.082/59.535s; all prior failures remain charged. Final sequence0.9602669383484082 GPUh. The [final ledger](supporting/attempt-002-final-ledger-v1.json) corrects the immutable cold readback's omitted0.000061877s shell overhead. Raw tree1,996,777,128bytes, under3GiB. Failed fixture/shell records and predecessor bytes remain preserved.
