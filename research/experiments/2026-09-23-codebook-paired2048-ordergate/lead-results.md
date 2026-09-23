# Corrected order-gate rerun accepted: partial geometry progress

2026-09-23. Lead independently accepts the corrected-objective evidence. Disposition: partial geometry-validity progress, no overall detection/recurrence rescue and no promotion. This is not the invalidated mean-hinge experiment. Its prior fitted outputs are not used as corrected-arm evidence; only the exact unchanged source baseline is reused.

| val200 | Source | Corrected early472 | Corrected late472 |
|---|---:|---:|---:|
| AP |0.462946|0.448846|0.446772|
| AP50 |0.623166|0.606585|0.612554|
| AP75 |0.497391|0.481639|0.477867|
| AR100 |0.514020|0.498796|0.494708|
| Pre-drop invalid geometry |567|40|332|
| Equality-invalid |559|14|308|
| Reversal-invalid |8|26|24|
| Invalid-geometry images |5|6|8|
| Malformed spans |238|247|2|
| Exact-row revisits |269|401|401|
| Annotation-owner revisits |25|27|51|

Early invalid-span count falls527/567=92.95%, primarily equality errors. This is not broad elimination: early32/40 invalids and246/247 malformed spans concentrate on capped image18380; late316/332 invalids concentrate on capped image7511. Invalid-image incidence and reversal counts are higher than source. Both arms have lower AP/AP50/AP75/AR100 and known-positive coverage; clean known-positive images remain73/200 each. Reduced geometry-invalid spans do not establish owner recovery or reduced recurrence. UNKNOWN stays annotation-unmatched, not physically false. Official post-parser accepted-invalid counters(source3,trained0) are distinct from the pre-drop counts above.

The intended conditional objective is implemented and exercised: CE1/typegate0.2/conditional_order_gate0.2, old mean hinge/Gaussian0. It penalizes illegal coordinate mass using the actual teacher-prefix x1/y1 token, including equality. There is no generation mask or coordinate correction. This is the correct token-level surrogate for the user's stated direction, but a finite-weight teacher-forced penalty cannot guarantee valid greedy output on generated prefixes. The observations are consistent with improved equality behavior and residual rollout failures; they do not isolate loss causality, injection timing, or establish a duplication-burst mechanism. Neither a universal rejection nor a claim that raising the weight will solve remaining errors follows.

Both arms used fixed2048train/four epochs/472calls,3776packs/8192image presentations, seed1729 and matched common recipe; no endpoint selection. All selected training identities occur in mature-source SFT population; exact earlier exposure unknown. No train-native completion result or convergence evidence exists. The absence of a same-data no-injection order-gate fit limits component attribution. The next research choice remains with the user; no new loss, training arm, dose or successor is launched by this closure.

## Independent verification

Root `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-23-codebook-paired2048-ordergate`.
Candidate SHA256 `6c3da77d740790990e9f737d14d38c870c4e9b6bccd9f3314d5938bddb0b6e08`.
Lead receipt `lead-verification-v1.json`.

Lead rehashed132 candidate bindings and14 payload-readback input bindings, reran both new official evaluations into fresh lead-evaluation directories, and exactly reproduced AP/AP50/AP75/AR100 and row/object/drop counters. Source official metrics were previously independently replayed and exact source artifacts remain hash-bound. Lead directly recounted geometry/equality/reversal from all600 parse-diagnostic rows and verified matched200 IDs/prompt,3084/RP1.0 and benchmark eligibility. No model calls were made. Initial one-off binding-reader schema assumptions failed before verification; corrected dictionary/hash-string scans passed, with no producer changes.

Technical source/mask/caller smoke is [lead accepted](lead-smoke-acceptance.md). Saved payload mode/finiteness and official adapter/selected-delta composition remain bound. Separately persisted post-load codebook tensor hashes and full-logit reload parity were not measured; do not relabel the exercised loader or saved tensors as those measurements. Ten focused gate tests remain accepted; no full-suite claim. The worker's late AR100 omission in its message is resolved by the official receipt:0.4947081259061161.

Recomputed allocated21608.313884GPU-seconds(6.002309GPUh),4057.989163model-wall seconds. All8 producer intervals terminal exit0,16 recorded PIDs absent,no owned GPU overlap. Both fits ended before new inference; old source reuse is not new computation. Prior invalid group remains a305-file verified archive. Research knowledge/layout checks and diff verification apply to closure separately from scientific acceptance. No automatic promotion,successor,commit or publication.

See [candidate-results.md](candidate-results.md) for per-image details and all remaining evidence boundaries.
