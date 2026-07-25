# Prefix-local long training and matched final-horizon evaluation

The four authorized 1,440-event runs completed all 90 optimizer updates and
retained ten milestone checkpoints. Every recorded update is finite and
applied. The owning result is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-24-prefix-local-and-on-policy-owner-set-training/results.md`

The 40-checkpoint, 512-new-token panel was useful as an expansion-pressure
diagnostic. Complete-action pairwise training at learning rate `1e-5` grew from
two length stops at step 9 to thirteen at step 90, and grouped
owner-conditioned training grew from three to twelve. The lower-rate pairwise
arm stayed between one and four, while the first-divergence transition control
retained Source's single short-horizon stop. A 512 stop is not itself a final
failure: Source's one capped boat sequence naturally closes before 3084.

Final-horizon comparisons must freeze batch policy as well as prompt, decode,
and token horizon. A same-checkpoint real replay showed that batch size 16 and
batch size 4 produced different raw greedy text on 57 of 64 images even though
their aggregate stop counts matched. The first table, which compared a batch-16
Source against batch-4 treatments, is rejected as conclusion evidence. Its
artifact root is retained only as provenance:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/owner-comparisons-64-long-promotion-max3084-b4-v2/`

The conclusion-bearing table uses batch size 4 for both Source and treatment,
greedy decoding, repetition penalty `1.0`, `max_new_tokens=3084`, the original
one-prompt/one-completion task, and the canonical row schema. Source has 389
matched owners and 64 natural stops. Only first-divergence transition step 36
has zero treatment length stops. It gains 21 Source-missed owners, loses 18
Source owners, and is therefore `+3` net owners, with 155 fewer predictions and
14 fewer strict duplicate candidates. All other promoted or final checkpoints
have one to ten length stops. The largest apparent gains come from the
complete-action and grouped final checkpoints, which add roughly 2,450
predictions and have nine or ten stops; do not promote them as set expansion.

The authoritative matched comparison is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/owner-comparisons-64-long-promotion-max3084-matched-b4-v3/`

The HF batch-16 failures were a separate runtime defect:
`compute_transition_scores` stacked every generation step before fp32
log-softmax, causing a 16.23-GiB transient allocation on a 1784-step batch.
Policy likelihood extraction now calls the same Transformers method in
32-step chunks. A focused regression and all 12 current `HFBackendSession`
tests pass. The original 3084-step real failure replay completes with no
artifact failure and peak reserved CUDA memory of 69,128,421,376 bytes. This
fix does not make batch sizes behaviorally interchangeable. The legacy
`tests/inference/test_backend_trace.py` remains stale against removed backend
interfaces and is outside this bounded correction.

The result is training-panel evidence only. It does not establish development
or held-out transfer, objective superiority, a final architecture, or genuine
on-policy utility. The user previously required a discussion after this round.
No development, held-out, full-pool, or successor-direction run has been
launched. The strongest continuation candidate is transition step 36, and the
next decision is whether its small bounded gain warrants matched Source versus
treatment evaluation on development and held-out images.
