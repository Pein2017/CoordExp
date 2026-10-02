# Full-label self-rollout: accepted 16-update observation

The fresh constant-rate trajectory is technically complete and lead-accepted. It does not achieve full-label fitting or stable preservation: endpoint TP increases247→261, while F1 decreases0.514048→0.466488, annotation FP doubles144→288 and valid predictions increase391→549. Of247 baseline matches,215 remain,46 other labels are gained and32 baseline labels are lost. Missing-label proxies remain deferred.

This is one18-image/570-annotation trajectory from the named step256 anchor, not pooled with the separate2-update qualification. All17 raw versions and checkpoints are retained; no checkpoint selection, quality stop or rollback occurred. Annotation-unmatched predictions are not verified physical false positives.

## Complete descriptive trajectory

| Update | TP | FP annotation | FN | Valid rows | F1 | Invalid | Valid repeats | Malformed | Caps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 247 | 144 | 323 | 391 | 0.514048 | 5 | 10 | 0 | 0 |
| 1 | 230 | 155 | 340 | 385 | 0.481675 | 1 | 29 | 130 | 0 |
| 2 | 240 | 109 | 330 | 349 | 0.522307 | 1 | 0 | 0 | 0 |
| 3 | 244 | 107 | 326 | 351 | 0.529859 | 2 | 0 | 0 | 0 |
| 4 | 240 | 150 | 330 | 390 | 0.500000 | 11 | 8 | 0 | 0 |
| 5 | 246 | 161 | 324 | 407 | 0.503582 | 9 | 19 | 0 | 0 |
| 6 | 244 | 121 | 326 | 365 | 0.521925 | 5 | 1 | 0 | 0 |
| 7 | 256 | 114 | 314 | 370 | 0.544681 | 2 | 1 | 0 | 0 |
| 8 | 258 | 116 | 312 | 374 | 0.546610 | 1 | 0 | 0 | 0 |
| 9 | 256 | 165 | 314 | 421 | 0.516650 | 6 | 21 | 0 | 0 |
| 10 | 258 | 146 | 312 | 404 | 0.529774 | 5 | 1 | 0 | 0 |
| 11 | 266 | 251 | 304 | 517 | 0.489420 | 206 | 41 | 1 | 1 |
| 12 | 251 | 265 | 319 | 516 | 0.462247 | 47 | 98 | 1 | 0 |
| 13 | 251 | 306 | 319 | 557 | 0.445430 | 47 | 110 | 1 | 0 |
| 14 | 250 | 334 | 320 | 584 | 0.433276 | 23 | 124 | 1 | 0 |
| 15 | 262 | 302 | 308 | 564 | 0.462081 | 55 | 60 | 0 | 0 |
| 16 | 261 | 288 | 309 | 549 | 0.466488 | 0 | 30 | 0 | 0 |

[Trajectory figure](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/figures/fresh16-trajectory.png) · [CSV](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/figures/fresh16-trajectory.csv)

Version1 has130 malformed rows; version11 has206 invalid rows,242 complete literal repeats (41 valid) and one capped output. The fixed endpoint returns to zero invalid/malformed rows but retains30 valid repeats. The image4134/annotation294005 tie is correct in versions0–5 and lost in6–16. The full per-image baseline/adjacent/recovered-ID ledgers remain in the machine report.

## Interpretation and optimization decision

Observation: training can add labeled owners, but gains exchange with previously covered owners and growing output debt. This16-update result is not convergence evidence or a proof of representational impossibility. The intervention changes full-label supply and coordinate objective together, so it does not isolate ranking efficacy.

Potential explanations include excessive early updates after resetting AdamW moments, later step-size effects, changing self-generated prefixes, and competing objective directions. Bursts alone do not identify an optimizer cause. Gradient magnitudes are not learning-rate prescriptions: in the completed2-update qualification, input/output delta gradient L2 is4.225/1.019 at step1, but actual update RMS is4.511e-6/4.789e-6. Both are independent1004×2048 FP32 additive deltas; these ratios are not changes to total pretrained embeddings. Language DoRA plus these two deltas are the only trainable surfaces. Vision/projector/base weights remain frozen.

[Saved-update evidence](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/cpu-01/lr-observed-updates-01.json). AdamW is fresh, with fixed1e-5 DoRA /5e-6 deltas, clip1, betas(.9,.999),eps1e-8,WD0. The16 recorded total preclip norms are all greater than1. Relative A/B/magnitude coordinate movement is parameterization-dependent; functional/logit sensitivity has not been isolated.

The next lead-selected comparison changes only the learning-rate time profile; its exact contract is in unit.md and state.json. Separate fresh16 runs compare constant1, linear warmup4 then constant1, and constant0.90625 (the same nominal learning-rate sum as warmup4). Input/output delta rates remain equal; other loss/data/decode/model settings remain fixed. This separates a time-profile comparison from gross nominal-dose reduction, without pretending equal summed rates imply equal AdamW updates or matched generated histories. One run per arm is a bounded diagnostic; any apparent advantage remains a candidate for replication. No GPU launch for this follow-up is yet released.

## Technical acceptance and provenance

Execution commit: `a00cd21550f6824c6bd4edb94d497af67d59202c`; recipe `a2b982a8526ca53efa2ed68ac04bd1fda8df81937122b3bbdb804313b56d8141`; full-label snapshot `1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792`. The lead verified24 report artifact hashes,22 terminal artifact hashes, all17 export payload identities,128 rank-update receipts, eight identity-aligned vLLM generate/refresh sequences,306 frozen requests and all720 forward receipts.28 exact owner/descendant identities were absent. A separate read-only raw recomputation reproduced the complete offline JSON, all570 denominator IDs, per-image metrics and owner transitions.

[Observation report](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/native-observation-16-worker-report.json), SHA256 `e15c03af0cbaae8992b930ba0d3602d3f112b17d59b02a442965d8a4697af4fa`.
[Lead acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/native-observation-16-owner/lead-terminal-acceptance-01.json), SHA256 `4b5c7b3b78e6625a2245c160ef5a874a22ee531ccb0948960d3757f1da76c0e4`.
[Separate qualification acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/native-qualification-02-owner/lead-terminal-acceptance-01.json).

Observed work:75376 generated tokens,720 HF forwards,1232571 HF context tokens,710182 visual tokens. Maximum context4764 and selected positions953; selected FP32-logit payload581978040 bytes. Operation-recorded peak allocated memory7779111424 bytes does not measure whole-device/training VRAM; sampled aggregate descendant RSS65457328 KiB.

External wall871.8426322713494 seconds is the charge authority, giving1.93742807171411 GPU-hours; this unit including qualification uses2.351383263286617. Earlier cumulative usage plus this unit is4.54863659661995, with no open reservations or live native job. At2026-10-02T09:20:46Z the user removed the cumulative GPU-hour ceiling; costs are informational and do not stop the research. The completed invocations retained their frozen operational bounds.
