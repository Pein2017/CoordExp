# Full-label self-rollout: accepted finite observations

## Completed learning-rate-profile comparison

All three independent fresh 16-update runs are technically complete and lead-accepted. None achieves full-label fitting or stable preservation on these 18 images / 570 annotations. Each independently measured baseline is TP 247, FN 323, valid rows 391 and F1=0.514048. All 17 versions per arm remain evidence; no quality stop, rollback, checkpoint selection or extension occurred.

| Profile | Endpoint TP | FN | F1 | Valid rows | Gained vs baseline | Lost vs baseline | Invalid | Complete repeats | Valid repeats | Malformed | Caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 256 | 314 | 0.392939 | 733 | 54 | 45 | 548 | 708 | 176 | 3 | 2 |
| warmup4 | 258 | 312 | 0.416465 | 669 | 60 | 49 | 244 | 321 | 86 | 1 | 1 |
| constant_dose | 266 | 304 | 0.398801 | 764 | 56 | 37 | 397 | 506 | 122 | 2 | 2 |

[All-version trajectory figure](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/optimizer-lr-shape-01/figures/lr-profile-trajectories.png) · [All 51 observations as CSV](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/optimizer-lr-shape-01/figures/lr-profile-trajectories.csv). F1 uses all strict-valid rows, including duplicates; invalid/malformed rows are separate burdens. Annotation-unmatched predictions are not verified physical false positives.

Warmup4 has lower endpoint repetition/invalid burden and higher F1 than the two current constant profiles, but loses 49 baseline owners versus 45 and 37. Constant_dose has the highest endpoint TP 266 and fewest baseline losses 37, while still having 397 invalid rows and 506 complete repeats. All profiles have late bursts; warmup's maximum invalid count is 483, despite its lower endpoint 244. This is a tradeoff across single trajectories, not a stable or replicated scheduler winner. The earlier nominally same-rate constant observation below ended at TP 261 / F1=0.466488 / lost 32; it remains separate evidence and is not pooled with this repeat.

Most endpoint invalid rows are exact repeats: constant: 532/548; warmup4: 235/244; constant_dose: 384/397. The latter 397 rows comprise 13 distinct invalid category-box identities across three images. This overlap is descriptive: the counts must not be read as hundreds of independent geometry failures, nor as proof that incomplete duplicate supervision caused the bursts. All three endpoints lose the image 4134 / annotation 294005 tie. The full per-image and baseline/adjacent/recovered-ID ledgers remain in the reports.

The observed actual LR factors are constant 1; warmup .25,.5,.75,1 then 1; and constant .90625. Their nominal sums are 16/14.5/14.5 for both parameter groups. Recipes were independently checked equal after removing lr_profile; equal nominal sums do not imply equal adaptive updates or generated histories. These runs do not test a large LR reduction or layer freezing. A proposed next discriminator is constant 0.1 of both LR groups with all layers/objectives fixed; it must show learning as well as stability. Per-occurrence duplicate supervision is a separate proposed factor, preserving the first trusted positive and actual later prefixes. No such new native run, loss change, sampling change or missing-label proxy has been released.

## Profile technical acceptance and accounting

Executed source: `da504d0c0dfb26e4527bb0d953187577455de0f4`. The lead verified each of 336 report artifact hashes and 22 terminal hashes per arm, all 51 export identities/raw versions, 384 actual rank-update receipts, eight generate/refresh sequences per arm and 2187 finite forward receipts. Independent maintained recomputation exactly reproduced all three full offline artifacts after JSON key normalization, all 570 denominator IDs, all per-image metrics, owner transitions and endpoint raw-to-credit observations. Nine run/readback/offline stages exited 0, invocations were strictly sequential and all 84 exact owner/descendant identities were absent. No operational failures or retries occurred.

[Combined worker report](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/optimizer-lr-shape-01/lr-sequence-worker-report-01.json), SHA256 `fabd559b035b68623801e115c2768c5202cfd5239d20931a850b73a92bdedc30`. [Lead sequence acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/optimizer-lr-shape-01/lead-sequence-acceptance-01.json), SHA256 `3394e0de360ba83a0a12bd00f0dd8b30e99009208212ad0d28206a2f4f82c95c`. Per-arm runtime, metric and invalid-repeat proofs are linked from the lead acceptance.

The three external invocations used 3343.9020067602396 seconds at eight charged slots, or 7.430893348356088 GPU-hours. Completed cumulative usage is 11.979529944976038 GPU-hours; usage is informational under the user's removed cumulative cap. No reservation or live native job remains. Recorded VRAM peaks cover vLLM operations only; RSS is sampled descendant aggregate, and selected-logit FP32 sizes are shape-derived envelopes, not measured allocations. HF score witnesses are local to checked original prefixes and do not certify native argmax parity.

---

## Initial fresh16 observation (historical)

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

The subsequent lead-selected comparison changed only the learning-rate time profile; its completed contract is in unit.md and state.json. Separate fresh16 runs compare constant1, linear warmup4 then constant1, and constant0.90625 (the same nominal learning-rate sum as warmup4). Input/output delta rates remain equal; other loss/data/decode/model settings remain fixed. This separates a time-profile comparison from gross nominal-dose reduction, without pretending equal summed rates imply equal AdamW updates or matched generated histories. One run per arm is a bounded diagnostic; any apparent advantage remains a candidate for replication. The exact three-arm follow-up is now complete and accepted above.

## Technical acceptance and provenance

Execution commit: `a00cd21550f6824c6bd4edb94d497af67d59202c`; recipe `a2b982a8526ca53efa2ed68ac04bd1fda8df81937122b3bbdb804313b56d8141`; full-label snapshot `1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792`. The lead verified24 report artifact hashes,22 terminal artifact hashes, all17 export payload identities,128 rank-update receipts, eight identity-aligned vLLM generate/refresh sequences,306 frozen requests and all720 forward receipts.28 exact owner/descendant identities were absent. A separate read-only raw recomputation reproduced the complete offline JSON, all570 denominator IDs, per-image metrics and owner transitions.

[Observation report](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/native-observation-16-worker-report.json), SHA256 `e15c03af0cbaae8992b930ba0d3602d3f112b17d59b02a442965d8a4697af4fa`.
[Lead acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/native-observation-16-owner/lead-terminal-acceptance-01.json), SHA256 `4b5c7b3b78e6625a2245c160ef5a874a22ee531ccb0948960d3757f1da76c0e4`.
[Separate qualification acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/native-qualification-02-owner/lead-terminal-acceptance-01.json).

Observed work:75376 generated tokens,720 HF forwards,1232571 HF context tokens,710182 visual tokens. Maximum context4764 and selected positions953; selected FP32-logit payload581978040 bytes. Operation-recorded peak allocated memory7779111424 bytes does not measure whole-device/training VRAM; sampled aggregate descendant RSS65457328 KiB.

External wall871.8426322713494 seconds is the charge authority, giving1.93742807171411 GPU-hours; this unit including qualification uses2.351383263286617. Earlier cumulative usage plus this unit is4.54863659661995, with no open reservations or live native job. At2026-10-02T09:20:46Z the user removed the cumulative GPU-hour ceiling; costs are informational and do not stop the research. The completed invocations retained their frozen operational bounds.

## Existing-error supervision coverage — read-only diagnosis

The user asked whether persistent invalid/repeated outputs require low-temperature multi-rollout training. The existing run produces one fresh greedy trajectory per image per update, trains its current plans once, and refreshes; old error prefixes are not pooled for replay. The endpoint16 plans have no following update. Thus the evidence is not repeated fitting of each error until removal.

The lead recomputed all 17 saved credit-plan sets. Version11 has242 complete literal repeats but only16 selected redirect positions;226 further occurrences share an already-selected identity. Each image uses the first eligible position per exact description/coordinate identity, with1/K redirect normalization. Near-repeats are measured but not additional negative events. Invalid geometry receives legal-set probability loss and0.1-weighted max-illegal/max-legal softplus at certified erroneous sites. Malformed spans receive direct corrections only where the parser certifies eligible slots; notably all130 version1 malformed rows did produce130 type-error sites, so these were not simply dropped from supervision.

Among144 actually trained redirects (versions0–15),83 first diverge in description and61 in coordinates. The full-label recipe retains the description bad-versus-good softplus but replaces the coordinate pair margin with owner-region ranking. Using the actual annotation GT,22 of those61 negative coordinate tokens remain inside the target's prefix-completable region;39 are outside. This shows absence of explicit exclusion at those22 first divergences, not zero total gradient or a measured causal fraction. Later positive supervision follows the corrected row prefix. A coordinate can permit a valid new-owner completion even when it also begins a fully repeated row; blindly penalizing that coordinate globally would mislabel legal alternatives.

[Reproducible coverage receipt](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/cpu-01/loss-coverage-diagnostic-01.json), SHA256 `2afed9ea4717cc33c2bd0dfe1bd43b2ec10658986f91f650ea84620aa402f73a`, binds the executed source, full annotation file and all136 credit files. The region is computed from the annotation GT by annotation_id, not the sampled positive target box.

Low-temperature sampling may broaden encountered error prefixes, but cannot by itself change event eligibility, normalization or the local coordinate objective. The cheapest next mechanism discriminator is same-error-prefix before/after scoring: unpenalized events, penalized-but-unimproved margins, and repaired-old-prefix/new-prefix errors imply different next actions. A future diversity comparison would retain greedy evaluation and match updates/supervised token exposure, with only certified invalid/repeated actions negative and unknown/unmatched neutral. No sampling or loss change is authorized by this diagnosis; the frozen LR comparison remains separate.
