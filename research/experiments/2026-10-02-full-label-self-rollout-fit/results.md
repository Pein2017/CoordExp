# Full-label self-rollout: accepted finite observations

## Current discussion for the next lead

**Handoff state, 2026-10-03 (Asia/Saigon):** the completed training and inference contrasts reported here are technically accepted; their jobs are terminal and cleaned. Broader research remains user-paused for discussion with a new lead; there is no new launch or loss-change grant. This section is a continuation pointer, not execution authority. The current decision is no selected LR winner and norm OFF retained; full-label fitting remains unachieved.

The scientific target remains fewer true physical false negatives on incomplete or unlabeled data. This 18-image / 570-label training panel is a bounded fitting diagnostic, not held-out generalization or physically exhaustive truth. Its matcher is geometric cardinality-first IoU>=.5 assignment followed by exact-description credit. Preserve separate owner gains/losses, annotation-unmatched, literal-valid/complete repetition, near-repeat pairs, invalidity and stopping outcomes. The original, LR-profile, balanced and fresh-norm-OFF baselines are different observations and must not be pooled.

Three unresolved questions should frame the next discussion:

1. **Preservation versus useful learning.** The balanced lower-LR arm is quieter and retains more baseline IDs but gains fewer new owners. What change would improve acquisition without merely reducing learning or replacing already covered owners?
2. **Coverage and effect of correction at repetition-producing prefixes.** Current rollout refresh visits new prefixes each update; it does not repeatedly fit each old error until corrected. Redirects select the first eligible exact duplicate identity, and owner-region coordinate supervision does not necessarily exclude a repeated coordinate prefix. The [existing-error supervision diagnosis](#existing-error-supervision-coverage--read-only-diagnosis) establishes this computation/coverage gap, not its causal share in late bursts. The cheapest proposed read-only discriminator is whether the first relevant repeated decision was supervised with a target that distinguishes a new owner, before considering gradient effects or more trajectories. This is a discussion candidate, not a released experiment.
3. **Geometry, owner preference and greedy rollout can change differently.** Order gate raises legal coordinate probability mass (weight .01); Gmax pushes the best legal token above the best illegal token at certified errors (weight .1). Neither is an instance-level deduplicator. Owner-region accepts only coordinates admitting a matching-owner completion. These constraints can be jointly satisfied, although order gate can locally oppose owner preference on a legal-but-wrong-owner coordinate. Current evidence does not establish harmful shared-parameter loss conflict. Median norm's differing effects on351017 and7511 reinforce the need to keep symptoms separate.

The [unit](unit.md#computation-and-unchanged-reductions) owns the exact loss/reduction contract; [state](state.json) owns accepted receipts, consumed grants, worker identity and informational costs. Source for the balanced training pair is `b8d4d8cbb0dffbb70adee691c795fb36ffc8ce44`; norm inference is `f731f2de73ab5ae8927456ec0b1b723883028e58`, with separately bound CPU scorer repair `52e6f4c49e96fefa4933f4e6991fceb22466a67b`. Maintained unit code is `probes/full_label_fit`, shared execution is `probes/online_row_credit.py` and `src/qwen`; outputs are evidence, never importable source. Recheck live Git/ownership before editing. Current AGENTS.md owns routing: decision-bearing science/design, model-forward or infrastructure reasoning, and mathematical derivations use Astra at high/max; frozen direct implementation may use Luna-Max or GPT-6.1-Sol. The earlier user-selected 102-worker is an execution worker; its completed grants do not transfer to a new task.

## Same-checkpoint coordinate norm (latest inference contrast)

The user reopened one inference-only comparison after the research-analysis pause: original-LR balanced endpoint16, the same18 inputs/570 labels, and fresh OFF versus median output-coordinate normalization. All8 resident TP1 ranks use identical frozen endpoint16 image groups; prompt/media, weights, BF16 runtime, greedy settings and3084-token caps are fixed. There are exactly36 requests and no optimizer, HF replay, weight update, or native retry. Native source is `f731f2de73ab5ae8927456ec0b1b723883028e58`; the separate CPU scorer repair is `52e6f4c49e96fefa4933f4e6991fceb22466a67b`.

| Metric | Fresh OFF | Median norm |
|---|---:|---:|
| TP / FN | 264 /306 |270 /300 |
| Valid predictions / annotation-unmatched |653 /389 |727 /457 |
| Annotation F1 |0.431725 |0.416345 |
| Complete literal repeats |497 |257 |
| Strict-valid literal repeats |84 |200 |
| Near-repeat occurrence pairs |126 |3278 |
| Geometry-invalid / malformed rows |429 /2 |66 /1 |
| Generated tokens |9923 |7305 |
| EOS / capped requests |16 /2 |17 /1 |

Median retains240 fresh-OFF owners, gains30 and loses24: net+6 TP with substantial turnover. Image351017 supplies233 of the aggregate240 fewer complete repeats (233→0), invalid175→0, valid repeats60→0, and cap→EOS; its TP15→16 contains4 gains and3 losses. Image7511 remains capped: complete repeats264→255, invalid253→65, valid repeats24→198 and near-repeat pairs8→3277, with TP9→11. This shows mixed burden changes, not proof that identical invalid boxes became valid duplicates. Near-repeat pairs are combinatorial occurrence pairs, not independent objects.

**Lead ruling:** technical comparison accepted; retain the existing OFF default. Median helps geometry and length on part of this panel but does not establish a general or owner-preserving improvement. Full-label fitting remains unachieved. This is one fixed-checkpoint policy contrast, not a training intervention, replicated winner, norm-origin causal explanation, or physically exhaustive evaluation. Annotation-unmatched predictions are not verified physical false positives.

The historical endpoint16 is a separate reference (TP265). Fresh OFF differs in tokens on5 images (2299,4134,309264,351017,477415), with1 owner gained and2 lost versus that reference; it is not pooled or substituted as control. Both current conditions miss image4134/annotation294005. Scoring retains class-agnostic cardinality-first IoU>=.5 one-to-one matching followed by exact-description credit, all570 annotation IDs, and per-image gain/loss ledgers.

The native run exited0 and sealed all36 outputs; the original scorer and wrapper exited1 on missing `arm`. A separately tested CPU-only projection adds `arm=greedy` in memory after sealed validation, preserving policy in `producer.condition`; scoring then exits0 on unchanged raw/frozen bytes. The lead reproduced the saved scores/owner ledgers at the JSON consumer boundary, checked all8 active median witnesses and both-condition grouping, and confirmed all67 exact owner/descendant identities absent. Factor range is0.9404875–1.0227903. Existing CPU sensitivity/regression evidence is reused; the native witness proves active coordinate scaling, while noncoordinate invariance is CPU-tested and preserved by the implementation rather than full-backend parity.

The one native owner took162.407836s, charged0.360906303 informational GPU-hours, and generated17228 tokens. The separately disclosed tiny CUDA unit-test exception loaded no Qwen model and added no image request; its unmeasured allocation time is excluded from the native-owner accounting. No model remains live and no further native work is released; the broader user-requested analysis pause continues.

Evidence: [lead acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/coord-norm-01/lead-acceptance-01.json), [paired metrics and all owner IDs](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/coord-norm-01/native-01/metrics.json), [terminal candidate](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/coord-norm-01/native-terminal-candidate-01.json), [contract](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/coord-norm-01/contract.json), [accounting](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/coord-norm-01/coord-norm-accounting-01.json). The cumulative informational native-owner total is16.5546464023 GPU-hours; this unit is14.3573930689, reservation0, no cumulative cap.

## Lower-LR balanced pair

Both independent fresh 16-update runs are technically complete and lead-accepted on source `b8d4d8cbb0dffbb70adee691c795fb36ffc8ce44`. Only LR scale differs: original language DoRA/delta rates 1e-5/5e-6 versus 0.1 times both; warmup0, objectives, modules, full18/570 labels, anchor and greedy decoder remain fixed. Both use `previous_rollout_tokens_lpt_v1`, taking scheduling history only from their own previous rollout. Independently acquired baselines happen to match at TP237 and the same 237 annotation IDs; old unbalanced TP247 trajectories remain separate evidence.

| Fixed endpoint16 | Original LR | 0.1 × LR |
|---|---:|---:|
| TP / FN | 265 / 305 | 241 / 329 |
| Baseline owners retained / lost | 206 / 31 | 215 / 22 |
| New owners vs baseline | 59 | 26 |
| Valid predictions / annotation FP | 648 / 383 | 399 / 158 |
| F1 | 0.435140 | 0.497420 |
| Complete literal repeats (valid subset) | 497 (80) | 20 (20) |
| Geometry-invalid / malformed rows | 436 / 2 | 0 / 31 |
| Capped requests | 2 | 0 |

Observation: lower LR retains 9 more baseline owners but gains 33 fewer new ones and ends with 24 fewer TP. It does learn 26 new owners, so zero learning is false; reduced learning remains a strong alternative to improved stability. No full-label fit, stable owner preservation, replicated winner, or causal mechanism is established. F1 includes duplicate strict-valid rows and does not count invalid/malformed rows; annotation-unmatched is not verified physical FP.

The full trajectories separate symptoms: original-LR TP grows mainly from update8 onward, then complete repeats jump to398/498/497 at updates14/15/16. Lower-LR TP fluctuates231–245 after the baseline, new-owner gains remain20–27, and malformed counts return to127/137/137/130/138 at updates1/3/6/9/11. Lowering LR therefore does not suppress all output failures. Both arms lose the named image4134/annotation294005 baseline owner at endpoint16. These are descriptive single-run observations, without post-hoc checkpoint selection.

Across all17 versions, baseline owners continuously retained are168 at original LR versus185 at lower LR;69 versus52 are lost at least once. Ever-gained nonbaseline owners are99 versus66, with59 versus26 present at endpoint;40 ever-gained owners per arm are absent at endpoint. Adjacent loss events total291 versus301, so the smaller LR does not remove owner churn. These quantities describe label coverage and do not distinguish disappearance from category/geometry threshold changes. Original-LR endpoint497 complete repeats occur entirely in images7511 and351017; those two account for434/436 invalid rows. Lower-LR endpoint31 malformed rows and earlier malformed spikes are all image351017. [Bound owner-trajectory analysis](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/lower-lr-balanced-01/owner-trajectory-analysis-01.json), SHA256 `111c41a2bc262f03007d5ac8612772f54f56e32afcb116b141b6b96cf569226a`. Matching endpoint TP and gain totals cannot test endpoint preservation because retained=TP−gains; conditioning on these post-treatment quantities does not identify the LR mechanism.

[All-version figure](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/lower-lr-balanced-01/figures/lower-lr-trajectories.png) · [All 34 observations as CSV](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/lower-lr-balanced-01/figures/lower-lr-trajectories.csv).

The lead reused each arm's accepted runtime/identity/update/export evidence and checked the final metric consumers, receipt bindings and sequential accounting. Independent maintained recomputation exactly matched both complete parsed offline JSON objects: 570 denominator IDs, 306 image-version rows per arm, all owner transitions and the named tie. All six run/readback/offline stages exited0; 612 requests, 34 exports, 256 rank-update receipts and 1346 finite HF forwards completed; all56 exact owner/descendant identities are absent. [Lead pair acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/lower-lr-balanced-01/lead-pair-acceptance-02.json), SHA256 `e99a3625eaa1797ecde4a5dafdf7bb91fd3fc3a7719a441a903aaaf9283c5058`, links the core and metric evidence. The initial CPU preparation incorrectly reported record validation passing; the enum failure and false claim were corrected before release, with original evidence retained.

External owner time is1669.165141440928seconds (constant992.304222, lower676.860919), charged at eight slots:3.709255869868729 informational GPU-hours. Including the inter-arm CPU gap, first issue to last finish is1744.381117seconds. [Final accounting](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/lower-lr-balanced-01/lower-lr-accounting-02.json) records cumulative16.193740099274315 and unit13.996486765940984 GPU-hours, zero reservation and no cumulative cap. These timings do not isolate an inference-balancing speedup; trajectories and generated work differ.

The user requested a pause for joint analysis after this pair. No live native job or further release remains. The next decision is which failure to explain first: loss of previously covered owners, or failure to eliminate certified invalid/repeated continuations. Same-prefix before/after scoring is a proposed discriminator between an uncorrected trained event and a new error reached through a changed prefix; it has not been run or authorized by this record. Further LR sweeps, layer freezing, per-occurrence supervision, sampling and missing-label proxies remain proposals, not scheduled experiments.

## Rollout balancing technical acceptance (historical qualification)

The `probes.full_label_fit` package and `previous_rollout_tokens_lpt_v1` inference assignment are technically accepted on source `7a51e091477cb714a20b5d6aef5dc035662557a6`. Custom DoRA vLLM performs acquisition; HF performs learning, then refreshes vLLM from the next same-version parameter snapshot. Assignment changes inference ownership only; the original sorted-stride learner/storage partition, 8/18 weights and final-job synchronization remain intact.

One fresh invocation with 8 ranks and 2 updates completed run/readback/offline with exits 0/0/0 in 227.229428 seconds, within the 900-second bound. The lead verified 301 bound files, 54 raw requests, exports 0..2, 16 rank-update receipts, three consistent/distinct parameter versions, 24 generate batches and 16 refreshes. Independent reconstruction matched all three schedules and 24 rank copies; 46/54 requests were generated on a different rank from their fixed learner rank. Actual work was 13143 generated tokens (maximum 1966/request) and 89 finite singleton HF forwards, maximum context 3725. All 28 exact owner/descendant identities are absent.

| Version | TP | FP annotation | FN | Valid rows | F1 | Baseline owners lost |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 237 | 153 | 333 | 390 | 0.493750 | 0 |
| 1 | 239 | 138 | 331 | 377 | 0.504752 | 17 |
| 2 | 237 | 105 | 333 | 342 | 0.519737 | 24 |

These are qualification observations, not evidence of quality improvement. Endpoint 24 gains exchange with 24 baseline losses; the image 4134 / annotation 294005 tie is lost at version 2. F1 includes all strict-valid rows and excludes invalid/malformed rows, whose counts remain in the report. The fresh baseline TP 237 differs from the earlier unbalanced baseline TP 247: changed batching can change greedy trajectories, so old runs are not equivalent controls. This run has no matched wall-time control and establishes no measured speedup. The fixed-trace CPU projection remains a separate work proxy.

[Lead native acceptance](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/rollout-balance-01/lead-native-acceptance-02.json), SHA256 `cc67cde8296bcb5e771b146187af581935ca61f091ccc8acb35b6a13c24a348e`. [Terminal report](/data/CoordExp/.worktrees/research-probes/outputs/research/physical-fn-recovery/2026-10-02/full-label-self-rollout-fit-01/rollout-balance-01/native-terminal-candidate-02.json) binds per-image/annotation transitions, burdens, timing, resources and all receipts. Cost is 0.5049542844 informational GPU-hours; cumulative usage is 12.4844842294, with no reservation or live job. The 144-site HF/native witness has 23 argmax mismatches and maximum HF gap 0.25; this remains bounded replay evidence. RSS is sampled process aggregate and allocator peaks are separate process high-water marks. No new 16-update research trajectory is released.

## Completed learning-rate-profile comparison (earlier unbalanced runs)

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
