# Scale-check candidate

Status: **candidate; lead acceptance pending**, 2026-09-22. Authority: [unit.md](unit.md). Execution owner: 922-worker, Astra/low. All model work is finished; no successor is launched.

The fixed recipe improved natural fitting across the training panel, but failed prospective format/duplication eligibility. This supports broader-panel learning, not complete fitting or automatic promotion. Ordinary refitting remains sufficient; no address-component causal, ordinary-SFT superiority or transfer claim follows.

## Fixed endpoint

| Panel | Source IoU50 / IoU80 | Epoch32 IoU50 / IoU80 | Source clean | Epoch32 clean |
| --- | --- | --- | --- | --- |
| Training1024,9519 targets | 5580 /3515 | 7574 /6450 | 320 | 613 |
| Retained32,679 targets | 264 /158 | 353 /229 | 4 | 9 |
| Additions992,8840 targets | 5316 /3357 | 7221 /6221 | 316 | 604 |
| Validation256,2033 targets | 1224 /800 | 1155 /687 | 104 | 87 |

Training IoU50 improves on624images, worsens on44, and has equal match counts on356; gained2348 and lost354 annotation owners are reported separately. Validation improves40/worsens85/equal131, gained148/lost217. These are class-consistent annotation proxies, not new physical adjudication. UNKNOWN remains unmatched, not verified false.

Training teacher token-weighted CE1.66641→0.66795; validation1.65458→2.20110. Training natural coverage and IoU80 improve independently of teacher loss. The retained32 do not reproduce predecessor full completion under this broader training exposure:9/32 clean. This does not retroactively alter the accepted predecessor. Validation degradation is descriptive; it is not the eligibility veto.

## Prospective guardrails and severity

| Panel | New bad /limit | New cap /limit | New owner-recurrent /limit | New severe /limit | Eligibility |
| --- | --- | --- | --- | --- | --- |
| Training | 75 /51 | 6 /10 | 47 /51 | 1 /10 | fail |
| Validation | 41 /12 | 2 /2 | 17 /12 | 0 /2 | fail |

Counts are paired source-negative→epoch32-positive, with complete denominators. Bad means parser/geometry/non-natural EOS/cap; UNKNOWN excluded. Severe means annotation-owner IoU50-proxy consecutive run≥5. Epoch4/16 sentinels remain diagnostic, never alternative selected endpoints.

Training parser-drop incidence76→97images and malformed-span incidence35→56; validation21→54 and11→36. Aggregate parser drops fall7946→345 training and2635→113 validation; invalid geometry7850→272 and2624→71. Training exact revisits3514→83, annotation-owner revisits736→117; validation1067→1 and361→35. Aggregate reductions therefore do not cancel new bad-image incidence.

Existing source-bad training76images account for7946→44 parser drops and902750→41521 dropped-span characters; existing source-bad validation21images account for2635→18 and300656→29411. Per-image spans, run lengths, gains/losses, ordinary/middle/dense strata and new failures are retained in the saved reduction and paired severity receipt. A parser drop is not a fixed-size output burden.

Natural EOS/cap: training990/34→1018/6; validation245/11→253/3. Generated tokens: training200731→110503; validation54733→31823. UNKNOWN: training7994→2158; validation2073→881. No cap, parser drop or unsuccessful output was excluded.

## Execution and evidence boundary

The fresh corrected attempt completed1968 finite/applied optimizer calls,32epochs,15744global packs and32768image presentations. The run metadata's3936consumed packs is rank-local; four ranks give15744. Warmup call1 has zero pre-step LR and advances optimizer state without parameter movement. Step62 is1033presentations, not an exact epoch boundary. Saved steps62/123/246/984/1968; only1968 is definitive. Nominal settings/source/seed/cache are unchanged; no extra fit or seed.

The original first attempt failed probe instrumentation after one zero-LR optimizer call per rank and before logging/checkpoint. Its bytes/costs remain preserved. The repair separates pre-step LR checks from post-step delta checks. The actual CPU optimizer/scheduler callback regression accepts zero-LR call1, requires movement on positive-LR call2, and rejects a positive-LR no-op. Seven instrumentation/supervision tests and nine reducer tests passed in retained receipts. An earlier CPU fixture failure remains visible. Source evaluation continued without replay.

All2752analytical cells are complete:2720new and32qualified reused source; zero missing/mutated cells. The reused source evidence is not new replication. Four isolated queue directories preserve exact proposal bytes and independent claim state. No qualification model replay was performed. Fresh evaluation processes load the intended saved checkpoint; CPU payload readback validates captured tensor identities.

All22producer intervals are terminal (one preserved failed attempt,21successful); producer and supervisor PIDs are absent. Model-execution wall8886.383215427399seconds (2.468440hours); allocated GPU66392.10316848755seconds (18.442251GPU-hours), including failed work/loading. Original wall start1790076495.1041024; no reset. No ceiling was reached. Directly instrumented evaluation model/vision-forward counts and actual decode throughput by output length are in paired-severity-telemetry-v1.json; training counts describe logical packs/optimizer calls and do not pretend to count activation-checkpoint recomputation forwards.

## Artifacts and replay

Root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale` (R). Frozen packet: sibling `2026-09-22-coordinate-codebook-scale-preparation/v3` (V). Candidate manifest binds launch/source captures, all checkpoint files, cost/jobs, test receipts, notes and exact changed paths. `reductions/final-v1.json` binds the exact2752planned inputs, including reused-source hashes. `paired-severity-telemetry-v1.json` provides supplementary paired and execution summaries. No unrelated output-directory scan enters the primary denominator.

```bash
CUDA_VISIBLE_DEVICES='' python -B -m probes.training_set_completion.coordinate_codebook_alignment.scale_reduce --manifest "$V/manifest.json" --run-root "$R" --output "$R/reductions/reviewer-fresh.json"
CUDA_VISIBLE_DEVICES='' python -B -m probes.training_set_completion.coordinate_codebook_alignment.readback --input-root "$R/production" --output "$R/reviewer-payload-fresh.json"
python -B scripts/research/check_research_knowledge.py check
python -B -m src.artifacts.output_layout --root /data/CoordExp/outputs --root /data/CoordExp/.worktrees/research-probes/outputs
git diff --check
```

Use fresh output paths; immutable receipts refuse overwrite. Exact training/evaluation argv and environment are in launch-v1/v2 and cost.json. Model launches are evidence commands, not authorization to rerun. Saved-output reduction replay, payload readback and final check statuses are bound in the candidate manifest.

Known exposure:959/1024training and248/256validation identities occurred in mature SFT; validation is excluded from this phase's updates, not historically unseen. Validation dense cases do not cover the heaviest training scenes. Incomplete annotation and prior numerical-not-bitwise resume limitations remain; this fit started fresh. No repository-wide passing-suite claim is made. Concurrent other-owner knowledge issues, if present at final checks, are disclosed without editing their records.

Final verification: saved reduction and fresh replay are byte-identical, SHA256 `c46f78a140f009d38aec07c938634eecd9000cde1134142836deff387696af3e`. Payload readback passes all2720new cells, SHA256 `b11edda5fe6729e87c0d74f862928e7a53173fb39eecd0813207e0484348f31c`;32reused cells retain their accepted payload lineage and exact input hashes. Fresh knowledge/layout/diff checks all pass. Earlier unrelated concurrent knowledge failures were observed during execution, but are absent in the final saved check.
