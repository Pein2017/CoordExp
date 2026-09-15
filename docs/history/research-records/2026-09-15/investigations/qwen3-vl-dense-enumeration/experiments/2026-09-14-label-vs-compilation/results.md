# Label versus compilation: accepted stages and current decision

Status: **lead-accepted and closed: physical32, frozen supply and eight conditional continuations completed**. No checkpoint promotion, training, raw-label mutation or exhaustive annotation is claimed. The [unit](unit.md) is the immutable, hash-bound launch contract; this completion record owns current disposition. Its launch-time status is preserved so exact consumer replay remains possible.

## Physical32 stage: completed

Eight source-blind Luna-max workers judged four images each, using original-resolution single-sample views and individual overlays. All32 image decisions and607 once-only proposal assignments passed the existing consumer's identity, alias and viewed-file hash checks before source unblinding. Batch01 had saved all four images before its unresponsive return phase was interrupted; root recovered and validated its119 proposals rather than redoing or discarding them. A long wait alone is not a reason to interrupt unfinished work.

The authoritative [physical accounting](physical/result.md) links the version2 machine receipt under the2026-09-14 output root, SHA256 `8c0d3abd085c4e1ee629235e56da34d0a9a6f1756fbb6d65813903b1e5ee81ea`. Root independently replayed it into `physical/root-replay.json` and obtained exact equality. The underlying frozen comparison SHA256 is `b4da082ae70e63cdc0e6001b26e937966993bd14b006c8f6ea384ec27c21cbaf`.

| Exactly the same32 images | N16 | A | B |
|---|---:|---:|---:|
| Reviewed atomic owners present |187|167|170|
| Atomic gains/losses versus N16 |—|5/25|10/27|
| GT50 TP |139|137|133|
| GT50 F1 |0.676399|0.688442|0.663342|
| Dense-group coverage, separate |11|11|10|

These surfaces disagree materially: A's GT50 F1 rises while reviewed physical coverage loses20 instances net. B loses17 versus N16; B's small physical advantage over A on this32-image panel does not establish an overall B advantage.

The direction is not driven only by caveated clusters. Removing **every** caveated atomic cluster leaves61 clusters and N16/A/B presence57/48/50, or net−9/−7 versus N16. Nine unresolved proposals stay neutral, three per arm; their conservative extra-distinct-owner sensitivity leaves N16→A in[−23,−17] and N16→B in[−20,−14]. These are descriptive sensitivities, not statistical confidence intervals or a replacement benchmark.

Exact source-index joins find that19/25 A-lost and20/27 B-lost physical clusters had **no GT50-matched N16 source prediction**. The corresponding physical net changes partition into matched/unmatched−4/−16 for A and−5/−12 for B. This explains why annotated TP can miss much of the physical loss. It does **not** identify all those instances as genuinely unannotated: class, box extent, threshold and assignment can also prevent a GT50 match. Retained clusters can change their GT match, so these strata are not a full algebraic decomposition of TP change.

Acceptance is limited to this frozen generated-proposal union and these model-assisted physical judgments. Owners missed by all arms, exhaustive scene recall, human-replicated annotation quality and training-label causality remain unmeasured. Physical32 is not exported into training and is no longer an untouched future tuning panel.

## Frozen supply stage: completed

Root replayed [supply/result.json](supply/result.json) exactly and verified11 source bindings and seven existing suffix-card paths. The image waterfall is4096 frozen→198 nominated→186 candidate-immediate-w→39 root-admitted; the final bank is53 packages. Most upstream non-nomination and370/429 unknown-neutral physical HOLD groups do not identify missing labels. Seven selected suffixes have stored machine burden, but the original admission verified c and immediate w, not the semantic quality of every later row. No supply expansion or external annotation was launched.

## Interpretation and the admitted diagnostic

**Observation:** correcting the measurement surface does not erase the physical regression; on A it reveals loss obscured by the GT F1 improvement. **Inference:** evaluation needs an independent physical overlay, while algorithmic localization is still warranted. **Unidentified:** whether historical missing-label SFT was the main upstream cause.

The chosen diagnostic uses four old-training conditions, disjoint from physical32, and N16 versus A. Their complete c rows already score all-token argmax at both endpoints. Under identical execution, inserting c therefore cannot be an independent rescue treatment. The [root ruling](unit.md) removes this duplicate comparison and admits eight free-h continuations, with no new training or root rollout. It separately measures c realization, immediate-w preservation and downstream known-owner coverage; supplied h gets no free credit.

Decision interpretation is frozen before generation:

- c works but A loses later obligations at the same h: evidence for damaged conditional continuation on that condition, not merely failure to access h from root.
- Conditional obligations survive while natural-root coverage regresses: motivates discovery/compilation on actual reachable histories; it does not prove that literal h is the only correct route.
- Neither endpoint yields a useful continuation: immediate c+w admission was not enough to certify the desired complete trajectory.
- Free c contradicts the bound sequential-argmax scores: first resolve the affected execution/parity discrepancy; do not count it as a scientific negative.

These four selected regressions cannot establish a population mechanism, and the new45 conditions' incomplete literal-row realization remains a separate known issue. No single result here automatically authorizes broader annotation, refreshed-prefix training, a loss sweep or KV surgery.

## Conditional stage: completed and independently recomputed

Root launched the sealed eight-cell packet once in detached tmux `label-vs-compilation-diag-v1`. Eight unique workers on GPUs0–7 completed with exit0. Root reran the original native consumer and the CPU analyzer, comparing their exact results to the immutable committed outputs without rewriting execution artifacts. Root also independently recomputed the four-owner clarification.

Primary artifacts under `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/diagnostic/`:

- `execution-v1/result.json`, SHA256 `90bb5b686fc75d329e23845d33e62896e66f75b223253e8e8002f932ee429818`.
- `root-acceptance-v1.json`: exact native-consumer and analyzer acceptance.
- `analysis-v1.json`, SHA256 `b69d4b66add516f73ce388bdf0bb3b4905a02c95d6a39ea2963e7bec25768706`.
- `clarification.json`, SHA256 `f7f29c7eaf5616949c713a28ae2a0decbcdd91e7466eb207d13acbe108707741`.

| Fixed h / image | Free GT50 owners N16/A | Registered obligations retained N16/A | First differing free row | Other observation |
|---|---:|---:|---:|---|
|477415|12/12|9/9 of10|4|Same free owner set despite changed rows.|
|351017|5/3|2/2 of2|3|Two N16 free owners absent from A's GT50 matches, beyond the original witness obligations.|
|417044|10/9|10/9 of11|4|A loses two N16 GT owners and gains one. Reviewed unlabeled c is separate from these counts.|
|388795|2/2|1/1 of1|3|A repeats exact c at free row4; N16 emits it only at row1.|

**Entry is not the failure in these eight reads.** Every free continuation begins with exact c. Both endpoints also produce the immediate registered-w geometry: IoU1.000,0.826,0.860,0.988 in the four cases respectively. Only477415 uses exact literal w; the other three are geometrically valid alternatives, not token-string failures.

All cells reach EOS. Across116 valid free rows there are0 geometry-invalid rows,0 other malformed rows,0 caps and1 strict repeat (A/388795's exact c). No supplied-h GT50 owner is re-emitted; supplied h itself earns no free credit. Actual generation totals1122 free tokens/1122 model forwards/8 image forwards. Outer wall time75.963s; summed worker time0.127808 GPU-hours, not measured GPU utilization. Peak cell allocated/reserved memory is about8.75/9.00GiB. No fallback call or retry was used.

The four N16-only GT owners in351017/417044 have A best-any-class IoUs0.3791,0.0095,0.0899,0.0719. Thus these are not merely global matching reassignment with an eligible IoU≥0.5 edge. The first is moderately threshold-sensitive; the others have little spatial support under this geometry test. They remain **GT/geometry evidence, not newly visually certified physical losses**. Shorter output co-occurs on these two images, but a pure unchanged-prefix truncation does not explain the earlier divergent rows; EOS causality is not identified.

## Closed decision and next proposal (not launched)

1. **Measurement:** retain a separate physical-owner evaluation overlay. Current GT F1 can reward a model that covers fewer reviewed instances. Do not convert GT-unmatched to negative supervision, nor convert every unmatched physical proposal directly into a new annotation.
2. **Learning:** on these old fitted conditions, strengthening c alone would target an already-realized action. The next decision-bearing target is a later harmful branch and post-completion behavior, while preserving useful downstream owners. Literal divergence alone is not harmful: two cases change rows but preserve owner sets.
3. **Discovery:** the next proposal is to find and verify complete useful alternatives from actual prefixes using the user's low-temperature discovery set{0.1,0.3,0.7}, then test whether training compiles the verified route into natural greedy behavior. Verified unlabeled owners may enter a separate provenance-bearing positive bank; this32-image confirmation panel must not supply training labels.

The strongest remaining alternative is distribution/trajectory interference rather than a universal missing owner-ledger mechanism. A repeats c despite successful c and immediate w, but neither that observation nor the four-case read identifies a KV circuit or proves a particular regularizer will fix it. Historical missing-label causality remains open; the present supply ledger cannot decide its prevalence.

The stop is reached: all three bounded lanes completed, no GPU continuation remains pending, no extra census or training started. A future discovery/compilation or label-completion pilot needs its own frozen contrast and acceptance condition; the completed diagnostic is not an automatic launch grant.
