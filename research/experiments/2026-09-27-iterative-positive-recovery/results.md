# Iterative noisy-positive recovery results

## First fixed-bank round: technically accepted, scientific result mixed

Source7420b44017d90dc9250ade7b5cb4bca87a78113b. Root:/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-27/iterative-positive-01. Lead replay rehashed1278 artifacts and2sources, verified72source receipts,896scheduled training forwards with common input/mask/weight identity, and exactly recomputed all126natural outputs' offline results. Exit0/3948seconds; no OOM; all72rankPIDs and supervisor absent. The first50-second loader failure and authorized metadata-keyword correction remain preserved; corrected save/reload slice passed independently.

Primary category-agreeing annotation-ID results (hidden/302,visible/268):

| Run | Hidden | Visible | Frozen preservation utility |
|---|---:|---:|---:|
| Qualified zero |58|143|0|
| Control4 |59|146|-1|
| Treatment4 |60|148|-1|
| Control8 |56|161|-8|
| Treatment8 |60|159|-3|
| Control16 |57|160|-6|
| Treatment16 |58|162|-3|

Treatment-minus-control utility is0,+5,+3; absolute treatment utility stays negative. Endpoint Human13 hidden48->43 versus control44; refined5 hidden10->15 versus control13. The relative endpoint advantage is not consistent adaptation-panel recovery. Total category coverage does improve201->220 versus control217; retain this broader observation, but the large shared visible and invalid-row improvements cannot be attributed specifically to pseudo positives. Geometry-invalid rows303->4 treatment/2control; hundreds of literal repeats and one cap remain. Both arms share corrected losses and preservation replay. No isolation of the old axis hinge's causal role.

The objective was applied:128pseudo-image/512row presentations, finite gradients, all13images' paired first/last pseudo loss falls (mean2.128->1.982). There are15nonzero-LR updates and a zero-LR step16 endpoint. Conditional fitting alone is not natural acquisition.

## Decision-changing candidate diagnostic

After selection and all outputs were frozen, the lead independently matched the52selected proposals:16category-agreeing hidden IDs;14 were ALREADY covered by the qualified ordinary-greedy zero baseline. Thus only2 supported selected IDs were initially missing. Neither arm acquires either missing ID; supported-set coverage zero14,control16=13,treatment16=12. Broader same-category overlap union has19IDs;zero17,control16=16,treatment16=14. Treatment's reproduction advantage for the actual selected boxes is confined to boxes without qualifying annotation overlap, which does not prove physical false positives.

This bank's high saved-query support often rewards targets already naturally emitted. The experiment supplied little direct additional-discovery opportunity despite its label-novelty filter. It does not show that all noisy-positive learning fails or that further scalar calibration is required. No per-item GT-based label correction/admission follows this diagnostic.

Accepted evidence:lead-round-01-replay.json,lead-bank-proxy.json,lead-selected-target-analysis-candidate.json and independently recomputedlead-selected-target-replay.json; workerround-01-candidate-receipt.json binds raw training/evaluation/payloads. Refined5 remains a development monitor with historical exposure and redraw ambiguity. Fresh converted zero remains the baseline; historical FP32 outputs are not substituted.

Decision: do not recursively train the worse endpoint or run an additional52-row conditional-scoring diagnostic. Perform one matched candidate-policy discriminator from the SAME zero teacher: withhold candidates covered by ordinary greedy predictions, using prediction geometry/category only. Keep low pseudo weight and all common updates unchanged. This is a revised first-round acquisition policy, NOT demonstrated iterative self-improvement or a new checkpoint promotion.
