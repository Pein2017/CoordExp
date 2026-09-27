# Accepted rollout-row-credit results

The fixed-history eight-update family is technically accepted. Terminal FN repair did not improve natural acquisition over matched-row reinforcement; the registered small D/G addition did not recover that loss. This is a bounded recipe result, not a rejection of FN learning or stronger rule supervision.

All counts below are category-agreeing annotation-ID assignments on the same18-image laboratory. Full-reference denominators are retained513/hidden57. The zero covers182/19, total201; all16selected retained FNs are initially uncovered.

| Arm/checkpoint | Retained | Hidden | Total | Selected FN | Caps |
|---|---:|---:|---:|---:|---:|
| A4 | 197 | 19 | 216 | 0 | 1 |
| A8 | 191 | 21 | 212 | 0 | 3 |
| B4 | 176 | 21 | 197 | 0 | 3 |
| B8 | 182 | 19 | 201 | 1 | 6 |
| C4 | 176 | 21 | 197 | 0 | 3 |
| C8 | 181 | 19 | 200 | 1 | 6 |

A's aggregate gains come mainly from refined5. Human13 total coverage changes by-1/-7 at4/8, while refined5 changes by+16/+18. A8 also increases geometry-invalid rows to580 versus303 at zero; no checkpoint is promoted. B8 has1198invalid rows and1795complete literal repeats. C8 has1044invalid rows and1786repeats, with5162nonidentical near-overlap occurrence pairs. These overlap counts are not physical duplicate labels. The selected FN recovered by B8/C8 is image16228,annotation-45. Both saved checkpoints, all per-image gains/losses, raw/category counterparts and full denominators are retained.

Conditional training success did not imply natural acquisition. Every selected F loss decreased (median relative drop28.9%); image-mean F fell2.857->1.999. B's M loss improved1.052->.985 versus A's1.052->.824. These observations are compatible with interference or conditioning mismatch, but do not measure per-branch parameter gradients. Weighted D logits gradients were roughly1e-7..1e-6, G roughly1e-4; M/F were much larger. No row-probability underflow was observed. A null C-minus-B result at this dose does not reject effective duplicate supervision.

Prediction/retained-only follow-up diagnosis finds all16F anchors precede the last frozen output anchor; terminal append backtracks in every case, including10otherwise monotonic outputs. Neither A8 nor B8 reaches any of the16exact old terminal prefixes, so exact-prefix drift is not specific to F. Ordered insertion will place9targets at prefixes that also receive positive M successor credit. The next contrast tests this placement package with that competition preserved; it does not isolate ordering from continuation-versus-termination conditioning.

Lead independently replayed604artifact hashes,72rank source receipts,720training input/mask/weight records,24finite synchronized updates, all checkpoint-zero equality and3540finite FP32 tensor changes across4/8. The actual offline entry reproduces both frozen manifests and all scores/contrasts exactly. All83owned PIDs were absent. Source was clean at afdcd43f1e37d73a5926b285318d31d40a68cc19; implementation a80b2a8c4260ee76d7842269ce351ad8d573dd50. Runtime cost:5064s wall,720training forwards and108natural requests; no retry or extra query.

Agent-routing provenance is amended: launch used Astra/low; recorded settings changed to Sol at10:31:45 UTC and xhigh at10:31:47 on2026-09-27. The11:23wake/validation turn used Sol/xhigh. The actor is not established. Original report and604-artifact receipt remain unchanged; lead independently accepted the model evidence and restored the required Astra/low through thread/settings/update with live readback. No claim that all historical worker validation used Astra/low remains.

Evidence root: /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-27/rollout-row-credit-01/family-01. Decisive entries: runtime-candidate-receipt.json (8d421dacf419add32495d200f573b0358511148c4fd656b93c3c1cb94bcebcc5), lead-runtime-acceptance-01.json, offline-results.json, diagnostic-summary.json, lead-placement-diagnosis-01.json, runtime-routing-amendment.json, lead-routing-timeline-02.json and lead-routing-restored-01.json.

Limits: one seed, one10%partition, eight fixed-history updates, previously examined adaptation images. Refined5 retains exposure/redraw ambiguity. No physical precision, unseen transfer or iterative convergence claim. Cans remain bottle. Hidden truth entered only frozen offline evaluation, not target choice or training.


## Accepted FN-insertion contrast and stage decision

The insertion I arm is technically accepted after one8-update training and two18-image evaluations. Category-agreeing retained/hidden/selected-F coverage is184/21/2 at4 and181/21/2 at8, versus A197/19/0 and191/21/0, and terminal B176/21/0 and182/19/1. I retained utility against zero is+2/-1; total coverage205/202 remains below A216/212. Both selected acquisitions are image16228 annotation-45 and image477415 annotation-1781023660733687. These are annotation-ID proxies, not physical-owner certifications.

Placement improves selected-target retrieval and output burdens relative to terminal B. I8 has477invalid rows,917complete literal repeats and3caps, compared with B8's1198/1795/6. All16conditional F losses fall; image-mean F1.825->1.425 and M1.052->.947. Both acquired targets lie among7non-competing prefixes, with0/9competing targets acquired; this post hoc grouping does not isolate competition. Placement also changes termination conditioning. No general rejection of FN supervision or attribution to a single mechanism follows.

The strict complete-failure criterion in the placement protocol is NOT fully met: acquisition improved and B-like burdens were reduced. The lead nevertheless closes the fixed16-target position/dose route for decision value: it has not produced net retained benefit over A, and another coefficient primarily tunes this narrow construction. No checkpoint is promoted. The broader user question now warrants one coherent90%-retained partial-label efficacy baseline, not a0%-hide memorization run.

This matters because513annotations were available, but previous updates credited184matched predicted rows plus at most16selected known FNs. Those recipes did not train on the entire retained annotation sequence. The next recipe will use all513retained annotations and keep57hidden; it jointly changes supervision coverage, geometry targets, conditioning and credit distribution, so any gain cannot be credited to ordering alone. Retained-label learning and hidden-label recovery will be reported separately.

Lead acceptance independently rehashed240artifacts and138source files, replayed the actual offline entry exactly, checked288training input/position/weight records and8synchronized590-parameter updates, compared checkpoint0 tensors to teacher and accepted zero, verified590finite FP32 changes at4 and again at8, and confirmed all29owned PIDs absent. Actual worker routing was Astra/low before launch and after wake. Runtime1672.143s,527000training input tokens,282560training visual tokens,36natural queries and23793generated tokens; no retry or extra model call.

Evidence root: /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-27/rollout-row-credit-01/family-insert-01. Accepted receipt: lead-runtime-acceptance-01.json. Frozen candidate receipt49bd67cd7ce1ec32dc8b85af8a9e93f400379e098c5b30a6589c21ea0b6033f8; offline-results.json1693b50f4cd3b93474dc266edf6134ec78a801f4be6d4bb5148ef50c1ef0d21f. Old artifacts and routing amendments remain unchanged.
