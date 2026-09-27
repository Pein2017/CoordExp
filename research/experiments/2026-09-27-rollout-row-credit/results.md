# Accepted finite A/B/C result

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
