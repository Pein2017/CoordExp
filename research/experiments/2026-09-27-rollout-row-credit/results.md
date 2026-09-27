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


## Accepted coherent retained-label S and bounded-stage synthesis

S is technically accepted. All513retained annotations were supervised as coherent sorted teacher forcing;57hidden annotations remained evaluation-only. This recipe differs from A/I in target coverage, annotation geometry, preceding context and credit distribution. Its result cannot isolate which of those changes causes a gain.

Category-agreeing full-reference counts, retaining all18images and denominators513/57:

| Recipe/checkpoint | Retained coverage | Retained gained/lost | Hidden coverage | Hidden gained/lost | Total | Caps |
|---|---:|---:|---:|---:|---:|---:|
| zero | 182 | 0/0 | 19 | 0/0 | 201 | 2 |
| A4 | 197 | 25/10 | 19 | 2/2 | 216 | 1 |
| A8 | 191 | 26/17 | 21 | 4/2 | 212 | 3 |
| I4 | 184 | 15/13 | 21 | 2/0 | 205 | 3 |
| I8 | 181 | 18/19 | 21 | 3/1 | 202 | 3 |
| S4 | 189 | 23/16 | 21 | 3/1 | 210 | 4 |
| S8 | 196 | 32/18 | 20 | 4/3 | 216 | 4 |

S improves retained coverage+7/+14 and hidden coverage+2/+1 versus zero. These are real finite annotation-match gains, not zero effect. Relative to A, S total coverage is-6/+4; the endpoint gain combines5more retained and1fewer hidden. Both checkpoints are retained, so S is not promoted from its favorable endpoint. S recovers1of the historical16selected FNs at both checkpoints, versus I2/16. Human13 drives S's retained gains: its retained coverage150->156/164, while refined5 is32->33/32. The earlier A gains were concentrated in refined5. This does not establish independent transfer: all18images are adaptation data.

The gains carry material generation costs. S4/S8 geometry-invalid counts572/786, complete literal repeats1274/1358, and caps4/4 exceed zero303/707/2. S8's four capped images7511,309264,351017,417044 account for722of786invalid rows and1246of1358complete repeats (about92percent each); no image is removed from scoring. S4's capped set differs by including13348instead of309264, so these are changing rollout failures, not a fixed discarded subset. Per-image IDs, unmatched predictions, raw/category differences and burdens remain in offline-results.json.

All-row conditional loss decreases1.690696->1.570236. This supports optimization on rendered retained prefixes, not reliable natural hidden retrieval. Excluding EOS credit may contribute to longer continuations, but all tested recipes exclude EOS and their burdens differ. Geometry changes and recurrent generated states remain alternatives. No EOS, duplication-burst, or shared latent mechanism has been established. The tiny complete-row D signal in C limits that negative result; it does not reject effective duplicate/geometry constraints.

Lead independently rehashed228candidate artifacts,138source files and202qualifier/22control bindings, replayed both offline artifacts exactly, checked144actual input/atom/position/row-loss records (513rows/4800atoms each update), eight finite synchronized590parameter updates, checkpoint0teacher/Azero equality,590finite FP32 changes at4 and again at8, and24composition hash multisets against saved payloads. All29owned PIDs were absent. Accepted source launch e4ddea052ce06bbd043e20297f6764d3d5ebe1ee, implementation41e645a23f434bc2132ab0572d3472be649ba776. No extra model call or retry occurred. Runtime1119.697s,144training forwards/230080input/141280visual tokens,36natural queries/33341generated tokens. Actual peak training allocation6343777280bytes; evaluation peak is unavailable.

Evidence root: /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-27/rollout-row-credit-01/retained-sft-01. Lead acceptance: lead-runtime-acceptance-01.json. Frozen candidate receipt cbe30e24b1023de84fef433ec35bf04a9cf4a727852f4744b795fd3c0ddf5273; lead-stage-summary-01.json contains the exact aggregate and capped-image concentration readback. Original artifacts and prior routing amendments remain unchanged.

Stage decision: close this finite10%-hidden investigation as technically accepted and scientifically mixed. Positive supervision improves some available-label coverage; moving FN targets helps specific recovery; none of A/B/C/I/S establishes robust missing-label recovery with controlled rollout burdens. This is a bounded laboratory conclusion, not a general negative result for incomplete-label learning. No model/label promotion,20%-hidden expansion, teacher refresh, longer dose or new runtime is released.

The highest-value future question is whether additional useful detections occur before the added repetitive tails or are interleaved with them. A separate proposal could inspect the existing frozen trajectories before spending more model compute. An evaluator-only truncation opportunity would not itself define a deployable stopping rule, and hidden annotations could not be used to choose per-image training/deployment cutoffs. This remains a proposal, not an active execution package.
