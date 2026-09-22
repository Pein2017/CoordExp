# V3 result: three fewer dev predictions than coordinate F1, same owner sets

Status: lead-accepted bounded result. This final matched package is
stopped. No retry, extra step, description-only arm, dose extension or promotion.
Grant: `root-owner-v3-full-action-f1-20260908-a`. Frozen protocol:
`protocol-v3-full-action-f1.md`.

## Observation

Primary category-agnostic pixel-IoU>=0.50 one-to-one pooled outcomes:

| Panel | Arm | Owners / GT | Valid predictions | Recall | Annotation-relative F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| dev64 | Source | 295 / 498 | 574 | .592369 | .550373 |
| dev64 | immediate v1 | 293 / 498 | 583 | .588353 | .542091 |
| dev64 | downstream v1 | 295 / 498 | 575 | .592369 | .549860 |
| dev64 | coordinate F1 v2 | 295 / 498 | 571 | .592369 | .551918 |
| dev64 | full-action F1 v3 | 295 / 498 | 568 | .592369 | .553471 |
| train16 | Source | 93 / 124 | 130 | .750000 | .732283 |
| train16 | immediate v1 | 94 / 124 | 131 | .758065 | .737255 |
| train16 | downstream v1 | 93 / 124 | 131 | .750000 | .729412 |
| train16 | coordinate F1 v2 | 92 / 124 | 131 | .741935 | .721569 |
| train16 | full-action F1 v3 | 92 / 124 | 131 | .741935 | .721569 |

- Versus coordinate F1 v2: **exact same owner sets** on both panels (dev295,
  train92; zero gains/losses). Dev valid predictions decrease3 and F1 rises
  .001553239. Train pooled counts/F1 are identical; this is not a claim of
  identical raw trajectories.
- Versus Source/downstream v1: dev gains `1987423` in image `070033`, loses
  `1656471` in image `131580`, retains294 (net0), the same owner turnover seen
  in v2. Prediction deltas are-6/-7 and F1 deltas+.003097785/+.003610714.
  Train loses `1926376` in image `532132` and gains none; retains92.
- Versus immediate v1: dev gains `1987423`/`070033` and `1194977`/`255904`,
  loses0, retains293. Predictions decrease15 and F1 rises .011380263. Train
  loses `1926253` and `1926376`, both in image `532132`, and gains none.
- All80 new trajectories terminate `im_end`; zero caps and strict geometry
  repeats. Train has0 parser drops; dev has3 drops in2 images, unchanged from
  Source/coordinate F1. Dev unmatched valid predictions are273 versus276 in
  coordinate F1 and279 in Source. These counts do not label physical false objects.
- Category-consistent dev owners remain286 versus coordinate F1, one below
  Source287. Category-consistent F1 is .536585366 versus v2 .535079514 and
  Source .535447761, due to the smaller prediction denominator.

## Interpretation and stop

Broadening direct credit from four coordinates to the whole sampled action
did not recover additional owners over coordinate F1 or original Source. The
measured difference is a further small dev annotation-relative burden/F1
improvement with unchanged v2 owner sets; the training deficit remains.
This result neither establishes a description-choice mechanism nor proves
that full-action credit cannot work at another support/dose. Such settings
were not tested and are not an automatic successor.

The root's existing v2 visual adjudication remains unchanged: those gain/loss
regions are visible book/media spines and v2 changes were local geometric
assignment changes, not semantic discovery. V3 reproduces the same owner-set
turnover, but equal owner IDs alone do not prove equal boxes or causes.

The final A16 credit-surface contrast is answered within its reused-dev scope.
No net owner-recovery success or robust superiority is claimed. Root owns the
global next direction/stop decision; this package schedules nothing further.

## Actual scope, execution and evidence

One invocation completed PASS in **166.977114 seconds** within360 seconds,
with nine child processes all exit0: one eight-rank Source-started training
entry and eight cold evaluators. No mechanical failure, repair or retry.

The actual saved training receipt declares `full_sampled_action_tokens` and
the raw all-action sum objective. Counters:64 forwards/backwards,16 images,
eight ranks, exactly one optimizer step, **626 credited/action tokens** and
**256 coordinate metadata tokens**. All588 permitted DoRA tensors changed;
parameter delta L2 .010588935, raw gradient norm1.76429808, clipped norm
approximately1.0. Full-action scope/counters/objective were validated before
save, by each actual cold-load adapter consumer, and by the final reducer.
This closes the real receipt seam beyond the prelaunch CPU fixture.
Measured maximum training-rank allocation25.540925GiB and host RSS11.800499GiB;
rank credited-token counts are78,76,74,84,77,79,80,78 (sum626).

Fresh `full_action.py reduce` exited0/PASS, yielding eight paired comparisons.
It revalidated v1/v2 under their original plans and required exact historical
results, then checked the new train/save/cold-load/row/media/metric identities.
The original Source and all previous updated controls were neither retrained
nor regenerated. Protected confirmation512 was not used.
Root independently replayed the full reduction to `lead-results.json`, exactly
matching the `results.json` bytes/hash below, and verified actual counters,
the saved checkpoint, all child exits and released GPUs.

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-owner-outcome-autonomous/learning/trial-v3-full-action-f1/`

- `plan.json` SHA256:
  `1b12583a1bb76d3d401cdc7bdb3ce87c33904e146280f52f4a1f6afc898980c8`.
- `results.json` SHA256:
  `53aa81015c20c92e352d1a25fbf27a503660fa45bdbe1571978a6f2956651037`.
- `full_action_f1/adapter/adapter_model.safetensors` SHA256:
  `c0ab281e8506ee766835994c06f78bb670d72b1c8b8566023eb7bdf5cfdd5963`.
- `execution.json`, `commands.json`, `controller.log`, `logs/`: grant,
  code hashes, exact child commands/exits and raw runtime evidence.
- `full_action_f1/train-receipt.json`, optimizer and rank payloads: original
  Source lineage, gradients, exact credited versus coordinate counters and save.
- `full_action_f1/eval/`:80 fresh rows and eight actual cold-load receipts.

Rough team receipt: one inherited Astra L1 worker, no L2 agents or delegated
review; inherited effort not independently exposed/verified in worker tools.
Rough L1 implementation-to-final time about12 minutes including root readiness
coordination and the167-second execution; L2 time0.
Root froze the contrast, replayed13 prelaunch tests and owned the GPU grant.
The worker's13 train/credit/consumer plus8 evaluator tests passed with no
prelaunch or runtime correction. This is a collaboration receipt, not a price
or model benchmark.
