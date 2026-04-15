# `cxcy_logw_logh` Retrain Reanalysis (2026-04-15)

## Scope

- checkpoint under test:
  - `coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-center_logwh-merged`
- question:
  - does the retrained `cxcy_logw_logh` checkpoint still show the old pathologically low detection quality?
  - if not, what is still preventing recovery to the `0.28+` baseline band?

## Main Result

The retrained checkpoint is **not** in the prior catastrophic regime.

On a reproducible `2080`-image prefix bundle:

- run:
  - `output/infer/coco1024_valprefix2080_center_logwh_retrain_merged`
- `coco_real` metrics:
  - `bbox_AP = 0.2069369334`
  - `bbox_AP50 = 0.3435000859`
  - `bbox_AP75 = 0.2172249901`
  - `f1@0.30 = 0.5071784440`
  - `f1@0.50 = 0.4469797471`

This is orders of magnitude above the old invalid run:

- old failed run on a same-style prefix:
  - `output/infer/coco1024_valprefix_old_center_log_size_merged`
  - `bbox_AP = 5.583774e-05`
  - `bbox_AP50 = 2.590468e-04`

But it is still clearly below the strong 2B baseline:

- baseline same-prefix reference:
  - `output/infer/coco1024_valprefix_stage1_2b_ckpt1564_merged`
  - `bbox_AP = 0.4026465592`
  - `bbox_AP50 = 0.5592252027`
  - `bbox_AP75 = 0.4269396423`
- baseline prior full-val reference:
  - `output/infer/coco1024_valfull_lvis_proxy_stage1_2b_ckpt1564_merged`
  - `bbox_AP = 0.3731727356`

## Evidence That The Old Failure Mode Is Gone

- the old bad run was effectively nonfunctional:
  - near-zero AP
  - near-zero `f1@0.50`
  - poor semantic accuracy on matched boxes
- the retrained run now has:
  - `f1@0.50 = 0.44698`
  - `f1@0.50_sem_acc_on_matched = 0.99007`
  - valid boxes and normal parsing counters

Interpretation:

- the model now usually predicts the right semantic category once a box is matched
- the current issue is no longer "the pipeline or decode is fundamentally broken"
- the remaining issue is a mix of:
  - duplication-heavy tail collapse on a subset of scenes
  - broader localization / size-calibration weakness relative to baseline

## Why It Still Misses The `0.28` Baseline

### 1. A duplication-heavy tail is still present

On the `2080`-image prefix scored artifact:

- `27` images hit the hard prediction cap of `128`
- those `27` images contribute `3456 / 19382 = 17.83%` of all predictions
- `74` images have `max_same_desc_count >= 50`
- those `74` images contribute `7706 / 19382 = 39.76%` of all predictions
- `126` images have `max_same_desc_count >= 20`
- those `126` images contribute `9384 / 19382 = 48.42%` of all predictions

Typical burst classes:

- `person`
- `book`
- `chair`
- `bottle`
- `car`
- `boat`

Strict same-prefix (`1536` images) comparison:

- current retrain:
  - `at_cap_128 = 18`
  - `max_same_desc >= 50` on `52` images
  - burst-50 images contribute `37.42%` of all predictions
- old failed `cxcy_logw_logh`-family run:
  - `at_cap_128 = 13`
  - `max_same_desc >= 50` on `27` images
  - burst-50 images contribute `26.93%` of all predictions
- strong 2B center-parameterization baseline:
  - `at_cap_128 = 6`
  - `max_same_desc >= 50` on `23` images
  - burst-50 images contribute `20.16%` of all predictions

Interpretation:

- this retrained run is much better than the old failed run on AP
- but it is **not** cleaner on duplication-like burst behavior
- in fact, on the same prefix it is more burst-heavy than both:
  - the old invalid run
  - the strong center-parameterization baseline

Representative worst cases:

- row `1972`: `128x "boat"`
- row `1631`: `128x "person"`
- row `1036`: `128x "book"`
- row `234`: `125x "car"`
- row `488`: `124x "chair"`

Visual evidence:

- [vis_0000.png](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_merged/vis_review_worst8/vis_0000.png)
- [vis_0002.png](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_merged/vis_review_worst8/vis_0002.png)
- [vis_0004.png](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_merged/vis_review_worst8/vis_0004.png)

### 2. Duplication is important, but it is not the whole story

I split the same `2080`-image prefix into two subsets:

- `burst50` subset:
  - images where `max_same_desc_count >= 50`
  - `74` images
- `keep_nonburst50` subset:
  - remaining `2006` images

Results:

- burst-only:
  - [summary](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_burst50/proxy_eval_bundle_summary.json)
  - `bbox_AP = 0.0475570777`
  - `f1@0.50 = 0.0743889479`
- non-burst:
  - [summary](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_keep_nonburst50/proxy_eval_bundle_summary.json)
  - `bbox_AP = 0.2323205805`
  - `f1@0.50 = 0.5709570957`

Interpretation:

- the duplication tail is a major drag on the overall metric
- but even after removing the worst burst scenes, AP only recovers to `0.2323`
- therefore the remaining shortfall vs `0.28+` is **not** explained by duplication alone

### 3. Localization / size quality remains weaker than baseline

The retrained run has strong semantic fidelity once matched:

- `f1@0.50_sem_acc_on_matched = 0.99007`

But localization quality remains behind the baseline:

- retrained prefix:
  - `AP50 = 0.34350`
  - `AP75 = 0.21722`
  - gap `= 0.12628`
- baseline prefix:
  - `AP50 = 0.55923`
  - `AP75 = 0.42694`
  - gap `= 0.13229`

The absolute localization quality is much lower, especially for small and medium objects:

- retrained prefix:
  - `APs = 0.01465`
  - `APm = 0.10062`
  - `APl = 0.32962`
- baseline prefix:
  - `APs = 0.09401`
  - `APm = 0.25818`
  - `APl = 0.53828`

Interpretation:

- semantics are mostly correct
- boxes are still too weakly calibrated to survive the stricter IoU thresholds at baseline quality
- the deficit is strongest on small/medium objects, not only on large crowded scenes

### 3.5. The deficit is not only in `logw/logh`; center localization is also weaker

Using same-prefix (`1536` images) same-desc greedy matches:

- current retrain (`IoU >= 0.5` matched pairs):
  - `center_l2_over_diag_median = 0.0354`
  - `abs_logw_ratio_median = 0.0625`
  - `abs_logh_ratio_median = 0.0602`
- strong 2B center-parameterization baseline (`IoU >= 0.5`):
  - `center_l2_over_diag_median = 0.0209`
  - `abs_logw_ratio_median = 0.0368`
  - `abs_logh_ratio_median = 0.0308`

The same pattern holds at `IoU >= 0.3`:

- current retrain:
  - `center_l2_over_diag_median = 0.0422`
  - `abs_logw_ratio_median = 0.0711`
  - `abs_logh_ratio_median = 0.0667`
- baseline:
  - `center_l2_over_diag_median = 0.0240`
  - `abs_logw_ratio_median = 0.0421`
  - `abs_logh_ratio_median = 0.0344`

Interpretation:

- center error is roughly `1.7x` worse than the center-parameterization baseline
- width error is also roughly `1.7x` worse
- height error is closer to `1.9x` worse

So the current weakness is **not** "center is fine but only `logw/logh` are bad".
Both center and size are worse, with height/size calibration somewhat more degraded.

This matters because the strongest in-repo evidence against a "center itself is the problem" explanation is that the strong 2B baseline here is already a center-based parameterization run (`center_size_1024`) and still reaches:

- same-prefix:
  - `bbox_AP = 0.4026465592`
- prior full-val:
  - `bbox_AP = 0.3731727356`

So a center-based chart can work well in this stack; the regression appears more specific to the `cxcy_logw_logh` regime than to "using center coordinates at all".

### 4. Token-level coordinate diagnostics look healthy, but detection quality still lags

From the retrain log:

- [logging.jsonl](/data/CoordExp/output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcy_logw_logh-pure_ce/epoch_4-cxcy_logw_logh-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v3-20260414-102133/logging.jsonl)
- final eval-side signals:
  - `eval_coord_diag/coord_vocab_mass = 0.99352154`
  - `eval_coord_diag/expected_bin_mae = 26.44988779`
  - `eval_coord_monitor/flip_coord_to_noncoord = 0.00018903`
  - `eval_token_acc = 0.84111257`

Interpretation:

- the model is learning the token chart in-distribution
- but healthy token-space metrics do not guarantee strong detection geometry after full sequence generation
- this again points to optimization / decoding behavior rather than a decode inversion bug

## Current Bottom Line

- the retrained `cxcy_logw_logh` checkpoint is **valid** and no longer exhibits the old invalid near-zero-AP failure
- it still underperforms the prior baseline substantially
- the residual gap is explained by two interacting issues:
  - a duplication-heavy burst tail on a small but influential subset of images
  - a broader localization / size-calibration deficit that persists even off the burst subset

## Artifacts

- retrained prefix bundle:
  - [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_merged/proxy_eval_bundle_summary.json)
- burst-only subset:
  - [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_burst50/proxy_eval_bundle_summary.json)
- non-burst subset:
  - [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_keep_nonburst50/proxy_eval_bundle_summary.json)
- worst-scene visual review:
  - [vis_review_worst8](/data/CoordExp/output/infer/coco1024_valprefix2080_center_logwh_retrain_merged/vis_review_worst8)

## Pending

- the earlier full-val infer run under `output/infer/coco1024_valfull_lvis_proxy_center_logwh_retrain_merged` stopped after partial shard emission (`2330 / 4951` rows merged-equivalent coverage was not reached)
- a clean full-val rerun is still needed for the official replacement of the prefix estimate
