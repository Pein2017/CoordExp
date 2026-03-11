# Rollout Duplication Thresholds on `ul_res_1024` vs `ul_res_1024_v2`

Date: 2026-03-11

## Question

Using the Oracle-K rollout sweeps already generated for the first-200 validation subset, is the suspicious duplication illness mainly an `IoU > 0.99` phenomenon, or does most of the bad mass live earlier in the overlap tail?

Inputs:

- `output/oracle_k/stage2_ab_prod_first200_sweep_20260310/ul_res_1024_ckpt300_t*_s*/gt_vs_pred.jsonl`
- `output/oracle_k/stage2_ab_prod_first200_sweep_20260310/ul_res_1024_v2_ckpt300_t*_s*/gt_vs_pred.jsonl`

Coverage per checkpoint:

- `8` runs
- `200` records per run
- `1600` rollout records total

Study artifacts:

- `output/analysis/rollout_duplication_thresholds_20260311/ul_res_1024_summary.json`
- `output/analysis/rollout_duplication_thresholds_20260311/ul_res_1024_v2_summary.json`

## Metric

For each rollout record:

- parse all predicted bbox objects
- compute pairwise predicted-object overlap within the same record
- track the suspicious subset as `different-desc` predicted pairs
- report threshold counts for IoU and CIoU

This is intentionally semantics-agnostic in the thresholding step, but the primary lens is still `different-desc` because that is the observed post-UL pathology.

## Main Result

### `ul_res_1024`

This checkpoint does **not** have a large `IoU > 0.99` illness.

Aggregated over all `1600` rollout records:

- predicted objects: `9394`
- predicted pairs: `66129`
- different-desc predicted pairs: `31428`
- different-desc pairs with `IoU >= 0.95`: `11`
- different-desc pairs with `IoU >= 0.99`: `4`
- affected rollout records with any different-desc `IoU >= 0.99`: `4 / 1600 = 0.25%`

Distribution of suspicious different-desc IoU tail (`IoU >= 0.90`):

- `0.90-0.95`: `9` pairs
- `0.95-0.97`: `4` pairs
- `0.97-0.98`: `0`
- `0.98-0.99`: `3`
- `>=0.99`: `4`

Interpretation:

- the tail exists, but it is tiny
- `IoU > 0.99` is only `4` pairs total
- this checkpoint does not look dominated by exact-overlap multi-label duplication

### `ul_res_1024_v2`

This checkpoint **does** have a clear severe exact-overlap subset, but that is still only the tip of the larger duplication problem.

Aggregated over all `1600` rollout records:

- predicted objects: `19542`
- predicted pairs: `582979`
- different-desc predicted pairs: `557299`
- different-desc pairs with `IoU >= 0.95`: `881`
- different-desc pairs with `IoU >= 0.99`: `38`
- affected rollout records with any different-desc `IoU >= 0.99`: `19 / 1600 = 1.19%`
- affected rollout records with any different-desc `IoU >= 0.95`: `123 / 1600 = 7.69%`

Distribution of suspicious different-desc IoU tail (`IoU >= 0.90`):

- `0.90-0.95`: `1656` pairs
- `0.95-0.97`: `517`
- `0.97-0.98`: `188`
- `0.98-0.99`: `138`
- `>=0.99`: `38`

Tail share:

- `>=0.99`: only `1.5%` of the suspicious different-desc tail
- `>=0.95`: `34.7%` of the suspicious different-desc tail
- `0.90-0.95` alone: `65.3%` of the suspicious different-desc tail

Interpretation:

- yes, `ul_res_1024_v2` has a real `IoU > 0.99` sickness
- no, the illness is not mainly above `0.99`
- most of the suspicious mass starts earlier, especially in `0.90-0.95`

## IoU vs CIoU

CIoU barely changes the conclusion in this rollout tail.

Examples:

- `ul_res_1024`
  - `IoU >= 0.99`: `4` pairs
  - `CIoU >= 0.99`: `4` pairs
- `ul_res_1024_v2`
  - `IoU >= 0.95`: `881` pairs
  - `CIoU >= 0.95`: `876` pairs
  - `IoU >= 0.98`: `176` pairs
  - `CIoU >= 0.98`: `173` pairs
  - `IoU >= 0.99`: `38` pairs
  - `CIoU >= 0.99`: `38` pairs

So the severe tail is essentially the same whether you gate by IoU or CIoU.

## Worst Examples

Worst different-desc pair for `ul_res_1024`:

- run: `ul_res_1024_ckpt300_t0p5_s101`
- image: `images/val2017/000000000285.jpg`
- pair: `bear` vs `person`
- `IoU = 0.996746`
- `CIoU = 0.996746`

Worst different-desc pair for `ul_res_1024_v2`:

- run: `ul_res_1024_v2_ckpt300_t0p2_s101`
- image: `images/val2017/000000001532.jpg`
- pair: `decoration` vs `blowtorch`
- `IoU = 1.0`
- `CIoU = 1.0`

This matches the qualitative heavy-FP inspection: `ul_res_1024_v2` often produces narrow-region hotspot stacks with semantic drift, including exact or near-exact coordinate reuse.

## Temperature Pattern

For `ul_res_1024_v2`, the exact-overlap tail is most concentrated at low temperature in this sweep:

- `t=0.2`: `14` records with different-desc `IoU >= 0.99`
- `t=0.5`: `2`
- `t=0.8`: `1`
- `t=1.0`: `2`

That suggests the worst exact-duplicate pathology is not simply “higher temperature causes more overlap spam”. It is at least partly structural in the checkpoint itself.

For `ul_res_1024`, `IoU >= 0.99` only appears once per temperature bucket:

- `t=0.2`: `1`
- `t=0.5`: `1`
- `t=0.8`: `1`
- `t=1.0`: `1`

## Recommendation

If the goal is to catch the most egregious exact-duplicate semantic drift:

- `IoU >= 0.99` is a very clean severe-case slice
- especially for `ul_res_1024_v2`

If the goal is to address the majority of suspicious duplication behavior:

- `IoU >= 0.99` is too narrow
- for `ul_res_1024_v2`, it only covers `38 / 2537 = 1.5%` of the suspicious different-desc tail above `0.90`
- the larger disease starts around `0.90-0.95`

Practical take:

- use `>=0.99` for a high-confidence severe-duplication bucket
- do not mistake that bucket for the whole pathology
- if you want to shape the dominant behavior, a softer or broader penalty band around `0.95` is likely more aligned with what the rollout distribution is actually doing
