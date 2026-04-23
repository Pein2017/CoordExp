# `ul_res_1024_v2` Rollout Temperature Refinement

Date: 2026-03-11

## Goal

Choose a practical rollout temperature for training-time sampling with:

- checkpoint: `output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged`
- dataset: `public_data/coco/rescale_32_1024_bbox_max60/val.first200.coord.jsonl`

The target is not just highest recall. We want the best balance among:

- full-match quality on the eval subset
- parse stability
- suspicious duplication tail, especially different-desc pairs with very high IoU

## Sweep

Refinement sweep run on `8` GPUs:

- `0.6 x 2 seeds`
- `0.7 x 2 seeds`
- `0.8 x 2 seeds`
- `0.9 x 2 seeds`

Common knobs:

- `top_p = 0.95`
- `max_new_tokens = 2048`
- `repetition_penalty = 1.1`
- `batch_size = 4`

Artifacts:

- run root:
  - `output/analysis/v2_temp_trainrollout_refine_20260311`
- aggregate summary:
  - `output/analysis/v2_temp_trainrollout_refine_20260311/aggregate_summary.json`
  - `output/analysis/v2_temp_trainrollout_refine_20260311/aggregate_summary.csv`

## Main Result

The best balance point in this refinement sweep is **`temperature = 0.7`**.

Mean metrics by temperature:

| temp | mean recall_full@0.50 | mean precision_full@0.50 | mean f1_full@0.50 | mean empty_pred | mean dup records @ IoU>=0.95 | mean dup records @ IoU>=0.99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.6` | `0.5544` | `0.7069` | `0.6203` | `14.5` | `17.0 / 200` | `2.5 / 200` |
| `0.7` | `0.5630` | `0.7139` | `0.6295` | `13.5` | `18.5 / 200` | `3.0 / 200` |
| `0.8` | `0.5152` | `0.7189` | `0.5999` | `19.0` | `14.5 / 200` | `1.5 / 200` |
| `0.9` | `0.5038` | `0.7335` | `0.5970` | `23.5` | `9.5 / 200` | `0.0 / 200` |

## Interpretation

### Why `0.7` wins

`0.7` gives:

- the best mean `recall_full@0.50`
- the best mean `f1_full@0.50`
- slightly better parse stability than `0.6`
- only a modest increase in duplication compared with `0.6`

In other words, `0.7` improves useful coverage more than it worsens the duplication tail.

### Why not `0.8`

`0.8` was the best prior from the earlier broader sweep, but in this focused refinement it underperformed:

- lower recall than `0.6` and `0.7`
- lower F1 than `0.6` and `0.7`
- worse `empty_pred` than `0.6` and `0.7`

It does reduce the duplication tail somewhat, but the quality loss is larger than the duplication gain.

### Why not `0.9`

`0.9` is the cleanest on duplication:

- fewest records with suspicious high-IoU different-desc overlaps
- zero `IoU >= 0.99` records in this refinement sweep

But the cost is too large:

- lowest recall among the tested refinement temperatures
- near-worst F1
- worst `empty_pred`

That makes it too conservative for a real rollout temperature unless your priority becomes “minimize duplication almost at any cost.”

## Practical Recommendation

Use:

- **primary training rollout temperature: `0.7`**

Keep in reserve:

- **fallback conservative temperature: `0.6`**
  - if later training becomes too duplication-sensitive
- **high-precision low-duplication temperature: `0.9`**
  - only if you decide rollout cleanliness matters more than recall / useful stochastic coverage

## Stability Notes

Per-seed spread is reasonable at `0.7`:

- `t0p7_s301`: recall `0.5637`, F1 `0.6262`, empty `17`
- `t0p7_s302`: recall `0.5623`, F1 `0.6329`, empty `10`

That is more stable than `0.6`, where one seed fell much lower on recall/F1:

- `t0p6_s301`: recall `0.5242`, F1 `0.6100`
- `t0p6_s302`: recall `0.5845`, F1 `0.6306`

So `0.7` is not only best on the mean; it is also a cleaner center point across the two refinement seeds.

## Bottom Line

For this checkpoint and this rollout configuration:

- `0.7` is the best temperature for real training rollout right now
- the meaningful operating band is `0.6 - 0.8`
- `0.9` is cleaner but too conservative
- `0.8` is no longer the best choice once we refine around the useful range
