# 2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project

## Scope
Latest local comparable prediction sequence family under `/data/CoordExp/outputs/infer`: `stage1_compare_*_ckpt1832_merged_val200*`.
All 10 variants have 200 aligned records. The Label Studio first batch selects 12 records: line 0 plus highest cross-variant disagreement cases.

## Best Variants
- Best bbox AP: `desc_first_t07` AP=0.2591, AP50=0.3635, F1-full@0.50=0.5737.
- Best F1-full@0.50: `desc_first_t07` F1=0.5737, AP=0.2591.

## Family Means
- `desc_first` n=5: AP=0.2531, AP50=0.3445, F1-full@0.50=0.5662, pred_total@0.50=944.0.
- `geometry_first` n=5: AP=0.2168, AP50=0.2987, F1-full@0.50=0.4921, pred_total@0.50=676.4.

## Variant Metrics
| alias | AP | AP50 | F1-full@0.50 | F1-loc@0.50 | pred_total | sem_acc_matched |
|---|---:|---:|---:|---:|---:|---:|
| `desc_first_t07` | 0.2591 | 0.3635 | 0.5737 | 0.5754 | 946 | 0.9970 |
| `desc_first_t02` | 0.2546 | 0.3448 | 0.5636 | 0.5696 | 935 | 0.9895 |
| `desc_first_t05` | 0.2525 | 0.3427 | 0.5647 | 0.5681 | 932 | 0.9940 |
| `desc_first_t00` | 0.2505 | 0.3354 | 0.5651 | 0.5753 | 953 | 0.9824 |
| `desc_first_t03` | 0.2487 | 0.3364 | 0.5638 | 0.5697 | 954 | 0.9896 |
| `geometry_first_t02` | 0.2237 | 0.3052 | 0.4931 | 0.4969 | 671 | 0.9923 |
| `geometry_first_t03` | 0.2186 | 0.2931 | 0.4914 | 0.4962 | 667 | 0.9904 |
| `geometry_first_t07` | 0.2180 | 0.3040 | 0.4909 | 0.4938 | 654 | 0.9942 |
| `geometry_first_t00` | 0.2164 | 0.3011 | 0.4972 | 0.5019 | 685 | 0.9906 |
| `geometry_first_t05` | 0.2075 | 0.2900 | 0.4880 | 0.4927 | 705 | 0.9905 |

## Selected Label Studio Tasks
| rank | line_idx | image | gt | pred_count_range | F1-full@0.50 range | disagreement |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0 | `images/val2017/000000000139.jpg` | 20 | 8-10 | 0.414-0.533 | 18.95 |
| 1 | 146 | `images/val2017/000000015272.jpg` | 1 | 1-6 | 0.000-1.000 | 110.50 |
| 2 | 147 | `images/val2017/000000015278.jpg` | 1 | 1-10 | 0.182-1.000 | 108.82 |
| 3 | 150 | `images/val2017/000000015440.jpg` | 3 | 1-4 | 0.000-1.000 | 108.50 |
| 4 | 13 | `images/val2017/000000001425.jpg` | 2 | 1-3 | 0.000-1.000 | 106.00 |
| 5 | 106 | `images/val2017/000000010977.jpg` | 2 | 1-3 | 0.000-1.000 | 105.00 |
| 6 | 167 | `images/val2017/000000017031.jpg` | 1 | 1-3 | 0.000-1.000 | 104.50 |
| 7 | 113 | `images/val2017/000000011615.jpg` | 1 | 1-2 | 0.000-1.000 | 103.50 |
| 8 | 59 | `images/val2017/000000006213.jpg` | 2 | 2-2 | 0.000-1.000 | 103.00 |
| 9 | 80 | `images/val2017/000000007888.jpg` | 2 | 1-2 | 0.000-1.000 | 103.00 |
| 10 | 86 | `images/val2017/000000008532.jpg` | 2 | 1-2 | 0.000-1.000 | 103.00 |
| 11 | 122 | `images/val2017/000000012667.jpg` | 1 | 1-1 | 0.000-1.000 | 100.50 |

## Files
- Label config: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/label_config.xml`
- Project API payload: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/project_payload.json`
- Import JSON array: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/import/tasks.json`
- Import JSONL: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/import/tasks.jsonl`
- First task: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/import/first_task.json`
- First task rollout text: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/import/first_task_rollout_sequences.txt`
- Metrics table: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/comparison/metrics_summary.csv`
- Per-image comparison: `/data/CoordExp/outputs/label_studio_reviews/2026-05-24_stage1_compare_ckpt1832_rollout_sequences_first_project/comparison/per_image_variant_summary.jsonl`

## Validation
- All selected task image paths resolved under `/data/CoordExp/public_data/...`.
- Label Studio API project creation was not attempted because `localhost:8080` returned 502 and no `LABEL_STUDIO_API_KEY` was set.
