# Determined Proxy Mapping Report

- Decision rule version: `v1_semantic_proxy_rank_2026-04-01`
- Semantic mappings considered: `53`
- Strict mappings exported: `8`
- Plausible mappings exported: `13`
- Rejected semantic mappings: `32`

## Strict Mappings

- `bus_(vehicle) -> bus` | score=0.925 | n_match=117 | precision=0.929 | coverage=0.842 | mean_iou=0.918 | iou>=0.75=0.957
- `soccer_ball -> sports ball` | score=0.922 | n_match=21 | precision=1.000 | coverage=0.875 | mean_iou=0.924 | iou>=0.75=1.000
- `train_(railroad_vehicle) -> train` | score=0.916 | n_match=97 | precision=0.990 | coverage=0.776 | mean_iou=0.889 | iou>=0.75=0.887
- `mug -> cup` | score=0.914 | n_match=71 | precision=0.973 | coverage=0.845 | mean_iou=0.890 | iou>=0.75=0.930
- `urinal -> toilet` | score=0.910 | n_match=19 | precision=1.000 | coverage=1.000 | mean_iou=0.877 | iou>=0.75=0.895
- `wine_bottle -> bottle` | score=0.908 | n_match=51 | precision=1.000 | coverage=0.911 | mean_iou=0.862 | iou>=0.75=0.863
- `water_bottle -> bottle` | score=0.904 | n_match=54 | precision=1.000 | coverage=0.806 | mean_iou=0.874 | iou>=0.75=0.907
- `beer_bottle -> bottle` | score=0.869 | n_match=35 | precision=0.946 | coverage=0.814 | mean_iou=0.849 | iou>=0.75=0.857

## Plausible Mappings

- `tennis_ball -> sports ball` | score=0.899 | n_match=70 | precision=1.000 | coverage=0.946 | mean_iou=0.820 | iou>=0.75=0.729
- `baseball -> sports ball` | score=0.884 | n_match=36 | precision=1.000 | coverage=0.923 | mean_iou=0.823 | iou>=0.75=0.778
- `birthday_cake -> cake` | score=0.856 | n_match=21 | precision=0.955 | coverage=0.875 | mean_iou=0.806 | iou>=0.75=0.810
- `ball -> sports ball` | score=0.854 | n_match=20 | precision=1.000 | coverage=0.741 | mean_iou=0.828 | iou>=0.75=0.800
- `control -> remote` | score=0.847 | n_match=78 | precision=0.987 | coverage=0.650 | mean_iou=0.815 | iou>=0.75=0.731
- `cab_(taxi) -> car` | score=0.837 | n_match=16 | precision=1.000 | coverage=0.889 | mean_iou=0.792 | iou>=0.75=0.688
- `soap -> bottle` | score=0.837 | n_match=30 | precision=1.000 | coverage=0.405 | mean_iou=0.874 | iou>=0.75=0.933
- `stove -> oven` | score=0.833 | n_match=49 | precision=0.961 | coverage=0.620 | mean_iou=0.807 | iou>=0.75=0.714
- `deck_chair -> chair` | score=0.822 | n_match=27 | precision=0.964 | coverage=0.730 | mean_iou=0.805 | iou>=0.75=0.667
- `motor_scooter -> motorcycle` | score=0.813 | n_match=25 | precision=0.926 | coverage=0.758 | mean_iou=0.804 | iou>=0.75=0.640
- `tablecloth -> dining table` | score=0.813 | n_match=55 | precision=0.932 | coverage=0.579 | mean_iou=0.808 | iou>=0.75=0.673
- `gull -> bird` | score=0.789 | n_match=15 | precision=0.938 | coverage=0.484 | mean_iou=0.849 | iou>=0.75=0.800

## Notes

- This file ranks only `semantic_evidence` mappings selected by the current learned mapping artifact.
- Exact/canonical LVIS->COCO mappings remain available in `learned_mapping.json` and are not duplicated here.
- The current determination uses support + precision + coverage + geometry metrics from matched instance evidence.
- A later anchor/containment study can refine `plausible` into `anchor_proxy` vs pure `cue_only_proxy`.
