# LVIS to COCO-80 Projection Report

## Caveats
- Local LVIS JSONs expose `neg_category_ids` and `not_exhaustive_category_ids`, but not `pos_category_ids`.
- Matching is box-based; segmentation is preserved only as an output summary, not as a matching signal.
- Exact/canonical mappings and semantic evidence-backed mappings are inferred separately and stay explicit in `learned_mapping.json`.

## Top Exact/Canonical Mappings
- person -> person (strict, n_match=4611, precision=0.994, mean_iou=0.860)
- traffic_light -> traffic light (usable, n_match=4208, precision=1.000, mean_iou=0.774)
- cow -> cow (strict, n_match=4057, precision=0.992, mean_iou=0.880)
- elephant -> elephant (strict, n_match=3962, precision=0.999, mean_iou=0.900)
- umbrella -> umbrella (usable, n_match=3738, precision=0.992, mean_iou=0.798)
- kite -> kite (usable, n_match=3712, precision=0.988, mean_iou=0.833)
- zebra -> zebra (strict, n_match=3688, precision=0.999, mean_iou=0.895)
- chair -> chair (usable, n_match=3642, precision=0.942, mean_iou=0.839)
- horse -> horse (strict, n_match=3597, precision=0.989, mean_iou=0.884)
- giraffe -> giraffe (strict, n_match=3481, precision=0.999, mean_iou=0.918)
- doughnut -> donut (strict, n_match=3442, precision=0.985, mean_iou=0.904)
- book -> book (usable, n_match=3403, precision=0.998, mean_iou=0.811)

## Top Semantic-but-Supported Mappings
- car_(automobile) -> car (usable, n_match=4509, precision=0.911, mean_iou=0.833)
- bus_(vehicle) -> bus (strict, n_match=2545, precision=0.977, mean_iou=0.906)
- tennis_ball -> sports ball (usable, n_match=1737, precision=0.999, mean_iou=0.833)
- train_(railroad_vehicle) -> train (strict, n_match=1544, precision=0.993, mean_iou=0.884)
- tablecloth -> dining table (usable, n_match=1231, precision=0.903, mean_iou=0.810)
- wine_bottle -> bottle (usable, n_match=1207, precision=0.994, mean_iou=0.843)
- mug -> cup (strict, n_match=1188, precision=0.979, mean_iou=0.902)
- control -> remote (usable, n_match=1096, precision=0.986, mean_iou=0.820)
- baseball -> sports ball (usable, n_match=822, precision=0.999, mean_iou=0.839)
- water_bottle -> bottle (usable, n_match=809, precision=0.988, mean_iou=0.858)
- stove -> oven (usable, n_match=749, precision=0.977, mean_iou=0.825)
- soccer_ball -> sports ball (strict, n_match=540, precision=0.998, mean_iou=0.911)

## Rejected / Ambiguous Mappings
- jacket: rejected. Top candidate person had n_match=3228, precision=0.966. Reason: insufficient evidence: mean_iou 0.497 < 0.750; median_iou 0.459 < 0.800; iou_ge_05_rate 0.408 < 0.850; iou_ge_075_rate 0.060 < 0.500
- glass_(drink_container): rejected. Top candidate cup had n_match=3029, precision=0.798. Reason: insufficient evidence: precision_like 0.798 < 0.850
- shirt: rejected. Top candidate person had n_match=2527, precision=0.959. Reason: insufficient evidence: mean_iou 0.490 < 0.750; median_iou 0.463 < 0.800; iou_ge_05_rate 0.416 < 0.850; iou_ge_075_rate 0.045 < 0.500
- ski: rejected. Top candidate skis had n_match=2453, precision=0.948. Reason: insufficient evidence: mean_iou 0.626 < 0.750; median_iou 0.615 < 0.750; iou_ge_05_rate 0.697 < 0.800; iou_ge_075_rate 0.292 < 0.400
- jersey: rejected. Top candidate person had n_match=2319, precision=0.976. Reason: insufficient evidence: mean_iou 0.445 < 0.750; median_iou 0.410 < 0.800; iou_ge_05_rate 0.289 < 0.850; iou_ge_075_rate 0.019 < 0.500
- wet_suit: rejected. Top candidate person had n_match=2306, precision=0.993. Reason: insufficient evidence: mean_iou 0.641 < 0.750; median_iou 0.660 < 0.800; iou_ge_05_rate 0.801 < 0.850; iou_ge_075_rate 0.243 < 0.500
- trousers: rejected. Top candidate person had n_match=1748, precision=0.971. Reason: insufficient evidence: mean_iou 0.406 < 0.750; median_iou 0.374 < 0.800; iou_ge_05_rate 0.122 < 0.850; iou_ge_075_rate 0.025 < 0.500
- coat: rejected. Top candidate person had n_match=1586, precision=0.974. Reason: insufficient evidence: mean_iou 0.521 < 0.750; median_iou 0.497 < 0.800; iou_ge_05_rate 0.492 < 0.850; iou_ge_075_rate 0.069 < 0.500
- dress: rejected. Top candidate person had n_match=1571, precision=0.973. Reason: insufficient evidence: mean_iou 0.539 < 0.750; median_iou 0.532 < 0.800; iou_ge_05_rate 0.573 < 0.850; iou_ge_075_rate 0.080 < 0.500
- jean: rejected. Top candidate person had n_match=1570, precision=0.952. Reason: insufficient evidence: mean_iou 0.412 < 0.750; median_iou 0.376 < 0.800; iou_ge_05_rate 0.131 < 0.850; iou_ge_075_rate 0.037 < 0.500
- monitor_(computer_equipment) computer_monitor: rejected. Top candidate tv had n_match=1463, precision=0.809. Reason: insufficient evidence: precision_like 0.809 < 0.850
- sweater: rejected. Top candidate person had n_match=1070, precision=0.959. Reason: insufficient evidence: mean_iou 0.525 < 0.750; median_iou 0.511 < 0.800; iou_ge_05_rate 0.521 < 0.850; iou_ge_075_rate 0.062 < 0.500

## Crowded-Scene Recovery Examples
- image 95341 (train2017): 395 recovered instances; categories=banana
- image 204436 (train2017): 315 recovered instances; categories=apple
- image 522788 (train2017): 277 recovered instances; categories=banana
- image 546824 (train2017): 276 recovered instances; categories=book|chair|clock
- image 121769 (train2017): 257 recovered instances; categories=banana|truck
- image 64622 (train2017): 247 recovered instances; categories=carrot
- image 403301 (train2017): 243 recovered instances; categories=dining table|umbrella
- image 110013 (train2017): 235 recovered instances; categories=book
- image 105820 (train2017): 215 recovered instances; categories=book|cat|vase
- image 243373 (train2017): 177 recovered instances; categories=carrot|cup|knife
- image 524500 (train2017): 160 recovered instances; categories=apple
- image 519479 (train2017): 160 recovered instances; categories=book|bottle
