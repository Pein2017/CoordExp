# LVIS/COCO80 Relation Candidates

## Summary

```json
{
  "anchor_or_cue_n_match": 3737,
  "anchor_or_cue_pair_count": 10,
  "hard_or_soft_same_extent_n_match": 196345,
  "hard_or_soft_same_extent_pair_count": 133,
  "pair_count": 3792,
  "policy_counts": {
    "cue_or_box_only_after_packaging_audit": 5,
    "do_not_train_as_coco_box": 7,
    "existence_anchor_no_hard_box": 5,
    "hard_proxy_desc1_coord1": 93,
    "manual_audit_then_tier": 101,
    "reject_or_collect_more_evidence": 3541,
    "soft_same_extent_desc05_coord03_or_verify": 40
  },
  "relation_counts": {
    "contained_surface_anchor": 3,
    "container_or_product_cue": 5,
    "container_subtype_low_support": 1,
    "contents_or_preparation_substitution": 1,
    "food_shape_substitution": 2,
    "functional_overlap_anchor": 1,
    "hyponym_same_extent": 31,
    "low_support_or_unstable": 3540,
    "related_object_substitution": 3,
    "same_extent_alias": 14,
    "same_extent_empirical_strong": 28,
    "same_extent_exact_name": 59,
    "same_extent_or_local_anchor_needs_audit": 101,
    "same_object_name_variant": 1,
    "table_family_extent_mismatch": 1,
    "visual_neighbor_or_equipment_cue": 1
  },
  "relation_support": {
    "contained_surface_anchor": {
      "mean_coverage_like": 0.37539511159192407,
      "mean_precision_like": 0.9278453663958821,
      "n_match_total": 1438,
      "pair_count": 3,
      "recovered_strict_total": 0,
      "recovered_usable_total": 1059
    },
    "container_or_product_cue": {
      "mean_coverage_like": 0.2821912166274731,
      "mean_precision_like": 0.9075136612021858,
      "n_match_total": 427,
      "pair_count": 5,
      "recovered_strict_total": 45,
      "recovered_usable_total": 915
    },
    "container_subtype_low_support": {
      "mean_coverage_like": 0.6904761904761905,
      "mean_precision_like": 0.9731543624161074,
      "n_match_total": 145,
      "pair_count": 1,
      "recovered_strict_total": 0,
      "recovered_usable_total": 62
    },
    "contents_or_preparation_substitution": {
      "mean_coverage_like": 0.164079822616408,
      "mean_precision_like": 0.940677966101695,
      "n_match_total": 222,
      "pair_count": 1,
      "recovered_strict_total": 0,
      "recovered_usable_total": 1132
    },
    "food_shape_substitution": {
      "mean_coverage_like": 0.1843138567253419,
      "mean_precision_like": 0.8663092917478883,
      "n_match_total": 103,
      "pair_count": 2,
      "recovered_strict_total": 0,
      "recovered_usable_total": 558
    },
    "functional_overlap_anchor": {
      "mean_coverage_like": 0.6909090909090909,
      "mean_precision_like": 0.9755501222493888,
      "n_match_total": 798,
      "pair_count": 1,
      "recovered_strict_total": 0,
      "recovered_usable_total": 401
    },
    "hyponym_same_extent": {
      "mean_coverage_like": 0.7190353079064516,
      "mean_precision_like": 0.978522297439633,
      "n_match_total": 17249,
      "pair_count": 31,
      "recovered_strict_total": 1729,
      "recovered_usable_total": 4435
    },
    "low_support_or_unstable": {
      "mean_coverage_like": 0.04735716382812091,
      "mean_precision_like": 0.20663341690445972,
      "n_match_total": 58606,
      "pair_count": 3540,
      "recovered_strict_total": 0,
      "recovered_usable_total": 70
    },
    "related_object_substitution": {
      "mean_coverage_like": 0.2503539281170484,
      "mean_precision_like": 0.7580418598117713,
      "n_match_total": 343,
      "pair_count": 3,
      "recovered_strict_total": 0,
      "recovered_usable_total": 178
    },
    "same_extent_alias": {
      "mean_coverage_like": 0.7627329162307435,
      "mean_precision_like": 0.9829978450881217,
      "n_match_total": 27150,
      "pair_count": 14,
      "recovered_strict_total": 9786,
      "recovered_usable_total": 1596
    },
    "same_extent_empirical_strong": {
      "mean_coverage_like": 0.793861237521392,
      "mean_precision_like": 0.9844071735599258,
      "n_match_total": 5237,
      "pair_count": 28,
      "recovered_strict_total": 640,
      "recovered_usable_total": 1462
    },
    "same_extent_exact_name": {
      "mean_coverage_like": 0.7250772772458567,
      "mean_precision_like": 0.9730638723413202,
      "n_match_total": 145535,
      "pair_count": 59,
      "recovered_strict_total": 22157,
      "recovered_usable_total": 50402
    },
    "same_extent_or_local_anchor_needs_audit": {
      "mean_coverage_like": 0.41116133294802576,
      "mean_precision_like": 0.9140711187584045,
      "n_match_total": 11878,
      "pair_count": 101,
      "recovered_strict_total": 1873,
      "recovered_usable_total": 10579
    },
    "same_object_name_variant": {
      "mean_coverage_like": 0.6621545403271292,
      "mean_precision_like": 0.9865546218487395,
      "n_match_total": 1174,
      "pair_count": 1,
      "recovered_strict_total": 0,
      "recovered_usable_total": 647
    },
    "table_family_extent_mismatch": {
      "mean_coverage_like": 0.4014953271028037,
      "mean_precision_like": 0.8217291507268554,
      "n_match_total": 1074,
      "pair_count": 1,
      "recovered_strict_total": 0,
      "recovered_usable_total": 105
    },
    "visual_neighbor_or_equipment_cue": {
      "mean_coverage_like": 0.6554054054054054,
      "mean_precision_like": 0.9897959183673469,
      "n_match_total": 194,
      "pair_count": 1,
      "recovered_strict_total": 0,
      "recovered_usable_total": 72
    }
  }
}
```

## hard_proxy_desc1_coord1

- person -> person relation=same_extent_exact_name tier=strict n=4883 prec=0.994 cov=0.751 iou75=0.831 rec_strict=1770 rec_usable=0
- cow -> cow relation=same_extent_exact_name tier=strict n=4295 prec=0.992 cov=0.802 iou75=0.880 rec_strict=1095 rec_usable=8
- elephant -> elephant relation=same_extent_exact_name tier=strict n=4160 prec=0.999 cov=0.882 iou75=0.910 rec_strict=649 rec_usable=0
- zebra -> zebra relation=same_extent_exact_name tier=strict n=3910 prec=0.999 cov=0.903 iou75=0.900 rec_strict=511 rec_usable=0
- horse -> horse relation=same_extent_exact_name tier=strict n=3754 prec=0.988 cov=0.884 iou75=0.883 rec_strict=534 rec_usable=6
- giraffe -> giraffe relation=same_extent_exact_name tier=strict n=3679 prec=0.999 cov=0.927 iou75=0.937 rec_strict=352 rec_usable=0
- doughnut -> donut relation=same_extent_alias tier=strict n=3573 prec=0.985 cov=0.545 iou75=0.948 rec_strict=2768 rec_usable=1
- bird -> bird relation=same_extent_exact_name tier=usable n=3416 prec=0.997 cov=0.619 iou75=0.812 rec_strict=0 rec_usable=1899
- sheep -> sheep relation=same_extent_exact_name tier=strict n=3392 prec=0.993 cov=0.677 iou75=0.860 rec_strict=1607 rec_usable=9
- necktie -> tie relation=same_extent_alias tier=usable n=3036 prec=0.999 cov=0.793 iou75=0.822 rec_strict=0 rec_usable=845
- bowl -> bowl relation=same_extent_exact_name tier=strict n=3022 prec=0.949 cov=0.697 iou75=0.913 rec_strict=1147 rec_usable=4
- skateboard -> skateboard relation=same_extent_exact_name tier=strict n=2893 prec=0.993 cov=0.845 iou75=0.872 rec_strict=590 rec_usable=0
- airplane -> airplane relation=same_extent_exact_name tier=strict n=2872 prec=0.998 cov=0.833 iou75=0.854 rec_strict=672 rec_usable=1
- vase -> vase relation=same_extent_exact_name tier=strict n=2862 prec=0.941 cov=0.675 iou75=0.893 rec_strict=1185 rec_usable=0
- tennis_racket -> tennis racket relation=same_extent_exact_name tier=strict n=2803 prec=0.996 cov=0.916 iou75=0.823 rec_strict=322 rec_usable=0
- pizza -> pizza relation=same_extent_exact_name tier=strict n=2752 prec=0.983 cov=0.775 iou75=0.889 rec_strict=764 rec_usable=45
- knife -> knife relation=same_extent_exact_name tier=strict n=2710 prec=0.974 cov=0.824 iou75=0.850 rec_strict=561 rec_usable=41
- bus_(vehicle) -> bus relation=hyponym_same_extent tier=strict n=2662 prec=0.974 cov=0.843 iou75=0.930 rec_strict=438 rec_usable=13
- laptop_computer -> laptop relation=same_extent_alias tier=strict n=2600 prec=0.987 cov=0.899 iou75=0.953 rec_strict=250 rec_usable=0
- fork -> fork relation=same_extent_exact_name tier=strict n=2431 prec=0.954 cov=0.828 iou75=0.888 rec_strict=439 rec_usable=28
- cup -> cup relation=same_extent_exact_name tier=strict n=2388 prec=0.928 cov=0.671 iou75=0.916 rec_strict=842 rec_usable=64
- cellular_telephone -> cell phone relation=same_extent_alias tier=strict n=2338 prec=0.992 cov=0.816 iou75=0.828 rec_strict=22 rec_usable=565
- dog -> dog relation=same_extent_exact_name tier=strict n=2297 prec=0.986 cov=0.882 iou75=0.938 rec_strict=287 rec_usable=1
- apple -> apple relation=same_extent_exact_name tier=strict n=2266 prec=0.974 cov=0.293 iou75=0.868 rec_strict=5283 rec_usable=213
- teddy_bear -> teddy bear relation=same_extent_exact_name tier=strict n=2262 prec=0.983 cov=0.657 iou75=0.910 rec_strict=923 rec_usable=0
- bench -> bench relation=same_extent_exact_name tier=usable n=2192 prec=0.935 cov=0.638 iou75=0.802 rec_strict=0 rec_usable=1287
- cat -> cat relation=same_extent_exact_name tier=strict n=2188 prec=0.986 cov=0.890 iou75=0.941 rec_strict=249 rec_usable=0
- frisbee -> frisbee relation=same_extent_exact_name tier=strict n=2170 prec=0.997 cov=0.903 iou75=0.925 rec_strict=228 rec_usable=0
- toilet -> toilet relation=same_extent_exact_name tier=usable n=2100 prec=0.993 cov=0.922 iou75=0.858 rec_strict=0 rec_usable=286
- television_set -> tv relation=same_extent_alias tier=strict n=2067 prec=0.995 cov=0.914 iou75=0.907 rec_strict=195 rec_usable=0
- sofa -> couch relation=same_extent_alias tier=strict n=2059 prec=0.925 cov=0.849 iou75=0.877 rec_strict=212 rec_usable=10
- orange_(fruit) -> orange relation=same_extent_alias tier=strict n=2009 prec=0.991 cov=0.354 iou75=0.879 rec_strict=3650 rec_usable=0
- wineglass -> wine glass relation=same_extent_alias tier=strict n=1966 prec=0.973 cov=0.750 iou75=0.832 rec_strict=534 rec_usable=28
- computer_keyboard -> keyboard relation=same_extent_alias tier=strict n=1819 prec=0.976 cov=0.638 iou75=0.878 rec_strict=1088 rec_usable=0
- spoon -> spoon relation=same_extent_exact_name tier=strict n=1650 prec=0.972 cov=0.796 iou75=0.893 rec_strict=408 rec_usable=9
- train_(railroad_vehicle) -> train relation=hyponym_same_extent tier=strict n=1641 prec=0.993 cov=0.854 iou75=0.878 rec_strict=294 rec_usable=0
- refrigerator -> refrigerator relation=same_extent_exact_name tier=strict n=1573 prec=0.989 cov=0.905 iou75=0.916 rec_strict=174 rec_usable=0
- remote_control -> remote relation=same_extent_alias tier=strict n=1571 prec=0.984 cov=0.713 iou75=0.887 rec_strict=546 rec_usable=94
- snowboard -> snowboard relation=same_extent_exact_name tier=strict n=1557 prec=0.963 cov=0.799 iou75=0.830 rec_strict=2 rec_usable=419
- mouse_(computer_equipment) -> mouse relation=same_extent_alias tier=strict n=1520 prec=0.995 cov=0.826 iou75=0.891 rec_strict=351 rec_usable=0

## soft_same_extent_desc05_coord03_or_verify

- car_(automobile) -> car relation=hyponym_same_extent tier=usable n=4741 prec=0.910 cov=0.666 iou75=0.789 rec_strict=0 rec_usable=2027
- traffic_light -> traffic light relation=same_extent_exact_name tier=usable n=4438 prec=1.000 cov=0.733 iou75=0.658 rec_strict=0 rec_usable=2003
- umbrella -> umbrella relation=same_extent_exact_name tier=usable n=3904 prec=0.992 cov=0.639 iou75=0.693 rec_strict=0 rec_usable=2613
- kite -> kite relation=same_extent_exact_name tier=usable n=3878 prec=0.989 cov=0.594 iou75=0.781 rec_strict=0 rec_usable=2756
- chair -> chair relation=same_extent_exact_name tier=usable n=3855 prec=0.944 cov=0.640 iou75=0.790 rec_strict=0 rec_usable=2234
- book -> book relation=same_extent_exact_name tier=usable n=3587 prec=0.997 cov=0.364 iou75=0.730 rec_strict=0 rec_usable=6442
- bottle -> bottle relation=same_extent_exact_name tier=usable n=3432 prec=0.986 cov=0.589 iou75=0.837 rec_strict=0 rec_usable=2319
- boat -> boat relation=same_extent_exact_name tier=usable n=3256 prec=0.994 cov=0.655 iou75=0.628 rec_strict=0 rec_usable=2005
- suitcase -> suitcase relation=same_extent_exact_name tier=usable n=3190 prec=0.875 cov=0.591 iou75=0.833 rec_strict=0 rec_usable=1801
- carrot -> carrot relation=same_extent_exact_name tier=usable n=3157 prec=0.996 cov=0.410 iou75=0.730 rec_strict=0 rec_usable=4812
- surfboard -> surfboard relation=same_extent_exact_name tier=usable n=3143 prec=0.985 cov=0.844 iou75=0.794 rec_strict=0 rec_usable=725
- motorcycle -> motorcycle relation=same_extent_exact_name tier=usable n=3042 prec=0.969 cov=0.848 iou75=0.797 rec_strict=0 rec_usable=642
- banana -> banana relation=same_extent_exact_name tier=usable n=2739 prec=0.988 cov=0.238 iou75=0.687 rec_strict=0 rec_usable=9122
- backpack -> backpack relation=same_extent_exact_name tier=usable n=2394 prec=0.877 cov=0.666 iou75=0.611 rec_strict=0 rec_usable=1143
- handbag -> handbag relation=same_extent_exact_name tier=usable n=2295 prec=0.948 cov=0.663 iou75=0.751 rec_strict=0 rec_usable=1198
- baseball_bat -> baseball bat relation=same_extent_exact_name tier=usable n=2258 prec=0.995 cov=0.827 iou75=0.722 rec_strict=0 rec_usable=581
- bicycle -> bicycle relation=same_extent_exact_name tier=usable n=2236 prec=0.964 cov=0.794 iou75=0.742 rec_strict=0 rec_usable=709
- clock -> clock relation=same_extent_exact_name tier=usable n=2181 prec=0.999 cov=0.789 iou75=0.780 rec_strict=0 rec_usable=676
- baseball_glove -> baseball glove relation=same_extent_exact_name tier=usable n=2120 prec=0.998 cov=0.839 iou75=0.785 rec_strict=0 rec_usable=433
- tennis_ball -> sports ball relation=hyponym_same_extent tier=usable n=1807 prec=0.999 cov=0.865 iou75=0.815 rec_strict=0 rec_usable=308
- sink -> sink relation=same_extent_exact_name tier=usable n=1727 prec=0.985 cov=0.789 iou75=0.712 rec_strict=0 rec_usable=647
- bed -> bed relation=same_extent_exact_name tier=usable n=1627 prec=0.963 cov=0.757 iou75=0.710 rec_strict=0 rec_usable=381
- broccoli -> broccoli relation=same_extent_exact_name tier=usable n=1525 prec=0.996 cov=0.516 iou75=0.650 rec_strict=0 rec_usable=1625
- control -> remote relation=same_object_name_variant tier=usable n=1174 prec=0.987 cov=0.662 iou75=0.771 rec_strict=0 rec_usable=647
- truck -> truck relation=same_extent_exact_name tier=usable n=1047 prec=0.857 cov=0.679 iou75=0.883 rec_strict=0 rec_usable=312
- baseball -> sports ball relation=hyponym_same_extent tier=usable n=858 prec=0.999 cov=0.878 iou75=0.825 rec_strict=0 rec_usable=128
- parking_meter -> parking meter relation=same_extent_exact_name tier= n=652 prec=0.988 cov=0.616 iou75=0.586 rec_strict=0 rec_usable=0
- oven -> oven relation=same_extent_exact_name tier=usable n=609 prec=0.944 cov=0.684 iou75=0.603 rec_strict=0 rec_usable=335
- cupcake -> cake relation=hyponym_same_extent tier=usable n=551 prec=0.953 cov=0.435 iou75=0.822 rec_strict=0 rec_usable=581
- deck_chair -> chair relation=hyponym_same_extent tier=usable n=472 prec=0.901 cov=0.533 iou75=0.727 rec_strict=0 rec_usable=407
- duck -> bird relation=hyponym_same_extent tier=usable n=353 prec=0.994 cov=0.636 iou75=0.799 rec_strict=0 rec_usable=206
- cab_(taxi) -> car relation=hyponym_same_extent tier=usable n=308 prec=0.994 cov=0.790 iou75=0.744 rec_strict=0 rec_usable=83
- motor_scooter -> motorcycle relation=hyponym_same_extent tier=usable n=302 prec=0.891 cov=0.731 iou75=0.768 rec_strict=0 rec_usable=108
- pigeon -> bird relation=hyponym_same_extent tier=usable n=280 prec=1.000 cov=0.535 iou75=0.804 rec_strict=0 rec_usable=252
- dining_table -> dining table relation=same_extent_exact_name tier=usable n=256 prec=0.934 cov=0.757 iou75=0.684 rec_strict=0 rec_usable=100
- gull -> bird relation=hyponym_same_extent tier=usable n=230 prec=0.996 cov=0.594 iou75=0.757 rec_strict=0 rec_usable=161
- wedding_cake -> cake relation=hyponym_same_extent tier=usable n=106 prec=1.000 cov=0.785 iou75=0.755 rec_strict=0 rec_usable=32
- hair_dryer -> hair drier relation=same_extent_alias tier=usable n=106 prec=0.981 cov=0.721 iou75=0.698 rec_strict=0 rec_usable=53
- wall_clock -> clock relation=hyponym_same_extent tier=usable n=58 prec=1.000 cov=0.659 iou75=0.793 rec_strict=0 rec_usable=34
- race_car -> car relation=hyponym_same_extent tier= n=1 prec=1.000 cov=0.167 iou75=1.000 rec_strict=0 rec_usable=0

## existence_anchor_no_hard_box

- tablecloth -> dining table relation=contained_surface_anchor tier=usable n=1286 prec=0.904 cov=0.548 iou75=0.685 rec_strict=0 rec_usable=1055
- table -> dining table relation=table_family_extent_mismatch tier=usable n=1074 prec=0.822 cov=0.401 iou75=0.736 rec_strict=0 rec_usable=105
- stove -> oven relation=functional_overlap_anchor tier=usable n=798 prec=0.976 cov=0.691 iou75=0.736 rec_strict=0 rec_usable=401
- mattress -> bed relation=contained_surface_anchor tier=usable n=102 prec=0.936 cov=0.302 iou75=0.539 rec_strict=0 rec_usable=4
- bedspread -> bed relation=contained_surface_anchor tier= n=50 prec=0.943 cov=0.276 iou75=0.600 rec_strict=0 rec_usable=0

## cue_or_box_only_after_packaging_audit

- soap -> bottle relation=container_or_product_cue tier=strict n=243 prec=0.996 cov=0.271 iou75=0.926 rec_strict=45 rec_usable=603
- shampoo -> bottle relation=container_or_product_cue tier=usable n=83 prec=1.000 cov=0.355 iou75=0.855 rec_strict=0 rec_usable=150
- can -> bottle relation=container_or_product_cue tier=usable n=49 prec=0.583 cov=0.040 iou75=0.714 rec_strict=0 rec_usable=40
- medicine -> bottle relation=container_or_product_cue tier=usable n=29 prec=1.000 cov=0.223 iou75=0.862 rec_strict=0 rec_usable=101
- hot_sauce -> bottle relation=container_or_product_cue tier=usable n=23 prec=0.958 cov=0.523 iou75=0.826 rec_strict=0 rec_usable=21

## do_not_train_as_coco_box

- sausage -> hot dog relation=contents_or_preparation_substitution tier=usable n=222 prec=0.941 cov=0.164 iou75=0.838 rec_strict=0 rec_usable=1132
- parasail_(sports) -> kite relation=visual_neighbor_or_equipment_cue tier=usable n=194 prec=0.990 cov=0.655 iou75=0.820 rec_strict=0 rec_usable=72
- telephone -> cell phone relation=related_object_substitution tier=usable n=128 prec=0.889 cov=0.128 iou75=0.797 rec_strict=0 rec_usable=48
- binder -> book relation=related_object_substitution tier=usable n=109 prec=0.965 cov=0.416 iou75=0.697 rec_strict=0 rec_usable=124
- duffel_bag -> backpack relation=related_object_substitution tier=usable n=106 prec=0.421 cov=0.207 iou75=0.736 rec_strict=0 rec_usable=6
- bagel -> donut relation=food_shape_substitution tier=usable n=70 prec=0.864 cov=0.314 iou75=0.914 rec_strict=0 rec_usable=102
- pear -> apple relation=food_shape_substitution tier=usable n=33 prec=0.868 cov=0.055 iou75=0.879 rec_strict=0 rec_usable=456

## manual_audit_then_tier

- glass_(drink_container) -> cup relation=same_extent_or_local_anchor_needs_audit tier= n=3140 prec=0.796 cov=0.615 iou75=0.907 rec_strict=0 rec_usable=0
- monitor_(computer_equipment) computer_monitor -> tv relation=same_extent_or_local_anchor_needs_audit tier= n=1561 prec=0.810 cov=0.527 iou75=0.841 rec_strict=0 rec_usable=0
- armchair -> chair relation=same_extent_or_local_anchor_needs_audit tier= n=727 prec=0.785 cov=0.681 iou75=0.908 rec_strict=0 rec_usable=0
- pickup_truck -> truck relation=same_extent_or_local_anchor_needs_audit tier= n=522 prec=0.760 cov=0.651 iou75=0.883 rec_strict=0 rec_usable=0
- jar -> bottle relation=same_extent_or_local_anchor_needs_audit tier= n=387 prec=0.816 cov=0.262 iou75=0.860 rec_strict=0 rec_usable=0
- magazine -> book relation=same_extent_or_local_anchor_needs_audit tier=usable n=335 prec=0.977 cov=0.330 iou75=0.746 rec_strict=0 rec_usable=554
- urinal -> toilet relation=same_extent_or_local_anchor_needs_audit tier=strict n=334 prec=0.971 cov=0.865 iou75=0.781 rec_strict=0 rec_usable=58
- bow-tie -> tie relation=same_extent_or_local_anchor_needs_audit tier=usable n=262 prec=1.000 cov=0.738 iou75=0.824 rec_strict=0 rec_usable=97
- lemon -> orange relation=same_extent_or_local_anchor_needs_audit tier=strict n=208 prec=0.924 cov=0.155 iou75=0.913 rec_strict=996 rec_usable=0
- street_sign -> stop sign relation=same_extent_or_local_anchor_needs_audit tier= n=207 prec=0.831 cov=0.030 iou75=0.942 rec_strict=0 rec_usable=0
- pot -> bowl relation=same_extent_or_local_anchor_needs_audit tier=usable n=185 prec=0.853 cov=0.143 iou75=0.789 rec_strict=0 rec_usable=985
- quilt -> bed relation=same_extent_or_local_anchor_needs_audit tier= n=180 prec=0.833 cov=0.350 iou75=0.556 rec_strict=0 rec_usable=0
- glove -> baseball glove relation=same_extent_or_local_anchor_needs_audit tier=usable n=169 prec=0.867 cov=0.031 iou75=0.757 rec_strict=0 rec_usable=5057
- awning -> umbrella relation=same_extent_or_local_anchor_needs_audit tier= n=155 prec=0.901 cov=0.044 iou75=0.587 rec_strict=0 rec_usable=0
- notebook -> book relation=same_extent_or_local_anchor_needs_audit tier=usable n=150 prec=0.987 cov=0.517 iou75=0.827 rec_strict=0 rec_usable=132
- shoulder_bag -> handbag relation=same_extent_or_local_anchor_needs_audit tier= n=145 prec=0.759 cov=0.487 iou75=0.717 rec_strict=0 rec_usable=0
- folding_chair -> chair relation=same_extent_or_local_anchor_needs_audit tier=usable n=145 prec=0.884 cov=0.633 iou75=0.738 rec_strict=0 rec_usable=86
- coffee_table -> dining table relation=same_extent_or_local_anchor_needs_audit tier= n=138 prec=0.807 cov=0.187 iou75=0.841 rec_strict=0 rec_usable=0
- shopping_bag -> handbag relation=same_extent_or_local_anchor_needs_audit tier=usable n=115 prec=0.920 cov=0.362 iou75=0.817 rec_strict=0 rec_usable=178
- chicken_(animal) -> bird relation=same_extent_or_local_anchor_needs_audit tier=strict n=98 prec=0.951 cov=0.547 iou75=0.888 rec_strict=75 rec_usable=0
- alarm_clock -> clock relation=same_extent_or_local_anchor_needs_audit tier= n=93 prec=0.979 cov=0.578 iou75=0.656 rec_strict=0 rec_usable=0
- tote_bag -> handbag relation=same_extent_or_local_anchor_needs_audit tier=usable n=92 prec=0.885 cov=0.447 iou75=0.826 rec_strict=0 rec_usable=91
- ladle -> spoon relation=same_extent_or_local_anchor_needs_audit tier=usable n=91 prec=0.989 cov=0.535 iou75=0.780 rec_strict=0 rec_usable=76
- peach -> apple relation=same_extent_or_local_anchor_needs_audit tier= n=89 prec=0.840 cov=0.210 iou75=0.865 rec_strict=0 rec_usable=0
- dispenser -> bottle relation=same_extent_or_local_anchor_needs_audit tier=strict n=88 prec=0.978 cov=0.160 iou75=0.898 rec_strict=430 rec_usable=22
- saltshaker -> bottle relation=same_extent_or_local_anchor_needs_audit tier= n=85 prec=0.752 cov=0.157 iou75=0.882 rec_strict=0 rec_usable=0
- alcohol -> bottle relation=same_extent_or_local_anchor_needs_audit tier= n=76 prec=0.844 cov=0.373 iou75=0.908 rec_strict=0 rec_usable=0
- jet_plane -> airplane relation=same_extent_or_local_anchor_needs_audit tier=usable n=73 prec=1.000 cov=0.924 iou75=0.795 rec_strict=0 rec_usable=10
- black_sheep -> sheep relation=same_extent_or_local_anchor_needs_audit tier=strict n=69 prec=0.972 cov=0.523 iou75=0.928 rec_strict=56 rec_usable=0
- dishwasher -> oven relation=same_extent_or_local_anchor_needs_audit tier= n=65 prec=0.823 cov=0.193 iou75=0.862 rec_strict=0 rec_usable=0
- pie -> cake relation=same_extent_or_local_anchor_needs_audit tier= n=62 prec=0.765 cov=0.337 iou75=0.758 rec_strict=0 rec_usable=0
- teacup -> cup relation=same_extent_or_local_anchor_needs_audit tier=usable n=60 prec=0.870 cov=0.513 iou75=0.817 rec_strict=0 rec_usable=45
- kayak -> boat relation=same_extent_or_local_anchor_needs_audit tier=usable n=59 prec=0.937 cov=0.738 iou75=0.780 rec_strict=0 rec_usable=19
- highchair -> chair relation=same_extent_or_local_anchor_needs_audit tier= n=59 prec=0.756 cov=0.590 iou75=0.695 rec_strict=0 rec_usable=0
- trunk -> suitcase relation=same_extent_or_local_anchor_needs_audit tier=strict n=57 prec=0.950 cov=0.383 iou75=0.965 rec_strict=67 rec_usable=27
- lime -> orange relation=same_extent_or_local_anchor_needs_audit tier=usable n=57 prec=0.905 cov=0.121 iou75=0.772 rec_strict=0 rec_usable=270
- mandarin_orange -> orange relation=same_extent_or_local_anchor_needs_audit tier=strict n=56 prec=1.000 cov=0.467 iou75=0.911 rec_strict=48 rec_usable=0
- flute_glass -> wine glass relation=same_extent_or_local_anchor_needs_audit tier=usable n=55 prec=0.965 cov=0.714 iou75=0.800 rec_strict=0 rec_usable=22
- muffin -> cake relation=same_extent_or_local_anchor_needs_audit tier= n=54 prec=0.818 cov=0.199 iou75=0.759 rec_strict=0 rec_usable=0
- racket -> tennis racket relation=same_extent_or_local_anchor_needs_audit tier=usable n=52 prec=1.000 cov=0.675 iou75=0.712 rec_strict=0 rec_usable=17
