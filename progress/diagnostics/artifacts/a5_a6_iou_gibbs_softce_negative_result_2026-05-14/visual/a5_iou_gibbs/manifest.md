# Poor Raw IoU@0.50 Visualizations: A5 iou-gibbs

- run_dir: `/data/home/xiaoyan/AIteam/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a5_iou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu`
- source: `/data/home/xiaoyan/AIteam/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a5_iou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/vis_resources/gt_vs_pred.jsonl`
- selected_resource: `/data/home/xiaoyan/AIteam/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a5_iou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/vis_resources/poor_raw_iou50_top12/gt_vs_pred.jsonl`
- ranking: lowest per-image F1@0.50 using `tp_full`, `fp_full`, `fn_full`; ties by larger FP+FN.

| rank | record_idx | image | gt | pred | TP | FP | FN | F1@0.50 | png |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 27 | images/val2017/000000002299.jpg | 22 | 12 | 0 | 12 | 22 | 0.0000 | `vis_0000.png` |
| 2 | 4 | images/val2017/000000000776.jpg | 4 | 1 | 0 | 1 | 4 | 0.0000 | `vis_0001.png` |
| 3 | 23 | images/val2017/000000002149.jpg | 2 | 2 | 0 | 2 | 2 | 0.0000 | `vis_0002.png` |
| 4 | 57 | images/val2017/000000006012.jpg | 2 | 1 | 0 | 1 | 2 | 0.0000 | `vis_0003.png` |
| 5 | 113 | images/val2017/000000011615.jpg | 1 | 1 | 0 | 1 | 1 | 0.0000 | `vis_0004.png` |
| 6 | 61 | images/val2017/000000006471.jpg | 16 | 35 | 1 | 34 | 15 | 0.0392 | `vis_0005.png` |
| 7 | 47 | images/val2017/000000005001.jpg | 17 | 19 | 1 | 18 | 16 | 0.0556 | `vis_0006.png` |
| 8 | 54 | images/val2017/000000005586.jpg | 14 | 20 | 1 | 19 | 13 | 0.0588 | `vis_0007.png` |
| 9 | 118 | images/val2017/000000012120.jpg | 24 | 23 | 2 | 21 | 22 | 0.0851 | `vis_0008.png` |
| 10 | 178 | images/val2017/000000017959.jpg | 24 | 20 | 2 | 18 | 22 | 0.0909 | `vis_0009.png` |
| 11 | 81 | images/val2017/000000007977.jpg | 6 | 16 | 1 | 15 | 5 | 0.0909 | `vis_0010.png` |
| 12 | 140 | images/val2017/000000014439.jpg | 20 | 23 | 2 | 21 | 18 | 0.0930 | `vis_0011.png` |
