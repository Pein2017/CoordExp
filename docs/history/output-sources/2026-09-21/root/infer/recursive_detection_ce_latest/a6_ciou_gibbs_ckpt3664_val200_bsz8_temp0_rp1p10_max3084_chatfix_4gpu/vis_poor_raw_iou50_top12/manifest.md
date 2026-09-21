# Poor Raw IoU@0.50 Visualizations: A6 ciou-gibbs

- run_dir: `/data/home/xiaoyan/AIteam/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/a6_ciou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu`
- source: `/data/home/xiaoyan/AIteam/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/a6_ciou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/vis_resources/gt_vs_pred.jsonl`
- selected_resource: `/data/home/xiaoyan/AIteam/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/a6_ciou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/vis_resources/poor_raw_iou50_top12/gt_vs_pred.jsonl`
- ranking: lowest per-image F1@0.50 using `tp_full`, `fp_full`, `fn_full`; ties by larger FP+FN.

| rank | record_idx | image | gt | pred | TP | FP | FN | F1@0.50 | png |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 27 | images/val2017/000000002299.jpg | 22 | 13 | 0 | 13 | 22 | 0.0000 | `vis_0000.png` |
| 2 | 4 | images/val2017/000000000776.jpg | 4 | 2 | 0 | 1 | 4 | 0.0000 | `vis_0001.png` |
| 3 | 23 | images/val2017/000000002149.jpg | 2 | 2 | 0 | 2 | 2 | 0.0000 | `vis_0002.png` |
| 4 | 57 | images/val2017/000000006012.jpg | 2 | 1 | 0 | 1 | 2 | 0.0000 | `vis_0003.png` |
| 5 | 113 | images/val2017/000000011615.jpg | 1 | 1 | 0 | 1 | 1 | 0.0000 | `vis_0004.png` |
| 6 | 54 | images/val2017/000000005586.jpg | 14 | 21 | 1 | 20 | 13 | 0.0571 | `vis_0005.png` |
| 7 | 61 | images/val2017/000000006471.jpg | 16 | 35 | 2 | 33 | 14 | 0.0784 | `vis_0006.png` |
| 8 | 118 | images/val2017/000000012120.jpg | 24 | 23 | 2 | 21 | 22 | 0.0851 | `vis_0007.png` |
| 9 | 178 | images/val2017/000000017959.jpg | 24 | 22 | 2 | 20 | 22 | 0.0870 | `vis_0008.png` |
| 10 | 181 | images/val2017/000000018380.jpg | 53 | 36 | 4 | 32 | 49 | 0.0899 | `vis_0009.png` |
| 11 | 140 | images/val2017/000000014439.jpg | 20 | 24 | 2 | 22 | 18 | 0.0909 | `vis_0010.png` |
| 12 | 158 | images/val2017/000000016010.jpg | 5 | 15 | 1 | 14 | 4 | 0.1000 | `vis_0011.png` |
