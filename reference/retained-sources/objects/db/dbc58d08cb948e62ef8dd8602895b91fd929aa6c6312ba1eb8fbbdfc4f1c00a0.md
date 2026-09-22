# COCO22 blocked launch packet

Status: BLOCKED. Do not launch the main arm from result.json status alone.

## Already frozen

- Main manifest: `closeout-v1/main-training-manifest.json` (S, fresh optimizer, 256 updates, 22 images, microbatch1, 8 ranks).
- Teacher: `data-v1/bank.json`; annotation: `annotations-v5/annotations.jsonl`; evaluation: `evaluation-preparation-v1/preparation.json`.
- Fixed256 main authorization exists. Source is not triggered by this technical failure.

## Missing gates

1. Resolve the readback result publication equality defect and replay the existing 66 immutable rows through the real collector. Preserve the failed terminal. Generation itself completed; do not automatically regenerate 66 rows. This repair is not performed in this closeout.
2. Produce the admitted cold-source step0 endpoint for this teacher using the existing runtime, then evaluate zero-update completion.
3. Publish a successful runtime successor admission and a lead release tied to the final trial digest.

## Existing real launch command, held until those gates pass

Run inside the named main tmux specified by `coco22_readback.TRIAL_TMUX`, in `/data/CoordExp/.worktrees/research-probes`, with CUDA devices0..7. The production controller command is:

```bash
python -m probes.training_set_completion.coco22_trial controller \
  --trial /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1/trial.json \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/trial-v1 \
  --release /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion/closeout-v1/main-release.json
```

Here `R` means `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion`. `trial.json` and `main-release.json` deliberately do not exist because the gates failed. The frozen exact training config and 8-rank command are recorded in `final-receipt.json`; that lower-level command must not bypass admission.
