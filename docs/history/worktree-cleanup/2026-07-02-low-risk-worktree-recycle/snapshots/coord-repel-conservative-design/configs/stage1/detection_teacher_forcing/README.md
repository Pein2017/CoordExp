# Stage-1 Detection Teacher Forcing

`stage1_detection_teacher_forcing` is the canonical clean-break public surface
for retained compact-full Stage-1 detection teacher-forcing configs.

Historical `recursive_detection_ce` configs have been quarantined under
`configs/archive/detection_scene_clean_break/stage1/` by the clean-break cleanup
gate. New canonical configs should be authored under this directory and should
keep `objective.id: teacher_forcing`.

## Coord-Repel V1

Coord-repel V1 experiment configs live under `coord_repel/`.

Primary packed smoke and launch leaves:

- `coord_repel/smoke/packed_len12000_2b_coordexp_smoke.yaml`
- `coord_repel/smoke/packed_len12000_2b_base_smoke.yaml`
- `coord_repel/smoke/packed_len12000_2b_coordexp_stability.yaml`
- `coord_repel/prod/packed_len12000_2b_coordexp.yaml`
- `coord_repel/prod/packed_len12000_2b_base.yaml`
- `coord_repel/ablation/compact_box_end_packed_len12000_2b_coordexp.yaml`

These configs use the COCO `rescale_32_1024_bbox_len12000` dataset, static
packing, exact teacher-forcing atom remap, `coord_repel.weight: 0.05`, and
LLM-tower-only LoRA (`freeze_vit: true`, `freeze_aligner: true`).

Run entrypoint:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/coord_repel/smoke/packed_len12000_2b_coordexp_smoke.yaml
```

Production launch is gated on successful real-GPU smoke and longer stability
checks. On 8 GPUs, `per_device_train_batch_size: 1` and
`effective_batch_size: 32` produce 32 physical packed rows per optimizer step.
