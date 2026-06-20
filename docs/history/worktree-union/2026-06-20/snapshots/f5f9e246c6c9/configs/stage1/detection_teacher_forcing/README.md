# Stage-1 Detection Teacher Forcing

`stage1_detection_teacher_forcing` is the canonical clean-break public surface
for retained compact-full Stage-1 detection teacher-forcing configs.

Historical `recursive_detection_ce` configs have been quarantined under
`configs/archive/detection_scene_clean_break/stage1/` by the clean-break cleanup
gate. New canonical configs should be authored under this directory and should
keep `objective.id: teacher_forcing`.

For `detection_template.id: compact_full`, the native answer surface is
compact-full no-newline: object rows are concatenated directly inside the
assistant payload. Keep `template.template: qwen3_vl` for the Qwen3-VL
multimodal chat wrapper; it controls role/image token wrapping and does not
replace the compact-full detection payload contract.
