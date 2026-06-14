# Stage-1 Detection Teacher Forcing

`stage1_detection_teacher_forcing` is the canonical clean-break public surface
for retained compact-full Stage-1 detection teacher-forcing configs.

Historical `recursive_detection_ce` configs have been quarantined under
`configs/archive/detection_scene_clean_break/stage1/` by the clean-break cleanup
gate. New canonical configs should be authored under this directory and should
keep `objective.id: teacher_forcing`.

## Prefix-Denoising V1

Prefix-denoising SFT V1 is authored as a narrow teacher-forcing extension under
this directory. It keeps `objective.id: teacher_forcing` and
`objective.profile: hard_sft`, but enables `prefix_denoising` so the runtime
materializes paired `clean_full` and `noisy_full` branches with clean labels.
Production configs start from:

```text
model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
```

Use these leaves for the first launch-health ladder:

- `prod/compact_full_prefix_denoising_ce_only.yaml`: packed clean/noisy hard CE,
  with `prefix_denoising.current_object_kl.weight: 0.0`.
- `prod/compact_full_prefix_denoising_kl_w0p05.yaml`: the same packed CE path
  plus sparse local coordinate KL with weight `0.05`.
- `smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`: tiny packed CE-only
  launch check.
- `smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`: tiny packed CE+KL
  launch check.

The clean baseline is not recreated here. Compare against the existing matched
or historical clean teacher-forcing artifact, and label that comparison scope
explicitly. Prefix-denoising launch-health evidence should first prove config
resolution, hybrid materialization, packed sidecar offsets, finite CE/KL, and
the standard `llm_loss` plus top-1/top-5 token accuracy monitors.
