# Stage-1 Detection Teacher Forcing

`stage1_detection_teacher_forcing` is the canonical clean-break public surface
for retained compact Stage-1 detection teacher-forcing configs.

Historical `recursive_detection_ce` configs have been quarantined under
`configs/archive/detection_scene_clean_break/stage1/` by the clean-break cleanup
gate. New canonical configs should be authored under this directory and should
keep `objective.id: teacher_forcing`.

## Prefix-Denoising V1

Prefix-denoising SFT V1 is authored as a narrow teacher-forcing extension under
this directory. It keeps `objective.id: teacher_forcing` and
`objective.profile: hard_sft`, but enables `prefix_denoising` so the runtime
materializes paired `clean_full` and `noisy_full` branches with clean labels.
When `training.packing: true`, static packing is used for sample grouping,
length eligibility, and throughput-oriented dataloader scheduling. The
trainer-side objective still replays the clean and noisy branches as isolated
model forwards before CE/KL, so no branch can attend to another branch or sample
answer span through a physical packed row.
Prefix-denoising runs also build a cacheable eligibility/length index before
the static packing plan. In distributed launches, rank 0 writes this cache under
`training.static_packing_cache.root_dir/prefix_denoising_eligibility/` while
other ranks wait and load it. This precompute estimates exact hybrid lengths
from the compact text template, tokenizer chat template, and deterministic
Qwen-VL visual-token count; it does not materialize or multimodally encode every
clean/noisy branch before training starts. Progress is written to
`progress.json`; the final artifact is `eligibility.json`. The current fast
estimator runs serially even when a larger precompute worker count is requested,
because measured tokenizer-heavy threaded precompute was slower; launch logs
report both the requested and effective worker counts.
Use an experiment-family `training.static_packing_cache.root_dir` for these runs,
not a trial/run output directory. Prefix-denoising eligibility and static packing
caches are data/template/noise/length artifacts; checkpoint cadence, run name,
artifact subdir, and KL-loss-only ablation knobs should not force a full
precompute rebuild. Rank 0 may promote a compatible eligibility cache into the
current exact-fingerprint directory so other ranks can load the same alias.
Production cache-root names should stay loss-neutral, for example omitting
`kl_w0p05`, so CE-only and KL ablations can share the same family cache.
For the current full-wrapper retraining target, `detection_template.id` is
`compact_object_box_closed`. The assistant target renderer emits the full
schema wrapper tokens `<|object_ref_end|>` and `<|box_end|>`, while preserving
marker-delimited compact rows: object rows are concatenated directly with no
`\n` separator.
Production full-wrapper configs start from:

```text
/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

The authored leaves also set `model.model_type: qwen3_vl` so cfg-only launch
checks do not have to infer the local checkpoint family.

Use these leaves for the first launch-health ladder:

- `prod/compact_object_box_closed_prefix_denoising_kl_w0p05_k2_2b_base_sorted_2epoch.yaml`:
  production launch leaf for natural-adjacent 2B coord-base, sorted
  full-wrapper compact rows, no-newline marker-delimited rows, 2 epochs, KL
  weight `0.05`, `num_objects_per_image: 2`, deterministic
  `debug.val_sample_limit: 512`, static packing for grouped train sampling,
  shared-memory-safe dataloader settings, and `training.eval_packing: false`.
  Eval packing is intentionally disabled because the current
  `teacher_forcing` schema guard rejects `training.eval_packing=true` until
  exact atom-position packing mapping is implemented for eval.
- `prod/compact_object_box_closed_prefix_denoising_kl_w0p05_k2_2b_base_random_4epoch.yaml`:
  random-permutation full-wrapper variant. Clean and noisy branches share the
  same sampled object order per image.
- `prod/compact_object_box_closed_prefix_denoising_ce_only.yaml`: packed
  clean/noisy hard CE on full-wrapper compact rows,
  with `prefix_denoising.current_object_kl.weight: 0.0`.
- `prod/compact_object_box_closed_prefix_denoising_kl_w0p05_k2.yaml`: the same
  packed CE path plus sparse local coordinate KL with weight `0.05` over two
  selected objects per image. This strengthens KL coverage across a trajectory
  instead of increasing the KL scale at one sampled object.
- The older `prod/compact_full_prefix_denoising_*` leaves remain as historical
  compact marker-only references and are not the preferred full-wrapper target.
- `smoke/compact_object_box_closed_prefix_denoising_kl_w0p05_k2_tiny.yaml`:
  tiny full-wrapper launch-health check for config resolution, packed sidecars,
  finite CE/KL, and wrapper-token rendering.
- `smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`: tiny packed CE-only
  legacy compact marker-only launch check.
- `smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`: tiny packed CE+KL
  legacy compact marker-only launch check.

The clean baseline is not recreated here. Compare against the existing matched
or historical clean teacher-forcing artifact, and label that comparison scope
explicitly. Prefix-denoising launch-health evidence should first prove config
resolution, hybrid materialization, packed sidecar offsets, finite CE/KL, and
the standard `llm_loss` plus top-1/top-5 token accuracy monitors.
For production-style gradient accumulation, interpret objective scale from
`llm_loss` and `prefix_denoising/*` metrics. The generic Trainer `loss` field can
reflect accumulation-scaled logging and is not the preferred CE/KL diagnostic.
