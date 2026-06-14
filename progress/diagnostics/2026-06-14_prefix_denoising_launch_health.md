# Prefix-Denoising SFT V1 Launch-Health Smoke

Date: 2026-06-14

Scope: tiny launch-health only. This note verifies that the V1 prefix-denoising SFT runtime can launch, pack paired clean/noisy views, preserve the standard `llm_loss` and token-accuracy monitors, and emit the configured local-window KL diagnostics. It is not rollout evidence and should not be interpreted as exposure-bias improvement.

## Configs

CE-only smoke:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml
```

KL-on smoke:

```bash
python -m src.sft --config configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml
```

Both smoke configs used the production-style coord base checkpoint:

```text
/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
```

Both smoke configs resolved the training data path to:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
```

The training provenance recorded `sample_limit=4`; the source JSONL existed with `size_bytes=169492296`, and SHA256 was skipped with `sha256_skipped_reason=file_too_large`.

## Artifact Roots

CE-only:

```text
temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v10-20260614-195716
```

KL-on:

```text
temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v5-20260614-200052
```

Static packing plan caches:

```text
temp/static_packing_smoke/prefix_denoising_ce_only_tiny/global_max_length_12000/train/7a1bd80b8d8b6feab6cb37f33b2de0fd024c5dbffe9ca59ddc88e9637f81504a/plan_ws1_drop0.json
temp/static_packing_smoke/prefix_denoising_kl_w0p05_tiny/global_max_length_12000/train/7a1bd80b8d8b6feab6cb37f33b2de0fd024c5dbffe9ca59ddc88e9637f81504a/plan_ws1_drop0.json
```

## Runtime Contract

Both runs materialized prefix-denoising static packing in `effective_runtime.json`:

| Config | `enabled` | `kl_weight` | `packing_enabled` | `packing_mode` |
| --- | ---: | ---: | ---: | --- |
| CE-only tiny | `true` | `0.0` | `true` | `static` |
| KL-on tiny | `true` | `0.05` | `true` | `static` |

Both static-packing plans packed the four tiny samples into one pack:

| Config | `aligned_plan` | `avg_fill` | `single_long` | `skipped_long` | `pad_needed` |
| --- | --- | ---: | ---: | ---: | ---: |
| CE-only tiny | `[[0, 1, 2, 3]]` | `0.9335` | `0` | `0` | `0` |
| KL-on tiny | `[[0, 1, 2, 3]]` | `0.9335` | `0` | `0` | `0` |

## CE-Only Smoke Metrics

The CE-only run completed 2/2 train steps.

| Step | `loss` | `llm_loss` | clean CE | noisy CE | balanced CE | top1 | top5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `10.53304005` | `10.53304005` | `11.3485527` | `9.71752739` | `10.53304005` | `0.38288289` | `0.48648649` |
| 2 | `10.53855038` | `10.53855038` | `11.3485527` | `9.72854805` | `10.53855038` | `0.38288289` | `0.48648649` |

No KL metrics were emitted, as expected for `kl_weight=0.0`.

## KL-On Smoke Metrics

The KL-on run completed 2/2 train steps. The standard `loss`, `llm_loss`, CE, top1, and top5 metrics remained present, and local-window KL diagnostics were emitted.

| Step | `loss` | `llm_loss` | clean CE | noisy CE | balanced CE | top1 | top5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `10.53304195` | `10.53304195` | `11.3485527` | `9.71752739` | `10.53304005` | `0.38288289` | `0.48648649` |
| 2 | `10.53855038` | `10.53855038` | `11.3485527` | `9.72854805` | `10.53855038` | `0.38288289` | `0.48648649` |

KL diagnostics:

| Step | raw KL | weighted KL | site count | candidate sites | support bins | edge truncation | identical-prefix sites |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.00002897` | `0.00000145` | `16.0` | `16.0` | `16.0625` | `0.125` | `2.0` |
| 2 | `0.00000062` | `0.00000003` | `16.0` | `16.0` | `16.5625` | `0.0625` | `1.0` |

Interpretation: this is enough to verify that the asymmetric local-window KL path is numerically wired and monitored. It is not enough to judge KL usefulness. In this tiny run the teacher/student full coordinate-vocab GT probabilities and support masses are extremely small, so the KL magnitude should be treated as launch evidence only.

## Dirty-State Notes

The CE-only artifact recorded `git_dirty=true` at `git_sha=d2f61e22bfeea95d9ffe78be768cefc1f9f01701` because the run happened before the packed-runtime repair commit was finalized. The repair is now committed in `2068607c`.

The KL-on artifact recorded `git_dirty=true` at `git_sha=d8b0521985dc900dd618045a2cde79dc53128af0`; at that point the only observed dirty item was the unrelated untracked audit note:

```text
progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md
```

## Counters And Gaps

No inference/eval was run in this smoke, so there are no parse/drop counters to report.

Static packing did report `skipped_long=0`, `single_long=0`, and `pad_needed=0` for both smoke plans.

The smoke artifacts did not emit explicit noising or sample-skip counters such as `zero_object_hybrid_sample` or `noise_infeasible_4coord_changed`. This is a remaining artifact gap for the two historical smoke runs: V1 should warn/skip if the noising maker ever produces an impossible sample, but this smoke only confirms that the packed tiny samples launched and trained.

Post-smoke implementation note: the branch now emits a dataset-construction warning with `skip_counters` whenever the prefix-denoising eligibility index filters rows. The historical smoke artifacts above still predate that warning, so this note should not be read as evidence that those two tiny runs observed no noising/sample-policy skips.

## Launch-Health Read

Status: healthy for V1 launch wiring.

Evidence:

- Both CE-only and KL-on packed smoke configs launch and complete 2/2 steps.
- Prefix-denoising static packing is active in runtime metadata.
- The A/B paired packed sample shape is accepted by the Qwen3-VL training path.
- Standard `llm_loss`, top1, and top5 token-accuracy monitors are preserved.
- KL-on config emits local-window raw/weighted KL, site-count, support, edge-truncation, and teacher/student probability diagnostics.

Residual risks:

- This is not a training-quality result.
- Explicit noising/sample-skip counters should be added before treating skip-rate monitoring as complete.
- The KL probability diagnostics are numerically present but not meaningful in a tiny two-step run.
