# Stage-1 MP Production-Like Packing Benchmark

- Generated: `2026-04-28T09:23:39.883033+00:00`
- Output root: `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark`
- Winner by optimizer steps/s: `smart_batched_exact`
- Winner by estimated logical samples/s: `smart_batched_exact`

| run | status | steps | runtime_s | opt steps/s | est logical samples/s | mem reserved GiB | fill ratio | padding forwards | artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smart_batched_exact | completed | 6/6 | 398.309 | 0.015 | 1.928 | n/a | n/a | n/a | `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark/smart_batched_exact_8gpu/prodlike-smart-batched-exact-8gpu/v0-20260428-084922` |
| online_rank_microbatch_packed | completed | 6/6 | 418.334 | 0.014 | 1.836 | n/a | n/a | n/a | `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark/online_rankmb_packed_8gpu/prodlike-online-rankmb-packed-8gpu/v0-20260428-085735` |
| offline_sample_packed | completed | 6/6 | 761.400 | 0.008 | 1.836 | n/a | n/a | n/a | `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark/offline_sample_packed_8gpu/prodlike-offline-sample-packed-8gpu/v0-20260428-090607` |

## Notes

- `smart_batched_exact`: production baseline: 8 GPUs x per_device 8 x grad_accum 2 = 128 raw samples/update
  Logical throughput estimate: raw_samples = per_device_train_batch_size * gradient_accumulation_steps * visible_world_size * completed_optimizer_steps.
- `online_rank_microbatch_packed`: online cross-sample candidate packing: raw samples/update computed from effective_runtime per_device x grad_accum x visible_world_size
  Logical throughput estimate: raw_samples = per_device_train_batch_size * gradient_accumulation_steps * visible_world_size * completed_optimizer_steps.
- `offline_sample_packed`: offline sample-packed envelopes: physical packs/update computed from effective_runtime; logical raw samples/update estimated from the sample-packing manifest
  Logical throughput estimate: first completed_steps * gradient_accumulation_steps * visible_world_size aligned pack positions from sample-packing manifest.

## Equivalence Interpretation

- This benchmark compares runtime strategies on production-like data/configuration.
- It does not certify real-Qwen same-batch mathematical equivalence by itself; that requires the separate no-step parity harness with identical encoded candidates and gradient comparison.
- Packed candidate scoring is expected to preserve candidate-selection and loss semantics by construction and by the existing synthetic/fake-trainer parity tests, but production adoption remains gated on real same-batch parity.
