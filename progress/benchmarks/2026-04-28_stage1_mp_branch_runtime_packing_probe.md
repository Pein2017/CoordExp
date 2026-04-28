---
doc_id: progress.benchmarks.stage1_mp_branch_runtime_packing_probe_2026_04_28
layer: progress
doc_type: benchmark-study
status: rough-evidence
domain: training
summary: Rough 8-GPU production-like comparison of Stage-1 MP smart batching versus online/offline packed-varlen experiments.
tags: [stage1, set-continuation, branch-runtime, smart-batching, packing, qwen3-vl]
updated: 2026-04-28
---

# Stage-1 MP Branch Runtime Packing Probe (2026-04-28)

This note records a rough production-like throughput probe for Stage-1
set-continuation MP candidate scoring. It is intentionally a benchmark note, not
a production recommendation to adopt packed-varlen branch execution.

## Scope

- Model: Qwen3-VL-2B coord-token checkpoint
  `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`
- Data: COCO coord-token train surface
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- GPUs: 8
- Steps: 6 optimizer steps per run
- Purpose: throughput/stability comparison of branch-runtime mechanisms under
  a production-like Stage-1 MP distribution.

The trainer `train_runtime` values below are from the training loop and exclude
most process startup and offline sample-pack planning. The offline sample-pack
run still uses a different logical batch geometry, so compare it as rough
throughput evidence only.

## Results

| Runtime | Status | Train runtime | Opt steps/s | Est. logical raw samples/s | Logged memory | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `smart_batched_exact` | `6/6` | `398.309s` | `0.015` | `1.928` | `44.70 GiB` | Current production default and fastest in this probe. |
| `online_rank_microbatch_packed` | `6/6` | `418.334s` | `0.014` | `1.836` | `61.07 GiB` | Stable but slower and much higher memory. |
| `offline_sample_packed` | `6/6` | `761.400s` | `0.008` | `1.836` | `47.83 GiB` | Dense sample packs, but current MP loss/scoring path is slower per optimizer step. |

Offline sample packing did achieve the intended dense-envelope shape:

- raw samples: `2048`
- raw packs: `1418`
- DDP-aligned packs: `1424`
- target tokens per rank forward: `14000`
- mean sample-pack fill: `0.981`
- min fill: `0.630`
- max fill: `1.000`

## Interpretation

`smart_batched_exact` remains the best production choice from this probe. It is
also the lower-risk mathematical path: candidate branches are independent batch
rows with ordinary padding and attention masks, not concatenated varlen segments
whose isolation depends on FlashAttention `cu_seq_lens` boundaries.

The packed-varlen work validated useful mechanics, especially that offline
sample packing can make dense 12k-14k envelopes, but it did not yet beat smart
batching end to end. The bottleneck shifted from fill ratio to the current
Stage-1 MP candidate-scoring/loss-assembly cost inside dense packed envelopes.

Do not infer same-batch mathematical equivalence from these independent smoke
runs. Real-Qwen same-batch no-step parity, including per-candidate scores and
gradient checks, remains the needed gate before any packed-varlen production
adoption.

## Artifacts

- Aggregate Markdown:
  [`artifacts/2026-04-28_stage1_mp_branch_runtime_packing_probe_aggregate.md`](artifacts/2026-04-28_stage1_mp_branch_runtime_packing_probe_aggregate.md)
- Aggregate JSON:
  [`artifacts/2026-04-28_stage1_mp_branch_runtime_packing_probe_aggregate.json`](artifacts/2026-04-28_stage1_mp_branch_runtime_packing_probe_aggregate.json)
- Smart artifact:
  `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark/smart_batched_exact_8gpu/prodlike-smart-batched-exact-8gpu/v0-20260428-084922`
- Online rank-microbatch packed artifact:
  `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark/online_rankmb_packed_8gpu/prodlike-online-rankmb-packed-8gpu/v0-20260428-085735`
- Offline sample-packed artifact:
  `/data/CoordExp/output_remote/stage1_2b/set_continuation_prodlike_benchmark/offline_sample_packed_8gpu/prodlike-offline-sample-packed-8gpu/v0-20260428-090607`

## Recommendation

Keep `smart_batched_exact` as the Stage-1 MP production default. Retire the
current packed-varlen feature worktree unless a future effort specifically
targets real-Qwen same-batch parity plus a faster packed MP scoring/loss path.
