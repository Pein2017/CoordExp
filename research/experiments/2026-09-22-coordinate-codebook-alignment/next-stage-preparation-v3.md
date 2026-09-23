# 1024/256 launch-packet proposal v3

Status: CPU-prepared candidate for lead freeze, 2026-09-22. No next-stage model launch is authorized or executed. The accepted 32-image package and all v1/v2 notes/artifacts remain immutable. This proposal implements [the updated brief](lead-next-stage-preparation.md); its 32-epoch ceiling and 2752 analytical cells supersede the v2 exposure/decode proposal only.

## Fixed question and data

Can the fixed recipe learn natural enumeration on the broader training panel while keeping format/duplication regression within prospective tolerances? Training natural coverage, coordinate fidelity, valid output and completion are primary. Validation coverage/CE alone neither veto stronger training fit nor select settings. This is a recipe-scale fitting test; multiple trainable surfaces preclude component-causal claims.

Reuse the accepted split exactly: 1024 training images (retained corrected32 and additions992 reported separately), 256 next-phase gradient-excluded validation images. The source/data/media/sort and annotation precedence remain bound by `candidate-v2.json`, `admission-v2.json`, and `distributions-v2.json` at `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation`. Canonical identity and original/processed content hashes are disjoint across these panels; validation excludes the prior fit/monitor. Historical exposure remains explicit: 959/1024 training and 248/256 validation identities occur in the mature source's SFT data; these are not untouched images. Validation's dense subset does not cover the heaviest training scenes. UNKNOWN is not a verified false positive.

## Exact CPU packing and training proposal

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/training-config.json` passes the maintained config loader. `config-delta.json` confirms only data paths, output identity, epochs/max-step mode and checkpoint steps differ from the accepted nominal config. Source, selected rows, live promoted PEFT policy, DoRA/codebook architecture, full-response segment-balanced objective, EOS, optimizer/LRs, warmup10, seed1729, BF16/FA2 and clipping remain unchanged. Start fresh from the mature untied+axis001 source, never the closed overfit checkpoint.

The maintained CPU entry produced **492 packs** from all1024 images exactly once, with no omitted/duplicated image within a logical epoch. Actual cache:110010221 bytes; sequence tokens1452903, image tokens1005617, supervised atoms89910 per epoch, including1024 EOS atoms. Serialized canonical response count90934 is a distinct count, not the runtime loss denominator. `packing-exposure.json` binds the cache, every pack's identities/counts, complete shuffled global order, and rank-equivalence checks for1/2/4/8 ranks.

Propose four training GPUs and gradient accumulation2, effective batch8 packs. **32 epochs =15744 packs =32768 image presentations =1968 updates**, no final tail padding. This is not dose-equivalent to the closed32-image run's roughly273 passes and does not guarantee sufficient fitting.

| Epoch label | Save update | Image presentations | Boundary |
| --- | ---: | ---: | --- |
| 1 | 62 | 1033 | four packs/nine images from epoch2 also consumed |
| 2 | 123 | 2048 | exact |
| 4 | 246 | 4096 | exact |
| 16 | 984 | 16384 | exact |
| 32 | 1968 | 32768 | exact |

The epoch1 excess is explicit in the receipt, with its nine IDs; no hidden padding or omitted tail. Save events were resolved again from the final config and match all five updates. Epoch32 is definitive; epochs1/2 are cheap checkpoint-only saves, epochs4/16 have sentinel diagnostics. No conditional LR fits or lucky full-panel checkpoint selection.

CPU packing took17.253 seconds, allocated GPU seconds0, model/vision forwards0. The helper requested four CPU workers through an environment setting, but the maintained pipeline used its default16; the actual cache receipt preserves this discrepancy. Peak child RSS1860332KiB is a per-process resource maximum, not aggregate16-worker RSS. The producer and supervisor finished successfully; no rerun was needed.

## Native queues and source reuse

`sentinel-and-queues-v3.json` freezes96 training sentinel identities: retained32 plus hash-selected24 ordinary,24 dense same-class,8 dense other,8 middle. `analytical-cells.json` lists all2752 analytical cells. The concrete maintained-evaluator queues are:

- `queue-source.json`:1248 new cells (992 training +256 validation).
- `queue-epoch32.json`:1280 cells (1024 training +256 validation).
- `queue-epoch4.json` and `queue-epoch16.json`:96 each.
- Exactly32 source cells are reused from the closed package, for **2720 new cells** total.

CPU `plan_examples` and `prepare_native_inputs(record_media_identity=True)` verified all32 reusable cases against serialized prompt token IDs, image bytes/paths/geometry, row semantics, source checkpoint, declared live runtime and decode policy. Case-level old paths/hashes/checks are retained. No old-route Mixin score is reused. `evaluation-admission.json` carries the same source runtime contract with the new data binding.

Every new cell uses empty-prefix greedy decoding, cap3084, unchanged parser/prompt, repetition penalty1 and teacher metrics after generation. Queue order is frozen; atomic claims make workers work-conserving without duplicate cells. Completion order may differ. GPUs0..3 train while4..7 drain source/available sentinel work; after training all8 drain fixed remaining queues, prioritizing the definitive endpoint over remaining diagnostic work. No early result changes data, order, settings or endpoint. Launch receipts and global guards must be bound by the lead before execution.

## Prospective decision and guardrails

`guardrails.json` proposes paired source-negative to epoch32-positive incidence limits, independently for train/validation: newly bad≤5%, newly capped≤1%, newly annotation-owner-recurrent≤5%, newly severe-run≤1%. Floor counts are **51/10/51/10** on training and **12/2/12/2** on validation. Bad means any parser/malformed-object drop, invalid geometry, non-natural EOS or cap; UNKNOWN excluded. Severe means consecutive annotation-owner IoU50-proxy run≥5. These are operational tolerances, not statistical guarantees or zero-regression requirements. Redundant net-incidence gates are omitted.

Report coverage/localization gains and losses, retained32/additions992, ordinary/dense strata, image incidence and type of malformed output, malformed-span characters, exact versus annotation-owner repeats, run lengths, generated length, EOS/cap and UNKNOWN. Existing source failures retain paired severity comparisons; passing incidence limits does not compel acceptance. Separate insufficient training fit, fit with excessive format/duplication regression, and fit within limits despite validation coverage/CE regression. Neither lower teacher loss nor lower aggregate repeat/invalid counts alone promotes the recipe. Incomplete cells remain HOLD at the original denominator.

## Budget and limits

`forecast.json` binds measured closed-run training and long-decode timings. Linear update scaling estimates training **7.725 GPU-hours**, **1.931 wall-hours** on4 GPUs. Base training+decode estimate is **28.858 GPU-hours**. A conservative plan uses the slower early-fit timing for all trained training/sentinel cells, measured source/monitor timings, +25% training, +20% decode, and explicit startup/tail/persistence reserves: **45.649 GPU-hours / 5.706 ideal work-conserving wall-hours**.

Propose the new-stage8-hour/64-allocated-GPU-hour envelope, not yet started. The forecast is not a guarantee: new shapes, long malformed outputs, cap incidence, contention and loading tails can differ. An all-cap-like300-second/cell stress case alone costs226.67 decode GPU-hours and cannot fit. Global admission must reserve outstanding allocations/failures, stop admitting by7.75wall-hours or earlier budget exhaustion, and persist/join within the ceiling. No lower token cap, denominator shrink or automatic budget extension.

## Commands and scope

CPU packing already completed through:

```sh
CUDA_VISIBLE_DEVICES='' python -B -m probes.training_set_completion.coordinate_codebook_alignment.prepare_cache --config /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/packing-config.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/packing-receipt.json
```

Do not overwrite that receipt. Future launch form, **not executed or authorized here**:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 coordexp_infras_PACK_CACHE_ROOT=/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/packing-cache python -B -m torch.distributed.run --standalone --nproc_per_node=4 -m src.train --config /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/training-config.json
```

Maintained evaluation command form (queue workers share one queue; checkpoint and output resolve in the lead-frozen launch packet):

```sh
python -B -m probes.training_set_completion.coordinate_codebook_alignment.evaluation --admission /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/evaluation-admission.json --checkpoint source --queue /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/v3/queue-source.json --output NEW_RUN_OUTPUT --device cuda:0 --teacher
```

Use the corresponding fixed checkpoint queue for trained outputs; never claim/modify proposal queues before launch freeze. Required checks and independent readback are bound by `v3/manifest.json`. Fresh output-layout and diff checks pass. The global knowledge check reports only the two previously disclosed other-owner recurrence-history-location issues (invalid not_authorized list and absent frontier link); no edit was made to those records. Final-config CPU readback verifies reuse of the492-pack cache without rebuilding it. Trainer `data.eval` remains null; native validation is exclusively in the frozen evaluator queues.

Only this new note and new v3 artifacts were authored for this refinement; no implementation, closed candidate, source index, lead state/frontier/catalog or v1/v2 file was changed. No new GPU allocation/model call/scientific update occurred.
