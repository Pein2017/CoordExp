# Val18380 Full-Span Slot State Patch

## Scope

This note records the latest completed processing before pausing the research loop. It extends the `18380` false-negative guidance case from a single post-x1/pre-y1 question into a narrow full object-span slot probe.

- Checkpoint: `/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908`
- Rollout source: `/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase`
- Image/sample: COCO val `18380`, target gt9 `wine glass`, target bbox bins `[516, 287, 557, 426]`
- Variants: `original` full generated history versus `no_history`
- Probe slots: y1 after forced `[516]`, x2 after forced `[516, 287]`, y2 after forced `[516, 287, 557]`, and box_end after forced `[516, 287, 557, 426]`

## Artifacts

- Slot-row plan:
  `/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_full_span_state_patch/v1_18380_slot_rows`
- Baseline continuations:
  `/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_full_span_state_patch/v1_18380_slot_baseline_continuation_gpu0`
- State patch panel:
  `/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_full_span_state_patch/v1_18380_nohistory_to_original_slots_y1_x2_y2_layers20_24_27_alpha0_0p5_1_gpu1`

## Baseline Continuation Result

The baseline is already diagnostic. Forcing the true prefix coordinates does not make the full-history `original` row complete the target box.

| Variant | Forced prefix | Target slot | Greedy continuation | Read |
| --- | --- | --- | --- | --- |
| original | `[516]` | y1=`287` | `[946, 526, 999] <box_end>` | catastrophic y1 miss |
| original | `[516, 287]` | x2=`557` | `[546, 999] <box_end>` | y1 guidance is insufficient |
| original | `[516, 287, 557]` | y2=`426` | `[686] <box_end>` | y2 remains poisoned |
| original | `[516, 287, 557, 426]` | box_end | `<box_end>` | closure works if all coords are forced |
| no_history | `[516]` | y1=`287` | `[281, 536, 317] <box_end>` | compact nearby glass basin |
| no_history | `[516, 287]` | x2=`557` | `[536, 319] <box_end>` | target x2/y2 still not exact |
| no_history | `[516, 287, 557]` | y2=`426` | `[319] <box_end>` | lower/nearby basin persists |
| no_history | `[516, 287, 557, 426]` | box_end | `<box_end>` | closure works if all coords are forced |

Interpretation: this is not just a missing y1 onset. The full object span is not represented as a stable target box program. The no-history variant moves away from the full-history bottom-of-image collapse, but it prefers a nearby compact wine-glass basin rather than the gt9 box.

## Same-Slot No-History-to-Original State Patch

Patch setup:

- Receiver: `original`
- Donor: `no_history`
- Slots: y1, x2, y2
- Donor layers: `20,24,27`
- Receiver patch layers: `20,24,27`
- Alpha sweep: `0,0.5,1`
- Rows: `81`, all patch/readout status `ok`

Counts from the patch summary:

- y1: 1/27 rows made the target coordinate the coord-readout top1 and generated the target as first token.
- x2: 0/27 rows made the target coordinate top1 or first token.
- y2: 0/27 rows made the target coordinate top1 or first token.
- Full coordinate tail completion: 0/81 rows.

Best y1 transfer:

- Donor L27 into receiver L24 at alpha `0.5`
- Coord-readout top1: `287`, target rank `1`, margin versus anchor contrast `+0.625`
- Greedy continuation: `<|coord_287|><|coord_546|><|coord_999|><|box_end|>`

This is the cleanest causal transfer in the panel. It moves the receiver from the original y1 collapse (`946`, target rank `949`) to the correct first y1 token. But the continuation then falls back to the original-style x2/y2 tail (`546`, `999`), not the target `[557,426]`.

Best x2 movement:

- Best coord-readout row: donor L20 into receiver L27 at alpha `0.5`
- Coord-readout top1 near target: `560`, target rank `5`, distance `3`
- Greedy continuation still begins with `<|coord_546|>` and then `<|coord_999|><|box_end|>`

Best y2 movement:

- Best rank row: donor L20 into receiver L27 at alpha `1.0`
- Coord-readout top1 `460`, target rank `47`, distance `34`
- Greedy continuation left the clean coordinate grammar path (`Pixels<|object_ref_end|>...`) rather than recovering target y2.

## Current Mechanistic Read

The latest evidence supports a split mechanism:

1. A late-layer coordinate-basin state can causally override the y1 onset. The y1 basin is not immutable; a same-image no-history state can move the original full-history receiver from a bottom collapse to the correct y1 at one tuned patch site.
2. That y1 repair is not a complete object-binding repair. Once the first coordinate is fixed, the continuation still follows the receiver history's x2/y2 geometry (`546,999`) instead of the target box tail (`557,426`).
3. No-history itself is not a perfect visual perception oracle here. It prefers the nearby compact glass `[516,281,536,317]`/slot variants, so the donor state contains a local nearby-object basin, not the full gt9 target state.
4. Closure is intact. Both variants emit `<|box_end|>` correctly after all four true coordinates are forced, so the failure is not wrapper syntax or termination once the coordinate sequence is supplied.

Working hypothesis after this step: prefix history can overwrite the first coordinate basin, but the deeper failure is distributed object-span binding. The model carries slot-local coordinate attractors and nearby-object geometry rather than a persistent latent "selected object" program that reliably conditions all four coordinate slots.

## Stop Point

No further jobs were launched after this patch panel. The helper scripts used for this note remain under ignored `temp/`; the durable committed surface is this note plus the artifact paths above.
