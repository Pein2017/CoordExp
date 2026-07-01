## 1. Investigation
- [x] Confirm current dataset flow in `src/sft.py`, `BaseCaptionDataset`, `FusionCaptionDataset`; note available fields (`length`, grids, pixel values).
- [x] Trace template packing flags and collator expectations (padding_free, position_ids) for VL models.

## 2. Design & Config Wiring
- [x] Define config knobs for packing (enable flag, packing_length default, buffer size, min_fill_ratio, drop_last_packed, allow_single_long).
- [x] Decide pack timing with fusion (post-schedule) and epoch reset behavior.

## 3. Wrapper Implementation Plan
- [x] Specify wrapper API (inputs/outputs), buffering, bin-packing, treatment of unfinished bins, handling of oversized samples.
- [x] Describe metric/logging changes (aggregate only) and batch-size constraints (`per_device_train_batch_size=1`, grad_acc to preserve effective batch).
- [x] Document multimodal merge expectations (pixel_values, grid THW, position_ids for Qwen VL) and truncation masking rules.

## 4. Spec Deltas
- [x] Author `packing-dataset` spec with ADDED requirements and scenarios.
- [x] Validate with `openspec validate add-vl-packing-wrapper --strict`.

## 5. Handoff/Next Steps
- [x] Note follow-up implementation tasks and smoke-test expectations for future apply stage.
- [x] Draft unit test plan (pytest) covering: pack length/fill ratio, carry-over vs drop-last, oversized sample allow/skip, batch-size auto-set, fusion set_epoch rebuild, multimodal collator compatibility, eval default off, metrics scope (aggregate only), and telemetry logs (fill ratios, single-long/skip counts).
