# Design: VL Packed Dataset Wrapper

## Goal
Provide a rank-local packing wrapper for vision-language samples (prompt + image → text) that maximizes sequence utilization without data rewrites, while remaining compatible with ms-swift collator behavior.

## Key Decisions
- **Wrapper vs. length cache:** Start with wrapper that consumes already-encoded samples; caching lengths is optional future optimization.
- **Scope:** Training-only packing; evaluation stays un-packed unless explicitly enabled.
- **Rank-local packing:** No cross-rank bin-packing; each DDP rank packs its shard to avoid communication overhead.
- **Batch contract:** `per_device_train_batch_size` forced to 1 when packing is on; effective batch recovered via `gradient_accumulation_steps`.
- **Bin-packing heuristic:** Reuse ms-swift `calculate_matched_group` (binpacking.to_constant_volume) on buffered samples; carry over last underfilled bin; optional drop-last at epoch end.
- **Fill policy:** Min fill ratio (e.g., 0.6–0.7) to avoid tiny packs; cap per-sample length at `packing_length`, allow single-long sample to emit alone with proper masking.
- **Fusion timing:** Build fusion schedule first (respect ratios per epoch), then wrap the resulting dataset; forward `set_epoch` so packing rebuilds after fusion reshuffle.
- **Multimodal merge:** Preserve `pixel_values`, `image_grid_thw`/`video_grid_thw`, `channel`; rely on template.packing padding_free collator to produce correct position_ids (Qwen override keeps mRoPE logic).
- **Truncation safety:** No wrapper-side truncation; depends on `template.encode` max_length/truncation_strategy; labels beyond any cut must be masked (already handled by template.encode).

## Data Flow
1) Base/Fusion dataset yields encoded dicts with `input_ids`, `labels`, `length`, `pixel_values`, grids, etc.
2) Wrapper buffers N samples → bin-packs into groups where sum(length) ≤ packing_length and fill ratio ≥ threshold.
3) Yields `List[Dict]` per packed item; dataloader batch size = 1; ms-swift collator (`padding_free`) flattens and builds position_ids.
4) Trainer sees one packed sample per step; metrics remain aggregate loss/token_acc only.

## Config Surface (proposed)
- `training.packing` (bool): enable wrapper.
- `training.packing_length` (int|null): default template.max_length.
- `training.packing_buffer` (int): default 512.
- `training.packing_min_fill_ratio` (float): default 0.65.
- `training.packing_drop_last` (bool): default true.
- `training.packing_allow_single_long` (bool): default true (emit lone long sample even if > packing_length would have been truncated earlier).

## Validation Hooks
- Log pack fill ratios and % of single-long packs per epoch.
- Warn/auto-set `per_device_train_batch_size=1` when packing enabled.
- Optional smoke test: small fused dataset, verify packed lengths ≤ packing_length and collator output contains position_ids.
