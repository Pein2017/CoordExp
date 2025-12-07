# Packed Dataset Notes (CoordExp)

Goal: enable ms-swift style sequence packing for CoordExp training on public fused datasets, reporting only overall loss/token_acc (no per-dataset metrics), to improve efficiency and reduce padding waste.

## What we found
- `training.packing: true` in YAML currently does nothing because `src/sft.py` builds datasets directly (`BaseCaptionDataset` / `FusionCaptionDataset`) and bypasses ms-swift’s `_post_process_datasets` that wraps with `PackingDataset` (`src/sft.py:350-427`, `src/sft.py:726-788`).
- Each sample is encoded on-the-fly and returned singly; there is no bin-packing or cached `length` column that ms-swift packers expect (`src/datasets/dense_caption.py:214-323`).
- Fusion still emits one encoded record at a time with `dataset`/`base_idx` metadata for mining, but no packing (`src/datasets/unified_fusion_dataset.py:399-551`).
- ms-swift reference packing: `PackingDataset` precomputes `length` and bin-packs indices with `binpacking.to_constant_volume`, then collator flattens via `Template.packing_row`; sets `template.packing = True` and `padding_free = True` (`swift/llm/dataset/utils.py:120-185`, `swift/llm/template/base.py:553-571`).
- Qwen VL template handles packed multimodal position_ids (`swift/llm/template/template/qwen.py:414-456`); compatible with flash-attn and padding_free.

## Gaps to close
- No path today to reuse ms-swift packing; the flag is ignored.
- Datasets lack precomputed `length` and a packed index list.
- Need batch-size guard: packing expects one packed sample per batch item.
- Fusion ratios should apply before packing so packed buckets mirror the mixed schedule.

## Proposed minimal design
1) **Wrapper dataset** `PackedCaptionDataset` (new):
   - Input: a base dataset that already yields encoded dicts with `length`.
   - Build `packed_idx` via the same logic as `PackingDataset.create_packed_idx`.
   - `__getitem__` returns a list of encoded rows; sets `template.packing = True` and `template.padding_free = True` once.
2) **Pre-encode step**:
   - Reuse existing template encode to produce a cached list (or HF Dataset) containing `input_ids`, `labels`, `pixel_values`, `image_grid_thw`, `length`.
   - For fusion, apply ratio sampling first, then encode, then pack over the combined pool (so no per-dataset metrics are needed).
3) **Dataloader contract**:
   - Force `per_device_train_batch_size = 1` when packing to avoid nesting multiple packs.
   - Keep `data_collator = template.data_collator`; it will flatten packed lists via `packing_row`.
4) **Telemetry**:
   - Drop per-dataset logging in this mode; keep only aggregate loss/token_acc (default trainer metrics).

## Implementation sketch
- Add `PackedCaptionDataset` in `src/datasets/wrappers/packed_caption.py` (new file) modeled on ms-swift `PackingDataset`.
- Add a builder utility (e.g., `build_packed_dataset(base_dataset, template, num_proc, packing_length=None)`) to:
  - map `base_dataset` → encoded rows with `length`
  - create pack index
  - return wrapped dataset and updated template flags.
- In `src/sft.py`:
  - If `training.packing` is true, wrap the constructed train dataset with the new builder; set `train_args.per_device_train_batch_size = 1` if not provided; warn if `lazy_tokenize` is enabled.
  - Keep eval un-packed (padding_free=false) unless explicitly requested.
- Guard fusion: run packing after `FusionCaptionDataset` schedule is built so ratios are preserved.

## Open questions for alignment
- Acceptable to pre-encode to an in-memory list vs HF cached dataset? (trade speed vs memory)
- Preferred `packing_length`: use `global_max_length` or a separate knob?
- Keep `dataset/base_idx` fields for hard-sample mining during packed training, or strip them to save memory?
- Should we also support padding_free (no bin-packing) as a fallback knob?

## Ready-to-do checklist (when approved)
- [ ] Implement wrapper dataset + builder.
- [ ] Integrate gating in `src/sft.py` for training.packing.
- [ ] Add config note in `docs` and an example YAML showing packing with fusion.
- [ ] Smoke test a short run to confirm packed batches, no per-dataset metrics, and stable memory.
