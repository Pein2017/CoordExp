# Packed Dataset Plan (CoordExp)

## Goal & Constraints
- Enable packing to reduce padding waste for large/public detection corpora (LVIS/COCO/Objects365, etc.).
- We only need aggregated `loss` / `token_acc` (no per-dataset metrics), so losing dataset labels inside packs is acceptable.
- Keep the existing YAML-driven flow; avoid touching HF/transformers internals. Reuse ms-swift template collator behavior.

## Current Behavior (CoordExp)
- `src/sft.py` builds `BaseCaptionDataset` or `FusionCaptionDataset` (torch `Dataset`), then feeds directly to trainer with padding-only batching.
- Samples carry `length`, `sample_id`, `dataset`, `base_idx` from template.encode (`return_length=True`), but there is **no** precomputed length column or HF `Dataset` view.
- Fusion reshuffles per-epoch via `FusionCaptionDataset.set_epoch`; any wrapper must respect epoch rebuilds.

## What ms-swift Packing Expects
- `swift.llm.dataset.PackingDataset` / `IterablePackingDataset` consume a dataset **with a `length` column** (HF Dataset), bin-pack via `binpacking.to_constant_volume`, and rely on the template collator when `template.packing=True` to concat `input_ids/labels/position_ids`.
- Collator drops most metadata; only core tensors survive (fine for our aggregated metrics use-case).

## Gaps vs. CoordExp
- We have torch datasets (lazy encode) → no columnar lengths.
- Padding-only is baked into docs/specs; configs still set `training.packing: true` but it is effectively ignored.
- Fusion requires epoch-aware rebuilding; packing bins must stay in sync with `set_epoch`.

## Minimal Path to “ms-swift-like” Packing
1) **Wrapper (iterable)**  
   - Add `src/datasets/packing.py` with `IterablePackedDataset` that wraps any torch dataset.  
   - Responsibilities:
     - Set `template.packing = True` and `template.padding_free = True`.
     - Stream samples, keep a buffer of `(sample, length)` (length from encoded dict).
     - Greedy/bin-pack buffered samples up to `packing_length` (use `binpacking.to_constant_volume` if available, else simple first-fit).
     - Yield `List[Dict]` packs; ms-swift collator will flatten and build position_ids.
   - Honor `set_epoch`: when the underlying dataset exposes it (Fusion), forward calls so the schedule rebuilds before packing.

2) **Config surface**  
   - Add `custom.packing` (e.g., `enabled`, `packing_length` defaulting to `template.max_length`, `buffer_size`, optional `strict`).
   - Keep `training.packing` in YAML aligned (set true when custom packing is enabled) to appease ms-swift expectations, but guard in our code so we do not rely on HF Dataset.

3) **sft.py integration**  
   - After dataset construction (base or fusion), if `custom.packing.enabled`: wrap with `IterablePackedDataset`.  
   - Log a warning that per-dataset metrics and sample-level telemetry will be dropped when packing is on.

4) **Metrics**  
   - With packing, only aggregated `loss/token_acc` remain. HSM uses `sample_id`/`dataset`—if HSM is enabled, either disable packing or explicitly copy those fields into the packed rows before yielding (optional toggle; default off since user said per-dataset tracking not needed).

5) **Length accuracy**  
   - Length comes from `template.encode(..., return_length=True)` per sample. For augmented Fusion, lengths can vary per epoch; since we pack after `set_epoch`, the buffer reflects the active schedule/augment RNG for that epoch. No pre-pass needed.

## Risks / Mitigations
- **Spec/doc drift**: Current docs say packing removed. If we proceed, update docs/configs and (if required) run through OpenSpec to reintroduce packing as an opt-in experimental path.
- **Throughput vs. latency**: Buffering (e.g., 512–2048 samples) adds startup latency; keep buffer configurable and small by default (e.g., 512) and consider a background worker if needed.
- **Metadata loss**: Expected/acceptable for this request; flag in logs.
- **Augmentation variance**: Packing decisions are per-epoch; acceptable for infinite/large datasets.

## Concrete Next Steps
- Implement `src/datasets/packing.py` iterable wrapper with buffer+binpack, metadata passthrough optional.
- Wire `custom.packing` in config schema; gate in `sft.py`.
- Add docs note + warning log about dropped per-dataset metrics when packing is on.
- Quick smoke: run a tiny fusion config with `custom.packing.enabled=true`, confirm collator builds packed tensors and trainer runs.

