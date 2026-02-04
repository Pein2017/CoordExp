# CoordExp

CoordExp extends Qwen3-VL with coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant matching to push open-vocabulary detection/grounding toward state of the art across public datasets.

## Why
- **Better geometry**: Softmax-on-coordinate-subvocab + expectation gives continuous boxes and smooth gradients (L1/GIoU) without extra detection heads.
- **Order-invariant**: Hungarian/OT matching supervises object sets, not sequences, reducing wasted supervision.
- **Practical training**: Stays in the standard SFT pipeline (ms-swift), no heavy RL; compatible with native chat templates.
- **Dataset focus**: Supports single-source JSONL or multi-dataset fusion via `custom.fusion_config`.

## Repo layout
- `src/` – training stack (datasets, callbacks, config loader, SFT entry `sft.py`; optional fusion dataset support)
- `configs/` – YAMLs (base, LoRA variants)
- `scripts/` – model utilities (e.g., `scripts/tools/expand_coord_vocab.py`)
- `public_data/scripts/` – data utilities (converters, resize, coord-token conversion)
- `docs/` – documentation index + runbooks; see `docs/README.md` (standards live in `docs/standards/`)
- `docs/standards/CODE_STYLE.md` – code + architecture style guidelines (Transformers-inspired “Option A”)
- `docs/notes/patent/` – background draft for CoordExp method
- `AGENTS.md` – project instructions

## Quick start
1) **Environment**: activate `ms` conda env (`/root/miniconda3/envs/ms`), transformers in that env, ms-swift at `/data/ms-swift`.
2) **Expand vocab once** (creates coord tokens 0–999 + optional wildcard and saves a new checkpoint):
   ```bash
   cd .
   python scripts/tools/expand_coord_vocab.py \
     --src /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct \
     --dst /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp
   ```
3) **Train (example)**:
   ```bash
   python -m src.sft \
     --config configs/dlora/sft_base.yaml \
     --base_config configs/base.yaml
   ```
   - Set `custom.train_jsonl` / `custom.val_jsonl` in the YAMLs to your datasets (single-source).
   - Or set `custom.fusion_config` to a fusion YAML/JSON to train/eval on multiple datasets (see `docs/data/FUSION_DATASET.md`).

### Data prep: LVIS end-to-end (raw → resized JSONL → coord tokens → tiny)
- After `public_data/scripts/download_lvis.py`, run:
  ```bash
  bash public_data/scripts/lvis_full_pipeline.sh
  # or with a larger budget:
  MAX_BLOCKS=1024 bash public_data/scripts/lvis_full_pipeline.sh
  ```
- Outputs land in `public_data/lvis/rescale_<FACTOR>_<MAX_BLOCKS>/`:
  - `{train,val}.jsonl` (smart-resized, polygons capped to `POLY_MAX_POINTS`, grid-aligned to `FACTOR`)
  - `{train,val}.coord.jsonl` (coord tokens)
  - `{train,val}_tiny.jsonl` and `{train,val}_tiny.coord.jsonl` (random `TINY` subset)
- All geometry in the emitted JSONLs is rounded to nearest integers by default, so they are directly safe for `<|coord_*|>` conversion.
- Tunables via env: `FACTOR` (default 32), `MAX_BLOCKS` (pixel budget, default 768), `MIN_BLOCKS` (default 4), `POLY_MAX_POINTS` (default 20), `TINY` (default 256), `NUM_WORKERS`, `RAW_ROOT`, `OUTPUT_BASE`, `SPLITS`.

4) **Key config knobs**
- `custom.emit_norm`: coordinate normalization mode (default `norm1000` uses a 0–999 integer grid; 1000 bins)
- `custom.coord_tokens.*`: opt-in coord-token mode (`enabled`, `skip_bbox_norm`) to consume pre-quantized `<|coord_k|>` data without double normalization
   - `training.*`: ms-swift trainer settings (deepspeed, schedulers, etc.)

### Coord-offset tuning (opt-in)
- Purpose: lets coord token rows learn without touching the rest of the vocab. Adds trainable offsets on `embed_tokens` and `lm_head` for coord IDs 151670–152669 (skips 151669 `<|coord_*|>`).
- How to enable:
  ```yaml
  extends: configs/dlora/sft_base.yaml
  training:
    optimizer: multimodal_coord_offset   # keeps dlora buckets + coord offsets
  custom:
    coord_offset:
      enabled: true
      ids: { start: 151670, end: 152669 }   # override if you changed vocab
      embed_lr: 4.0e-4                      # tune per run
      head_lr: 4.0e-4
      weight_decay: 0.0
      dtype: auto                           # use model dtype by default
  ```
- Saved with the adapter: coord offsets live under `coord_offset_adapter` and are included via `modules_to_save`; no sidecar files.
- Defaults are no-op when `coord_offset.enabled: false` and optimizer stays `multimodal`.

### Merging LoRA + coord offsets (export)
Standard `swift export --merge_lora` drops the coord offsets, so use the helper script that patches shards in-place:
```bash
ADAPTERS=output/debug/coord/<run>/checkpoint-* \
OUTPUT_DIR=output/debug/coord_merged \
GPU_DEVICES=3 \
bash scripts/merge_coord.sh
```
What it does:
- Runs `swift export` to merge LoRA.
- Patches `embed_tokens.weight` and `lm_head.weight` shards with the trained coord offsets (no full model load).
- Rewrites only the affected safetensor shards; final merged model lives in `$OUTPUT_DIR`.
Notes:
- If `$OUTPUT_DIR` already exists, `scripts/merge_coord.sh` will refuse to overwrite it unless you set `ALLOW_OVERWRITE=1`.

## Notes
- Uses the model’s native chat templates; no custom tokenizer hacks beyond added coord tokens.
- Keep the expanded checkpoint as your canonical init to avoid token-ID drift.

## License
Pending project decision; inherits upstream licensing until specified.
