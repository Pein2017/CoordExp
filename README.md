# CoordExp

CoordExp extends Qwen3-VL with coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant matching to push open-vocabulary detection/grounding toward state of the art across public datasets.

## Why
- **Better geometry**: Softmax-on-coordinate-subvocab + expectation gives continuous boxes and smooth gradients (L1/GIoU) without extra detection heads.
- **Order-invariant**: Hungarian/OT matching supervises object sets, not sequences, reducing wasted supervision.
- **Practical training**: Stays in the standard SFT pipeline (ms-swift), no heavy RL; compatible with native chat templates.
- **Dataset scale**: Designed to fuse multiple detection datasets to validate broad generalization.

## Repo layout
- `src/` – training stack (datasets, fusion loader, callbacks, config loader, SFT entry `sft.py`)
- `configs/` – YAMLs (base, LoRA variants, fusion presets)
- `scripts/` – utilities (e.g., `expand_coord_vocab.py` to add coord tokens & resize embeddings)
- `patent/` – background draft for CoordExp method
- `AGENTS.md` – project instructions

## Quick start
1) **Environment**: activate `ms` conda env (`/root/miniconda3/envs/ms`), transformers in that env, ms-swift at `/data/ms-swift`.
2) **Expand vocab once** (creates coord tokens 1–1000 + wildcard and saves a new checkpoint):
   ```bash
   cd /data/home/xiaoyan/AIteam/data/CoordExp
   python scripts/expand_coord_vocab.py \
     --src /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct \
     --dst /data/home/xiaoyan/AIteam/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp
   ```
3) **Train (example)**:
   ```bash
   python -m src.sft \
     --config configs/dlora/sft_base.yaml \
     --base_config configs/base.yaml
   ```
   - Set `custom.train_jsonl` / `custom.val_jsonl` in the YAMLs to your datasets.
   - To train on fused datasets, point `custom.fusion_config` to a file in `configs/fusion/` or your own fusion YAML.

4) **Key config knobs**
   - `custom.emit_norm`: coordinate normalization mode (default `norm1000` for coord tokens)
   - `custom.fusion_config`: enable multi-source training via fusion loader
   - `training.*`: ms-swift trainer settings (deepspeed, schedulers, etc.)

## Notes
- Uses the model’s native chat templates; no custom tokenizer hacks beyond added coord tokens.
- Keep the expanded checkpoint as your canonical init to avoid token-ID drift.

## License
Pending project decision; inherits upstream licensing until specified.
