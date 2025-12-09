# Packed dataset run issues (tests/packed_tiny)

## Context
- We’re debugging a post-build smoke run for the packed dataset path. Packing was wired in, but runtime failures block training before any metrics. Goal: get a minimal debug run (tiny LVIS) to complete without touching upstream ms-swift sources.

- **Packed IterableDataset + ms-swift DataLoaderShard conflict**  
  When packing is enabled, the train dataset becomes an `IterableDataset` (PackedCaptionDataset). ms-swift’s trainer always attaches a `BatchSamplerShard` whenever `__len__` exists, and `torch.utils.data.DataLoader` rejects `batch_sampler` for `IterableDataset`, raising:  
  `ValueError: DataLoader with IterableDataset: expected unspecified batch_sampler option`.
  - Config-only workaround: disable packing for this run (`training.packing: false`).  
  - Otherwise needs trainer code change to skip batch_sampler for IterableDataset.

- **IterableDataset requires explicit max_steps**  
  After avoiding the batch_sampler issue (if code were patched), HF requires a finite `args.max_steps` when the dataloader has no reliable length. With packing on and no map-style length, we hit:  
  `ValueError: args.max_steps must be set to a positive value if dataloader does not have a length, was -1`.
  - Config-only workaround: set `training.max_steps: <positive>` when using an IterableDataset.

- **Dataset path sanity**  
  The single-source LVIS tiny JSONLs live under `public_data/lvis/rescale_32_768_poly_20/`. Use those paths (e.g., `train.jsonl` / `val.jsonl`) to avoid missing-file errors encountered with older bbu paths.

- **Next safe config to try (no code edits)**  
  - Turn off packing for this smoke run.  
  - Point to `public_data/lvis/rescale_32_768_poly_20/train.jsonl` and `val.jsonl`.  
  - Keep `per_device_train_batch_size` small for 2 GPUs; no special knobs needed.

## Debug CLI to rerun (no code changes)
```bash
cd /data/home/xiaoyan/AIteam/data/CoordExp
config=tests/packed_tiny gpus=0,1 debug=true bash scripts/train.sh
```
If packing stays on, also set in YAML: `training.max_steps: <small>`; otherwise turn packing off to bypass the IterableDataset batch_sampler conflict.

