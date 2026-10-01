# CoordExp

CoordExp extends Qwen3-VL with coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant matching to push open-vocabulary detection/grounding toward state of the art across public datasets.

## Why
- **Better geometry**: Softmax-on-coordinate-subvocab + expectation gives continuous boxes and smooth gradients (L1/GIoU) without extra detection heads.
- **Order-invariant**: Hungarian/OT matching supervises object sets, not sequences, reducing wasted supervision.
- **Canonical infrastructure**: coordexp-infras owns the active training,
  inference, evaluation, packing, loss, and artifact paths on repository
  `main`. The old MS-Swift-centered implementation is preserved on the
  `ms-swift` archive branch.
- **Dataset focus**: Defaults to single-source JSONL training; multi-dataset training uses offline-merged JSONL; runtime `custom.fusion_config` is dormant in the supported training surface.

## Repo layout
- `src/` - importable CoordExp library code for config loading, datasets,
  training, inference, evaluation, metrics, and visualization helpers.
- `configs/` - YAML-first training, inference, evaluation, benchmark, and
  analysis configs. Current Swift training surfaces live under
  `configs/coordexp_infras/`; older `configs/stage1/` and `configs/stage2/`
  families are compatibility or historical routes.
- `scripts/` - stable user-facing entrypoints plus maintained wrappers and
  utilities. See `scripts/README.md`.
- `public_data/` - dataset tooling and local raw/processed public datasets.
  Processed data reproducibility is recorded in
  `manifests/public_data_provenance/`.
- `docs/` - current operator-facing documentation and standards. Start with
  `docs/README.md`.
- `progress/` - legacy/deprecated archive of old historical notes,
  diagnostics, audits, and benchmark evidence. Do not add new records here;
  migrate or synthesize useful material into `research/`, and never treat this
  tree as current behavior.
- `openspec/` - stable compatibility-sensitive contracts and active contract
  deltas.
- `outputs/` - ignored artifacts owned by a physical worktree. Infrastructure runs
  use `/data/CoordExp/.worktrees/coordexp-infras/outputs`; `/data/CoordExp/outputs/`
  is reserved for explicitly selected shared assets.
- `ops/` - workstation and agent-runtime policy helpers that are not CoordExp
  training, inference, evaluation, or artifact entrypoints.
- `.codex/skills/` - tracked repo-local agent skills. Other `.codex/` runtime
  state is local-only.
- `AGENTS.md` - project instructions for coding agents.

## Quick start

1) **Environment**: activate the `ms` conda environment with the installed
   Transformers/PyTorch runtime. The canonical Swift path does not require a
   sibling `/data/ms-swift` checkout as its application entrypoint.
2) **Expand vocab once** (creates coord tokens 0–999 + optional wildcard and saves a new checkpoint):
   ```bash
   cd .
   python scripts/tools/expand_coord_vocab.py \
     --src /data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct \
     --dst /data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp
   ```
3) **Train (infrastructure smoke example)**:
   ```bash
   cd /data/CoordExp/.worktrees/coordexp-infras
   conda run -n ms python -m src.train \
     --config configs/smoke/eight_gpu_geo_sorted_xy_untied.yaml
   ```
   - For branch-owned runs, enter that branch's physical worktree instead;
     relative training artifact paths resolve from the launch directory.
   - In the infrastructure checkout, use `configs/train/` for training and
     `configs/infer/` with `src.infer` for inference. Config families differ
     across branches; qualify the selected checkout and inputs before launch.
   - The fixed val200 inference/eval run is the accepted V1 validation gate;
     tiny smokes are implementation evidence only.

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

4) **Key Swift config surfaces**
- `custom.emit_norm`: must be `none` (runtime normalization is disabled; training assumes pre-normalized norm1000 coords)
- `custom.coord_tokens.*`: required (`enabled`, `skip_bbox_norm`) to consume pre-quantized coords without double normalization
- `custom.json_format`: required (currently only `standard`; typo-guard for deterministic parsing)
- `custom.object_field_order`: required (`desc_first|geometry_first`); keep train/infer parity with `infer.object_field_order`
- `training.*`: coordexp-infras training settings; backend/runtime derivation is
  recorded in the resolved and effective runtime artifacts.

The old MS-Swift launch commands and legacy config roots remain available on
the `ms-swift` archive branch and in explicitly labeled historical/reference
documentation. They are not the current `main` workflow.

### Token-embeddings adapter tuning (opt-in)
- Purpose: lets role-resolved special token rows learn without touching the rest of the vocab. Adds trainable offsets on `embed_tokens` and `lm_head` for coord IDs 151670-152669 and any compact schema tokens required by the template.
- How to enable:
  ```yaml
  extends: configs/stage1/sft_base.yaml
  training:
    optimizer: multimodal_token_embeddings_adapter
  custom:
    token_embeddings_adapter:
      enabled: true
      groups:
        coord_geometry:
          role: coord_geometry
          start_token: "<|coord_0|>"
          end_token: "<|coord_999|>"
          expected_start: 151670
          expected_end: 152669
      embed_lr: 4.0e-4                      # tune per run
      head_lr: 4.0e-4
      weight_decay: 0.0
      dtype: auto                           # use model dtype by default
  ```
- Saved with the adapter: token offsets live under `token_embeddings_adapter` and are included via `modules_to_save`; no sidecar files.
- Defaults are no-op when `token_embeddings_adapter.enabled: false` and optimizer stays `multimodal`.

### Checkpoint inputs

Use the selected checkout's inference entry and model-composition contract.
Current infrastructure inference consumes adapter and special-token payloads;
see [the infrastructure guide](docs/coordexp_infras.md). The older Swift export
recipe belongs to repository history and is not an entry in the retained
infrastructure checkout.

## Notes
- Uses the model’s native chat templates; no custom tokenizer hacks beyond added coord tokens.
- Keep the expanded checkpoint as your canonical init to avoid token-ID drift.

## License
Pending project decision; inherits upstream licensing until specified.
