# CoordExp

CoordExp extends Qwen3-VL with coordinate-specialized tokens to push open-vocabulary detection/grounding across public datasets. Boxes are emitted as quantized norm1000 coordinate tokens (`<|coord_0|>`..`<|coord_999|>`) and supervised at the token level, with no extra detection head.

## Why
- **Coordinate tokens**: Specialized 0..999 coordinate tokens keep geometry on the language surface, so no extra detection head is needed.
- **Token-level supervision**: Base cross-entropy with optional token-type gating and an optional coordinate Gaussian/CRPS-RPS term supervise coordinates without box-regression heads.
- **Canonical infrastructure**: CoordExp-Swift owns the active training,
  inference, evaluation, packing, loss, and artifact paths on repository
  `main`. The old MS-Swift-centered implementation is preserved on the
  `ms-swift` archive branch.
- **Dataset focus**: Defaults to single-source JSONL training; multi-dataset training uses offline-merged JSONL (runtime fusion config authoring is removed from the supported training surface).

> Removed mechanisms: the earlier expectation-based continuous box decoding
> (softmax-on-coordinate-subvocab with L1/GIoU box regression) and order-invariant
> Hungarian/OT set matching were removed in the CoordExp-Swift rebuild. The
> current Swift path supervises coordinate tokens directly with CE, token-type
> gating, and optional Gaussian/RPS, as described in
> [`docs/COORDEXP_SWIFT.md`](docs/COORDEXP_SWIFT.md).

## Repo layout
- `src/` - importable CoordExp library code for config loading, datasets,
  training, inference, evaluation, metrics, and visualization helpers.
- `configs/` - YAML-first training, inference, evaluation, benchmark, and
  analysis configs. Current Swift training surfaces live under
  `configs/coordexp_swift/`; older `configs/stage1/` and `configs/stage2/`
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
- `outputs/` - local experiment artifacts and Baidu Netdisk sync surface. This
  is not source-controlled.
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
3) **Train (canonical Swift example)**:
   ```bash
   conda run -n ms python -m src.train \
     --config configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_2step_warmup0p1_eval_patchproof.yaml
   ```
   - Use `configs/coordexp_swift/prod/` for production-style training and
     `configs/coordexp_swift/infer/` with `src.infer` for inference.
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
- `template.object_field_order`: required (`desc_first|geometry_first`)
- `template.object_ordering`: required (`source_order|geo_sorted|random`)
- `data.train_order`: currently only `source_order` (authored order)
- Training assumes the JSONL is already pre-normalized to norm1000
  (`*.coord.jsonl` coord-token or `*.norm.jsonl` raw-text surfaces); there is
  no runtime normalization switch and no `custom.*` config section.
- `training.*`: CoordExp-Swift training settings; backend/runtime derivation is
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

### Merging LoRA + token-embeddings adapter offsets (export)
Standard `swift export --merge_lora` drops the token-embeddings adapter offsets, so use the helper script that patches shards in-place:
```bash
ADAPTERS=outputs/debug/coord/<run>/checkpoint-* \
OUTPUT_DIR=outputs/debug/coord_merged \
GPU_DEVICES=3 \
bash scripts/merge_coord.sh
```
What it does:
- Runs `swift export` to merge LoRA.
- Patches `embed_tokens.weight` and `lm_head.weight` shards with the trained token-embeddings adapter offsets (no full model load).
- Rewrites only the affected safetensor shards; final merged model lives in `$OUTPUT_DIR`.
Notes:
- If `$OUTPUT_DIR` already exists, `scripts/merge_coord.sh` will refuse to overwrite it unless you set `ALLOW_OVERWRITE=1`.

## Notes
- Uses the model’s native chat templates; no custom tokenizer hacks beyond added coord tokens.
- Keep the expanded checkpoint as your canonical init to avoid token-ID drift.

## License
Pending project decision; inherits upstream licensing until specified.
