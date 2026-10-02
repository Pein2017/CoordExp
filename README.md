# CoordExp

CoordExp extends Qwen3-VL with coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant matching to push open-vocabulary detection/grounding toward state of the art across public datasets.

## Why
- **Better geometry**: Softmax-on-coordinate-subvocab + expectation gives continuous boxes and smooth gradients (L1/GIoU) without extra detection heads.
- **Order-invariant**: Hungarian/OT matching supervises object sets, not sequences, reducing wasted supervision.
- **Canonical infrastructure**: coordexp-infras owns the active training,
  inference, evaluation, packing, loss, and artifact paths integrated on
  repository `main`.
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
- `research/` - earlier OKF-style research source material retained in `main`;
  [its index](research/index.md) routes current maintained research to the
  canonical Research Probes worktree.
- `progress/` - legacy/deprecated archive of old historical notes,
  diagnostics, audits, and benchmark evidence. Do not add new records here or
  treat this tree as current behavior.
- `openspec/` - stable compatibility-sensitive contracts and active contract
  deltas.
- `outputs/` - per-worktree ignored artifacts; the root directory is reserved
  for explicitly selected shared assets. See
  [`docs/OUTPUT_STORAGE_POLICY.md`](docs/OUTPUT_STORAGE_POLICY.md).
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
   - In the infrastructure checkout, use its typed configs under `configs/train/`
     and `configs/infer/`. Config families differ across branches; qualify the
     selected checkout and inputs before launch. See
     [`docs/coordexp_infras.md`](docs/coordexp_infras.md) for the current `main`
     config route and contracts.
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

4) **Current typed configs**: Use the selected checkout's schema-versioned
   YAML configs. The `main` config roots and strict schema are documented in
   [`docs/coordexp_infras.md`](docs/coordexp_infras.md#config-routes) and the
   [config runtime spec](openspec/specs/coordexp-infras-config-runtime/spec.md).
   Current adapter and selected-token payload fields are owned by the
   [adapter spec](openspec/specs/coordexp-infras-adapters-embeddings-optim/spec.md).
   Older MS-Swift commands and `configs/stage1/` examples are historical
   references, not current typed-config examples.

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
