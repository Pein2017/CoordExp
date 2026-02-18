# Data Contract + Dataset Flow (Memory)

Role separation:
- Memory role: retrieval-first reminders for data assumptions in code changes.
- Canonical docs: `docs/data/JSONL_CONTRACT.md`, `docs/data/INTAKE_PIPELINE.md`, `docs/data/README.md`.
- Canonical code paths: `src/datasets/`, `src/datasets/geometry.py`, `src/datasets/builders/jsonlines.py`.
- Update trigger: when JSONL schema, preprocessing steps, or dataset loader behavior changes.

Training assumptions (must hold):
- Contract fields: `images`, `objects`, `width`, `height`.
- One geometry per object (`bbox_2d` or `poly`), with non-empty `desc`.
- Runtime normalization is disabled (`custom.emit_norm: none`), so geometry must already be norm1000 or coord-tokenized.

Runtime flow (single JSONL):
- Loader path: `src/datasets/dense_caption.py`.
- Preprocessor chain may include validation/augmentation/object-capping.
- Optional object ordering + optional coord-token annotation.
- Message build via `src/datasets/builders/jsonlines.py`, then template encode.

Fusion mode:
- Use `custom.fusion_config` only when needed; treat as legacy/experimental path.
- Multi-dataset loading path: `src/datasets/fusion.py` + `src/datasets/unified_fusion_dataset.py`.

Operational note:
- `src/sft.py` auto-sets `ROOT_IMAGE_DIR` from dataset path hints when unset.
