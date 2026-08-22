# Repository ownership index

Use the smallest owner for the surface being changed:

- `src/training/`, `src/packing/`, and `src/artifacts/`: training and
  artifact contracts.
- `src/inference/`, `src/eval/`, `src/vis/`, and their command-line
  entrypoints: inference and direct evaluation.
- `public_data/` and `manifests/public_data_provenance/`: COCO/LVIS data
  preparation and provenance.
- `configs/`, `scripts/`, `docs/`, `openspec/`, and `research/`:
  repository surfaces and current contracts.

When ownership or a compatibility decision is unclear, stop and obtain the
current task owner rather than inferring authority from an old artifact.
