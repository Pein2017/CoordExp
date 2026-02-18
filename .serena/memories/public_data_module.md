# Public Data Runner (Memory)

Role separation:
- Memory role: quick operational contract for public-data intake artifacts.
- Canonical docs: `public_data/README.md`, `docs/data/INTAKE_PIPELINE.md`, `docs/data/JSONL_CONTRACT.md`.
- Update trigger: when runner stages, output layout, or plugin contract changes.

Primary interface:
- `./public_data/run.sh <dataset> <command> [runner-flags] [-- <passthrough-args>]`
- Dataset plugin boundary: `public_data/datasets/<dataset>.sh`

Output contract reminders:
- Raw artifacts under `public_data/<ds>/raw/`.
- Preset artifacts under `public_data/<ds>/<preset>/` include `*.raw.jsonl`, `*.norm.jsonl`, `*.coord.jsonl`.
- Repro metadata includes `pipeline_manifest.json`.
- JSONL image paths must remain relative to JSONL directory.

Operational controls:
- Optional max-object filtering via `PUBLIC_DATA_MAX_OBJECTS=<N>`.
- Validation can be structure-only (`--skip-image-check`) when image assets are unavailable.
