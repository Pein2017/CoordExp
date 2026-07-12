---
name: coordexp-public-data-provenance
description: Use when auditing, updating, or handing off CoordExp public_data provenance manifests, JSONL-only checksums, active-root cleanup guardrails, or cross-node regeneration from raw datasets.
---

# CoordExp Public Data Provenance

Use this for processed `public_data/` reproducibility. The rule is: raw data is fetched locally, processed data is regenerated from Git-tracked manifests, and routine sync uses JSONL checksums instead of Baidu mirroring.

## Sources Of Truth

- Policy: `docs/standards/OUTPUT_SYNC_AND_DATA_PROVENANCE.md`
- Manifest contract: `manifests/public_data_provenance/README.md`
- Schema: `manifests/public_data_provenance/schema.json`
- Validation: `tests/test_public_data_provenance_manifests.py`

Do not treat `public_data/raw`, image caches, or whole processed trees as normal Baidu sync targets. Use `baidu-netdisk-transfer mode=union-sync` for `outputs/`, not routine `public_data` recovery.

## Workflow

1. Bound the requested roots.
   - Identify the keep-set, delete-set, and any active training roots before touching data.
   - Require fresh explicit user approval for the exact delete-set immediately before destructive cleanup. Prior audit, planning, regeneration, or general cleanup approval is not deletion approval.
   - If a root is active, do not rewrite, move, regenerate, or delete it without a fresh explicit request naming that root and action.
   - Track only materialized roots. If a requested variant was never generated, mark it absent instead of adding a speculative manifest.

2. Map roots to manifests.
   - `public_data/<dataset>/<processed-dir>/` maps to `manifests/public_data_provenance/<dataset>/<processed-dir>.json`.
   - `public_data/<dataset>/images/<image-store>/` maps to an `image_store` manifest with `checksums: null` unless an explicit image-store audit is requested.
   - `public_data/<dataset>/views/<family>/<view>/` maps to an `annotation_view` manifest with sidecar metadata when available.

3. Keep checksum scope narrow.
   - Use `checksums.scope: jsonl_training_samples_only`.
   - Hash model-facing `*.jsonl` files and an aggregate over sorted `path sha256 size_bytes records lines`.
   - Do not hash raw images, resized images, caches, or the whole `public_data` tree for routine cross-node checks.

4. Validate locally.

```bash
python -m pytest tests/test_public_data_provenance_manifests.py -q
```

If the data root is absent, tests may skip local file hashing. If the root exists, every listed JSONL and sidecar must exist and match the manifest.

## Handoff Shape

For another machine, give a pull-first regeneration path:

```text
git pull --ff-only
python -m pytest tests/test_public_data_provenance_manifests.py -q
inspect manifests/public_data_provenance/<dataset>/<artifact>.json
prepare the required raw dataset locally
run the manifest's command for missing processed roots
python -m pytest tests/test_public_data_provenance_manifests.py -q
```

Report `tiny`, `val200`, proxy, or full scope only after naming the exact manifest, data root, and checksum status.

## Cleanup Guardrails

- Use `du -sh`, `/usr/bin/find`, and `git ls-files` to distinguish raw inputs, processed roots, generated caches, and Git-tracked manifests.
- Remove data only after the keep-set is explicit, active roots are excluded,
  and fresh approval names the exact delete-set.
- Leave `.json` manifests in Git; they are the portable contract.
- Keep one-off cleanup scripts under `temp/` and delete them when finished.
