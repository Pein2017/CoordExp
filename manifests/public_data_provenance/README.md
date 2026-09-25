# Public-data identity and recovery contracts

A checksum identifies bytes; it does not establish a restore path. Schema 2
separates **current, executable recovery** from **historical identity only**.
The [storage policy](../../docs/OUTPUT_STORAGE_POLICY.md) is the sole owner of
backup/transfer/retention rules. No processed-data mirror has been verified.

## Current supported assets

| Manifest under this directory | Recovery contract |
|---|---|
| `coco/rescale_32_1024_bbox.json` | Original noncrowd COCO bbox JSONLs and shared resized images, from the three checksum-bound official raw ZIPs. |
| `coco/rescale_32_1024_bbox_len12000.json` | Exact current train/val pixel, norm1000 and coordinate-token JSONLs; raw COCO plus the bound observed annotation delta, with the same shared-image layout. |

All retained production configs and their public-data smoke inputs use the second
asset. The other eight manifests have `support: historical` and no executable
recovery promise: no retained configuration consumes those versions, and no
independent replica was established. Their original checksums/metadata remain
unchanged, and `origin` gives the exact prior manifest Git commit/path/hash.
Historical entries are rejected by regeneration/verification commands instead of
falling back to a deleted producer or silently substituting a current dataset.

## Why the current view has a distinct version

On 2026-09-25 the two base JSONLs matched their original manifest, but the six
consumed len12000 files did not. Besides compact JSON serialization, the current
view contains added/removed annotations and revised boxes/order. Replacing them
with a raw-only regeneration would lose actual training input information.

`public_data/coco_annotation_delta.json` is a necessary **current data input**,
not an archive or a new scientific label admission. It stores only observed
object edits and order for affected records, with before/after content hashes.
The curated manifest binds its exact hash. The previous manifest remains
recoverable through `origin`; old receipts and all existing external files were
left untouched. Do not use the current version to rewrite a historical score.

The original frozen raw corpus dropped zero images at the recorded 12k limit.
The replacement recovers those exact image identities, then reapplies the observed
annotation edits. It is **not** a general length-budget filter, tokenizer, proxy
label generator or arbitrary-corpus factory. Whole-output checksums reject any
different corpus or conversion. A different view requires separate qualification.

## Restore on a new node

Prepare `annotations_trainval2017.zip`, `train2017.zip`, and `val2017.zip` from
the official source URLs recorded in `recovery.raw_archives`, or from a separately
verified mirror. Their full SHA-256 and sizes are mandatory. No authentication
material is stored here. The recovery command itself does not download anything.
Install the exact Python/Pillow/libjpeg-turbo versions declared by the manifest.
The contract checker additionally requires the `jsonschema` Python package; use
the normal project Python environment for the production-reader smoke. `zipfile`,
JSON and hashing use the Python standard library. No legacy training config,
mapping CSV, tokenizer or model checkpoint is needed.

From a checkout containing this implementation:

```bash
python -m public_data.recover_coco check
python -m public_data.recover_coco regenerate \
  --manifest manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json \
  --raw-archives /path/to/coco-zip-inputs \
  --destination /path/to/absent-workspace --dry-run
```

Dry-run validates all three archive hashes, implementation/data dependencies,
environment and six resized-image byte canaries, without creating the destination.
For explicitly authorized full restoration, run the same command without
`--dry-run`. It reads official annotation/image members directly, writes the
selected JSONLs and shared images under the destination workspace, verifies all
expected JSONL hashes and publishes `recovery.json` only on success. A failure
leaves `.recovery-incomplete`; never use that directory as accepted input.
Existing destinations are not overwritten or repaired in place.

```bash
python -m public_data.recover_coco verify \
  --manifest manifests/public_data_provenance/coco/rescale_32_1024_bbox_len12000.json \
  --workspace /path/to/restored-workspace
```

Verification hashes every declared JSONL, invokes the real reader on bounded
coordinate rows, opens their images at the declared dimensions, and verifies
the image canaries. It is not a full image checksum census. Use the restored
absolute input paths in a newly resolved training config; no script rewrites
existing configs or substitutes data into an old run.

`reconstruct-jsonl --manifest ... --raw-archives ...` is a separate read-only
whole-JSONL reconstruction check: it reads only the annotation ZIP and delta,
writes no images/data, and explicitly does not qualify image archives.

## Contract maintenance

`python -m public_data.recover_coco check` validates every schema and current
producer/delta dependency. Tests also require every retained COCO config input
to have a current recovery owner and preserve each historical manifest origin.
A passing static check is not raw-input availability, byte reconstruction or
new-node restoration; report those scopes separately.
