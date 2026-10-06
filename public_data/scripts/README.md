# Offline coordinate conversion

Legacy preset exports distinguish three representations:

- `<split>.jsonl`: pixel-space records after resize/filter;
- `<split>.norm.jsonl`: normalized integers on the 0..999 lattice;
- `<split>.coord.jsonl`: Qwen coordinate tokens on the same lattice.

From the repository root,
`conda run -n ms python -m public_data.scripts.convert_to_coord_tokens --help`
owns the conversion options. Pixel inputs
can produce both normalized and token outputs; `--assume-normalized` skips
pixel scaling for already-normalized inputs. These standalone outputs are
legacy/debug artifacts, not canonical COCO view JSONLs.

The bbox path uses floor/ceil to preserve extents, clamps to the lattice and
nudges collapsed endpoints to enforce strict width/height. Token parsing rejects
malformed or out-of-range tokens. Do not describe bbox conversion as a universal
no-clamping transform or use its nonempty output as proof of original geometry.
Current conversion details belong to the source and its geometry tests.

Canonical `public_data/coco/views/**` JSONLs store norm1000 integers in strict
JSON and resolve `images[]` through `public_data/coco/images/res-1024`.
Rendering them as Qwen coordinate tokens is a template boundary, not a different
stored view format. Training must consume the declared representation without
applying a second normalization.

The COCO 12k view policy counts image patch tokens, system/user tokens and the
rendered assistant sequence. `max_objects` / `max-60` is legacy artifact
provenance rather than this view's length policy.

See [LVIS geometry modes and exporters](../lvis/README.md), the
[public-data runner](../README.md), and the [data contract](../../docs/data/contract.md)
for their separate owners.
