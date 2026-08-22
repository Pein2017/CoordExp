# COCO/LVIS data and provenance

COCO and LVIS are the supported public-data routes. Their preparation,
transformation, validation, and local materialization live under
[`public_data/`](../../public_data/README.md); reproducibility records live
under
[`manifests/public_data_provenance/`](../../manifests/public_data_provenance/README.md).

The current data contract is:

- preserve raw and processed public data;
- make preparation and view generation explicit and reproducible;
- validate JSONL content and provenance before use; and
- keep geometry and coordinate-token conversion aligned with the configured
  training template.

Data preparation remains separate from direct evaluation, and no unrelated
dataset route is part of the default product surface.

See [training and packing](train.md) for the cache boundary and
[operations](operations.md) for validation expectations.
