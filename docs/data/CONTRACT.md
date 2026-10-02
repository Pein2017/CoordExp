# Data identity and geometry

Raw dataset records and rendered model text are different objects. Raw JSONL
remains valid JSON; rendering, tokenization and parsing use the selected
checkout's template and typed configuration. The executable field schema belongs
to `src/data/`, `src/templates/` and the relevant local OpenSpec.

Always distinguish pixel coordinates, normalized coordinates and discrete
coordinate-token IDs. Carry image dimensions and transform identity through
conversion; never infer a unit from an integer's appearance. Clipping, rounding,
resizing, object ordering and token mapping can change the learning problem and
must be explicit, not cosmetic cleanup.

A sample identity, its image bytes and its annotation version are separate
identities. Missing annotations are not evidence of physical absence. Preserve
original annotations and derived edits as distinct versions; do not rewrite an
old run's dataset identity to match newly published labels.

Resolve paths from their declared source and retain original/derived provenance.
A checksum establishes identity, not availability or recoverability. A maintained
input needs a verified copy or a dependency-complete regeneration route; commands
copied from retired source are not that route. See the checkout's
[public-data provenance owner](../../manifests/public_data_provenance/README.md)
and [storage policy](../OUTPUT_STORAGE_POLICY.md).

Packing changes physical layout, not the intended supervision. Preserve causal
isolation, span alignment, loss selection and normalization. Exact implementation
and cache identity are owned by local code and tests, not this explanation.
