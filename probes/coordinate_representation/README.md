# Coordinate representation

This family groups coordinate codebook/alignment, address-assisted readout,
coordinate order/legality and spatial-progress gating. It does not absorb every
experiment that happens to manipulate a coordinate token. Objectives, geometry,
injection sites, vocabularies and their baselines remain explicit in each profile.

`coordinate_codebook_alignment`, `coordinate_address_readout`,
`coordinate_order_knowledge` and `spatial_progress_gate` contain the retained
implementation and independent evaluation/readback paths. Existing study records
remain in the [catalog](../../research/experiments/catalog.jsonl), including
invalidated objectives, negative results and limits of completed comparisons.

## Retired one-time preparation

The closed LR packet builder, seed-repeat packet builder, early-edge packet
builder and old-Mixin loader bridge no longer have current commands. Their
decision-relevant details were extracted into the original study owners:
[alignment preparation](../../research/experiments/2026-09-22-coordinate-codebook-alignment/supporting/retired-preparation-methods.md)
and [early-edge preparation](../../research/experiments/2026-09-23-early-edge-codebook/supporting/retired-preparation-methods.md).
The bridge deliberately had no parity gate; retiring it does not promote its
diagnostics into a passed model-composition claim.

Other code is retained where it participates in the actual profile or independent
CPU verification closure. Presence does not mean every old frozen launch is
compatible with today's source. Current module imports use the new family;
historical schema strings and output/receipt paths intentionally retain their
original spelling. Source/path identity and scientific equivalence are different.

Tests remain next to the corresponding profile. Static imports and CPU fixtures
do not certify current full-model training or permit a new scale-up experiment.
