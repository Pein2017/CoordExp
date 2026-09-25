# External data boundary

Processed datasets, original images and annotation workspaces live outside the
tracked research workset. Their recorded identities are in
[research/assets.md](../research/assets.md) and `manifests/public_data_provenance`.
These external bytes and manifests were not changed by cleanup.

Old conversion factories and dataset-specific launchers are not current supported
interfaces. Recover their exact historical source through Git when interpreting
a dataset, not by restoring a pipeline archive into HEAD. New preparation needs
an explicit current algorithm, coordinate/path convention and derived-input
identity. Do not silently substitute similarly named presets or re-export live
annotation labels over a frozen experiment reference.
